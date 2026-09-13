from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen3-32B"
MAX_TOKENS = 32768                       # 초과 시 skip (meta에 기록)
HEAVY_FIELDS = ("prompt", "all_llm_output", "parsed_llm_output")

tok = None
model = None


def ensure_model():
    global tok, model
    if model is None:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(  # lm_head 없음 → logits 미계산
            MODEL, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        print("device:", model.device)
    return tok, model


# ------------------------------------------------------------- step boundaries

STEP_PAT = re.compile(r"(?mi)^(?:#+\s*|\*+\s*)?Step\s*(\d+)\s*[.:]")

def step_char_bounds(text: str) -> list[int]:
    if not text.strip():
        return [0, len(text)]

    starts = [m.start() for m in STEP_PAT.finditer(text)]
    if not starts:
        return [0, len(text)]
    if text[:starts[0]].strip():
        starts = [0] + starts
    else:
        starts[0] = 0
    return starts + [len(text)]


def char_to_token_bounds(char_bounds: list[int], offsets) -> list[int]:
    tok_bounds, k = [], 0
    for cb in char_bounds:
        while k < len(offsets) and offsets[k][0] < cb:
            k += 1
        tok_bounds.append(k)
    return tok_bounds


# ---------------------------------------------------------------- tokenization

def tokenize_episode(episode: dict, use_prompt_context: bool):
 
    ensure_model()
    prompt = episode["prompt"] if use_prompt_context else ""
    output = episode["all_llm_output"]

    enc = tok(prompt + output, add_special_tokens=False, return_offsets_mapping=True)

    char_bounds = [len(prompt) + b for b in step_char_bounds(output)]
    tok_bounds = char_to_token_bounds(char_bounds, enc.offset_mapping)
    assert tok_bounds[-1] == len(enc.input_ids), "unconsumed tokens"

    ctx = tok_bounds[0]                              # prompt 토큰 수
    boundaries = [b - ctx for b in tok_bounds]       # output 기준으로 shift
    return enc.input_ids, ctx, boundaries


# ------------------------------------------------------------------- extractor

@torch.no_grad()
def extract_hidden(ids, ctx):
    ensure_model()
    H = model(torch.tensor([ids], device=model.device)).last_hidden_state[0]
    return H[ctx:].to(torch.bfloat16).cpu().clone()


# ------------------------------------------------------------------------ meta

def build_meta(episode: dict) -> dict:
    return {k: v for k, v in episode.items() if k not in HEAVY_FIELDS}


def episode_status(episode: dict) -> str:

    r = episode.get("eval_result") or {}
    if "success" in r:
        return "success" if r["success"] else "failure"
    if "CR" in r:
        return "success" if r["CR"] == 1 else "failure"
    if episode.get("eval_error") is not None:
        return "failure"
    raise ValueError(f"cannot determine status from eval_result: {r!r}")


# ------------------------------------------------------------------------- run

def load_episodes(path: Path) -> list[dict]:

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    episodes = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def extract_run(
    data_dir: Path,
    out_root: Path,
    task: str,
    level: str,
    mode: str = "no_thinking",
    use_prompt_context: bool = True,
    chunk: int = 256,
):

    src = data_dir / f"{task}_{mode}.jsonl"
    all_episodes = load_episodes(src)
    
    episodes = [e for e in all_episodes
                if e.get("task") == task and e.get("env_name") == level]
    if not episodes:
        raise ValueError(f"no episodes with task == {task!r} and env_name == {level!r} "
                         f"in {src} ({len(all_episodes)} loaded total)")

    ctx_tag = "with_prompt" if use_prompt_context else "no_prompt"
    run_dir = out_root / task / level / ctx_tag
    for status in ("success", "failure"):
        d = run_dir / status
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("chunk_*.pt"):   # 재실행 시 이전 청크 개수와 안 맞게 남는 것 방지
            old.unlink()
    meta_path = run_dir / "meta.jsonl"

    buffers = {"success": {}, "failure": {}}
    chunk_idx = {"success": 0, "failure": 0}
    seen_seeds = {"success": set(), "failure": set()}

    def flush(status):
        if not buffers[status]:
            return
        out_path = run_dir / status / f"chunk_{chunk_idx[status]:04d}.pt"
        torch.save({"episodes": buffers[status],
                    "use_prompt_context": use_prompt_context, "model": MODEL}, out_path)
        buffers[status] = {}
        chunk_idx[status] += 1

    n_saved = n_skipped = 0
    with meta_path.open("w", encoding="utf-8") as mf:
        for episode in tqdm(episodes, desc="episodes", unit="episode"):
            meta = build_meta(episode)
            meta["src"] = src.name

            if episode.get("skipped"):               # 출력 자체가 없는 에피소드
                meta["extract_skipped"] = "no_output"
                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                n_skipped += 1
                continue

            status = episode_status(episode)
            meta["status"] = status

            ids, ctx, boundaries = tokenize_episode(episode, use_prompt_context)
            meta.update(
                n_tokens_total=len(ids),
                n_tokens_output=boundaries[-1],
                n_steps=len(boundaries) - 1,
                output_sha1=hashlib.sha1(
                    episode["all_llm_output"].encode()
                ).hexdigest(),
            )

            if len(ids) > MAX_TOKENS:
                meta["extract_skipped"] = "too_long"
                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                n_skipped += 1
                continue

            E = extract_hidden(ids, ctx)
            assert E.shape[0] == boundaries[-1]

            seed = episode["env_seed"]
            if seed in seen_seeds[status]:
                raise ValueError(f"id collision: {task}/{level}/{status} seed={seed}")
            seen_seeds[status].add(seed)
            buffers[status][seed] = {
                "E": E,                          # (output 토큰수) x d
                "boundaries": boundaries,        # 길이 T+1, output 토큰 기준
                "output_sha1": meta["output_sha1"],
            }
            meta["chunk"] = chunk_idx[status]        # 이 episode가 들어갈 청크 파일 인덱스
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_saved += 1

            if len(buffers[status]) >= chunk:
                flush(status)

    for status in buffers:
        flush(status)

    print(f"saved {n_saved} episodes under {run_dir} ({n_skipped} skipped)")


# ----------------------------------------------------------------- load helper

def load_chunk(pt_path: Path) -> dict:

    return torch.load(pt_path, map_location="cpu", weights_only=False)["episodes"]


def load_step_views(episode: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:

    E, b = episode["E"], episode["boundaries"]
    return E, [E[s:e] for s, e in zip(b, b[1:])]