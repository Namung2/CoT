from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from segment import split_steps
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


# ---------------------------------------------------------------- tokenization

def tokenize_episode(episode: dict, use_prompt_context: bool):
    """원문을 1회 토큰화하고 step 경계를 토큰 좌표로 매핑.

    반환:
      ids        : forward에 넣을 전체 토큰 (prompt 포함 여부는 flag에 따름)
      ctx        : output 시작 전 컨텍스트 토큰 수 (prompt 미사용 시 0)
      boundaries : 길이 T+1, output 토큰 기준 step 경계.
                   step t = E[boundaries[t]:boundaries[t+1]]
    """
    ensure_model()
    output = episode["all_llm_output"]
    steps = split_steps(output)
    assert "".join(steps) == output

    prompt = episode["prompt"] if use_prompt_context else ""
    text = prompt + output

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping

    # char 경계(누적 길이) → token 경계. 토큰의 시작 문자가 속한 구간에 배정.
    char_bounds = [len(prompt)]
    for s in steps:
        char_bounds.append(char_bounds[-1] + len(s))

    tok_bounds, k = [], 0
    for cb in char_bounds:
        while k < len(ids) and offs[k][0] < cb:
            k += 1
        tok_bounds.append(k)
    assert tok_bounds[-1] == len(ids), "unconsumed tokens"

    ctx = tok_bounds[0]                              # prompt 토큰 수
    boundaries = [b - ctx for b in tok_bounds]       # output 기준으로 shift
    return ids, ctx, boundaries


# ------------------------------------------------------------------ extractors

@torch.no_grad()
def extract_full_sequence(ids, ctx, boundaries):
    """전체 1회 forward. output 구간만 저장 → E: (output 토큰수) x d."""
    ensure_model()
    H = model(torch.tensor([ids], device=model.device)).last_hidden_state[0]
    return H[ctx:].to(torch.bfloat16).cpu().clone()


@torch.no_grad()
def extract_cumulative_prefix(ids, ctx, boundaries):
    """step t마다 prefix(ids[:ctx+b_{t+1}])로 forward, 해당 step 구간만 누적.

    full_sequence와 동일한 ids/boundaries를 쓰므로 토큰열이 완전히 일치.
    반환 shape은 동일하게 (output 토큰수) x d이나, 행마다 forward된
    prefix 길이가 다르다는 점은 분석 시 해석에 반영할 것.
    """
    ensure_model()
    rows = []
    for s, e in zip(boundaries, boundaries[1:]):
        H = model(
            torch.tensor([ids[: ctx + e]], device=model.device)
        ).last_hidden_state[0]
        rows.append(H[ctx + s : ctx + e].to(torch.bfloat16).cpu())
    return torch.cat(rows, dim=0)


EXTRACTORS = {
    "full_sequence": extract_full_sequence,
    "cumulative_prefix": extract_cumulative_prefix,
}


# ------------------------------------------------------------------------ meta

def build_meta(episode: dict) -> dict:
    """heavy 텍스트 필드만 빼고 전부 복사 (task별 확장 필드 자동 포함)."""
    return {k: v for k, v in episode.items() if k not in HEAVY_FIELDS}


def episode_status(episode: dict) -> str:
    """성공/실패 → 저장 디렉토리 분기.

    eval_result 포맷이 task마다 다름:
      - {"success": bool, ...}  → success 그대로 사용
      - {"CR": float, ...}      → CR == 1 을 성공으로 판정 (기준 바뀌면 여기 수정)
    """
    r = episode.get("eval_result") or {}
    if "success" in r:
        return "success" if r["success"] else "failure"
    if "CR" in r:
        return "success" if r["CR"] == 1 else "failure"
    raise ValueError(f"cannot determine status from eval_result: {r!r}")


# ------------------------------------------------------------------------- run

def load_episodes(data_dir: Path):
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"no such directory: {data_dir}")
    episodes = []
    for path in sorted(data_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append((path, json.loads(line)))
    return episodes


def extract_run(
    data_dir: Path,
    out_root: Path,
    task: str,
    level: str,
    method: str,
    use_prompt_context: bool = True,
):
    if method not in EXTRACTORS:
        raise ValueError(f"unknown method: {method!r} (choose from {list(EXTRACTORS)})")
    extractor = EXTRACTORS[method]

    all_episodes = load_episodes(data_dir)
    episodes = [(p, e) for p, e in all_episodes
               if e.get("task") == task and e.get("env_name") == level]
    if not episodes:
        raise ValueError(f"no episodes with task == {task!r} and env_name == {level!r} "
                         f"under {data_dir} ({len(all_episodes)} loaded total)")

    ctx_tag = "with_prompt" if use_prompt_context else "no_prompt"
    run_dir = out_root / task / level / method / ctx_tag
    for status in ("success", "failure"):
        (run_dir / status).mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.jsonl"

    n_saved = n_skipped = 0
    with meta_path.open("w", encoding="utf-8") as mf:
        for path, episode in tqdm(episodes, desc="episodes", unit="episode"):
            meta = build_meta(episode)
            meta["src"] = path.name

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

            E = extractor(ids, ctx, boundaries)
            assert E.shape[0] == boundaries[-1]

            out_path = run_dir / status / f"{episode['task']}_{episode['env_name']}_{episode['env_seed']}.pt"
            if out_path.exists():
                raise FileExistsError(f"id collision: {out_path}")
            torch.save(
                {
                    "E": E,                          # (output 토큰수) x d
                    "boundaries": boundaries,        # 길이 T+1, output 토큰 기준
                    "output_sha1": meta["output_sha1"],
                    "method": method,
                    "use_prompt_context": use_prompt_context,
                    "model": MODEL,
                },
                out_path,
            )
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_saved += 1

    print(f"saved {n_saved} episodes under {run_dir} ({n_skipped} skipped)")


# ----------------------------------------------------------------- load helper

def load_step_views(pt_path: Path) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """저장된 N x d 행렬과, boundaries로 자른 step별 view 리스트를 반환.

    view라서 복사 비용 없음. E_t = views[t] (n_t x d).
    """
    blob = torch.load(pt_path)
    E, b = blob["E"], blob["boundaries"]
    return E, [E[s:e] for s, e in zip(b, b[1:])]