from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
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
MAX_STEPS = 6

# 정답을 뱉는 문장(터미널). llms/utils.py 의 parser() 와 같은 마커지만,
# 마크다운 변형(### / ** 접두, **...is**: 처럼 볼드가 콜론 앞에서 닫힘)과
# 곡선 아포스트로피까지 흡수한다.
_APOS = r"['’ʼ´`]"
_SEP = r"[\s*_]+"                        # 단어 사이에 낀 볼드 마커/공백
TERMINAL_PAT = {
    "plan": re.compile(
        rf"(?i)the{_SEP}llm\s*{_APOS}?\s*s{_SEP}action{_SEP}sequence{_SEP}is\s*\**\s*:"),
    "predict": re.compile(
        rf"(?i)the{_SEP}agent\s*{_APOS}?\s*s{_SEP}final{_SEP}state{_SEP}is\s*\**\s*:"),
    "decompose": re.compile(r"(?i)<+\s*start\s*>+"),
}
END_PAT = re.compile(r"(?i)<+\s*end\s*>+")     # decompose 전용


def _line_start(text: str, i: int) -> int:
    return text.rfind("\n", 0, i) + 1


def step_char_bounds(text: str, task: str) -> tuple[list[int] | None, str | None]:
    """출력 텍스트를 구간 경계(문자 인덱스)로 쪼갠다.

    성공: (bounds, None). bounds[0] == 0, bounds[-1] == len(text) 이고
          마지막 구간이 정답을 뱉는 터미널 문장, 그 앞이 Step 1..N.
    실패: (None, 사유) — 호출부에서 그 궤적을 통째로 skip.

    깨끗한 궤적만 남긴다. 헤더가 없거나, Step 1 앞에 서문이 있거나,
    스텝 번호가 1,2,3.. 으로 안 이어지거나, 스텝이 MAX_STEPS 를 넘으면 버린다.

    정규식은 출력 텍스트에만 건다. 프롬프트에도 "Step 1." 류 목차가 들어 있어서
    (cot_*.py 의 프롬프트가 6단계 지시문) 이어붙인 전체 텍스트에 걸면 프롬프트
    쪽 헤더가 먼저 잡힌다.
    """
    if not text or not text.strip():
        return None, "empty_output"

    try:
        term_pat = TERMINAL_PAT[task]
    except KeyError:
        raise ValueError(f"no terminal marker defined for task {task!r}") from None

    marks = list(term_pat.finditer(text))
    if not marks:
        return None, "no_terminal_marker"
    # 마커가 여러 번 나오면 마지막 것. 마커가 걸린 줄을 통째로 터미널에 준다
    # (### / ** 접두가 마지막 스텝 쪽에 남지 않게).
    term = _line_start(text, marks[-1].start())

    if task == "decompose":
        ends = list(END_PAT.finditer(text, marks[-1].end()))
        if not ends:
            return None, "no_end_marker"
        if text[ends[-1].end():].strip():
            return None, "text_after_end"

    heads = list(STEP_PAT.finditer(text))
    if not heads:
        return None, "no_step_header"
    if text[:heads[0].start()].strip():
        return None, "preamble"

    nums = [int(m.group(1)) for m in heads]
    if nums != list(range(1, len(nums) + 1)):   # Step 1,2,4 처럼 끊기면 버린다
        return None, "step_sequence_break"
    if len(nums) > MAX_STEPS:
        return None, "too_many_steps"

    starts = [_line_start(text, m.start()) for m in heads]
    starts[0] = 0                               # 첫 헤더 앞 공백은 Step 1 이 흡수
    if starts[-1] >= term:                      # 헤더와 정답 마커가 같은 줄 → 길이 0 스텝
        return None, "step_in_terminal"

    bounds = starts + [term, len(text)]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        return None, "empty_segment"
    return bounds, None


def char_to_token_bounds(char_bounds: list[int], offsets) -> list[int]:
    tok_bounds, k = [], 0
    for cb in char_bounds:
        while k < len(offsets) and offsets[k][0] < cb:
            k += 1
        tok_bounds.append(k)
    return tok_bounds


# ---------------------------------------------------------------- tokenization

def render_prompt(episode: dict) -> str:
    """생성 시점에 모델이 실제로 본 입력을 재현한다.

    generation/scripts/cot_*.py 는 apply_chat_template(add_generation_prompt=True)
    결과를 vLLM 에 넘기고 jsonl 에는 raw 프롬프트만 저장한다. 그대로 쓰면
    <|im_start|> 류 특수 토큰과 (no_thinking 의) 빈 <think> 블록이 통째로 빠진다.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": episode["prompt"]}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(episode.get("thinking", False)),
    )


def tokenize_episode(episode: dict, char_bounds: list[int]):
    """prompt + output 을 통째로 토크나이즈하고 토큰 경계를 만든다.

    boundaries 규약 (길이 N+3):
        [0, 프롬프트 끝, step1 끝, ..., stepN 끝, 전체 끝]
    즉 첫 구간 = 프롬프트, 마지막 구간 = 정답 문장. 호출부가 이 규약을 안다고 가정한다.

    반환: (input_ids, tok_bounds, 사유). 사유가 있으면 그 궤적은 skip.
    """
    ensure_model()
    prompt = render_prompt(episode)
    output = episode["all_llm_output"]

    enc = tok(prompt + output, add_special_tokens=False, return_offsets_mapping=True)

    # 이어붙인 문자열을 한 번에 토크나이즈하므로 프롬프트 끝 글자와 출력 첫 글자가
    # 한 토큰으로 묶일 수 있다. 그러면 경계를 토큰 단위로 못 그어서 출력 첫 토큰이
    # 조용히 프롬프트 구간에 먹힌다. 템플릿이 \n\n 로 끝나 보통은 안 생기지만 검사한다.
    if any(s < len(prompt) < e for s, e in enc.offset_mapping):
        return None, None, "boundary_straddle"

    abs_bounds = [0] + [len(prompt) + b for b in char_bounds]   # char_bounds[0] == 0
    tok_bounds = char_to_token_bounds(abs_bounds, enc.offset_mapping)
    assert tok_bounds[-1] == len(enc.input_ids), "unconsumed tokens"
    return enc.input_ids, tok_bounds, None


# ------------------------------------------------------------------- extractor

@torch.no_grad()
def extract_hidden(ids):
    ensure_model()
    H = model(torch.tensor([ids], device=model.device)).last_hidden_state[0]
    return H.to(torch.bfloat16).cpu().clone()   # 프롬프트 구간 포함 전체


# ------------------------------------------------------------------------ meta

def build_meta(episode: dict) -> dict:
    return {k: v for k, v in episode.items() if k not in HEAVY_FIELDS}


# 앞에 있는 키를 먼저 쓴다 — decompose→PR, predict→success, plan→CR.
LABEL_PRIORITY = ("PR", "success", "CR")


def episode_status(episode: dict) -> tuple[str | None, str | None]:
    """(status, 판정에 쓴 키). 판정 불가면 (None, None).

    PR 을 먼저 보는 이유: decompose 의 CR 은 봇이 서브골을 추가해서라도 완주하면
    1 이라 "성공" 안에 LLM 분해가 불완전한 궤적이 섞인다 (GoTo 에서 CR 91% vs
    PR 15%). PR 은 봇 추가 0회만 성공으로 친다.

    eval_error 가 있는 궤적은 호출부에서 이미 걸렀다고 가정한다. CR·ACI 등
    나머지 지표는 meta 의 eval_result 에 그대로 남으므로 나중에 재분류할 수 있다.
    """
    r = episode.get("eval_result") or {}
    for key in LABEL_PRIORITY:
        if key in r:
            ok = bool(r[key]) if key == "success" else r[key] == 1
            return ("success" if ok else "failure"), key
    return None, None


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
    chunk: int = 256,
):

    src = data_dir / f"{task}_{mode}.jsonl"
    all_episodes = load_episodes(src)

    episodes = [e for e in all_episodes
                if e.get("task") == task and e.get("env_name") == level]
    if not episodes:
        raise ValueError(f"no episodes with task == {task!r} and env_name == {level!r} "
                         f"in {src} ({len(all_episodes)} loaded total)")

    run_dir = out_root / task / level
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
        torch.save({"episodes": buffers[status], "model": MODEL,
                    "boundary_layout": "prompt|steps|terminal",
                    "label_priority": LABEL_PRIORITY}, out_path)
        buffers[status] = {}
        chunk_idx[status] += 1

    n_saved = n_skipped = 0
    reasons = Counter()

    with meta_path.open("w", encoding="utf-8") as mf:

        def skip(meta, reason):
            nonlocal n_skipped
            meta["extract_skipped"] = reason
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            reasons[reason] += 1
            n_skipped += 1

        for episode in tqdm(episodes, desc="episodes", unit="episode"):
            meta = build_meta(episode)
            meta["src"] = src.name

            if episode.get("skipped"):               # 출력 자체가 없는 에피소드
                skip(meta, "no_output")
                continue

            if episode.get("truncated"):             # max_tokens 에서 잘림 → 마지막 구간 불완전
                skip(meta, "truncated")
                continue

            if episode.get("eval_error") is not None:
                # 파싱 실패 / 봇 실행 예외 / eval 타임아웃이 한 필드에 섞여 있다.
                # 타임아웃은 봇 리플랜 루프가 안 끝나는 환경 문제라 LLM 실패가
                # 아니고, 파싱 실패는 어차피 step_char_bounds 가 거른다. 통째로 뺀다.
                skip(meta, "eval_error")
                continue

            if not (episode.get("prompt") or "").strip():
                skip(meta, "empty_prompt")           # 프롬프트 구간이 길이 0이 된다
                continue

            status, label_key = episode_status(episode)
            if status is None:
                skip(meta, "no_label")               # eval_result 에 판정 키가 없다
                continue
            meta["status"] = status
            meta["label_key"] = label_key

            # 구조 검사는 토크나이즈 전에 — 안 맞는 궤적엔 모델을 안 태운다
            char_bounds, reason = step_char_bounds(episode["all_llm_output"], task)
            meta["output_sha1"] = hashlib.sha1(
                (episode["all_llm_output"] or "").encode()).hexdigest()
            if reason is not None:
                skip(meta, reason)
                continue

            ids, boundaries, reason = tokenize_episode(episode, char_bounds)
            if reason is not None:
                skip(meta, reason)
                continue
            meta.update(
                n_tokens_total=len(ids),
                n_tokens_prompt=boundaries[1],
                n_tokens_output=boundaries[-1] - boundaries[1],
                n_tokens_terminal=boundaries[-1] - boundaries[-2],
                n_steps=len(boundaries) - 3,         # 프롬프트·터미널 제외
            )

            if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
                skip(meta, "empty_token_segment")    # 문자로는 갈렸는데 토큰으로 뭉개진 구간
                continue
            if len(ids) > MAX_TOKENS:
                skip(meta, "too_long")
                continue

            E = extract_hidden(ids)
            assert E.shape[0] == boundaries[-1]

            seed = episode["env_seed"]
            if seed in seen_seeds[status]:
                raise ValueError(f"id collision: {task}/{level}/{status} seed={seed}")
            seen_seeds[status].add(seed)
            buffers[status][seed] = {
                "E": E,                          # (프롬프트+출력 토큰수) x d
                "boundaries": boundaries,        # [0, prompt, step1..N, terminal 끝]
                "output_sha1": meta["output_sha1"],
            }
            meta["chunk"] = chunk_idx[status]        # 이 episode가 들어갈 청크 파일 인덱스
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_saved += 1

            if len(buffers[status]) >= chunk:
                flush(status)

    for status in buffers:
        flush(status)

    n_success = sum(1 for s in seen_seeds["success"] for _ in (0,))
    print(f"saved {n_saved} episodes under {run_dir} ({n_skipped} skipped)")
    print(f"  success {len(seen_seeds['success'])} / failure {len(seen_seeds['failure'])}"
          f"  [label_priority={LABEL_PRIORITY}]")
    for reason, n in reasons.most_common():
        print(f"  skipped {reason}: {n}")


# ----------------------------------------------------------------- load helper

def load_chunk(pt_path: Path) -> dict:

    return torch.load(pt_path, map_location="cpu", weights_only=False)["episodes"]


def load_all_views(episode: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """[프롬프트, step1..stepN, 터미널] 전 구간."""
    E, b = episode["E"], episode["boundaries"]
    return E, [E[s:e] for s, e in zip(b, b[1:])]


def load_step_views(episode: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """스텝 구간만 — 프롬프트(첫 구간)와 터미널(마지막 구간)은 뺀다."""
    E, b = episode["E"], episode["boundaries"]
    return E, [E[s:e] for s, e in zip(b[1:-2], b[2:-1])]