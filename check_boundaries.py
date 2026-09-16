"""추출 파이프라인의 토큰 경계 점검.

모델은 안 태운다 (토크나이저만 필요). extract.py 와 같은 방식으로 토크나이즈한 뒤
  1. 특수 토큰의 offset 이 (0,0) 인지 실제 문자 범위인지
  2. 프롬프트/출력 경계가 토큰 단위로 정확히 갈리는지
  3. 각 step 경계가 "Step N" 헤더 직전에 놓이는지
를 눈으로 확인한다.

    python check_boundaries.py generation/trajectory/decompose_no_thinking.jsonl
    python check_boundaries.py <jsonl> --task decompose --n 5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

# ---- extract.py 에서 그대로 가져온 부분 (import 안 되는 환경도 있어서 복제) ----

STEP_PAT = re.compile(r"(?mi)^(?:#+\s*|\*+\s*)?Step\s*(\d+)\s*[.:]")
MAX_STEPS = 6

_APOS = r"['’ʼ´`]"
_SEP = r"[\s*_]+"
TERMINAL_PAT = {
    "plan": re.compile(
        rf"(?i)the{_SEP}llm\s*{_APOS}?\s*s{_SEP}action{_SEP}sequence{_SEP}is\s*\**\s*:"),
    "predict": re.compile(
        rf"(?i)the{_SEP}agent\s*{_APOS}?\s*s{_SEP}final{_SEP}state{_SEP}is\s*\**\s*:"),
    "decompose": re.compile(r"(?i)<+\s*start\s*>+"),
}
END_PAT = re.compile(r"(?i)<+\s*end\s*>+")


def _line_start(text: str, i: int) -> int:
    return text.rfind("\n", 0, i) + 1


def step_char_bounds(text, task):
    if not text.strip():
        return None, "empty_output"
    term_pat = TERMINAL_PAT[task]
    marks = list(term_pat.finditer(text))
    if not marks:
        return None, "no_terminal_marker"
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
    if nums != list(range(1, len(nums) + 1)):
        return None, "step_sequence_break"
    if len(nums) > MAX_STEPS:
        return None, "too_many_steps"
    starts = [_line_start(text, m.start()) for m in heads]
    starts[0] = 0
    if starts[-1] >= term:
        return None, "step_in_terminal"
    bounds = starts + [term, len(text)]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        return None, "empty_segment"
    return bounds, None


def char_to_token_bounds(char_bounds, offsets):
    tok_bounds, k = [], 0
    for cb in char_bounds:
        while k < len(offsets) and offsets[k][0] < cb:
            k += 1
        tok_bounds.append(k)
    return tok_bounds


# ----------------------------------------------------------------- 점검 본체

def show(tok, episode, task, ctx=4):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": episode["prompt"]}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(episode.get("thinking", False)),
    )
    output = episode["all_llm_output"]
    full = prompt + output

    char_bounds, reason = step_char_bounds(output, task)
    if reason is not None:
        print(f"  [skip] {reason}")
        return

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping

    print(f"  tokens={len(ids)}  prompt_chars={len(prompt)}  output_chars={len(output)}")

    # --- 1. 특수 토큰 offset 확인 -------------------------------------------
    special = tok.all_special_ids
    zero_span = [(i, tok.decode([t]), offs[i])
                 for i, t in enumerate(ids[:60])
                 if offs[i][0] == offs[i][1]]
    print("\n  [1] 앞 60토큰 중 offset 폭이 0인 것:",
          f"{len(zero_span)}개" if zero_span else "없음 (정상)")
    for i, s, o in zero_span[:6]:
        flag = " <-- special" if ids[i] in special else ""
        print(f"      idx={i:4d} {s!r:24s} offset={o}{flag}")

    print("\n      앞 8토큰 덤프:")
    for i in range(min(8, len(ids))):
        print(f"      idx={i:3d} {tok.decode([ids[i]])!r:26s} offset={offs[i]}")

    # --- 2. straddle 및 프롬프트 경계 ---------------------------------------
    straddle = [(i, tok.decode([ids[i]]), offs[i])
                for i, (s, e) in enumerate(offs) if s < len(prompt) < e]
    print("\n  [2] 프롬프트/출력 이음매:")
    if straddle:
        i, s, o = straddle[0]
        print(f"      STRADDLE! idx={i} {s!r} offset={o}  -> extract.py 가 이 궤적을 skip 함")
    else:
        print("      straddle 없음 (정상)")

    abs_bounds = [0] + [len(prompt) + b for b in char_bounds]
    tb = char_to_token_bounds(abs_bounds, offs)

    ok_len = tb[-1] == len(ids)
    print(f"      boundaries={tb}")
    print(f"      마지막 경계 == 토큰수 : {ok_len}  ({tb[-1]} vs {len(ids)})")

    recon = tok.decode(ids[:tb[1]])
    print(f"      decode(ids[:prompt_end]) == prompt : {recon == prompt}")
    if recon != prompt:
        print(f"        재구성 끝 40자: {recon[-40:]!r}")
        print(f"        원본   끝 40자: {prompt[-40:]!r}")

    # --- 3. 각 경계 앞뒤 토큰 -----------------------------------------------
    labels = ["prompt_end"] + [f"step{i+1}_start" for i in range(len(tb) - 3)] \
             + ["terminal_start", "eos"]
    print("\n  [3] 경계별 앞뒤 토큰:")
    for lab, b in zip(labels, tb[1:]):
        lo, hi = max(0, b - ctx), min(len(ids), b + ctx)
        before = tok.decode(ids[lo:b])
        after = tok.decode(ids[b:hi])
        print(f"      {lab:16s} idx={b:5d} | ...{before!r} || {after!r}...")

    # --- 4. 구간 길이 --------------------------------------------------------
    seg = [(labels[i - 1] if i else "prompt", tb[i + 1] - tb[i])
           for i in range(len(tb) - 1)]
    print("\n  [4] 구간 토큰 수:", ", ".join(f"{n}" for _, n in seg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--task", default=None, help="생략하면 첫 행의 task 사용")
    ap.add_argument("--n", type=int, default=3, help="점검할 에피소드 수")
    args = ap.parse_args()

    rows = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("skipped") or r.get("truncated"):
                continue
            if not (r.get("all_llm_output") or "").strip():
                continue
            rows.append(r)
            if len(rows) >= args.n:
                break

    if not rows:
        raise SystemExit("점검할 에피소드가 없다")

    task = args.task or rows[0]["task"]
    print(f"model={args.model}  task={task}  episodes={len(rows)}\n")
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"tokenizer fast={tok.is_fast}\n")

    for r in rows:
        print("=" * 78)
        print(f"{r['env_name']} seed={r['env_seed']} thinking={r.get('thinking')}")
        show(tok, r, task)
        print()


if __name__ == "__main__":
    main()