"""추출 파이프라인의 토큰 경계 점검.

모델은 안 태운다 (토크나이저만 필요). extract.py 와 같은 방식으로 토크나이즈한 뒤
  1. 특수 토큰의 offset 이 (0,0) 인지 실제 문자 범위인지
  2. 프롬프트/출력 경계가 토큰 단위로 정확히 갈리는지
  3. 각 step 경계가 ("Step N" 마커가 빠진) 스텝 첫 토큰 앞에 놓이고 "Step N" 토큰이 남지 않았는지
를 눈으로 확인한다. extract.prepare_output 이 "Step N" 마커를 입력에서 빼므로
여기서도 같은 함수로 벗긴 텍스트를 토크나이즈한다.

    python check_boundaries.py generation/trajectory/decompose_no_thinking.jsonl
    python check_boundaries.py <jsonl> --task decompose --n 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "inference"))
from extract import prepare_output, char_to_token_bounds  # noqa: E402

# ----------------------------------------------------------------- 점검 본체

def show(tok, episode, task, ctx=4):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": episode["prompt"]}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(episode.get("thinking", False)),
    )
    cleaned, reason = prepare_output(episode["all_llm_output"], task)
    if reason is not None:
        print(f"  [skip] {reason}")
        return
    output, char_bounds = cleaned.text, cleaned.bounds      # 마커가 빠진 출력
    full = prompt + output

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping

    print(f"  tokens={len(ids)}  prompt_chars={len(prompt)}  output_chars={len(output)}"
          f"  (마커 {cleaned.n_header_chars}자 제거)")

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
    abs_bounds = [0] + [len(prompt) + b for b in char_bounds]
    cuts = abs_bounds[1:-1]                      # 프롬프트 끝 + 각 스텝/터미널 시작
    straddle = [(c, i, tok.decode([ids[i]]), offs[i])
                for c in cuts for i, (s, e) in enumerate(offs) if s < c < e]
    print("\n  [2] 경계 이음매 (프롬프트/출력, 스텝 사이):")
    if straddle:
        for c, i, s, o in straddle:
            print(f"      STRADDLE! char={c} idx={i} {s!r} offset={o}  -> extract.py 가 이 궤적을 skip 함")
    else:
        print(f"      straddle 없음 (정상, 경계 {len(cuts)}개 검사)")

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
    # tb[1] = 프롬프트 끝 = step1 시작, tb[2..] = step2..N 시작, 터미널 시작, 끝
    labels = ["prompt_end"] + [f"step{i}_start" for i in range(2, len(tb) - 2)] \
             + ["terminal_start", "eos"]
    print("\n  [3] 경계별 앞뒤 토큰 (|| 오른쪽에 'Step' 이 보이면 헤더 제거가 실패한 것):")
    for lab, b in zip(labels, tb[1:]):
        lo, hi = max(0, b - ctx), min(len(ids), b + ctx)
        before = tok.decode(ids[lo:b])
        after = tok.decode(ids[b:hi])
        print(f"      {lab:16s} idx={b:5d} | ...{before!r} || {after!r}...")
    leaked = [i for i in range(tb[1], len(ids)) if "step" in tok.decode([ids[i]]).lower()]
    print(f"      출력 구간에 'step' 을 담은 토큰: {len(leaked)}개"
          + (f"  idx={leaked[:8]}" if leaked else "  (헤더 제거 정상)"))

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