"""segment.py 가 만든 span 이 원문과 정말 맞는지 검사한다.

두 층을 따로 본다.

1. char span — segments_action / segments_step 을 이어붙이면 all_llm_output 이
   그대로 나오는가. segment.py 안에 assert 가 있지만 그건 span 끼리의 연속성만
   보므로, 실제로 원문이 복원되는지는 여기서 확인한다.

2. token span — probe_gram.py 의 token_spans 로 char span 을 토큰 인덱스로
   바꾼 뒤, 빈틈·겹침 없이 [0,N) 을 덮는가. 그리고 각 블록을 디코드하면 원래
   char span 텍스트가 나오는가.

   BPE 토큰은 우리 세그먼트 경계를 존중하지 않는다. 경계를 걸치는 토큰이 있으면
   시작 offset 쪽 블록에 통째로 들어가고 다음 블록은 앞 몇 글자를 잃는다.
   원리상 피할 수 없으므로 없애는 대신 규모를 잰다 — 무시할 수준인지 판단하려면
   숫자가 있어야 한다.

토크나이저를 인자로 받는 이유
    실험은 CoT 를 쓴 모델로 인코딩한다. 토크나이저가 다르면 걸치는 토큰도
    달라지므로, 실제 쓸 모델로 재야 의미가 있다.

    python check_spans.py data/segmented                       # char span 만 (전수)
    python check_spans.py data/segmented --model Qwen/Qwen3-32B --sample 400
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from probe_gram import token_spans

KINDS = ("segments_action", "segments_step")


def iter_records(root: Path):
    for f in sorted(root.glob("*/*.jsonl")):
        if f.name == "_bad.jsonl":
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield f, json.loads(line)


def check_chars(records):
    bad = []
    for _, r in records:
        t = r["all_llm_output"]
        for k in KINDS:
            if "".join(t[s["start"]:s["end"]] for s in r[k]) != t:
                bad.append((r["id"], k))
                break
    return bad


def check_tokens(records, model, verbose):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    n_tok = n_strad = 0
    eps_strad = set()
    problems = {"gap": [], "overlap": [], "not_cover": [], "decode": []}
    leaks = []

    for _, r in records:
        t = r["all_llm_output"]
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        offs, ids = enc.offset_mapping, enc.input_ids
        N = len(ids)

        for k in KINDS:
            bs = token_spans(offs, r[k])
            if not bs:
                continue
            if bs[0][1] != 0 or bs[-1][2] != N:
                problems["not_cover"].append((r["id"], k))
            for (_, _, e1), (_, s2, _) in zip(bs, bs[1:]):
                if e1 < s2:
                    problems["gap"].append((r["id"], k)); break
                if e1 > s2:
                    problems["overlap"].append((r["id"], k)); break

            by_name = {s["name"]: s for s in r[k]}
            for name, s, e in bs:
                seg = by_name[name]
                want = t[seg["start"]:seg["end"]]
                got = tok.decode(ids[s:e])
                if got.strip() != want.strip():
                    problems["decode"].append((r["id"], k, name, want[-40:], got[-40:]))

        # 경계를 걸치는 토큰 (분절 종류와 무관하게 액션 기준으로 센다)
        bounds = {s["start"] for s in r["segments_action"]}
        bounds |= {s["end"] for s in r["segments_action"]}
        for a, b in offs:
            if b <= a:
                continue
            n_tok += 1
            inner = [x for x in bounds if a < x < b]
            if inner:
                n_strad += 1
                eps_strad.add(r["id"])
                leaks.append(min(b - x for x in inner))

    print(f"\n[token span]  tokenizer={model}")
    for key, label in (("gap", "span 사이 빈틈"), ("overlap", "span 겹침"),
                       ("not_cover", "[0,N) 미포함")):
        print(f"  {label:16} {len(problems[key])}")
    print(f"  {'디코딩 불일치':14} {len(problems['decode'])}")
    print(f"  {'경계 걸친 토큰':14} {n_strad} / {n_tok} "
          f"({n_strad / max(n_tok, 1):.4%})   에피소드 {len(eps_strad)}")
    if leaks:
        leaks.sort()
        print(f"  {'넘어간 글자수':14} median {leaks[len(leaks)//2]}  max {max(leaks)}")

    if verbose:
        for row in problems["decode"][:10]:
            print(f"\n  {row[0]} {row[1]} {row[2]}")
            print(f"    want ...{row[3]!r}")
            print(f"    got  ...{row[4]!r}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg_dir", type=Path, help="segment.py 출력 디렉토리")
    ap.add_argument("--model", default=None,
                    help="주면 token span 까지 검사한다 (토크나이저만 받으므로 가볍다)")
    ap.add_argument("--sample", type=int, default=0,
                    help="token span 검사 표본 수. 0 이면 전수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true", help="불일치 예시를 찍는다")
    args = ap.parse_args()

    records = list(iter_records(args.seg_dir))
    if not records:
        raise SystemExit(f"{args.seg_dir} 에 jsonl 이 없습니다")

    bad = check_chars(records)
    print(f"[char span]  레코드 {len(records)}   원문 복원 실패 {len(bad)}")
    for i, k in bad[:10]:
        print(f"    {i}  {k}")

    if args.model:
        sub = records
        if args.sample and args.sample < len(records):
            random.Random(args.seed).shuffle(sub := list(records))
            sub = sub[:args.sample]
            print(f"\n  (token span 은 {len(sub)}개 표본)")
        check_tokens(sub, args.model, args.verbose)

    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
