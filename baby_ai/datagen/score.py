"""
사후 채점 + 게이트. raw.jsonl 만 읽는다. GPU 불필요. 생성기와 무관하게 동작한다.

공개 함수 2개
    parse_actions(raw)  -> (actions, mode, n_dropped, reasoning, n_schema, n_boundary)
    main()              -> outcomes.jsonl + 게이트 출력

경계 규약
    CoT 와 답의 경계는 `The LLM's action sequence is: ` 종결구가 정한다
    (LLM-BabyBench `prompters/utils.py`). **포맷 사실이지 파서의 산물이 아니다.**
    없으면 unparsed 로 끝나고 reasoning 은 전문이 된다.
    동점 규칙도 원문을 따른다 — `llms/utils.py` 는 lower().find() 로 **첫 번째**
    match 를 잡는다. 다중 발생률은 `n_boundary_matches` 로 감사한다.

파싱 규약
    종결구 뒤를 콤마로 자르고 **정규 5개 명칭만** 받는다. 별칭 없음.
    LLM-BabyBench `evaluators/utils.py` 도 split(",") -> strip -> lower 후
    `Actions.__members__` 에 있는 것만 취하고 나머지는 버린다. 동일 규약이다.
    버려진 항목 수는 n_dropped 로 남겨 감사한다.

사용
    python -m datagen.score --dir data/P1
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .babyai_env import ACT_ID, replay
from .prompt import SCHEMA_TOKENS, answer_spec

_BLANK = re.compile(r"\n[ \t]*\n")


def parse_actions(raw: str, prompt: str = "bb"):
    """(actions, mode, n_dropped, reasoning, n_schema, n_boundary_matches).

    mode : "answer" | "no_answer"

    """
    # 경계 규칙은 변형마다 다르고, 각 레퍼런스의 추출 코드를 따른다.
    #   bb     llms/utils.py            lower().find(...)     -> 첫 번째
    #   balrog agents/chain_of_thought  split("ACTION:")[-1]  -> 마지막
    spec = answer_spec(prompt)
    ms = list(re.finditer(spec["regex"], raw, re.I))
    if not ms:
        return [], "no_answer", 0, raw, 0, 0
    m = ms[0] if spec["tie"] == "first" else ms[-1]

    items = []
    for w in re.split(r"[,\n;]+", m.group(1)):
        w = re.sub(r"[^a-z ]+", " ", w.lower())
        w = re.sub(r"\s+", " ", w).strip()
        if w:
            items.append(w)
    acts = [w for w in items if w in ACT_ID]
    n_schema = sum(1 for w in items if w in SCHEMA_TOKENS)   # 스키마를 그대로 베낌
    return acts, "answer", len(items) - len(acts), raw[: m.start()], n_schema, len(ms)


def _ngrams(text: str, n: int = 4):
    w = re.findall(r"\w+", text.lower())
    return set(zip(*[w[i:] for i in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    rows, reasonings = [], []
    with open(d / "raw.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            variant = r.get("prompt_variant", "bb")
            acts, mode, n_drop, reasoning, n_schema, n_bound = parse_actions(
                r["assistant_text"], variant)
            n_gen = r.get("n_chars", len(r["assistant_text"]))
            # 액션이 비어도 replay 를 부른다 — 공변량(n_targets_visible)은 reset 만으로
            # 나오므로 호출을 한 번으로 유지한다.
            rep = replay(r["level"], r["seed"], acts)
            if not acts:
                rep["outcome"] = "unparsed"
            rows.append({
                "id": r["id"], "seed": r["seed"],
                "parse_mode": mode, "n_actions": len(acts), "n_dropped": n_drop,
                "n_boundary_matches": n_bound,
                "action_seq": acts,
                "generator": r.get("generator"), "prompt_variant": variant,
                "n_chars": n_gen, "truncated": bool(r.get("truncated", False)),
                "stop_reason": r.get("stop_reason"),
                # thinking 이 꺼지지 않은 경우. assistant_text 가 CoT 가 아니라
                # "사고 블록 + 결론"이 되므로 분절도 채점도 의미를 잃는다.
                "think_leak": "<think>" in r["assistant_text"],

                "n_paragraphs": len([s for s in _BLANK.split(reasoning) if s.strip()]),
                "schema_echo": n_schema > 0,
                **rep,
            })
            reasonings.append(reasoning)

    if not rows:
        raise SystemExit(f"[중단] {d / 'raw.jsonl'} 에 레코드가 없습니다.")

    with open(d / "outcomes.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---- 에피소드 간 4-gram Jaccard (디코딩 확정 게이트)
    grams = [_ngrams(t) for t in reasonings]
    jac = [len(a & b) / len(a | b)
           for i, a in enumerate(grams) for b in grams[i + 1:] if (a | b)]

    n = len(rows)
    para = [x["n_paragraphs"] for x in rows]
    vis = {}
    for x in rows:
        v = vis.setdefault(x["n_targets_visible"], [0, 0])
        v[0] += 1
        v[1] += int(x["success"])

    def frac(key):
        return sum(x[key] for x in rows) / max(n, 1)

    tag_rate = sum(x["parse_mode"] == "answer" for x in rows) / max(n, 1)

    variants = sorted({x["prompt_variant"] for x in rows})
    print(f"=== {d}   n={n}   prompt={','.join(variants)} ===")
    print("\n-- 포맷 --")
    print(f"  답 블록 존재   {tag_rate:.1%}")
    print(f"  절단           {frac('truncated'):.1%}"
          f"   (stop_reason=max_tokens. 포맷 불이행과 구분해야 한다)")
    print(f"  think 누출     {frac('think_leak'):.1%}"
          f"   (<think> 가 assistant_text 에 남음)")

    print(f"  다중 경계      {np.mean([x['n_boundary_matches'] > 1 for x in rows]):.1%}"
          f"   (종결구 2회 이상)")

    print(f"  schema echo    {frac('schema_echo'):.1%}   (스키마 플레이스홀더를 그대로 베낀 비율)")
    print(f"  n_dropped 합   {sum(x['n_dropped'] for x in rows)}   "
          f"(태그 안 비액션 항목. 크면 별칭 도입 여부를 데이터로 판단)")
    print(f"  출력 길이      median {np.median([x['n_chars'] for x in rows]):.0f} chars")

    print("\n-- 분절 가능성 (`\\n\\n`) --")
    print(f"  단락 수/에피소드  median {np.median(para):.1f}  "
          f"[{min(para)},{max(para)}]  단일 단락 비율 {np.mean([p<=1 for p in para]):.1%}")

    print("\n-- 반복 (디코딩 확정) --")
    if jac:
        print(f"  4-gram Jaccard  mean {np.mean(jac):.3f}  p95 {np.percentile(jac,95):.3f}")

    print("\n-- 성공 --")
    print(f"  전체            {frac('success'):.1%}")
    print(f"  outcome         {dict(Counter(x['outcome'] for x in rows))}")
    print("  가시성별:")
    for k in sorted(vis, key=lambda x: (x is None, x)):
        a, b = vis[k]
        print(f"    n_targets_visible={str(k):>4}  {b}/{a}  {b/max(a,1):.1%}")

    print("\n-- 판정 --")
    v = []
    if frac("think_leak") > 0.0:
        v.append(f"FAIL <think> 누출 {frac('think_leak'):.1%} -> qwen.ENABLE_THINKING 이 "
                 "먹지 않았다. 추론이 thinking block 으로 빠지면 남는 것은 결론뿐이라 "
                 "CoT 분절 자체가 성립하지 않는다")

    if frac("truncated") > 0.0:
        v.append(f"FAIL 절단 {frac('truncated'):.1%} -> --max-new-tokens 상향. "
                 "절단분은 답 블록이 없으므로 no_answer 로 잘못 집계된다")
    if tag_rate < 0.7:
        v.append(f"FAIL 답 블록 {tag_rate:.1%} -> 포맷 지시 수정 또는 P4 우선순위 상승")
    if np.median(para) < 2:
        v.append("FAIL 분절: 단락 중앙값 < 2 -> `\\n\\n` 분절 불성립. 폴백 검토")
    if jac and np.mean(jac) > 0.3:
        v.append(f"WARN 반복 4-gram {np.mean(jac):.2f} -> 디코딩을 sample 로")
    if frac("schema_echo") > 0.1:
        v.append(f"WARN 스키마 복제 {frac('schema_echo'):.1%} -> 플레이스홀더 표기 변경 검토")
    print("\n".join("  " + x for x in v) or "  PASS")


if __name__ == "__main__":
    main()