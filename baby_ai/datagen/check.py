"""
파일럿 게이트 (setting.md §6 STEP 2).

assert 로 잡는 것 (하드 게이트) 과 지표로 판단하는 것 (소프트 게이트) 을 분리한다.
  하드: span 정합, id 유일성, steps/labels 정렬, outcome ⟺ success
  소프트: leakage rate, 반복률, 분절 step 수, 가시성별 성공률, 재방문율 ...

주의 (setting.md §1 B)
  * assert 는 인덱스 밀림만 잡고 lookup 은 못 잡는다. 그건 unique/total, dup%,
    4-gram 반복률이 잡는다.
  * 개수 등식 len(step_spans) == n_exec 는 걸지 않는다.
    hidden state 1개 = action 1개가 아니다.

사용
  python -m datagen.check --cell-dir data/P1
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .babyai_env import ACTIONS


def load(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


# ---------------------------------------------------------------- 하드 게이트


def hard_gates(steps, labels):
    errs = []
    ids = [r["id"] for r in steps]
    if len(set(ids)) != len(ids):
        errs.append("중복 id")
    if len(steps) != len(labels):
        errs.append(f"steps({len(steps)}) / labels({len(labels)}) 길이 불일치")
    for r, l in zip(steps, labels):
        if r["id"] != l["id"]:
            errs.append(f"id 정렬 불일치: {r['id']} vs {l['id']}")
            break
    for r in steps:
        if not r["span_report"]["ok"]:
            errs.append(f"span 불일치: {r['id']} "
                        f"({r['span_report']['n_bad']}/{r['span_report']['n']})")
        if (r["outcome"] == "success") != r["answer"]["success"]:
            errs.append(f"outcome/success 불일치: {r['id']}")
        for (s, e), (s2, e2) in zip(r["step_spans"], r["step_spans"][1:]):
            if e > s2:
                errs.append(f"span 겹침: {r['id']}")
                break
    return errs


# ---------------------------------------------------------------- 소프트 지표


def ngrams(text, n=4):
    w = re.findall(r"\w+", text.lower())
    return set(zip(*[w[i:] for i in range(n)]))


def soft_metrics(steps, labels, rejects):
    all_steps = [s for r in steps for s in r["steps"]]
    n_step = len(all_steps)

    # --- lookup 조기 경보
    uniq = len(set(all_steps))
    dup_in_ep = np.mean([
        1 - len(set(r["steps"])) / max(len(r["steps"]), 1) for r in steps
    ]) if steps else 0.0

    # 에피소드 간 4-gram 중복률 (greedy 반복 퇴화 탐지)
    grams = [ngrams(" ".join(r["steps"])) for r in steps]
    inter = []
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            u = grams[i] | grams[j]
            if u:
                inter.append(len(grams[i] & grams[j]) / len(u))

    # --- 누출
    leak = np.mean([r["leakage"]["has_action_leak"] for r in steps]) if steps else 0.0
    coord = np.mean([r["leakage"]["n_coord_mentions"] > 0 for r in steps]) if steps else 0.0
    syn = Counter(w for r in steps for w in r["leakage"]["action_synonyms"])

    # --- 분절
    n_seg = [len(r["steps"]) for r in steps]
    seg_mode = Counter(r["segment_mode"] for r in steps)
    tok_per_step = [e - s for r in steps for s, e in r["step_spans"]]

    # --- 길이-위치 상관 (setting.md B4 전제)
    corrs = []
    for r in steps:
        n = [e - s for s, e in r["step_spans"]]
        if len(n) > 2:
            corrs.append(np.corrcoef(np.arange(len(n)), n)[0, 1])

    # --- 파싱 / 절단
    allrec = steps + rejects
    parse_ok = np.mean([r.get("parse_ok", False) for r in allrec]) if allrec else 0.0
    trunc = np.mean([r.get("truncated", False) for r in steps]) if steps else 0.0

    # --- 가시성별 성공률
    vis = {}
    seen_ids = {r["id"] for r in steps}
    for l in labels:
        v = l.get("n_targets_visible")
        vis.setdefault(v, [0, 0])
        vis[v][0] += 1
        vis[v][1] += 1  # labels 에는 성공만 들어옴
    for r in rejects:
        v = r.get("n_targets_visible")
        if v is not None:
            vis.setdefault(v, [0, 0])
            vis[v][0] += 1

    # --- 라벨 chance level + 죽은 컬럼
    lab_stats = {}
    for key in ("action", "pos", "dir", "carrying", "front_obj"):
        vals = [tuple(v) if isinstance(v, list) else v
                for l in labels for v in l.get(key, [])]
        if not vals:
            continue
        c = Counter(vals)
        lab_stats[key] = {
            "classes": len(c),
            "majority": c.most_common(1)[0][1] / len(vals),
            "dead": len(c) <= 1,
        }

    # --- LLM 궤적 재방문율 (기준 5: state 중복 -> imagination 유의미성)
    rev = []
    for l in labels:
        st = list(zip([tuple(p) for p in l.get("pos", [])], l.get("dir", [])))
        if st:
            c = Counter(st)
            rev.append(sum(v - 1 for v in c.values()) / len(st))

    T = [l["n_exec"] for l in labels]
    eff = [l["n_exec"] / l["bot_reference_len"]
           for l in labels if l.get("bot_reference_len")]

    return {
        "n_success": len(steps),
        "n_rejected": len(rejects),
        "success_rate": len(steps) / max(len(allrec), 1),
        "lookup": {
            "n_steps_total": n_step,
            "unique_over_total": uniq / max(n_step, 1),
            "dup_within_episode": float(dup_in_ep),
            "inter_episode_4gram_jaccard_mean": float(np.mean(inter)) if inter else None,
            "inter_episode_4gram_jaccard_p95": float(np.percentile(inter, 95)) if inter else None,
        },
        "leakage": {
            "action_leak_rate": float(leak),
            "coordinate_mention_rate": float(coord),
            "top_synonyms": syn.most_common(8),
        },
        "segmentation": {
            "steps_per_episode": {
                "median": float(np.median(n_seg)) if n_seg else None,
                "mean": float(np.mean(n_seg)) if n_seg else None,
                "min": min(n_seg) if n_seg else None,
                "max": max(n_seg) if n_seg else None,
                "frac_single": float(np.mean([x <= 1 for x in n_seg])) if n_seg else None,
            },
            "mode_used": dict(seg_mode),
            "tokens_per_step_median": float(np.median(tok_per_step)) if tok_per_step else None,
        },
        "length_position_corr": {
            "mean": float(np.mean(corrs)) if corrs else None,
            "p95_abs": float(np.percentile(np.abs(corrs), 95)) if corrs else None,
        },
        "parsing": {"parse_ok_rate": float(parse_ok), "truncated_rate": float(trunc)},
        "visibility_success": {
            str(k): {"n": v[0], "success": v[1], "rate": v[1] / max(v[0], 1)}
            for k, v in sorted(vis.items(), key=lambda x: (x[0] is None, x[0]))
        },
        "labels": lab_stats,
        "revisit_rate": {
            "mean": float(np.mean(rev)) if rev else None,
            "max": float(np.max(rev)) if rev else None,
        },
        "T": {"median": float(np.median(T)) if T else None,
              "mean": float(np.mean(T)) if T else None},
        "efficiency_ratio": {"median": float(np.median(eff)) if eff else None},
    }


def verdict(m):
    """게이트 판정. 임계값은 잠정이며 파일럿 결과로 조정한다."""
    v = []
    seg = m["segmentation"]["steps_per_episode"]
    if seg["median"] is not None and seg["median"] < 2:
        v.append("FAIL 분절: step 중앙값 < 2. \\n\\n 분절 실패 -> marker 폴백 필요")
    if m["leakage"]["action_leak_rate"] > 0.05:
        v.append(f"FAIL 누출: 액션명 {m['leakage']['action_leak_rate']:.1%} "
                 "-> 프롬프트 수정 필요")
    if m["parsing"]["parse_ok_rate"] < 0.7:
        v.append(f"WARN 파싱: {m['parsing']['parse_ok_rate']:.1%} "
                 "-> P4(few-shot) 우선순위 상승 / 온도 0.5 하향")
    j = m["lookup"]["inter_episode_4gram_jaccard_mean"]
    if j is not None and j > 0.3:
        v.append(f"WARN 반복: 에피소드 간 4-gram Jaccard {j:.2f} "
                 "-> 디코딩을 temperature 0.7 로")
    if m["success_rate"] < 0.02:
        v.append(f"WARN 성공률: {m['success_rate']:.1%} -> 레벨 변종 S5N3 하향 검토")
    c = m["length_position_corr"]["mean"]
    if c is not None and abs(c) > 0.5:
        v.append(f"WARN 길이-위치 상관 {c:.2f} -> e_t 가 길이로 t 를 인코딩할 위험")
    for k, s in m["labels"].items():
        if s["dead"]:
            v.append(f"WARN 죽은 라벨: {k}")
    return v or ["PASS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    d = Path(args.cell_dir)
    steps, labels = load(d / "steps.jsonl"), load(d / "labels.jsonl")
    rejects = load(d / "rejects.jsonl")

    errs = hard_gates(steps, labels)
    m = soft_metrics(steps, labels, rejects)
    v = verdict(m)

    if args.json:
        print(json.dumps({"hard_errors": errs, "metrics": m, "verdict": v}, indent=2))
        return

    print(f"=== {d} ===")
    print(f"success {m['n_success']} / rejected {m['n_rejected']} "
          f"({m['success_rate']:.1%})")
    print("\n-- HARD --")
    print("  OK" if not errs else "\n".join("  " + e for e in errs[:10]))
    print("\n-- 분절 --")
    s = m["segmentation"]["steps_per_episode"]
    print(f"  step/에피소드  median {s['median']} mean {s['mean']} "
          f"[{s['min']},{s['max']}]  단일 step 비율 {s['frac_single']}")
    print(f"  분절 방식      {m['segmentation']['mode_used']}")
    print(f"  토큰/step      median {m['segmentation']['tokens_per_step_median']}")
    print("\n-- lookup 경보 --")
    lk = m["lookup"]
    print(f"  unique/total   {lk['unique_over_total']:.3f}")
    print(f"  에피소드내 중복 {lk['dup_within_episode']:.3f}")
    print(f"  4-gram Jaccard  mean {lk['inter_episode_4gram_jaccard_mean']} "
          f"p95 {lk['inter_episode_4gram_jaccard_p95']}")
    print("\n-- 누출 --")
    print(f"  액션명 {m['leakage']['action_leak_rate']:.1%}   "
          f"좌표 {m['leakage']['coordinate_mention_rate']:.1%}")
    print(f"  동의어 {m['leakage']['top_synonyms']}")
    print("\n-- 파싱 --")
    print(f"  parse_ok {m['parsing']['parse_ok_rate']:.1%}  "
          f"truncated {m['parsing']['truncated_rate']:.1%}")
    print("\n-- 가시성별 성공률 --")
    for k, x in m["visibility_success"].items():
        print(f"  n_targets_visible={k:>4}  {x['success']}/{x['n']}  {x['rate']:.1%}")
    print("\n-- 라벨 --")
    for k, x in m["labels"].items():
        print(f"  {k:10} classes={x['classes']:3d} majority={x['majority']:.1%}"
              + ("   <-- DEAD" if x["dead"] else ""))
    print("\n-- 궤적 --")
    print(f"  T median {m['T']['median']}   재방문율 {m['revisit_rate']['mean']}   "
          f"efficiency {m['efficiency_ratio']['median']}")
    print(f"  길이-위치 상관 {m['length_position_corr']['mean']}")
    print("\n-- 판정 --")
    for x in v:
        print("  " + x)


if __name__ == "__main__":
    main()