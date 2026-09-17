#!/usr/bin/env python
"""인접 구간 대표 벡터 사이의 유클리드 거리 / 코사인 유사도. 논문(Sun et al.) 4.1 대응.

대표 벡터는 probing.py 와 같다 — 각 구간의 마지막 토큰 E[end-1-offset].
구간 순서는 [prompt, step 1, ..., step N, answer] 이고, 인접한 쌍마다 두 값을 잰다.

    d_euc(a, b) = ||a - b||_2
    cos_sim(a, b) = a·b / (|a||b|)        (논문은 코사인 "거리" 1 - cos 를 쓴다)

집계는 논문 방식 — 전이마다 에피소드 전부에서 값을 모아 평균과 95% CI 를 낸다.
status(success/failure)로 나눠 두 그룹을 비교하고, 두 CI 가 겹치지 않으면 † 를 붙인다.
논문 Figure 2a 가 이 형태이고, 거기서는 후반 전이에서만 † 가 붙었다.

Δ 정의는 논문과 맞춘다: Δ = failure 평균 − success 평균 (논문의 Δ(I−C)).

경로: <hidden_root>/<task>/<level>/<status>/chunk_*.pt

Usage:
    python distance.py --pt 'latent/hidden_states/decompose/*/*/chunk_*.pt' \
        --n-steps 6 --output out/decompose_dist
    python distance.py --pt 'latent/hidden_states/plan/*/*/chunk_*.pt' \
        --n-steps 5 --output out/plan_dist --offset 1
"""
from __future__ import annotations

import json
import glob
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------- 로딩

def segment_names(n_seg: int) -> list[str]:
    """구간 이름. n_seg = N + 2 (prompt + step N개 + answer)."""
    return ["prompt"] + [f"step_{k}" for k in range(1, n_seg - 1)] + ["answer"]


def load_trajectories(patterns, offset, n_steps=None):
    """에피소드마다 구간 대표 벡터를 순서대로 모은다.

    반환: list of dict(vecs=(n_seg, D) float32, names=[...], status=str, gid=str)

    boundaries 규약 [0, 프롬프트끝, step1끝, ..., stepN끝, 전체끝] 을 그대로 쓴다.
    probing.py 의 load_pt 와 같은 위치·같은 offset 이라 두 결과가 맞물린다.
    """
    out, stats_ = [], Counter()

    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    if not paths:
        raise SystemExit(f"no files: {patterns}")

    for p in paths:
        p = Path(p)
        status, level, task = p.parent.name, p.parents[1].name, p.parents[2].name
        d = torch.load(p, map_location="cpu", weights_only=False)

        for seed, ep in d["episodes"].items():
            E = ep["E"].float().numpy()
            b = [int(x) for x in ep["boundaries"]]
            n_tok = E.shape[0]

            if len(b) < 4 or b[0] != 0 or b[-1] != n_tok:
                stats_["bad_boundaries"] += 1
                continue
            if any(x >= y for x, y in zip(b, b[1:])):
                stats_["nonmonotonic_boundaries"] += 1
                continue
            if n_steps is not None and len(b) - 3 != n_steps:
                stats_["wrong_n_steps"] += 1
                continue

            vecs, ok = [], True
            for s, e in zip(b, b[1:]):          # prompt, step 1..N, terminal
                i = e - 1 - offset
                if i < s:
                    ok = False
                    break
                vecs.append(E[i])
            if not ok:
                stats_["segment_too_short"] += 1
                continue

            stats_["episodes"] += 1
            out.append({"vecs": np.stack(vecs), "names": segment_names(len(vecs)),
                        "status": status, "gid": f"{task}/{level}/{status}/{seed}"})

    if not out:
        raise SystemExit("에피소드 없음 — boundaries 규약 확인")

    print(f"loaded: {dict(stats_)}")
    print(f"  구간 수 분포: {dict(sorted(Counter(len(o['vecs']) for o in out).items()))}")
    print(f"  status: {dict(Counter(o['status'] for o in out))}")
    return out


# ---------------------------------------------------------------- 지표

def euclidean(a, b):
    return float(np.linalg.norm(a - b))


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def collect(traj):
    """전이별 · status별 값. {transition: {status: {"euc": [...], "cos": [...]}}}"""
    acc = defaultdict(lambda: defaultdict(lambda: {"euc": [], "cos": []}))
    for ep in traj:
        V, names, st = ep["vecs"], ep["names"], ep["status"]
        for i in range(len(V) - 1):
            key = f"{names[i]}→{names[i + 1]}"
            acc[key][st]["euc"].append(euclidean(V[i], V[i + 1]))
            acc[key][st]["cos"].append(cosine_sim(V[i], V[i + 1]))
    return acc


def ci95(vals):
    """(평균, 하한, 상한, n). n<2 면 구간을 평균으로 닫는다."""
    v = np.asarray(vals, dtype=np.float64)
    n = v.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    m = float(v.mean())
    if n < 2:
        return m, m, m, n
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=stats.sem(v))
    return m, float(lo), float(hi), n


def summarize(acc, order):
    """전이별로 success/failure 평균·CI·Δ·CI 겹침 여부."""
    rows = []
    for key in order:
        per_status = acc[key]
        row = {"transition": key}
        for metric in ("euc", "cos"):
            g = {}
            for st in ("success", "failure"):
                m, lo, hi, n = ci95(per_status.get(st, {}).get(metric, []))
                g[st] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": n}
            s, f = g["success"], g["failure"]
            # 논문 Δ(I−C) 와 같은 부호: 실패 − 성공
            have_both = s["n"] > 1 and f["n"] > 1
            overlap = not (f["ci_hi"] < s["ci_lo"] or s["ci_hi"] < f["ci_lo"])
            g["delta"] = f["mean"] - s["mean"] if have_both else None
            g["ci_disjoint"] = (have_both and not overlap)
            # 전체(status 합산)
            allv = (per_status.get("success", {}).get(metric, [])
                    + per_status.get("failure", {}).get(metric, []))
            m, lo, hi, n = ci95(allv)
            g["all"] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": n}
            row[metric] = g
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 출력

def print_table(rows):
    for metric, label in (("euc", "Euclidean distance"), ("cos", "Cosine similarity")):
        print(f"\n{label}")
        print(f"{'transition':24s} {'success (95% CI)':>26s} {'failure (95% CI)':>26s} "
              f"{'Δ(F−S)':>10s}")
        print("-" * 90)
        for r in rows:
            g = r[metric]
            s, f = g["success"], g["failure"]
            fmt = lambda d: (f"{d['mean']:.4f} [{d['ci_lo']:.4f},{d['ci_hi']:.4f}]"
                             if d["n"] else "—")
            dl = "—" if g["delta"] is None else f"{g['delta']:+.4f}"
            if g["ci_disjoint"]:
                dl += "†"
            print(f"{r['transition']:24s} {fmt(s):>26s} {fmt(f):>26s} {dl:>10s}")
        print("† = 두 그룹의 95% CI 가 겹치지 않음")


def plot(rows, out_path, title):
    labels = [r["transition"] for r in rows]
    x, w = np.arange(len(labels)), 0.38

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(labels) * 1.5), 9))
    for ax, metric, ylab in ((axes[0], "euc", "Euclidean distance"),
                             (axes[1], "cos", "Cosine similarity")):
        for i, (st, color) in enumerate((("success", "#1f77b4"), ("failure", "#d62728"))):
            m = [r[metric][st]["mean"] for r in rows]
            lo = [r[metric][st]["mean"] - r[metric][st]["ci_lo"] for r in rows]
            hi = [r[metric][st]["ci_hi"] - r[metric][st]["mean"] for r in rows]
            n = rows[0][metric][st]["n"]
            ax.bar(x + (i - 0.5) * w, m, w, yerr=[lo, hi], capsize=3,
                   color=color, alpha=0.85, label=f"{st} (n={n})")
        for j, r in enumerate(rows):                      # † 표시
            if r[metric]["ci_disjoint"]:
                top = max(r[metric]["success"]["ci_hi"], r[metric]["failure"]["ci_hi"])
                ax.text(j, top, "†", ha="center", va="bottom", fontsize=12)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", nargs="+", required=True,
                    help="chunk_*.pt glob. success/failure 양쪽을 다 넣는다")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="구간 마지막 토큰에서 몇 칸 앞 (probing.py 와 같은 의미)")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="이 step 수인 에피소드만 (decompose=6, plan=5). "
                         "안 주면 구간 수가 섞여 전이 이름이 갈린다")
    a = ap.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)

    traj = load_trajectories(a.pt, a.offset, a.n_steps)
    acc = collect(traj)

    # 전이 순서: 가장 흔한 구간 수 기준으로 prompt→...→answer
    n_seg = Counter(len(o["vecs"]) for o in traj).most_common(1)[0][0]
    names = segment_names(n_seg)
    order = [f"{names[i]}→{names[i + 1]}" for i in range(n_seg - 1)]
    order = [k for k in order if k in acc]
    extra = [k for k in acc if k not in order]
    if extra:
        print(f"[warn] 구간 수가 섞여 있다 — 표에서 제외된 전이: {extra}")

    rows = summarize(acc, order)
    print_table(rows)

    with open(a.output / "distances.json", "w") as f:
        json.dump({"offset": a.offset, "n_steps": a.n_steps,
                   "n_episodes": len(traj), "rows": rows}, f, indent=2)

    plot(rows, a.output / "distances.png",
         f"Adjacent-segment transitions (offset={a.offset}, "
         f"episodes={len(traj)})")
    print(f"\nsaved → {a.output}")


if __name__ == "__main__":
    main()