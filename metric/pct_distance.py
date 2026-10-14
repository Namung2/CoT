#!/usr/bin/env python
"""pct_cache.py 캐시(구간별 q% 위치 벡터)로 벡터쌍 거리를 재고 정답/오답 궤적을 비교한다.

입력  <cache_dir>/<task>/<level>/<status>/<tag>/chunk_XXXX.pt   (metric/pct_cache.py 출력)
      success / failure 를 둘 다 읽는다. tag 는 --pct 로 정해진다 (캐시와 같은 목록이어야 한다).

거리 두 가지 (fp32 로 계산):
    euc = ||a − b||₂            cos = 1 − a·b / (|a||b|)   (코사인 "거리")

1) 전체 쌍 행렬 저장 (pairs.npz 하나). 에피소드마다 K = n_seg·n_pct (+1, --with-prompt 면 프롬프트
   마지막 토큰) 개 벡터의 K×K 거리 행렬을 만들어 통째로 저장한다. 아래 두 분석은 이 행렬의 슬라이스다.
       euc, cos          (n_ep, K, K) float32      거리 행렬
       row_seg, row_pct  (K,)                      행 k 가 어느 구간의 몇 % 인지
       seeds, status     (n_ep,)                   에피소드 순서 (status 는 "success" / "failure")
       seg_len           (n_ep, n_seg) int32       구간 토큰 수
       pos               (n_ep, n_seg, n_pct) int32  각 % 가 실제 가리킨 구간 안 토큰 위치
       pct, seg_names                              설정
   decompose 1.8만 에피소드 × 49×49 × 4B ≈ 180MB / 지표 → 파일 약 360MB. --no-save-pairs 로 생략.
   읽기: z = np.load("pairs.npz"); z["euc"][z["status"] == "success", k1, k2]

2) inter — 구간 사이 (Sun et al. "LLM Reasoning as Trajectories" 4.1 의 전 구간 버전).
   각 구간의 마지막 토큰(q=100) 끼리 인접 전이 [prompt→]Step 1→Step 2→…→Step N→answer 의 거리를
   status 별로 모아 평균·95% CI, Δ = failure − success. 두 CI 가 겹치지 않으면 †.
   100% 위치의 모든 구간쌍 평균 행렬(success / failure / Δ)도 같이 저장한다.
       inter.json, inter.png (전이별 막대 + CI), inter_matrix.json, inter_matrix.png

3) intra — 구간 안 (q 축). 구간마다 인접 % 쌍 5→10, 10→30, …, 95→100 의 거리를 status 별로 집계.
   짧은 구간에서 두 q 가 같은 토큰을 가리키면 거리가 정확히 0 이라 평균을 끌어내린다 — 기본은
   그런 쌍을 빼고 센다 (--keep-coincident 로 끔). 빠진 개수는 json 에 남는다.
       intra.json, intra.png (구간 열 × 지표 행, 선 + CI 막대)

출력 폴더 기본값: metric/output/pct_dist/<task>/<level>/<tag>[_prompt]

Usage (레포 루트에서):
    python metric/pct_distance.py --task decompose --level BabyAI-GoTo-v0
    python metric/pct_distance.py --task plan --level CustomBabyAI-GoToRedBall-Small-4Dists-v0 --with-prompt
    python metric/pct_distance.py --task plan --level ... --pct 10 20 40 60 80 100 --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from scipy import stats as sps                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "metric"))
from pct_cache import DEFAULT_PCT, STATUSES, check_pct, make_tag        # noqa: E402

COLOR = {"success": "#2a78d6", "failure": "#eb6834"}   
METRICS = (("euc", "Euclidean distance"), ("cos", "Cosine distance (1 − cos)"))


# ---------------------------------------------------------------------- 읽기

def load_cache(cache_dir: Path, task: str, level: str, tag: str, statuses, with_prompt: bool):
    """캐시 청크를 모아 (X (n_ep, K, d) fp16, rows, episodes 메타). K = [prompt] + n_seg·n_pct."""
    Xs, episodes = [], []
    pct = seg_names = None
    for status in statuses:
        files = sorted((cache_dir / task / level / status / tag).glob("chunk_*.pt"))
        if not files:
            raise SystemExit(f"no cache: {cache_dir / task / level / status / tag}  (pct_cache.py 먼저)")
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            if pct is None:
                pct, seg_names = list(d["pct"]), list(d["seg_names"])
            elif list(d["pct"]) != pct or list(d["seg_names"]) != seg_names:
                raise SystemExit(f"캐시 설정이 섞였다: {f}")
            V = d["V"].flatten(1, 2)                                   # (n, n_seg·n_pct, d)
            if with_prompt:
                V = torch.cat([d["prompt_last"][:, None, :], V], dim=1)
            Xs.append(V)
            for i, seed in enumerate(d["seeds"]):
                episodes.append({"seed": seed, "status": status, "file": str(f),
                                 "seg_len": d["seg_len"][i].tolist(), "pos": d["pos"][i].tolist()})
    rows = ([{"seg": "prompt", "pct": 100}] if with_prompt else []) + \
           [{"seg": s, "pct": q} for s in seg_names for q in pct]
    return torch.cat(Xs), rows, episodes, pct, seg_names


# ---------------------------------------------------------------------- 거리

@torch.no_grad()
def pairwise(X: torch.Tensor, device: str, batch: int = 512):
    """X (n, K, d) → (euc, cos) 각 (n, K, K) float32 numpy."""
    n = X.shape[0]
    euc = np.empty((n, X.shape[1], X.shape[1]), np.float32)
    cos = np.empty_like(euc)
    for s in range(0, n, batch):
        x = X[s:s + batch].to(device, torch.float32)
        euc[s:s + batch] = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist").cpu().numpy()   # mm 경로는 대각이 0.6 까지 틀린다
        xn = F.normalize(x, dim=-1)
        cos[s:s + batch] = (1.0 - xn @ xn.transpose(1, 2)).clamp_(min=0.0).cpu().numpy()
    return euc, cos


# ---------------------------------------------------------------------- 집계

def ci95(v: np.ndarray) -> dict:
    v = np.asarray(v, np.float64)
    n = int(v.size)
    if n == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "std": None, "n": 0}
    m = float(v.mean())
    if n < 2:
        return {"mean": m, "ci_lo": m, "ci_hi": m, "std": 0.0, "n": n}
    lo, hi = sps.t.interval(0.95, n - 1, loc=m, scale=sps.sem(v))
    return {"mean": m, "ci_lo": float(lo), "ci_hi": float(hi), "std": float(v.std(ddof=1)), "n": n}


def compare(vals: dict[str, np.ndarray]) -> dict:
    """{status: 값들} → status 별 CI + Δ(failure − success) + CI 겹침."""
    g = {st: ci95(vals.get(st, np.empty(0))) for st in STATUSES}
    s, f = g["success"], g["failure"]
    if s["n"] > 1 and f["n"] > 1:
        g["delta"] = f["mean"] - s["mean"]
        g["ci_disjoint"] = bool(f["ci_hi"] < s["ci_lo"] or s["ci_hi"] < f["ci_lo"])
    else:
        g["delta"] = None
        g["ci_disjoint"] = False
    return g


def fmt_ci(d: dict) -> str:
    return "—" if not d["n"] else f"{d['mean']:.4f} [{d['ci_lo']:.4f},{d['ci_hi']:.4f}]"


def fmt_delta(g: dict) -> str:
    if g["delta"] is None:
        return "—"
    return f"{g['delta']:+.4f}" + ("†" if g["ci_disjoint"] else "")


def inter_analysis(D: dict[str, np.ndarray], rows: list[dict], status: np.ndarray, pct, seg_names, with_prompt):
    """구간 마지막 토큰(q=100) 끼리. 인접 전이 표 + 전 구간쌍 평균 행렬."""
    last = {r["seg"]: k for k, r in enumerate(rows) if r["pct"] == 100}
    chain = (["prompt"] if with_prompt else []) + list(seg_names)
    idx = [last[s] for s in chain]

    transitions = []
    for a, b in zip(chain, chain[1:]):
        row = {"transition": f"{a}→{b}"}
        for m, _ in METRICS:
            v = D[m][:, last[a], last[b]]
            row[m] = compare({st: v[status == st] for st in STATUSES})
        transitions.append(row)

    matrix = {}
    for m, _ in METRICS:
        sub = D[m][:, idx][:, :, idx]                                  # (n_ep, S, S)
        ms = {st: sub[status == st].mean(0) for st in STATUSES}
        matrix[m] = {"labels": chain,
                     "success": ms["success"].tolist(), "failure": ms["failure"].tolist(),
                     "delta": (ms["failure"] - ms["success"]).tolist()}
    return transitions, matrix


def intra_analysis(D: dict[str, np.ndarray], rows: list[dict], status: np.ndarray, pos: np.ndarray,
                   pct, seg_names, keep_coincident: bool):
    """구간 안 인접 % 쌍. pos (n_ep, n_seg, n_pct) 로 같은 토큰을 가리키는 쌍을 걸러낸다."""
    key = {(r["seg"], r["pct"]): k for k, r in enumerate(rows)}
    out = []
    for t, seg in enumerate(seg_names):
        for qa, qb in zip(pct, pct[1:]):
            ka, kb = key[(seg, qa)], key[(seg, qb)]
            same = pos[:, t, pct.index(qa)] == pos[:, t, pct.index(qb)]
            keep = np.ones_like(same) if keep_coincident else ~same
            row = {"seg": seg, "transition": f"{qa}→{qb}",
                   "n_coincident": {st: int((same & (status == st)).sum()) for st in STATUSES}}
            for m, _ in METRICS:
                v = D[m][:, ka, kb]
                row[m] = compare({st: v[keep & (status == st)] for st in STATUSES})
            out.append(row)
    return out


# ---------------------------------------------------------------------- 출력

def print_inter(transitions):
    for m, label in METRICS:
        print(f"\n[inter] {label}  (q=100, 구간 마지막 토큰)")
        print(f"{'transition':18s} {'success (95% CI)':>28s} {'failure (95% CI)':>28s} {'Δ(F−S)':>10s}")
        for r in transitions:
            g = r[m]
            print(f"{r['transition']:18s} {fmt_ci(g['success']):>28s} {fmt_ci(g['failure']):>28s} "
                  f"{fmt_delta(g):>10s}")
    print("† = 두 그룹의 95% CI 가 겹치지 않음")


def print_intra(intra, seg_names):
    for m, label in METRICS:
        print(f"\n[intra] {label}  (구간 안 인접 % 쌍)")
        print(f"{'seg':8s} {'q':8s} {'success (95% CI)':>28s} {'failure (95% CI)':>28s} {'Δ(F−S)':>10s} {'excl S/F':>9s}")
        for r in intra:
            g, c = r[m], r["n_coincident"]
            print(f"{r['seg']:8s} {r['transition']:8s} {fmt_ci(g['success']):>28s} {fmt_ci(g['failure']):>28s} "
                  f"{fmt_delta(g):>10s} {c['success']:>4d}/{c['failure']:<4d}")
    print("excl = 같은 토큰을 가리켜 거리 0 인 쌍 (기본은 제외)")


def _bars(ax, rows_, m, labels):
    x, w = np.arange(len(rows_)), 0.38
    for i, st in enumerate(STATUSES):
        d = [r[m][st] for r in rows_]
        mean = [v["mean"] or 0 for v in d]
        lo = [(v["mean"] - v["ci_lo"]) if v["n"] else 0 for v in d]
        hi = [(v["ci_hi"] - v["mean"]) if v["n"] else 0 for v in d]
        ax.bar(x + (i - 0.5) * w, mean, w, yerr=[lo, hi], capsize=2, color=COLOR[st],
               label=f"{st} (n={d[0]['n']})", linewidth=0)
    for j, r in enumerate(rows_):
        if r[m]["ci_disjoint"]:
            ax.text(j, max(r[m]["success"]["ci_hi"], r[m]["failure"]["ci_hi"]), "†",
                    ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_inter(transitions, out: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.6 * len(transitions) + 3), 4))
    for ax, (m, label) in zip(axes, METRICS):
        _bars(ax, transitions, m, [r["transition"] for r in transitions])
        ax.set_title(label, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_inter_matrix(matrix, out: Path, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for i, (m, label) in enumerate(METRICS):
        lab = matrix[m]["labels"]
        S, Fm, Dl = (np.array(matrix[m][k]) for k in ("success", "failure", "delta"))
        vmax = max(S.max(), Fm.max())
        dmax = float(np.abs(Dl).max()) or 1.0
        for j, (M, name, cmap, vmin, vmx) in enumerate((
                (S, "success", "Blues", 0, vmax), (Fm, "failure", "Blues", 0, vmax),
                (Dl, "Δ (failure − success)", "RdBu_r", -dmax, dmax))):
            ax = axes[i, j]
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmx)
            ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab)))
            ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(lab, fontsize=8)
            ax.set_title(f"{label} — {name}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_intra(intra, seg_names, pct, out: Path, title: str):
    labels = [f"{a}→{b}" for a, b in zip(pct, pct[1:])]
    fig, axes = plt.subplots(2, len(seg_names), figsize=(2.6 * len(seg_names) + 1, 6.5), sharey="row")
    for i, (m, label) in enumerate(METRICS):
        for t, seg in enumerate(seg_names):
            ax = axes[i, t]
            rows_ = [r for r in intra if r["seg"] == seg]
            x = np.arange(len(rows_))
            for st in STATUSES:
                d = [r[m][st] for r in rows_]
                mean = np.array([v["mean"] if v["n"] else np.nan for v in d])
                lo = np.array([(v["mean"] - v["ci_lo"]) if v["n"] else 0 for v in d])
                hi = np.array([(v["ci_hi"] - v["mean"]) if v["n"] else 0 for v in d])
                ax.errorbar(x, mean, yerr=[lo, hi], color=COLOR[st], marker="o", ms=4, lw=2,
                            capsize=2, label=st)
            for j, r in enumerate(rows_):
                if r[m]["ci_disjoint"]:
                    ax.text(j, max(r[m]["success"]["ci_hi"], r[m]["failure"]["ci_hi"]), "†",
                            ha="center", va="bottom", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(seg, fontsize=10)
            if t == 0:
                ax.set_ylabel(label, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="q% 위치 벡터쌍 거리: inter(구간 사이) / intra(구간 안)")
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--pct", type=int, nargs="+", default=list(DEFAULT_PCT), metavar="Q",
                    help="캐시 tag 를 정한다 (pct_cache.py 와 같은 목록)")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "latent" / "pct_vectors")
    ap.add_argument("--output", type=Path, default=None, help="기본 metric/output/pct_dist/<task>/<level>/<tag>[_prompt]")
    ap.add_argument("--with-prompt", action="store_true", help="프롬프트 마지막 토큰을 prompt→Step 1 전이에 넣는다")
    ap.add_argument("--keep-coincident", action="store_true", help="같은 토큰을 가리키는 %% 쌍(거리 0)도 intra 에 포함")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-save-pairs", dest="save_pairs", action="store_false", help="pairs.npz (K×K 전체 쌍 행렬) 저장 생략")
    a = ap.parse_args()

    pct = check_pct(a.pct)
    tag = make_tag(pct)
    out = a.output or ROOT / "metric" / "output" / "pct_dist" / a.task / a.level / (tag + ("_prompt" if a.with_prompt else ""))
    out.mkdir(parents=True, exist_ok=True)

    X, rows, episodes, pct, seg_names = load_cache(a.cache_dir, a.task, a.level, tag, STATUSES, a.with_prompt)
    status = np.array([e["status"] for e in episodes])
    pos = np.array([e["pos"] for e in episodes], np.int32)             # (n_ep, n_seg, n_pct)
    print(f"loaded {len(episodes)} episodes {dict(Counter(status.tolist()))}  K={len(rows)}  d={X.shape[-1]}  "
          f"seg={seg_names} pct={pct}", flush=True)

    euc, cos = pairwise(X, a.device)
    del X
    D = {"euc": euc, "cos": cos}

    if a.save_pairs:
        np.savez(out / "pairs.npz", euc=euc, cos=cos,
                 row_seg=np.array([r["seg"] for r in rows]), row_pct=np.array([r["pct"] for r in rows], np.int32),
                 seeds=np.array([e["seed"] for e in episodes]), status=status,
                 seg_len=np.array([e["seg_len"] for e in episodes], np.int32), pos=pos,
                 pct=np.array(pct, np.int32), seg_names=np.array(seg_names))

    title = f"{a.task} / {a.level} / {tag}  (n={len(episodes)}: " + \
            ", ".join(f"{st} {int((status == st).sum())}" for st in STATUSES) + ")"

    transitions, matrix = inter_analysis(D, rows, status, pct, seg_names, a.with_prompt)
    print_inter(transitions)
    with open(out / "inter.json", "w") as f:
        json.dump({"task": a.task, "level": a.level, "tag": tag, "with_prompt": a.with_prompt,
                   "n": dict(Counter(status.tolist())), "transitions": transitions}, f, indent=1)
    with open(out / "inter_matrix.json", "w") as f:
        json.dump(matrix, f, indent=1)
    plot_inter(transitions, out / "inter.png", "inter: adjacent segments at q=100 — " + title)
    plot_inter_matrix(matrix, out / "inter_matrix.png", "inter: all segment pairs at q=100 — " + title)

    intra = intra_analysis(D, rows, status, pos, pct, seg_names, a.keep_coincident)
    print_intra(intra, seg_names)
    with open(out / "intra.json", "w") as f:
        json.dump({"task": a.task, "level": a.level, "tag": tag, "keep_coincident": a.keep_coincident,
                   "n": dict(Counter(status.tolist())), "rows": intra}, f, indent=1)
    plot_intra(intra, seg_names, pct, out / "intra.png", "intra: adjacent % within segment — " + title)

    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
