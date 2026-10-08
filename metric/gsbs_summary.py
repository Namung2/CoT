"""gsbs_batch.py 가 쌓은 jsonl 을 모아 "스텝 안에서 몇 개로 나뉘고 어디서 나뉘는지"를 집계한다.

집계 단위: (task, level, status, config). config = tokens | spectral_k{k}_{sign}.
level="ALL" 행은 그 task 의 모든 level 을 합친 것.

지표 (구간 = step 1..N + answer, skipped 구간은 제외):
    mean_splits   구간당 GSBS 내부 분할 수 평균
    ci95          그 평균의 95% 신뢰구간 반폭 (1.96 · std / √구간수)
    frac_front20  분할 위치가 구간 앞 20% 안에 있는 비율 (리셋 아티팩트 의심 지표)
    mean_rel_pos  분할 위치의 구간 내 상대 좌표 평균
    sat_rate      kmax 포화 구간 비율 (높으면 --kmax 를 올려야 함)

출력 (<out_dir>/):
    summary.csv        (task, level, status, config) 별 한 줄
    by_step.csv        위에 step 이름(Step 1..N, answer)까지 나눈 것
    relpos_hist.csv    분할 상대 위치 10칸 히스토그램 (개수)
    fig/<task>/<level>_splits.png    config 별 구간당 분할 수, 성공 vs 실패 (에러바 = 95% CI)
    fig/<task>/<level>_relpos.png    config 별 분할 상대 위치 분포, 성공 vs 실패
    fig/<task>/<level>_by_step.png   config 별 step 순번에 따른 구간당 분할 수

    python metric/gsbs_summary.py
    python metric/gsbs_summary.py --batch-dir latent/gsbs/batch --out-dir latent/gsbs/summary
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "metric"))

from gsbs_batch import parse_config_tag                                     # noqa: E402

N_BINS = 10
FRONT = 0.2
COLOR = {"success": "#2a78d6", "failure": "#eb6834"}     # 고정 (dataviz 팔레트 slot 1, 2)
STATUSES = ("success", "failure")


def config_order(tag: str) -> tuple:
    c = parse_config_tag(tag)
    km = c["kmax"] or 0
    if c["input"] == "tokens":
        return (km, 0, 0, "")
    return (km, 1, c["k"], c["sign_mode"])


# ---------------------------------------------------------------------- 읽기

def load_rows(batch_dir: Path) -> list[dict]:
    """구간 하나 = 행 하나. skipped 구간도 일단 들고 온다 (skip 비율 계산용)."""
    rows = []
    for f in sorted(batch_dir.glob("*/*/*/*.jsonl")):
        status, level, task = f.parent.name, f.parents[1].name, f.parents[2].name
        tag = f.stem
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                n_seg = ep["n_segments"]
                for s in ep["segments"]:
                    rows.append({"task": task, "level": level, "status": status, "config": tag,
                                 "seed": ep["seed"], "t": s["t"], "name": s["name"],
                                 "is_answer": s["t"] == n_seg - 1,
                                 "n_tokens": s["n_tokens"], "skipped": s["skipped"],
                                 "n_splits": len(s["splits"]), "rel_pos": s["rel_pos"],
                                 "kmax_saturated": s["kmax_saturated"]})
    return rows


# ---------------------------------------------------------------------- 집계

def agg(rows: list[dict]) -> dict:
    n_ep = len({r["seed"] for r in rows})
    ran = [r for r in rows if not r["skipped"]]
    splits = np.array([r["n_splits"] for r in ran], dtype=float)
    rel = np.array([p for r in ran for p in r["rel_pos"]], dtype=float)
    ci = 1.96 * splits.std(ddof=1) / np.sqrt(splits.size) if splits.size > 1 else float("nan")
    hist = np.histogram(rel, bins=N_BINS, range=(0.0, 1.0))[0] if rel.size else np.zeros(N_BINS, int)
    return {
        "n_episodes": n_ep, "n_segments": len(ran), "n_skipped": len(rows) - len(ran),
        "sat_rate": float(np.mean([r["kmax_saturated"] for r in ran])) if ran else float("nan"),
        "mean_splits": float(splits.mean()) if splits.size else float("nan"),
        "ci95": float(ci), "median_splits": float(np.median(splits)) if splits.size else float("nan"),
        "frac_zero": float((splits == 0).mean()) if splits.size else float("nan"),
        "n_splits_total": int(splits.sum()),
        "frac_front20": float((rel < FRONT).mean()) if rel.size else float("nan"),
        "mean_rel_pos": float(rel.mean()) if rel.size else float("nan"),
        "hist": hist.tolist(),
    }


def group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def with_pooled(rows: list[dict]) -> list[dict]:
    """task 마다 level="ALL" 복사본을 덧붙인다 (seed 충돌 방지로 level 을 seed 에 섞는다)."""
    pooled = [{**r, "level": "ALL", "seed": f"{r['level']}/{r['seed']}"} for r in rows]
    return rows + pooled


# ---------------------------------------------------------------------- 그림

def _configs(rows):
    return sorted({r["config"] for r in rows}, key=config_order)


def _label(tag: str) -> str:
    c = parse_config_tag(tag)
    lab = "tokens" if c["input"] == "tokens" else f"k{c['k']}\n{c['sign_mode']}"
    return lab + (f"\nkmax{c['kmax']}" if c["kmax"] else "")


def fig_splits(rows: list[dict], out: Path, title: str):
    import matplotlib.pyplot as plt
    cfgs = _configs(rows)
    by = group(rows, ("config", "status"))
    x = np.arange(len(cfgs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.1 * len(cfgs) + 2.5, 4))
    for i, st in enumerate(STATUSES):
        m, e = [], []
        for c in cfgs:
            a = agg(by.get((c, st), [])) if (c, st) in by else None
            m.append(a["mean_splits"] if a else np.nan)
            e.append(a["ci95"] if a else np.nan)
        ax.bar(x + (i - 0.5) * (w + 0.02), m, w, yerr=e, capsize=3, color=COLOR[st],
               label=st, linewidth=0, error_kw={"elinewidth": 1, "ecolor": "#52514e"})
    ax.set_xticks(x); ax.set_xticklabels([_label(c) for c in cfgs], fontsize=8)
    ax.set_ylabel("splits per segment (mean, 95% CI)")
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e5e5e2", linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_relpos(rows: list[dict], out: Path, title: str):
    import matplotlib.pyplot as plt
    cfgs = _configs(rows)
    by = group(rows, ("config", "status"))
    n = len(cfgs)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 1, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    w = 0.42 / N_BINS
    for ax, c in zip(axes, cfgs):
        for i, st in enumerate(STATUSES):
            if (c, st) not in by:
                continue
            h = np.array(agg(by[(c, st)])["hist"], dtype=float)
            dens = h / h.sum() if h.sum() else h
            ax.bar(centers + (i - 0.5) * w, dens, w, color=COLOR[st], label=st, linewidth=0)
        ax.axvline(FRONT, color="#52514e", linewidth=0.8, linestyle=":")
        ax.set_title(_label(c).replace("\n", " "), fontsize=8)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("fraction of splits")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title} — split position within segment (0=start, dotted=20%)", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_by_step(rows: list[dict], out: Path, title: str):
    import matplotlib.pyplot as plt
    cfgs = _configs(rows)
    names = sorted({r["name"] for r in rows}, key=lambda s: (s == "answer", s))
    by = group(rows, ("config", "status", "name"))
    n = len(cfgs)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 1, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(names))
    for ax, c in zip(axes, cfgs):
        for st in STATUSES:
            m, e = [], []
            for nm in names:
                a = agg(by[(c, st, nm)]) if (c, st, nm) in by else None
                m.append(a["mean_splits"] if a else np.nan)
                e.append(a["ci95"] if a else np.nan)
            ax.errorbar(x, m, yerr=e, color=COLOR[st], label=st, linewidth=2,
                        marker="o", markersize=4, capsize=2, elinewidth=0.8)
        ax.set_title(_label(c).replace("\n", " "), fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([nm.replace("Step ", "S") for nm in names], fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e5e5e2", linewidth=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("splits per segment")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title} — by step", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------- 실행

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", type=Path, default=ROOT / "latent" / "gsbs" / "batch")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "latent" / "gsbs" / "summary")
    ap.add_argument("--no-fig", dest="fig", action="store_false")
    a = ap.parse_args()

    rows = load_rows(a.batch_dir)
    if not rows:
        raise SystemExit(f"no jsonl under {a.batch_dir}")
    rows = with_pooled(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    base_keys = ("task", "level", "status", "config")
    summary, by_step, hist = [], [], []
    for key, rs in sorted(group(rows, base_keys).items(),
                          key=lambda kv: (kv[0][0], kv[0][1] != "ALL", kv[0][1], kv[0][2] != "success", config_order(kv[0][3]))):
        meta = dict(zip(base_keys, key))
        meta.update(parse_config_tag(key[3]))
        st = agg(rs)
        h = st.pop("hist")
        summary.append({**meta, **st})
        hist.append({**meta, **{f"bin{i}": v for i, v in enumerate(h)}})
        for nm, rs2 in sorted(group(rs, ("name",)).items(), key=lambda kv: (kv[0][0] == "answer", kv[0][0])):
            s2 = agg(rs2); s2.pop("hist")
            by_step.append({**meta, "name": nm[0], **s2})

    write_csv(a.out_dir / "summary.csv", summary)
    write_csv(a.out_dir / "by_step.csv", by_step)
    write_csv(a.out_dir / "relpos_hist.csv", hist)

    # 콘솔 표 (level 별 + ALL)
    print(f"{'task':9s} {'level':44s} {'status':8s} {'config':18s} {'ep':>4s} {'seg':>5s} "
          f"{'splits':>7s} {'±ci':>5s} {'front20':>7s} {'sat':>5s}")
    for r in summary:
        print(f"{r['task']:9s} {r['level'][:44]:44s} {r['status']:8s} {r['config']:18s} "
              f"{r['n_episodes']:4d} {r['n_segments']:5d} {r['mean_splits']:7.2f} "
              f"{r['ci95']:5.2f} {r['frac_front20']:7.0%} {r['sat_rate']:5.0%}")
    print(f"saved -> {a.out_dir}/summary.csv, by_step.csv, relpos_hist.csv")

    if not a.fig:
        return
    for (task, level), rs in group(rows, ("task", "level")).items():
        fdir = a.out_dir / "fig" / task
        fdir.mkdir(parents=True, exist_ok=True)
        title = f"{task}/{level}"
        fig_splits(rs, fdir / f"{level}_splits.png", title)
        fig_relpos(rs, fdir / f"{level}_relpos.png", title)
        fig_by_step(rs, fdir / f"{level}_by_step.png", title)
    print(f"saved figures -> {a.out_dir}/fig/")


if __name__ == "__main__":
    main()
