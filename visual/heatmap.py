"""probe_gram.py 가 만든 .pt 를 토큰 간 유사도 히트맵으로 그린다.

행렬은 N x N (N = 에피소드 전체 토큰 수). 세그먼트는 그 위에 긋는 선일 뿐이고,
대각 블록이 스텝 내부, 비대각 블록이 스텝 간 관계다.

계산과 분리해 둔 이유
    유사도 행렬을 얻으려면 3B 모델을 올려 forward 를 돌려야 한다. 색상·정규화·
    어느 분절을 볼지는 몇 번이고 바꿔보게 되는데, 그때마다 모델을 다시 올릴
    이유가 없다. probe_gram.py 는 .pt 까지만 만들고 그림은 여기서 그린다.

    python visual/heatmap.py visual/success_BabyAI-GoToObj-v0_1.pt
    python visual/heatmap.py visual/*.pt --kind action
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def block_stats(C, blocks):
    """대각 블록 평균 vs 비대각 평균. 그림 없이도 대비를 숫자로 본다."""
    diag = [C[s:e, s:e].mean() for _, s, e in blocks]
    off = [C[s1:e1, s2:e2].mean()
           for i, (_, s1, e1) in enumerate(blocks)
           for _, s2, e2 in blocks[i + 1:]]
    return float(np.mean(diag)), float(np.mean(off)) if off else float("nan")


def draw(ax, C, blocks, title, vmin, vmax):
    im = ax.imshow(C, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    for _, s, e in blocks:
        for v in (s, e):
            ax.axhline(v - 0.5, lw=0.6, c="k")
            ax.axvline(v - 0.5, lw=0.6, c="k")
    # 블록이 많으면 라벨이 겹쳐 못 읽는다 — 20개 넘어가면 눈금만 남긴다.
    if len(blocks) <= 20:
        pos = [(s + e) / 2 for _, s, e in blocks]
        names = [n for n, _, _ in blocks]
        ax.set_xticks(pos); ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticks(pos); ax.set_yticklabels(names, fontsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    d, o = block_stats(C, blocks)
    ax.set_title(f"{title}   {len(blocks)} blocks\ndiag {d:+.3f}  off {o:+.3f}", fontsize=10)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", type=Path, nargs="+")
    ap.add_argument("--kind", choices=["step", "action", "both"], default="both")
    ap.add_argument("--raw", action="store_true",
                    help="코사인 대신 내적 S 를 그린다 (노름 차이에 지배됨)")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    for path in args.pt:
        z = torch.load(path, map_location="cpu", weights_only=False)
        meta = z["meta"]
        M = (z["S"] if args.raw else z["C"]).numpy()
        N = M.shape[0]

        kinds = ["step", "action"] if args.kind == "both" else [args.kind]
        blocks = {k: z["blocks"][k] for k in kinds}

        # 내적은 값 범위가 제각각이라 분위수로 자른다. 코사인은 [-1,1] 고정.
        if args.raw:
            vmax = float(np.percentile(np.abs(M), 99)); vmin = -vmax
        else:
            vmin, vmax = -1.0, 1.0

        fig, axes = plt.subplots(1, len(kinds), figsize=(8.5 * len(kinds), 8.5),
                                 squeeze=False)
        for ax, k in zip(axes[0], kinds):
            im = draw(ax, M, blocks[k], k, vmin, vmax)
        fig.suptitle(f"{meta['id']}   task={meta['task']}  success={meta['success']}   "
                     f"N={N}  {'inner product' if args.raw else 'cosine'}")
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02)

        out = path.with_suffix(".raw.png" if args.raw else ".png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")
        for k in kinds:
            d, o = block_stats(M, blocks[k])
            print(f"  {k:6} diag {d:+.4f}  off {o:+.4f}  gap {d - o:+.4f}")


if __name__ == "__main__":
    main()
