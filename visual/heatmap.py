"""probe_gram.py 가 만든 .pt 를 토큰 간 유사도 히트맵으로 그린다.

행렬은 N x N (N = 에피소드 전체 토큰 수). 세그먼트는 그 위에 긋는 선일 뿐이고,
대각 블록이 스텝 내부, 비대각 블록이 스텝 간 관계다.

계산과 분리해 둔 이유
    유사도 행렬을 얻으려면 모델을 올려 forward 를 돌려야 한다. 색상·정규화·
    어느 분절을 볼지는 몇 번이고 바꿔보게 되는데, 그때마다 모델을 다시 올릴
    이유가 없다. probe_gram.py 는 .pt 까지만 만들고 그림은 여기서 그린다.

    python visual/heatmap.py visual/success_*.pt
    python visual/heatmap.py visual/*.pt --kind action --which cosine
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def block_labels(N, blocks):
    """토큰 -> 블록 번호. 어느 블록에도 안 들어가면 -1."""
    lab = np.full(N, -1, dtype=int)
    for k, (_, s, e) in enumerate(blocks):
        lab[s:e] = k
    return lab


def block_stats(C, blocks):
    """블록 내 쌍 평균 vs 블록 간 쌍 평균.

    자기유사도(i==j, 항상 1.0)는 뺀다. 크기 n 블록에서 1/n 을 깔고 들어가는데,
    작은 블록일수록 그 몫이 커서(5 토큰이면 +0.2) 없는 신호가 생긴다. 블록 크기가
    성공/실패에서 다르면 그 차이가 그대로 "구조 차이"로 둔갑한다.
    """
    N = C.shape[0]
    lab = block_labels(N, blocks)
    i, j = np.triu_indices(N, k=1)          # i<j 라 자기유사도가 애초에 없다
    same = (lab[i] == lab[j]) & (lab[i] >= 0)
    v = C[i, j]
    w = v[same].mean() if same.any() else np.nan
    b = v[~same].mean() if (~same).any() else np.nan
    return float(w), float(b)


def distance_matched(C, blocks, nbins=24, min_count=20):
    """토큰 거리를 맞춘 뒤 블록 내 vs 블록 간 비교.

    그냥 블록 내/간을 비교하면 인접성에 속는다 — 같은 블록의 토큰은 원래 서로
    가까이 있고, hidden state 는 거리가 가까울수록 닮는다. 블록 구조와 무관하게
    대각 근처가 밝다. 거리 d 를 고정하고 같은 d 안에서 블록 내 쌍과 블록 간 쌍을
    비교하면 그 효과가 상쇄된다.

    거리 분포가 근거리에 크게 쏠려 로그 간격으로 구간을 나눈다. 양쪽 표본이
    min_count 미만인 구간은 버린다(먼 거리는 블록 내 쌍이 거의 없다).
    반환: (구간 평균 gap, 사용된 구간 수, 구간별 표)
    """
    N = C.shape[0]
    lab = block_labels(N, blocks)
    i, j = np.triu_indices(N, k=1)
    d = j - i
    same = (lab[i] == lab[j]) & (lab[i] >= 0)
    v = C[i, j]

    edges = np.unique(np.round(np.logspace(0, np.log10(max(N, 2)), nbins)).astype(int))
    rows, gaps = [], []
    for lo, hi in zip(edges, edges[1:]):
        m = (d >= lo) & (d < hi)
        sw, sb = v[m & same], v[m & ~same]
        if len(sw) < min_count or len(sb) < min_count:
            continue
        gaps.append(sw.mean() - sb.mean())
        rows.append((lo, hi, len(sw), len(sb), sw.mean(), sb.mean()))
    return (float(np.mean(gaps)) if gaps else float("nan")), len(gaps), rows


def draw(ax, C, blocks, title, vmin, vmax):
    im = ax.imshow(C, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    for _, s, e in blocks:
        for v in (s, e):
            ax.axhline(v - 0.5, lw=0.6, c="k")
            ax.axvline(v - 0.5, lw=0.6, c="k")
    # 블록이 많으면 라벨이 겹쳐 못 읽는다 — 20개 넘어가면 눈금만 남긴다.
    if len(blocks) <= 20:
        pos = [(s + e) / 2 for _, s, e in blocks]
        ax.set_xticks(pos); ax.set_xticklabels([n for n, _, _ in blocks],
                                               rotation=90, fontsize=7)
        ax.set_yticks(pos); ax.set_yticklabels([n for n, _, _ in blocks], fontsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])

    w, b = block_stats(C, blocks)
    dm, nb, _ = distance_matched(C, blocks)
    sizes = [e - s for _, s, e in blocks]
    ax.set_title(f"{title}   {len(blocks)} blocks, size median {int(np.median(sizes))}\n"
                 f"within {w:+.4f}  between {b:+.4f}  gap {w - b:+.4f}\n"
                 f"거리 맞춘 gap {dm:+.4f}  ({nb} bins)", fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", type=Path, nargs="+")
    ap.add_argument("--kind", choices=["step", "action", "both"], default="both")
    # 기본이 centered 인 이유: residual stream 의 공통 평균 방향 때문에 raw 코사인은
    # 무엇이든 +0.4~0.6 으로 뭉쳐 나온다. 공통 성분을 뺀 쪽이 구조를 보여준다.
    ap.add_argument("--which", choices=["centered", "cosine", "inner"],
                    default="centered",
                    help="centered=중심화 코사인(기본) / cosine=원본 코사인 / inner=내적")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    for path in args.pt:
        z = torch.load(path, map_location="cpu", weights_only=False)
        meta = z["meta"]
        key = {"centered": "C_centered", "cosine": "C", "inner": "S"}[args.which]
        if key not in z:      # 옛 형식의 .pt
            raise SystemExit(f"{path}: '{key}' 없음 — probe_gram.py 를 다시 돌리세요")
        M = z[key].numpy()
        N = M.shape[0]

        kinds = ["step", "action"] if args.kind == "both" else [args.kind]
        blocks = {k: z["blocks"][k] for k in kinds}

        # 내적은 값 범위가 제각각이라 분위수로 자른다. 코사인은 [-1,1] 고정.
        if args.which == "inner":
            vmax = float(np.percentile(np.abs(M), 99)); vmin = -vmax
        else:
            vmin, vmax = -1.0, 1.0

        fig, axes = plt.subplots(1, len(kinds), figsize=(8.5 * len(kinds), 9),
                                 squeeze=False)
        for ax, k in zip(axes[0], kinds):
            im = draw(ax, M, blocks[k], k, vmin, vmax)
        fig.suptitle(f"{meta['id']}   task={meta['task']}  success={meta['success']}   "
                     f"N={N}  {args.which}  {meta['model']}")
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02)

        out = path.with_suffix(f".{args.which}.png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"{out}   N={N}")
        for k in kinds:
            w, b = block_stats(M, blocks[k])
            dm, nb, rows = distance_matched(M, blocks[k])
            sizes = [e - s for _, s, e in blocks[k]]
            print(f"  {k:6} blocks {len(blocks[k]):3} size median {int(np.median(sizes)):4}"
                  f"   within {w:+.4f}  between {b:+.4f}  gap {w - b:+.4f}"
                  f"   거리맞춤 {dm:+.4f} ({nb} bins)")
            # 거리 구간별 표는 항상 찍는다. 근거리에서만 gap 이 있고 멀어지면
            # 0 이면 그건 인접성이지 블록 구조가 아니다 — 요약값만 보면 구분이 안 된다.
            for lo, hi, nw, nbw, mw, mb in rows:
                print(f"      d[{lo:4},{hi:4})  n {nw:6}/{nbw:7}"
                      f"   within {mw:+.4f}  between {mb:+.4f}  gap {mw - mb:+.4f}")


if __name__ == "__main__":
    main()
