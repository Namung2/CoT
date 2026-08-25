"""probe_gram.py 가 만든 .pt 를 토큰 간 유사도 히트맵으로 그린다.

행렬은 N x N (N = 에피소드 전체 토큰 수). 세그먼트는 그 위에 긋는 선일 뿐이고,
대각 블록이 스텝 내부, 비대각 블록이 스텝 간 관계다.

세 가지를 낸다.
    *.<which>.png   전체 행렬. step / action 두 분절을 나란히.
    *.drift.png     --drift. prefix x prefix. 시점 s 의 상태와 시점 t 의 상태가
                    닮았는가 — 토큰끼리가 아니라 "여기까지 왔을 때의 요약"끼리다.
    *.growth.png    --growth. prefix 길이 k 에 따른 gap 곡선. step 과 action 을 한
                    축에 겹쳐 그린다 — 어느 쪽이 실제 단위인지는 나란히 봐야 안다.

prefix 스냅샷을 격자로 찍지 않는 이유
    Qwen 은 causal 이라 h_i 가 뒤 토큰의 영향을 받지 않고, 내적·코사인은 쌍마다
    독립이다. 그래서 앞 k 개로 만든 행렬은 전체 행렬의 왼쪽 위 k x k 와 숫자까지
    같다 — 그림을 늘어놔도 같은 한 장을 조금씩 드러내는 것뿐이다. k 에 따라 실제로
    변하는 것은 집합 전체에 의존하는 값(중심화 평균, 그리고 통계량)뿐이라,
    곡선으로 보는 것이 맞다.

계산과 분리해 둔 이유
    유사도 행렬을 얻으려면 모델을 올려 forward 를 돌려야 한다. 색상·정규화·
    어느 분절을 볼지는 몇 번이고 바꿔보게 되는데, 그때마다 모델을 다시 올릴
    이유가 없다.

    python visual/heatmap.py visual/*.pt
    python visual/heatmap.py visual/*.pt --drift --growth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

STORED = {"centered": "C_centered", "cosine": "C", "inner": "S"}


# ── 통계 ────────────────────────────────────────────────────────────────

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
    if N < 2 or len(blocks) < 2:
        return np.nan, np.nan
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
    if N < 4 or len(blocks) < 2:
        return float("nan"), 0, []
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


# ── prefix ──────────────────────────────────────────────────────────────

def clip_blocks(blocks, k):
    """앞 k 개 토큰 범위로 자른 블록. 완전히 밖이면 버리고 걸치면 끝을 자른다."""
    out = []
    for name, s, e in blocks:
        if s >= k:
            break
        out.append((name, s, min(e, k)))
    return out


def prefix_matrix(H, k, which, global_mean=None):
    X = H[:k]
    if which == "centered":
        X = X - (global_mean if global_mean is not None else X.mean(axis=0, keepdims=True))
    if which == "inner":
        return X @ X.T
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    X = X / n
    return np.clip(X @ X.T, -1.0, 1.0)


# ── 그리기 ──────────────────────────────────────────────────────────────

def draw(ax, C, blocks, title, vmin, vmax, labels=True):
    im = ax.imshow(C, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    for _, s, e in blocks:
        for v in (s, e):
            ax.axhline(v - 0.5, lw=0.5, c="k")
            ax.axvline(v - 0.5, lw=0.5, c="k")
    # 블록이 많으면 라벨이 겹쳐 못 읽는다 — 20개 넘어가면 눈금만 남긴다.
    if labels and len(blocks) <= 20:
        pos = [(s + e) / 2 for _, s, e in blocks]
        ax.set_xticks(pos); ax.set_xticklabels([n for n, _, _ in blocks],
                                               rotation=90, fontsize=7)
        ax.set_yticks(pos); ax.set_yticklabels([n for n, _, _ in blocks], fontsize=7)
    else:
        ax.set_xticks([]); ax.set_yticks([])

    w, b = block_stats(C, blocks)
    dm, nb, _ = distance_matched(C, blocks)
    sizes = [e - s for _, s, e in blocks] or [0]
    ax.set_title(f"{title}   {len(blocks)} blocks, size median {int(np.median(sizes))}\n"
                 f"within {w:+.4f}  between {b:+.4f}  gap {w - b:+.4f}\n"
                 f"distance-matched gap {dm:+.4f}  ({nb} bins)", fontsize=9)
    return im


def drift_matrices(K):
    """prefix 시점끼리의 유사도. 토큰 x 토큰이 아니라 시점 x 시점이다.

    시점 t 의 상태를 그때까지 본 토큰 전체의 요약으로 잡는다.
        G_t = E_t^T E_t = sum_{i<=t} h_i h_i^T      (d x d)
    토큰이 하나 늘 때마다 rank-1 항이 더해지므로 G_t 는 실제로 자란다 — 토큰 그람
    K 처럼 "부분행렬이라 값이 그대로"인 것과 다르다.

    G_t 를 만들지는 않는다. d=5120 이면 218 개가 22TB 다. 대신
        tr(h_i h_i^T h_j h_j^T) = (h_i . h_j)^2
    이라 d 가 식에서 사라지고
        <G_s, G_t>_F = sum_{i<=s} sum_{j<=t} K_ij^2
    가 된다. K 를 원소별 제곱해 2D 누적합을 한 번 만들면 모든 쌍이 O(1) 이다.

    두 가지를 낸다.
      C2  : 위 정의 그대로 정규화한 것.
      CKA : 각 prefix 를 자기 평균으로 중심화한 뒤 같은 비교. 중심화는 이중중심화
            J_s K[:s,:t] J_t 로 K 에서 직접 되므로 여기서도 H 가 필요 없다.
    """
    N = K.shape[0]
    s_i = np.arange(1, N + 1)[:, None]
    t_j = np.arange(1, N + 1)[None, :]

    def cum2(A):
        return np.cumsum(np.cumsum(A, 0), 1)

    P = cum2(K * K)
    dP = np.clip(np.diag(P), 1e-12, None)
    C2 = P / np.sqrt(np.outer(dP, dP))

    # ||J_s K[:s,:t] J_t||_F^2 을 누적합으로 편 것. 직접 이중중심화하면 쌍마다
    # O(N^2) 이라 전체 O(N^4) 인데, 이렇게 하면 O(N^2) 에 끝난다.
    T = cum2(K)
    RS = np.cumsum(K, axis=1); Q1 = np.cumsum(RS * RS, axis=0)
    CS = np.cumsum(K, axis=0); Q2 = np.cumsum(CS * CS, axis=1)
    F = P - Q1 / t_j - Q2 / s_i + T ** 2 / (s_i * t_j)
    dF = np.clip(np.diag(F), 1e-12, None)
    CKA = np.clip(F / np.sqrt(np.outer(dF, dF)), 0.0, 1.0)

    return {"C2": C2, "CKA": CKA}


def draw_drift(K, blocks, meta, N, out, dpi):
    mats = drift_matrices(K)
    fig, axes = plt.subplots(1, len(mats), figsize=(8.2 * len(mats), 7.4),
                             squeeze=False)
    for ax, (name, M) in zip(axes[0], mats.items()):
        # 값이 1 근처에 몰린다 — G_t 가 G_s 를 품고 있어서 공유 항이 지배한다.
        # 0~1 로 고정하면 전부 노랗게 뭉치므로 하위 분위수부터 색을 편다.
        off = M[np.triu_indices(N, k=1)]
        vmin = float(np.percentile(off, 1))
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=1.0,
                       interpolation="nearest")
        # 블록 경계는 prefix 축 위의 위치이기도 하다. 상태가 급변하면 그 자리에
        # 가로/세로 줄이 서고, 그 줄이 경계와 겹치는지가 볼 것이다.
        for _, s, _ in blocks:
            ax.axhline(s - 0.5, lw=0.5, c="w", alpha=0.8)
            ax.axvline(s - 0.5, lw=0.5, c="w", alpha=0.8)
        ax.set_title(f"{name}   median {np.median(off):.4f}", fontsize=11)
        ax.set_xlabel("prefix t"); ax.set_ylabel("prefix s")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{meta['id']}  success={meta['success']}  N={N}  "
                 f"prefix x prefix  {meta['model']}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")
    for name, M in mats.items():
        off = M[np.triu_indices(N, k=1)]
        print(f"  {name:4} median {np.median(off):.4f}  p10 {np.percentile(off,10):.4f}"
              f"  min {off.min():.4f}")


def draw_growth(H, blocks_by_kind, kinds, which, gm, meta, N, out, dpi, points):
    step = max(1, N // points)
    grid = list(range(4, N + 1, step))
    curves = {k: [] for k in kinds}
    for k in grid:
        C = prefix_matrix(H, k, which, gm)
        for kind in kinds:
            bl = clip_blocks(blocks_by_kind[kind], k)
            w, b = block_stats(C, bl)
            dm, _, _ = distance_matched(C, bl)
            curves[kind].append((w, b, dm))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, lw=0.6, c="k")
    for kind, color in zip(kinds, ("tab:blue", "tab:orange")):
        a = np.array(curves[kind])
        ax.plot(grid, a[:, 0] - a[:, 1], lw=1.8, c=color, label=f"{kind}  gap")
        ax.plot(grid, a[:, 2], lw=1.8, ls="--", c=color,
                label=f"{kind}  distance-matched gap")
    # 블록 경계에서 곡선이 꺾이는지 본다. action 은 step 을 세분한 것이라
    # action 경계를 그으면 step 경계도 그 안에 들어간다.
    for _, s, _ in blocks_by_kind[kinds[-1]]:
        ax.axvline(s, lw=0.4, c="gray", alpha=0.5)
    ax.set_xlabel("prefix length k (tokens)")
    ax.set_ylabel("mean cosine")
    ax.set_title(f"{meta['id']}  success={meta['success']}  N={N}  "
                 f"{'global' if gm is not None else 'prefix'}-centered  {meta['model']}")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")
    for kind in kinds:
        w, b, dm = curves[kind][-1]
        print(f"  {kind:6} k={grid[-1]}  within {w:+.4f}  between {b:+.4f}"
              f"  gap {w - b:+.4f}  distance-matched {dm:+.4f}")


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
    ap.add_argument("--drift", action="store_true",
                    help="prefix x prefix 히트맵(C2, CKA)을 추가로 낸다. S 만 있으면 됨")
    ap.add_argument("--growth", action="store_true",
                    help="k 에 따른 gap 곡선을 추가로 낸다. H 필요")
    ap.add_argument("--curve-points", type=int, default=150)
    ap.add_argument("--global-center", action="store_true",
                    help="prefix 마다가 아니라 전체 평균으로 중심화한다")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    for path in args.pt:
        z = torch.load(path, map_location="cpu", weights_only=False)
        meta = z["meta"]
        key = STORED[args.which]
        if key not in z:      # 옛 형식의 .pt
            raise SystemExit(f"{path}: '{key}' 없음 — probe_gram.py 를 다시 돌리세요")
        M = z[key].numpy()
        N = M.shape[0]
        kinds = ["step", "action"] if args.kind == "both" else [args.kind]
        blocks = {k: z["blocks"][k] for k in kinds}

        # ── 전체 행렬 ──────────────────────────────────────────────────
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
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)

        print(f"{out}   N={N}")
        for k in kinds:
            w, b = block_stats(M, blocks[k])
            dm, nb, rows = distance_matched(M, blocks[k])
            sizes = [e - s for _, s, e in blocks[k]]
            print(f"  {k:6} blocks {len(blocks[k]):3} size median {int(np.median(sizes)):4}"
                  f"   within {w:+.4f}  between {b:+.4f}  gap {w - b:+.4f}"
                  f"   distance-matched {dm:+.4f} ({nb} bins)")
            # 거리 구간별 표는 항상 찍는다. 근거리에서만 gap 이 있고 멀어지면
            # 0 이면 그건 인접성이지 블록 구조가 아니다 — 요약값만 보면 구분이 안 된다.
            for lo, hi, nw, nbw, mw, mb in rows:
                print(f"      d[{lo:4},{hi:4})  n {nw:6}/{nbw:7}"
                      f"   within {mw:+.4f}  between {mb:+.4f}  gap {mw - mb:+.4f}")

        # ── prefix x prefix (K 만 있으면 됨) ───────────────────────────
        if args.drift:
            draw_drift(z["S"].numpy().astype(np.float64), blocks[kinds[-1]], meta, N,
                       path.with_suffix(".drift.png"), args.dpi)

        # ── prefix 성장 곡선 (H 필요) ──────────────────────────────────
        if not args.growth:
            continue
        if "H" not in z:
            raise SystemExit(f"{path}: 'H' 없음 — --growth 는 hidden state "
                             f"원본이 필요합니다. probe_gram.py 를 다시 돌리세요")
        H = z["H"].float().numpy()
        gm = H.mean(axis=0, keepdims=True) if args.global_center else None

        draw_growth(H, blocks, kinds, args.which, gm, meta, N,
                        path.with_suffix(".growth.png"), args.dpi, args.curve_points)


if __name__ == "__main__":
    main()
