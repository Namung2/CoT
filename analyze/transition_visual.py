"""궤적 하나하나를 PCA 3차원 공간에 개별로 그려서 전이 구조를 눈으로 확인.
analyze.py 는 건드리지 않고 함수만 재사용.

시각화 전용으로 n_components=3 PCA를 새로 fit한다 (분석용 50차원 PCA와는 별개).
PCA 성분은 분산 순으로 정렬되므로, 몇 차원짜리로 fit하든 top-3는 항상 동일하다.

여러 궤적을 한 좌표계에 겹쳐 그리면 스파게티처럼 얽혀서 개별 궤적의 모양이
안 보이므로, 궤적마다 별도의 3D 서브플롯(작은 격자)으로 나눠 그린다. 점은 진행도
(t/T) 순서로 잇고, step 번호를 점 옆에 작게 표기한다. 선이 한 방향으로 부드럽게
이어지면 구조가 있다는 시각적 증거, 제자리에서 요동치면 구조가 약하다는 뜻.
(PC1~3 설명 분산이 낮으면 이 그림만으로 "구조 없음"이라 단정하면 안 됨 — 콘솔에
찍히는 explained variance 를 같이 봐야 함.)

사용법:
    python analyze/transition_visual.py --level gotoseq_step10 --case c3 --tag k8_scaled_signfix
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  3d projection 등록용
from sklearn.decomposition import PCA

from analyze import ROOT, discover_levels, discover_cases, load_meta, build_pool, SEED


def run(level: str, case: str, tag: str, method: str,
        data_dir: Path, emb_dir: Path, out_root: Path, n_trajs: int = 6):
    print(f"\n=== {level} / {case} ({method}_{tag}) ===")
    meta = load_meta(data_dir / level / "cases" / f"{case}.jsonl")
    emb_path = emb_dir / level / "cases" / case / f"{method}_{tag}.pt"
    X, labels, traj_ids, ts = build_pool(emb_path, meta, level, case)

    pca = PCA(n_components=3, svd_solver="randomized", random_state=SEED)
    Xp3 = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    print(f"  PC1-3 explained var: {ev.round(3).tolist()} (sum {ev.sum():.3f})")

    rng = np.random.RandomState(SEED)
    sids = np.unique(traj_ids)
    picked = rng.choice(sids, size=min(n_trajs, len(sids)), replace=False)

    ncols = min(3, len(picked))
    nrows = int(np.ceil(len(picked) / ncols))
    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows))
    cmap = plt.get_cmap("viridis")

    for k, sid in enumerate(picked):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        mm = (traj_ids == sid) & (ts > 0)   # mission 제외
        order = np.argsort(ts[mm])
        pts = Xp3[mm][order]
        steps = ts[mm][order]
        prog = steps.astype(float)
        prog = prog / prog.max() if prog.max() > 0 else prog

        for i in range(len(pts) - 1):
            ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], pts[i:i + 2, 2],
                    color=cmap(prog[i]), linewidth=1.5, alpha=0.9)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=[cmap(p) for p in prog], s=18)
        ax.scatter(*pts[0], color="black", s=45, marker="o")   # start
        ax.scatter(*pts[-1], color="red", s=45, marker="^")    # end
        for j in (0, len(pts) - 1):   # 시작/끝 step 번호만 표기 (중간은 어수선해짐)
            ax.text(*pts[j], f"t={steps[j]}", fontsize=7)

        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.set_zlabel("PC3", fontsize=8)
        ax.set_title(f"traj {sid[:8]}  (T={len(pts)})", fontsize=9)

    fig.suptitle(f"individual trajectories in PCA-3D  [{level}/{case} {tag}]\n"
                f"black=start red=end, color=progress (PC1-3 sum var={ev.sum():.1%})",
                fontsize=11)
    plt.tight_layout()

    out_dir = out_root / tag / level
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"pca3d_traj_{case}.png"
    plt.savefig(dst, dpi=150)
    plt.close()
    print(f"  saved {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--emb-dir", default=str(ROOT / "result" / "spectral_states"))
    ap.add_argument("--method", default="full")
    ap.add_argument("--tag", default="k8_scaled_signfix")
    ap.add_argument("--levels", default=None,
                    help="쉼표 구분. 생략 시 data-dir 에서 *_step<N> 자동 탐색")
    ap.add_argument("--n-trajs", type=int, default=6,
                    help="개별 서브플롯으로 그릴 궤적 수 (많으면 격자가 커짐)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "pca3d_out"),
                    help="analyze.py 기존 출력과 안 섞이도록 별도 디렉터리 기본값")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    emb_dir = Path(args.emb_dir)
    out_root = Path(args.out)

    levels = (args.levels.split(",") if args.levels
              else discover_levels(data_dir))
    if not levels:
        raise SystemExit(f"no <name>_step<N> level dirs found under {data_dir}")

    for level in levels:
        for case in discover_cases(data_dir, level):
            run(level, case, args.tag, args.method, data_dir, emb_dir, out_root, args.n_trajs)


if __name__ == "__main__":
    main()
