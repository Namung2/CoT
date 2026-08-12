"""decay 곡선을 3-way null 기준선과 함께 비교. analyze.py 는 건드리지 않고 재사용만 함.

"인접 step이 baseline보다 비슷하다"는 게 진짜 시간 순서 때문인지, 아니면
그냥 위치(t) 때문인지, 궤적 정체성 때문인지 셋을 갈라서 본다.

    null 종류        짝짓는 방법          제거되는 것      알려주는 것
    baseline         다른 궤적, 아무 t    궤적 + 위치      전체 초과분 (뭉뚱그려짐)
    궤적 내 셔플      같은 궤적, 순서만    순서만          순수 시간 순서 기여
    위치 매칭 교차    다른 궤적, 같은 t    궤적 정체성만    순수 위치 기여

위치 매칭 교차는 궤적 길이가 전부 같아야 의미가 있어서(*_step<N> 데이터라
정확히 같음), 길이가 다르면 assert 로 막는다.

사용법:
    python analyze/decay_nulls.py --level gotoseq_step10 --case c3 --tag k8_scaled_signfix
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analyze import (
    ROOT, discover_levels, discover_cases,
    load_meta, build_pool, traj_matrices, cross_traj_baseline,
)

MAX_LAG = 40
N_SAMPLES = 20000
SEED = 0


def real_decay_curve(trajs: dict, max_lag_cap: int = MAX_LAG):
    """실제 순서대로의 decay 곡선. analyze.py 의 adjacent_decay 와 동일한 집계 로직."""
    max_T = max(len(E) for E in trajs.values())
    max_lag = min(max_lag_cap, max_T - 1)
    lag_sum = np.zeros(max_lag + 1)
    lag_cnt = np.zeros(max_lag + 1)
    for E in trajs.values():
        T = len(E)
        S = E @ E.T
        for lag in range(1, min(max_lag, T - 1) + 1):
            v = np.diagonal(S, offset=lag)
            lag_sum[lag] += v.sum()
            lag_cnt[lag] += len(v)
    curve = np.where(lag_cnt[1:] > 0, lag_sum[1:] / np.maximum(lag_cnt[1:], 1), np.nan)
    return np.arange(1, max_lag + 1), curve


def within_traj_shuffle_baseline(trajs: dict):
    """같은 궤적, 순서만 무시(셔플). 궤적 안의 모든 off-diagonal 쌍 평균과 동치."""
    means = []
    for E in trajs.values():
        if len(E) < 2:
            continue
        S = E @ E.T
        off = S[~np.eye(len(E), dtype=bool)]
        means.append(float(off.mean()))
    return float(np.mean(means)), float(np.std(means))


def position_matched_baseline(trajs: dict, n_samples: int = N_SAMPLES, seed: int = SEED):
    """다른 궤적, 같은 t. 궤적 길이가 전부 동일해야 t 매칭이 의미 있다."""
    sids = list(trajs)
    lengths = {len(trajs[s]) for s in sids}
    assert len(lengths) == 1, (
        f"궤적 길이가 안 맞음 {lengths} — 위치 매칭 교차는 *_step<N> 데이터처럼 "
        f"길이가 전부 같아야 t 가 같은 의미를 가짐"
    )
    T = lengths.pop()
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_samples):
        t = rng.randint(T)
        a, b = rng.choice(len(sids), 2, replace=False)
        vals.append(float(trajs[sids[a]][t] @ trajs[sids[b]][t]))
    v = np.array(vals)
    return float(v.mean()), float(v.std())


def plot_decay_nulls(trajs: dict, level: str, case: str, tag: str, out_dir: Path):
    xs, real_curve = real_decay_curve(trajs)
    base_mean, base_std = cross_traj_baseline(trajs)
    shuf_mean, shuf_std = within_traj_shuffle_baseline(trajs)
    pos_mean, pos_std = position_matched_baseline(trajs)

    print(f"  real decay lag1={real_curve[0]:.4f} lagmax={real_curve[~np.isnan(real_curve)][-1]:.4f}")
    print(f"  baseline (다른궤적+아무t)     = {base_mean:.4f} ± {base_std:.4f}")
    print(f"  궤적 내 셔플 (순서만 제거)     = {shuf_mean:.4f} ± {shuf_std:.4f}")
    print(f"  위치 매칭 교차 (궤적만 제거)   = {pos_mean:.4f} ± {pos_std:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(xs, real_curve, marker=".", color="C0", label="실제 decay (진짜 순서)")
    plt.axhline(base_mean, color="gray", linestyle="--",
                label=f"baseline: 다른 궤적+아무 t ({base_mean:.3f})")
    plt.axhline(shuf_mean, color="C1", linestyle="--",
                label=f"궤적 내 셔플: 순서만 제거 ({shuf_mean:.3f})")
    plt.axhline(pos_mean, color="C2", linestyle="--",
                label=f"위치 매칭 교차: 궤적만 제거 ({pos_mean:.3f})")
    plt.xlabel("|t - t'| (lag)")
    plt.ylabel("mean cosine similarity")
    plt.title(f"decay vs 3 nulls  [{level}/{case} {tag}]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"decay_nulls_{case}.png"
    plt.savefig(dst, dpi=150)
    plt.close()
    print(f"  saved {dst}")


def run(level: str, case: str, tag: str, method: str,
        data_dir: Path, emb_dir: Path, out_root: Path):
    print(f"\n=== {level} / {case} ({method}_{tag}) ===")
    meta = load_meta(data_dir / level / "cases" / f"{case}.jsonl")
    emb_path = emb_dir / level / "cases" / case / f"{method}_{tag}.pt"
    X, labels, traj_ids, ts = build_pool(emb_path, meta, level, case)
    trajs = traj_matrices(X, traj_ids, ts)
    # 기존 analyze.py 출력(analyze/<tag>/analysis2_<level>/)과 안 섞이게 별도 트리에 저장
    out_dir = out_root / tag / level
    plot_decay_nulls(trajs, level, case, tag, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--emb-dir", default=str(ROOT / "result" / "spectral_states"))
    ap.add_argument("--method", default="full")
    ap.add_argument("--tag", default="k8_scaled_signfix",
                    help="spectral 스윕 태그 (예: k8, k8_scaled, k8_signfix, k8_scaled_signfix)")
    ap.add_argument("--levels", default=None,
                    help="쉼표 구분. 생략 시 data-dir 에서 *_step<N> 자동 탐색")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "decay_nulls_out"),
                    help="analyze.py 의 기존 출력(analyze/<tag>/analysis2_<level>/)과 "
                         "안 섞이도록 기본값을 별도 디렉터리로 둠")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    emb_dir = Path(args.emb_dir)
    out_root = Path(args.out)

    levels = (args.levels.split(",") if args.levels
              else discover_levels(data_dir))
    if not levels:
        raise SystemExit(f"no <name>_step<N> level dirs found under {data_dir}")

    for level in levels:
        cases = discover_cases(data_dir, level)
        for case in cases:
            run(level, case, args.tag, args.method, data_dir, emb_dir, out_root)


if __name__ == "__main__":
    main()
