"""e_t -> e_{t+1} 예측 가능성 검증. RSSM state 후보로서 e_t 가 쓸만한지 직접 테스트.

CTRLS e_t 를 PCA로 축소한 뒤 (Xp_t, action_t) -> Xp_{t+1} 선형회귀를 궤적 단위로
train/test 나눠서 적합하고, "상태가 안 변한다"(identity)/"그냥 평균" 두 베이스라인과
R² 를 비교한다. 회귀가 이 둘보다 유의미하게 나아야 "다음 상태를 예측할 수 있는
구조가 있다"고 말할 수 있다.

analyze.py 는 건드리지 않고 함수/상수만 재사용.

사용법:
    python analyze/transition_predict.py --level gotoseq_step10 --case c3 --tag k8_scaled_signfix
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from analyze import (
    ROOT, discover_levels, discover_cases,
    load_meta, build_pool, N_PC, SEED,
)

ACTIONS = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}
ALPHA = 1.0   # Ridge 정규화 강도


def onehot_action(a: str) -> np.ndarray:
    v = np.zeros(len(ACTIONS), dtype=np.float32)
    v[ACT_IDX[a]] = 1.0
    return v


def build_transitions(X, labels, traj_ids, ts):
    """궤적 경계를 넘지 않는 (row_t, action, row_{t+1}) 연속쌍을 궤적별로 묶어서 반환.
    mission/terminal 은 제외 (실제 action 라벨이 있는 구간만)."""
    by_traj: dict[str, list[int]] = {}
    for i, sid in enumerate(traj_ids):
        by_traj.setdefault(sid, []).append(i)

    pair_idx = []  # (i, j, action) — i: t, j: t+1, X 배열 인덱스
    for sid, rows in by_traj.items():
        rows = sorted(rows, key=lambda i: ts[i])
        # action 라벨이 있는(=mission/terminal 이 아닌) 연속 구간만 페어링
        clean = [i for i in rows if labels[i] not in ("mission", "terminal")]
        clean_sorted = sorted(clean, key=lambda i: ts[i])
        for a, b in zip(clean_sorted, clean_sorted[1:]):
            if ts[b] - ts[a] != 1:
                continue  # 혹시 몰라 연속성 체크
            pair_idx.append((a, b, labels[b]))  # labels[b] = 이 전이에 쓰인 action
    return pair_idx, by_traj


def run(level: str, case: str, tag: str, method: str,
        data_dir: Path, emb_dir: Path, out_root: Path,
        test_frac: float = 0.2):
    print(f"\n=== {level} / {case} ({method}_{tag}) ===")
    meta = load_meta(data_dir / level / "cases" / f"{case}.jsonl")
    emb_path = emb_dir / level / "cases" / case / f"{method}_{tag}.pt"
    X, labels, traj_ids, ts = build_pool(emb_path, meta, level, case)

    pca = PCA(n_components=min(N_PC, *X.shape), svd_solver="randomized",
              random_state=SEED)
    Xp = pca.fit_transform(X)
    print(f"  PCA -> {Xp.shape[1]} dims (explained var "
          f"{pca.explained_variance_ratio_.sum():.3f})")

    pair_idx, by_traj = build_transitions(X, labels, traj_ids, ts)
    print(f"  transition pairs: {len(pair_idx)} (from {len(by_traj)} trajs)")

    # 궤적 단위로 train/test 분리 (같은 궤적의 t, t+1 이 양쪽에 걸치면 정보 누수)
    rng = np.random.RandomState(SEED)
    sids = sorted(by_traj)
    rng.shuffle(sids)
    n_test = max(1, int(len(sids) * test_frac))
    test_sids = set(sids[:n_test])

    train_pairs = [(a, b, act) for a, b, act in pair_idx if traj_ids[a] not in test_sids]
    test_pairs = [(a, b, act) for a, b, act in pair_idx if traj_ids[a] in test_sids]
    print(f"  train pairs: {len(train_pairs)} ({len(sids) - n_test} trajs) / "
          f"test pairs: {len(test_pairs)} ({n_test} trajs)")

    def featurize(pairs):
        Xin = np.stack([np.concatenate([Xp[a], onehot_action(act)])
                        for a, b, act in pairs])
        Yout = np.stack([Xp[b] for a, b, act in pairs])
        X_t = np.stack([Xp[a] for a, b, act in pairs])  # identity 베이스라인용
        return Xin, Yout, X_t

    Xin_tr, Y_tr, _ = featurize(train_pairs)
    Xin_te, Y_te, Xt_te = featurize(test_pairs)

    model = Ridge(alpha=ALPHA, random_state=SEED)
    model.fit(Xin_tr, Y_tr)
    r2_model = r2_score(Y_te, model.predict(Xin_te))

    r2_identity = r2_score(Y_te, Xt_te)                       # e_{t+1} = e_t
    mean_pred = np.tile(Y_tr.mean(axis=0), (len(Y_te), 1))
    r2_mean = r2_score(Y_te, mean_pred)                        # e_{t+1} = train 평균

    print(f"  R^2  model(ridge)={r2_model:.4f}  identity={r2_identity:.4f}  "
          f"mean={r2_mean:.4f}")

    return dict(level=level, case=case, tag=tag, method=method,
                n_pairs_train=len(train_pairs), n_pairs_test=len(test_pairs),
                n_trajs_train=len(sids) - n_test, n_trajs_test=n_test,
                r2_model=round(r2_model, 4), r2_identity=round(r2_identity, 4),
                r2_mean=round(r2_mean, 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--emb-dir", default=str(ROOT / "result" / "spectral_states"))
    ap.add_argument("--method", default="full")
    ap.add_argument("--tag", default="k8_scaled_signfix")
    ap.add_argument("--levels", default=None,
                    help="쉼표 구분. 생략 시 data-dir 에서 *_step<N> 자동 탐색")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "transition_predict_out"),
                    help="analyze.py 기존 출력과 안 섞이도록 별도 디렉터리 기본값")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    emb_dir = Path(args.emb_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    levels = (args.levels.split(",") if args.levels
              else discover_levels(data_dir))
    if not levels:
        raise SystemExit(f"no <name>_step<N> level dirs found under {data_dir}")

    rows = []
    for level in levels:
        for case in discover_cases(data_dir, level):
            rows.append(run(level, case, args.tag, args.method, data_dir, emb_dir, out_root))

    summary_path = out_root / f"transition_predict_summary_{args.tag}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
