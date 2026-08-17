"""adjacent-step similarity를 progress bin(0~1) 근사 대신 실제 step 번호로 직접 계산.

analyze.py의 adjacent_decay()는 progress를 B=30개 bin으로 나눠 평균 내는데,
이 데이터셋은 레벨/케이스별로 전체 스텝 수(T)가 고정돼 있어서 bin 근사가 불필요하고
(T-1 < 30인 레벨은 빈 bin이 0으로 찍히는 아티팩트가 생김). 대신 실제 step 번호를
그대로 x축으로 써서 근사 없이 확인. analyze.py는 건드리지 않고 검증용으로 별도 작성.

사용법:
    python analyze/adjacent_by_step.py --level bosslevel_step30 --case c3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analyze import ROOT, load_meta, build_pool


def run(level: str, case: str, tag: str, method: str,
        data_dir: Path, emb_dir: Path, out_root: Path):
    meta = load_meta(data_dir / level / "cases" / f"{case}.jsonl")
    emb_path = emb_dir / level / "cases" / case / f"{method}_{tag}.pt"
    X, labels, traj_ids, ts = build_pool(emb_path, meta, level, case)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    sids = np.unique(traj_ids)
    Ts = {sid: int(ts[traj_ids == sid].max()) for sid in sids}
    print(f"{level}/{case}: n_traj={len(sids)}  T values={sorted(set(Ts.values()))}")

    max_t = max(Ts.values())
    sim_sum = np.zeros(max_t + 1)
    sim_cnt = np.zeros(max_t + 1)

    for sid in sids:
        mm = traj_ids == sid
        order = np.argsort(ts[mm])
        t_sorted = ts[mm][order]
        E = Xn[mm][order]
        for i in range(len(E) - 1):
            if t_sorted[i] > 0 and t_sorted[i + 1] == t_sorted[i] + 1:
                t = t_sorted[i]
                sim_sum[t] += float(E[i] @ E[i + 1])
                sim_cnt[t] += 1.0

    curve = np.where(sim_cnt > 0, sim_sum / np.maximum(sim_cnt, 1), np.nan)
    steps = np.arange(1, max_t + 1)
    curve = curve[1:]

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, curve, marker=".")
    plt.xlabel("step t"); plt.ylabel("mean cos(e_t, e_{t+1})")
    plt.title(f"adjacent-step similarity by real step t  [{level}/{case}]")
    plt.tight_layout()

    out_root.mkdir(parents=True, exist_ok=True)
    dst = out_root / f"adjacent_by_step_{level}_{case}.png"
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
    ap.add_argument("--level", required=True)
    ap.add_argument("--case", default="c3")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "adjacent_by_step_out"))
    args = ap.parse_args()
    run(args.level, args.case, args.tag, args.method,
        Path(args.data_dir), Path(args.emb_dir), Path(args.out))


if __name__ == "__main__":
    main()
