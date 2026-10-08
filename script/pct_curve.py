#!/usr/bin/env python
"""구간 길이의 q% 지점마다 probe 를 학습해 "q → 정확도/확신" 곡선을 만든다 (hidden h 와 gram e 를 같은 샘플 위에서).

predict/probing.py --source pct 를 q 마다 따로 부르면 저장본을 q 번 다시 읽는다. 여기서는 저장본을 한 번만 읽어
(에피소드마다, 구간마다, 저장된 모든 위치의 벡터) 메모리에 두고, q 마다 pct_position 으로 행을 골라
probing.py 의 fit / evaluate 를 그대로 돌린다. 분할·시드·지표는 probing.py 와 같다 (에피소드 단위 split).

입력: inference/spectral.py --pct … [--with-hidden] 로 저장한 파일. --input 에 hidden 과 spectral 을 둘 다 주면
      같은 에피소드·같은 위치에서 h(d 차원) 와 e(kd 차원) 를 번갈아 학습해 두 곡선을 한 그림에 겹친다.
      hidden 은 _h 태그 파일에만 있다.

메모리: 저장된 모든 위치의 벡터를 들고 있는다. hidden(bf16, d=5120) 은 에피소드당 ~600KB(10 지점 x 6 구간) 라
      1 만 에피소드에 6GB. spectral e 는 kd=40960 이라 그 8 배(fp32) — --n-episodes 로 줄이거나 --dtype bfloat16 저장본을 쓸 것.
      --input 하나가 끝나면 그 벡터는 버리고 다음 input 을 읽는다.

출력: <output>/curve.json      {input: {target: {q: {acc/auc/f1/margin: {mean, std}, n_pos_test, n_test}}}}
      <output>/curve.png       target 마다 한 열, 위 = acc, 아래 = margin, 선 = input
      <output>/rows.csv        같은 내용 평평하게

Usage:
    python script/pct_curve.py \
        --pt 'latent/spectral/decompose/BabyAI-GoToObj-v0/*/k8_scaled_sign-data_p1-5-10-20-50-80-90-95-99-100_h/chunk_*.pt' \
        --input hidden spectral --n-episodes 2000 --min-len 20 --max-step 6 --output out/pct_curve/decompose_gotoobj
    python script/pct_curve.py --pt '…_h/chunk_*.pt' --input hidden --pct 1 5 10 20 50 --seeds 42 --output out/tmp
"""
from __future__ import annotations

import sys
import json
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "predict"))
from spectral import pct_position                                                   # noqa: E402
import probing as P                                                                  # noqa: E402


# ---------------------------------------------------------------- 로딩 (모든 저장 위치)

def load_all_positions(patterns, inp, k=None, sign_mode=None, min_len=0):
    """저장본의 모든 (에피소드, 구간, 위치) 벡터를 한 번에 읽는다.

    반환 segs: 구간 하나당 dict(n, pos: list[int], X: (m, D) tensor, label, gid). 그리고 (pct 목록, stats)."""
    segs, stats, seen = [], Counter(), set()
    pct_saved = None
    for p in P._paths(patterns):
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not P._config_ok(d, p, k, sign_mode, seen, stats):
            continue
        if not d.get("pct"):
            raise SystemExit(f"{p}: --pct 로 저장한 파일이 아니다 (pct={d.get('pct')})")
        pct_saved = pct_saved or list(d["pct"])
        key = P._vec_key(d, p, inp)
        for seed, ep in d["episodes"].items():
            gid = P._gid(p, "edges", seed)
            seg, n_seg = ep["seg"], len(ep["e"])
            stats["episodes"] += 1
            for t in range(n_seg):
                n = seg[t + 1] - seg[t]
                if n < min_len:
                    stats["short_segment"] += 1
                    continue
                ps = ep["pos"][t]
                if not ps:
                    stats["no_positions"] += 1
                    continue
                segs.append(dict(n=n, pos=list(ps), X=ep[key][t].clone(),
                                 label=P._seg_label(t, n_seg), gid=gid))
        del d
    if not segs:
        raise SystemExit("구간 없음 — 경로/규약 확인")
    print(f"[{inp}] loaded: {dict(stats)}  segments={len(segs)}  D={segs[0]['X'].shape[1]}  pct_saved={pct_saved}")
    return segs, pct_saved, stats


def subsample_groups(segs, n, seed):
    """probing.subsample_episodes 와 같은 규칙 (success 우선, 모자라면 failure) 을 구간 리스트에 적용."""
    if n is None:
        return segs, None
    groups = np.unique([s["gid"] for s in segs])
    status = np.array([g.split("/")[2] for g in groups])
    rng = np.random.default_rng(seed)
    succ = rng.permutation(groups[status == "success"])
    fail = rng.permutation(groups[status != "success"])
    keep = set(np.concatenate([succ[:n], fail[:max(0, n - len(succ))]]).tolist())
    out = [s for s in segs if s["gid"] in keep]
    print(f"--n-episodes {n}: success {min(n, len(succ))}/{len(succ)} + failure "
          f"{max(0, n - len(succ))}/{len(fail)} → {len(keep)} episodes, segments {len(segs)} → {len(out)}")
    return out, sorted(keep)


def rows_at(segs, q, k_eig, stats):
    """q% 지점의 행만 모아 probing 의 data dict 로."""
    Xs, Ns, Gs = [], [], []
    for s in segs:
        want = pct_position(q, s["n"])
        if want not in s["pos"]:
            stats[f"pos_missing@{q}"] += 1
            continue
        if want + 1 < k_eig:
            stats[f"t_lt_k@{q}"] += 1
        Xs.append(s["X"][s["pos"].index(want)]); Ns.append(s["label"]); Gs.append(s["gid"])
    X = torch.stack(Xs)
    return dict(X=X, step_num=np.array(Ns, np.int32), group=np.array(Gs, dtype=object))


# ---------------------------------------------------------------- 학습 루프

METRICS = ("acc", "auc", "f1", "margin")


def run_input(inp, a, k_eig, curve, rows_out):
    segs, pct_saved, stats = load_all_positions(a.pt, inp, a.k, a.sign_mode, a.min_len)
    segs, _ = subsample_groups(segs, a.n_episodes, a.sample_seed)
    pcts = a.pct or pct_saved
    missing = [q for q in pcts if q not in pct_saved]
    if missing:
        raise SystemExit(f"{missing}% 는 저장본에 없다 (저장된 pct={pct_saved})")

    for q in pcts:
        data = rows_at(segs, q, k_eig, stats)
        if a.drop_above_max and a.max_step is not None:
            keep = data["step_num"] <= a.max_step
            data = P._take(data, np.flatnonzero(keep))
        labels = sorted(np.unique(data["step_num"]).tolist())
        if a.max_step is not None:
            labels = [l for l in labels if l <= a.max_step]
        labels = [l for l in labels if l > 0] + ([P.ANSWER_LABEL] if P.ANSWER_LABEL in labels else [])

        group = len(np.unique(data["group"])) >= 2
        if a.device == "cpu":
            Xnp, gd = data["X"].float().numpy(), None
        else:
            Xnp, gd = None, P.GpuData(data["X"], a.device)

        for label in labels:
            t = P.target_name(label)
            X, y, g = P.make_binary(data, label)
            if y.sum() < 2 or (1 - y).sum() < 2:
                continue
            per_seed = defaultdict(list)
            n_test = n_pos = 0
            for s in a.seeds:
                tr, te = P.split(len(y), y, g, a.test_size, s, group)
                if len(np.unique(y[te])) < 2:
                    continue
                clf = P.fit(Xnp, gd, y, tr, s, False, True)
                m = P.evaluate(clf, Xnp, gd, te, y[te])
                for kk in METRICS:
                    per_seed[kk].append(m[kk])
                n_test, n_pos = len(te), int(y[te].sum())
            if not per_seed["acc"]:
                continue
            ent = {kk: dict(mean=float(np.mean(v)), std=float(np.std(v))) for kk, v in per_seed.items()}
            ent.update(n_test=n_test, n_pos_test=n_pos, n_seeds=len(per_seed["acc"]))
            curve[inp][t][str(q)] = ent
            rows_out.append(dict(input=inp, target=t, pct=q, **{kk: ent[kk]["mean"] for kk in METRICS},
                                 **{f"{kk}_std": ent[kk]["std"] for kk in METRICS},
                                 n_test=n_test, n_pos_test=n_pos))
            print(f"[{inp}] q={q:3d}% {t:8s} acc={ent['acc']['mean']:.3f} auc={ent['auc']['mean']:.3f} "
                  f"margin={ent['margin']['mean']:.3f}  (n_test={n_test}, pos={n_pos})")
        del data, Xnp, gd
    print(f"[{inp}] stats: {dict(stats)}")
    del segs


def plot(curve, out_png, title):
    inputs = list(curve)
    targets = sorted({t for inp in inputs for t in curve[inp]},
                     key=lambda t: (t == "answer", t))
    if not targets:
        return
    fig, axes = plt.subplots(2, len(targets), figsize=(3.2 * len(targets), 6), squeeze=False, sharex=True)
    for j, t in enumerate(targets):
        for i, met in enumerate(("acc", "margin")):
            ax = axes[i][j]
            for inp in inputs:
                pts = curve[inp].get(t, {})
                qs = sorted(int(q) for q in pts)
                if not qs:
                    continue
                m = [pts[str(q)][met]["mean"] for q in qs]
                sd = [pts[str(q)][met]["std"] for q in qs]
                ax.errorbar(qs, m, yerr=sd, marker="o", ms=3, capsize=2, label=inp)
            ax.set_ylim(0.45, 1.02)
            ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.5)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(t)
            if j == 0:
                ax.set_ylabel(met)
            if i == 1:
                ax.set_xlabel("% of segment")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", nargs="+", required=True, help="spectral.py --pct 저장본 chunk_*.pt glob")
    ap.add_argument("--input", nargs="+", choices=("hidden", "spectral"), default=["hidden", "spectral"])
    ap.add_argument("--pct", type=int, nargs="+", default=None, help="돌릴 q 목록 (기본: 저장본의 전부)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--sign-mode", default=None, choices=("none", "first", "max", "data"))
    ap.add_argument("--min-len", type=int, default=0, help="이 길이 미만 구간 제외 (짧은 구간의 % 겹침 회피)")
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--drop-above-max", action="store_true")
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    # k 는 t<k 카운트용 — 첫 파일 헤더에서
    k_eig = torch.load(P._paths(a.pt)[0], map_location="cpu", weights_only=False)["k"]

    curve = defaultdict(lambda: defaultdict(dict))
    rows_out = []
    for inp in a.input:
        run_input(inp, a, k_eig, curve, rows_out)

    curve = {i: {t: dict(v) for t, v in d.items()} for i, d in curve.items()}
    with open(a.output / "curve.json", "w") as f:
        json.dump(curve, f, indent=2)
    if rows_out:
        with open(a.output / "rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader(); w.writerows(rows_out)
    plot(curve, a.output / "curve.png",
         f"probe vs % of segment  (min_len={a.min_len}, seeds={len(a.seeds)}, episodes={a.n_episodes or 'all'})")

    print("\n" + "=" * 70)
    for inp in curve:
        for t in curve[inp]:
            qs = sorted(int(q) for q in curve[inp][t])
            print(f"[{inp}] {t:8s} acc    " + " ".join(f"{q:3d}%:{curve[inp][t][str(q)]['acc']['mean']:.3f}" for q in qs))
            print(f"[{inp}] {t:8s} margin " + " ".join(f"{q:3d}%:{curve[inp][t][str(q)]['margin']['mean']:.3f}" for q in qs))
    print(f"\nsaved → {a.output}")


if __name__ == "__main__":
    main()