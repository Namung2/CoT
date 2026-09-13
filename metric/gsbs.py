"""hidden state 궤적에서 GSBS로 경계를 찾고, 텍스트 step 경계와 비교한다.

GSBS(Geerligs et al. 2021, Neuroimage)는 (시점 x 복셀) 행렬을 "안정적인 패턴을
유지하는 구간"으로 나누는 알고리즘이다. 대응은 이렇게 둔다.

    복셀  ->  은닉 차원 5120개
    시점  ->  토큰            (에피소드마다 개수가 다름)

경계를 몇 개 둘지와 어디에 둘지는 GSBS가 t-distance로 알아서 정한다.

--input 으로 복셀에 무엇을 둘지 고른다. 시점은 둘 다 토큰이다.

    tokens    원본 E                복셀 5120      step 정보 없음
    spectral  토큰별 누적 e_t        복셀 k*5120    step 시작에서 리셋됨

spectral 은 heatmap.py 와 같은 표현이라 step 시작마다 누적이 리셋된다. 따라서
GSBS 가 step 경계를 찾는 건 당연하고, 보고 싶은 건 "step 안에서 또 갈라지는
지점"이다 (substep_splits 로 집계).

그림 두 종류를 낸다.

  overlay : 토큰x토큰 코사인 유사도(원본 E) 위에 step 경계(빨강)와 GSBS 경계(흰색)를
            겹쳐 그린다. 행렬 자체에 step 정보가 안 들어가서 공정한 비교다.
  reset   : heatmap.py의 누적 e_i 히트맵을 두 장 그린다. 왼쪽은 리셋 지점이 step
            경계, 오른쪽은 GSBS 경계. 어느 쪽 블록이 더 또렷한지 본다.

    python metric/gsbs.py --task decompose --level BabyAI-GoToObj-v0 --status success
    python metric/gsbs.py --task decompose --level BabyAI-GoToObj-v0 --status success --input spectral -k 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "visual"))

from extract import load_chunk                                    # noqa: E402
from spectral import K_EIG, SCALE, SIGN_MODE, SIGN_MODES          # noqa: E402
from heatmap import cumulative_within_step, heatmap_matrix        # noqa: E402


# ---------------------------------------------------------------------- GSBS

def build_input(E: torch.Tensor, boundaries: list[int], inp: str,
                k: int, scale: bool, sign_mode: str) -> torch.Tensor:
    """GSBS에 넣을 (시점 x 복셀) 행렬. 시점은 두 경우 모두 토큰이다.

      tokens   : 원본 E                     -> 복셀 = 5120
      spectral : 토큰별 누적 e_t             -> 복셀 = k*5120

    spectral은 heatmap.py와 똑같이 step 시작에서 누적이 리셋된다. GSBS가 그 리셋
    지점을 찾는 건 당연하므로, 관심사는 "step 안에서 추가로 갈라지는 지점"이다."""
    if inp == "tokens":
        return E.float()
    return cumulative_within_step(E, boundaries, k, scale, sign_mode)[0]


def reduce_features(X: np.ndarray, r: int) -> np.ndarray:
    """행(시점) 중심화 후 SVD로 복셀 차원을 줄인다. GSBS 결과는 거의 안 바뀐다.

    GSBS가 쓰는 값(t-distance, wdists)은 전부 "시점마다 특징 방향으로 표준화한 뒤의
    시점간 내적"에서 나온다. 행 중심화 후 SVD는 그 내적을 정확히 보존하므로
    복셀 수만 줄고 GSBS가 보는 행렬은 그대로다. 5120 -> 427(=토큰수)만 해도 12배 빠르다.

    ★ sklearn PCA를 쓰면 안 된다. PCA는 열(특징) 방향을 중심화해서 "모든 토큰에
    공통인 성분"을 지워버리는데, 그 성분이 매우 커서(||열평균||이 행 노름의 77%)
    시점간 상관행렬이 완전히 달라진다 (원본과 상관 0.48).

    r <= 0 이면 full rank = min(시점수, 복셀수). 실측 일치도(상관, 427토큰 기준):
        r=427 -> 0.99999 | 200 -> 0.99991 | 100 -> 0.99944 | 50 -> 0.99813 | 25 -> 0.99443
    """
    Xc = X - X.mean(1, keepdims=True)
    full = min(Xc.shape)
    r = full if r <= 0 else min(r, full)

    # Gram 고유분해. d가 크면 SVD보다 싸고 (N x d) 우특이벡터를 안 만들어도 된다.
    w, V = np.linalg.eigh(Xc @ Xc.T)
    idx = np.argsort(w)[::-1][:r]
    return np.ascontiguousarray(V[:, idx] * np.sqrt(np.clip(w[idx], 0, None)), dtype=np.float64)


def run_gsbs(X_t: torch.Tensor, kmax: int | None, reduce: int | None,
             statewise: bool = False):
    """X_t: (시점, 복셀) -> (경계 인덱스 배열, GSBS 객체, 실제 입력 X).

    kmax=None이면 문서 권장값인 시점수/2. 경계는 "그 시점에서 상태가 바뀐다"는 뜻이라
    0번 시점에는 절대 안 붙는다. reduce=None이면 축소 안 함.

    statewise: GSBS의 statewise_detection. 켜면 경계를 한 번에 두 개씩 놓아 보느라
    모든 시점 쌍을 훑어서 시점수의 3.6제곱쯤으로 느려진다. 566토큰 실측:
        꺼짐 kmax=50  ->  13초,  켜짐 kmax=10 -> 710초.  결과(상태 2, 경계 533)는 동일.
    그래서 기본은 꺼짐. 켜진 쪽이 저자들의 개선판이니 최종 확인용으로만 켤 것."""
    from statesegmentation import GSBS

    X = X_t.float().numpy().astype(np.float64)
    if reduce is not None:
        X = reduce_features(X, reduce)
    X = np.ascontiguousarray(X, dtype=np.float64)

    n = X.shape[0]
    kmax = max(2, n // 2) if kmax is None else int(min(kmax, max(2, n // 2)))

    g = GSBS(kmax=kmax, x=X, statewise_detection=statewise)
    g.fit(showProgressBar=False)
    return np.nonzero(g.get_deltas())[0], g, X


def compare(pred: np.ndarray, true: np.ndarray, tol: int):
    """step 경계마다 tol 이내의 GSBS 경계를 하나씩 짝지어 준다 (중복 매칭 없음).

    가까운 쌍부터 확정하므로 순서에 안 휘둘린다."""
    pred, true = np.sort(pred).astype(int), np.sort(true).astype(int)
    cand = sorted((abs(int(t) - int(p)), int(t), int(p))
                  for t in true for p in pred if abs(int(t) - int(p)) <= tol)
    seen_t, seen_p, pairs = set(), set(), []
    for _, t, p in cand:
        if t not in seen_t and p not in seen_p:
            seen_t.add(t); seen_p.add(p); pairs.append((t, p))

    hit = len(pairs)
    prec = hit / pred.size if pred.size else 0.0
    rec = hit / true.size if true.size else 0.0
    f1 = 0.0 if hit == 0 else 2 * prec * rec / (prec + rec)
    return {"matched": [[t, p] for t, p in pairs], "n_matched": hit,
            "precision": prec, "recall": rec, "f1": f1, "tolerance": tol}


def substep_splits(pred: np.ndarray, true_b: list[int]):
    """step 하나 안에 GSBS 경계가 몇 개 들어갔는지. step 경계 자체는 제외(strict inside).

    input=spectral 일 때 핵심 지표다. 그 표현은 step 시작에서 리셋되므로 GSBS가
    step 경계를 찾는 건 당연하고, 의미 있는 건 step 내부의 추가 분할이다."""
    out = []
    for t, (s, e) in enumerate(zip(true_b, true_b[1:])):
        inside = [int(b) for b in pred if s < b < e]
        out.append({"step": t, "start": int(s), "end": int(e), "n_tokens": int(e - s),
                    "n_splits": len(inside), "splits": inside})
    return out


# ---------------------------------------------------------------------- 그림

def plot_overlay(X_in: torch.Tensor, true_b: list[int], pred_b: np.ndarray,
                 cmp_: dict, out_path: Path, title: str):
    """GSBS 가 실제로 본 표현의 시점x시점 코사인 유사도 위에 두 경계를 겹친다.

    input=tokens 면 원본 E 끼리(①), input=spectral 이면 누적 e_t 끼리(③)의
    히트맵과 같은 행렬이 된다."""
    import matplotlib.pyplot as plt

    S = heatmap_matrix(X_in.float()).numpy()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(S, cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="cosine similarity")

    # step 경계는 몇 개 안 되니 전체 선으로, GSBS 경계는 수십 개라 가장자리 눈금으로
    # (전체 선으로 그리면 행렬이 안 보인다). step 경계와 짝지어진 것은 연두색.
    for b in true_b[1:-1]:
        ax.axvline(b - 0.5, color="red", lw=1.2, alpha=0.8)
        ax.axhline(b - 0.5, color="red", lw=1.2, alpha=0.8)
    matched = {p for _, p in cmp_["matched"]}
    for b in pred_b:
        c = "lime" if b in matched else "white"
        ax.axvline(b - 0.5, ymin=0.0, ymax=0.045, color=c, lw=1.3)
        ax.axhline(b - 0.5, xmin=0.0, xmax=0.045, color=c, lw=1.3)

    ax.set_xlabel("token")
    ax.set_title(f"red line = CoT step ({len(true_b) - 1}) | edge tick = GSBS "
                 f"({pred_b.size + 1} states, lime = matched)", fontsize=9)
    fig.suptitle(f"{title}\n"
                 f"matched {cmp_['n_matched']}/{len(true_b) - 2} step bounds  "
                 f"(P={cmp_['precision']:.2f} R={cmp_['recall']:.2f} "
                 f"F1={cmp_['f1']:.2f}, tol={cmp_['tolerance']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reset(E: torch.Tensor, true_b: list[int], gsbs_b: list[int],
               k: int, scale: bool, sign_mode: str, out_path: Path, title: str):
    """heatmap.py의 누적 e_i 히트맵을 리셋 지점만 바꿔 두 장 그린다."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, bounds) in zip(axes, [("reset at CoT step", true_b),
                                         ("reset at GSBS boundary", gsbs_b)]):
        e_all, _ = cumulative_within_step(E, bounds, k, scale, sign_mode)
        sim = heatmap_matrix(e_all).numpy()
        im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
        for s, e in zip(bounds, bounds[1:]):
            ax.add_patch(patches.Rectangle((s - 0.5, s - 0.5), e - s, e - s,
                                           linewidth=1.3, edgecolor="red", facecolor="none"))
        ax.set_title(f"{name}  ({len(bounds) - 1} segments)", fontsize=10)
        ax.set_xlabel("token")
        fig.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- 실행

def load_episode(hidden_dir: Path, task: str, level: str, status: str,
                 seed: int | None, ctx_tag: str):
    h_dir = hidden_dir / task / level / ctx_tag / status
    files = sorted(h_dir.glob("chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"no chunk_*.pt in {h_dir}")
    for cf in files:
        eps = load_chunk(cf)
        if seed is None:
            return next(iter(eps.items()))
        if seed in eps:
            return seed, eps[seed]
    raise KeyError(f"seed {seed} not found under {h_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--status", default="success", choices=["success", "failure"])
    ap.add_argument("--seed", type=int, default=None, help="env_seed. 안 주면 첫 episode")
    ap.add_argument("--input", default="tokens", choices=["tokens", "spectral"],
                    help="tokens=원본 E(복셀 5120) | spectral=토큰별 누적 e_t(복셀 k*5120)")
    ap.add_argument("--ctx-tag", default="with_prompt")
    ap.add_argument("--kmax", type=int, default=None,
                    help="GSBS 최대 상태 수. 기본은 토큰수/2 (문서 권장값). 크면 느리다")
    ap.add_argument("--reduce", type=int, default=0,
                    help="행중심화 SVD로 복셀 축소. 0(기본)=full rank(=시점수)로 무손실 축소, "
                         "양수=상위 N개만, 음수=축소 안 함(원본 차원 그대로)")
    ap.add_argument("--tol", type=int, default=5, help="경계 매칭 허용 오차 (토큰)")
    ap.add_argument("--statewise", action="store_true",
                    help="GSBS statewise_detection 켜기. 50배 느리고 실측상 결과 동일 (run_gsbs 참고)")
    ap.add_argument("-k", type=int, default=K_EIG, help="reset 그림의 spectral 고유값 개수")
    ap.add_argument("--sign-mode", default=SIGN_MODE, choices=list(SIGN_MODES))
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "latent" / "gsbs")
    args = ap.parse_args()

    seed, ep = load_episode(args.hidden_dir, args.task, args.level, args.status,
                            args.seed, args.ctx_tag)
    E, true_b = ep["E"], ep["boundaries"]
    n = E.shape[0]
    print(f"seed={seed}  tokens={n}  dims={E.shape[1]}  CoT steps={len(true_b) - 1}")

    t0 = time.perf_counter()
    X_in = build_input(E, true_b, args.input, args.k, SCALE, args.sign_mode)
    pred_b, g, X = run_gsbs(X_in, args.kmax, None if args.reduce < 0 else args.reduce,
                            statewise=args.statewise)
    print(f"GSBS: {g.nstates} states, {pred_b.size} boundaries "
          f"(kmax={g.kmax}, voxels {X_in.shape[1]}->{X.shape[1]}, "
          f"{time.perf_counter() - t0:.0f}s)")
    if g.nstates >= g.kmax - 1:
        print(f"warning: 최적 k가 kmax({g.kmax})에 붙었다 — t-distance가 아직 오르는 중이라 "
              f"진짜 최적이 아닐 수 있음", file=sys.stderr)

    cmp_ = compare(pred_b, np.asarray(true_b[1:-1]), args.tol)
    print(f"step 경계 {len(true_b) - 2}개 중 {cmp_['n_matched']}개를 GSBS도 찾음 "
          f"(tol={args.tol})  P={cmp_['precision']:.3f} R={cmp_['recall']:.3f} "
          f"F1={cmp_['f1']:.3f}")

    splits = substep_splits(pred_b, true_b)
    print("step 내부 추가 분할:")
    for sp in splits:
        print(f"  step {sp['step']} [{sp['start']:4d}:{sp['end']:4d}] "
              f"{sp['n_tokens']:4d}토큰 -> {sp['n_splits']}개 분할 {sp['splits']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    itag = "tokens" if args.input == "tokens" else f"spectral_k{args.k}_{args.sign_mode}"
    if args.reduce != 0:
        itag += f"_red{args.reduce}"
    base = f"{args.task}_{args.level}_{args.status}_{seed}_{itag}"
    title = (f"{args.task}/{args.level}/{args.status} seed={seed} "
             f"(n_tok={n}, input={args.input}, voxels={X.shape[1]})")

    plot_overlay(X_in, true_b, pred_b, cmp_, args.out_dir / f"{base}_overlay.png", title)
    print(f"saved -> {args.out_dir / base}_overlay.png")

    gsbs_full = [0] + pred_b.tolist() + [n]
    plot_reset(E, true_b, gsbs_full, args.k, SCALE, args.sign_mode,
               args.out_dir / f"{base}_reset.png",
               f"{title}   k={args.k} sign={args.sign_mode}")
    print(f"saved -> {args.out_dir / base}_reset.png")

    out_json = args.out_dir / f"{base}.json"
    out_json.write_text(json.dumps({
        "task": args.task, "level": args.level, "status": args.status, "env_seed": int(seed),
        "n_tokens": int(n), "voxels_in": int(X_in.shape[1]),
        "voxels_used": int(X.shape[1]), "reduce": args.reduce, "statewise": args.statewise,
        "kmax": int(g.kmax), "kmax_saturated": bool(g.nstates >= g.kmax - 1),
        "n_states_gsbs": int(g.nstates), "n_steps_true": len(true_b) - 1,
        "input": args.input, "k": args.k, "sign_mode": args.sign_mode,
        "gsbs_bounds": pred_b.tolist(), "step_bounds": true_b[1:-1],
        "compare": cmp_, "substep_splits": splits,
        "tdists": [float(v) for v in g.tdists],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
