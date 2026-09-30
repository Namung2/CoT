"""토큰마다 스텝 시작점에서 리셋되는 누적 그람(SVD) 임베딩 → 토큰x토큰 히트맵.

5-1(spectral.py, 구간당 그람 1개)과 다르게, 토큰이 하나 생성될 때마다 "현재
구간의 첫 토큰부터 지금 토큰까지"만 다시 누적해서 그람을 계산한다 (에피소드
전체 누적이 아니라 구간 경계에서 리셋). 그러니 각 구간의 "마지막 토큰" 시점만
뽑으면 5-1의 구간당 e_t와 정확히 같아야 한다 — latent/spectral 저장본이 있으면 그것과
비교해서 검증까지 한다.

구간은 extract.gen_view 기준이다. 프롬프트는 빼고 step 1..N + 터미널(정답 문장).
프롬프트 토큰이 전체의 2/3 라 넣으면 관심 구간이 구석으로 밀린다.

step_similarity.py(레벨 전체 평균)와 다르게 episode 하나를 골라서 그 안의
토큰x토큰 유사도 행렬을 그린다 (success/fail 예시 하나씩 뽑아보는 용도).

    python visual/heatmap.py --task decompose --level BabyAI-GoToObj-v0 --status success --seed 5
    python visual/heatmap.py --task decompose --level BabyAI-GoToObj-v0 --status success --rep raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

from extract import load_chunk, gen_view, seg_labels                    # noqa: E402
from spectral import (make_tag, DEVICE, K_EIG, SCALE, SIGN_MODE, SIGN_MODES,  # noqa: E402
                      N_FRONT, N_BACK, tokens_cumulative, segment_last)


@torch.no_grad()
def cumulative_within_step(E: torch.Tensor, boundaries: list[int],
                           k: int, scale: bool, sign_mode: str):
    """토큰마다 e_i 계산 (구간 시작점부터 그 토큰까지 누적, 구간 바뀌면 리셋).

    계산은 inference/spectral.py 의 CumulativeSpectral(누적 그람 톱니)이 한다 (토큰마다 SVD 를
    다시 도는 대신 n x n 그람을 한 행씩 키워 eigh — 같은 값, CPU 에서 ~15배 빠름).
    이 함수는 그 결과를 heatmap / gsbs 가 쓰던 모양으로 돌려주는 얇은 껍데기다.

    반환: e_all (N x kd), last_of_step ({구간: 그 구간 마지막 토큰의 e_i}) —
    후자는 5-1의 e_t와 동일해야 함(같은 행렬이라 정의상 동일).
    """
    e_all, _ = tokens_cumulative(E, boundaries, k, scale, sign_mode, DEVICE)
    last_of_step = {t: (e_all[e - 1] if e > s else None)
                    for t, (s, e) in enumerate(zip(boundaries, boundaries[1:]))}
    return e_all, last_of_step


def verify_against_spectral_states(last_of_step: dict, spectral_e: dict,
                                   rtol: float = 1e-3, atol: float = 1e-3):
    """저장된 spectral 의 e_t(5-1, back[t][-1])와 각 구간 마지막 토큰의 e_i(5-2)가 실제로
    같은지 확인. 다르면 구현 버그.

    5-1 은 (n x d) SVD, 5-2 는 (n x n) 그람 eigh 라 float32 반올림 수준(상대 1e-4)
    의 차이는 정상이다. 그래서 절대 1e-4 가 아니라 상대 오차로 본다."""
    for t, e_5_2 in last_of_step.items():
        if e_5_2 is None or t not in spectral_e:
            continue
        e_5_1 = spectral_e[t]
        if not torch.allclose(e_5_1, e_5_2, rtol=rtol, atol=atol):
            diff = (e_5_1 - e_5_2).abs().max().item()
            raise AssertionError(f"segment {t}: 5-1과 5-2 마지막 토큰 불일치 (max diff={diff})")
    return True


def heatmap_matrix(e_all: torch.Tensor) -> torch.Tensor:
    un = torch.nn.functional.normalize(e_all, dim=1)
    return un @ un.T


def plot_heatmap(sim: torch.Tensor, seg: list[int], out_path: Path, title: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim.numpy(), cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="cosine similarity")

    # 구간 경계 — 옅은 선 대신 각 구간의 대각 블록(intra-segment 영역) 자체를
    # 빨간 사각형 테두리로 명시 (미팅 피드백: "옅은 흰 선으로는 부족, 사각형/라벨로")
    for s, e in zip(seg, seg[1:]):
        n = e - s
        rect = patches.Rectangle((s - 0.5, s - 0.5), n, n,
                                 linewidth=1.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

    mids = [(s + e) / 2 - 0.5 for s, e in zip(seg, seg[1:])]
    labels = seg_labels(seg)
    ax.set_xticks(mids); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(mids); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(hidden_dir: Path, task: str, level: str, status: str, seed: int | None = None,
        k: int = K_EIG, scale: bool = SCALE, sign_mode: str = SIGN_MODE,
        n_front: int = N_FRONT, n_back: int = N_BACK,
        spectral_dir: Path | None = None, rep: str = "spectral"):
    h_dir = hidden_dir / task / level / status
    chunk_files = sorted(h_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"no chunk_*.pt in {h_dir}")

    episode, chunk_name = None, None
    for cf in chunk_files:
        episodes = load_chunk(cf)
        if seed is None:
            seed, episode = next(iter(episodes.items()))
            chunk_name = cf.name
            break
        if seed in episodes:
            episode, chunk_name = episodes[seed], cf.name
            break
    if episode is None:
        raise KeyError(f"seed {seed} not found under {h_dir}")

    E, seg = gen_view(episode)      # 프롬프트 제외, step 1..N + 터미널

    if rep == "raw":
        # spectral 을 안 거친 원본 토큰 벡터(5120) 끼리의 코사인. 누적도 리셋도 없다.
        return heatmap_matrix(E.float()), seg, seed

    e_all, last_of_step = cumulative_within_step(E, seg, k, scale, sign_mode)

    if spectral_dir is not None:
        spectral_tag = make_tag(k, scale, sign_mode, n_front, n_back)   # spectral.py 와 같은 규칙
        s_path = spectral_dir / task / level / status / spectral_tag / chunk_name
        if s_path.exists():
            spectral_e = torch.load(s_path, map_location="cpu", weights_only=False)
            spectral_e = segment_last(spectral_e["episodes"][seed])
            verify_against_spectral_states(last_of_step, spectral_e)
        else:
            print(f"warning: {s_path} 없음 — 5-1 vs 5-2 검증 건너뜀 "
                  f"(같은 k/scale/sign_mode 로 spectral 을 먼저 돌렸는지 확인)", file=sys.stderr)

    return heatmap_matrix(e_all), seg, seed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--status", default="success", choices=["success", "failure"])
    ap.add_argument("--seed", type=int, default=None, help="episode env_seed. 안 주면 첫 episode")
    ap.add_argument("--rep", default="spectral", choices=["spectral", "raw"],
                    help="spectral=토큰별 누적 e_t 끼리 코사인(기본) | raw=원본 5120차원 토큰끼리 코사인")
    ap.add_argument("-k", type=int, default=K_EIG)
    ap.add_argument("--sign-mode", default=SIGN_MODE, choices=list(SIGN_MODES),
                    help="spectral 과 같은 값을 줘야 저장본 검증이 맞물림")
    ap.add_argument("--n-front", type=int, default=N_FRONT, help="검증에 쓸 spectral 저장본의 n_front")
    ap.add_argument("--n-back", type=int, default=N_BACK, help="검증에 쓸 spectral 저장본의 n_back (≥1)")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--spectral-dir", type=Path, default=ROOT / "latent" / "spectral")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "visual" / "heatmap")
    args = ap.parse_args()

    sim, seg, seed = run(args.hidden_dir, args.task, args.level, args.status,
                         seed=args.seed, k=args.k, sign_mode=args.sign_mode,
                         n_front=args.n_front, n_back=args.n_back, spectral_dir=args.spectral_dir,
                         rep=args.rep)

    # raw 는 k/부호와 무관하므로 raw/<status>/ 로 따로 둔다.
    if args.rep == "raw":
        tag = "raw"
        out_dir = args.out_dir / "raw" / args.status
    else:
        tag = make_tag(args.k, SCALE, args.sign_mode, args.n_front, args.n_back)
        # 한 디렉토리에 다 쌓이면 못 찾는다 → k / status / 부호모드 로 3단 분리.
        # 파일명에는 전체 tag 를 남겨서 파일 하나만 떼어 봐도 설정을 알 수 있게 둔다.
        out_dir = args.out_dir / f"k{args.k}" / args.status / args.sign_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.task}_{args.level}_{args.status}_{seed}_{tag}"
    plot_heatmap(sim, seg, out_dir / f"{name}.png",
                 title=f"{args.task}/{args.level}/{args.status} seed={seed} "
                       f"(n_tok={sim.shape[0]}, prompt 제외)\n{tag}")
    print(f"n_tokens={sim.shape[0]} n_segments={len(seg) - 1} "
          f"(steps={len(seg) - 2} + answer)")
    print(f"saved -> {out_dir / name}.png")


if __name__ == "__main__":
    main()