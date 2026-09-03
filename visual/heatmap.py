"""토큰마다 스텝 시작점에서 리셋되는 누적 그람(SVD) 임베딩 → 토큰x토큰 히트맵.

5-1(spectral.py, 스텝당 그람 1개)과 다르게, 토큰이 하나 생성될 때마다 "현재
스텝의 첫 토큰부터 지금 토큰까지"만 다시 누적해서 그람을 계산한다 (에피소드
전체 누적이 아니라 스텝 경계에서 리셋). 그러니 각 스텝의 "마지막 토큰" 시점만
뽑으면 5-1의 스텝당 e_t와 정확히 같아야 한다 — spectral_states가 있으면 그것과
비교해서 검증까지 한다.

step_similarity.py(레벨 전체 평균)와 다르게 episode 하나를 골라서 그 안의
토큰x토큰 유사도 행렬을 그린다 (옛날 팀원 방식과 동일 — success/fail 예시
하나씩 뽑아보는 용도).

    python visual/heatmap.py --task decompose --level BabyAI-GoToObj-v0 --status success --seed 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

from extract import load_chunk                                          # noqa: E402
from spectral import spectral_embedding, DEVICE, K_EIG, SCALE, FIX_SIGN  # noqa: E402


@torch.no_grad()
def cumulative_within_step(E: torch.Tensor, boundaries: list[int],
                           k: int, scale: bool, fix_sign: bool):
    """토큰마다 e_i 계산 (스텝 시작점부터 그 토큰까지 누적, 스텝 바뀌면 리셋).

    반환: e_all (N x kd), last_of_step ({스텝: 그 스텝 마지막 토큰의 e_i}) —
    후자는 5-1의 e_t와 동일해야 함(같은 슬라이스라 정의상 동일).
    """
    e_list, last_of_step = [], {}
    for t, (s, e) in enumerate(zip(boundaries, boundaries[1:])):
        for i in range(s, e):
            e_i, _, _ = spectral_embedding(E[s:i + 1].to(DEVICE), k, scale, fix_sign)
            e_list.append(e_i)
        last_of_step[t] = e_list[-1] if e > s else None
    return torch.stack(e_list), last_of_step


def verify_against_spectral_states(last_of_step: dict, spectral_e: dict, atol: float = 1e-4):
    """spectral_states의 e_t(5-1)와 각 스텝 마지막 토큰의 e_i(5-2)가 실제로
    같은지 확인. 다르면 구현 버그."""
    for t, e_5_2 in last_of_step.items():
        if e_5_2 is None or t not in spectral_e:
            continue
        e_5_1 = spectral_e[t]
        if not torch.allclose(e_5_1, e_5_2, atol=atol):
            diff = (e_5_1 - e_5_2).abs().max().item()
            raise AssertionError(f"step {t}: 5-1과 5-2 마지막 토큰 불일치 (max diff={diff})")
    return True


def heatmap_matrix(e_all: torch.Tensor) -> torch.Tensor:
    un = torch.nn.functional.normalize(e_all, dim=1)
    return un @ un.T


def plot_heatmap(sim: torch.Tensor, boundaries: list[int], out_path: Path, title: str):
    import matplotlib.pyplot as plt

    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim.numpy(), cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="cosine similarity")

    # 스텝 경계 — 옅은 선 대신 각 스텝의 대각 블록(intra-step 영역) 자체를 빨간
    # 사각형 테두리로 명시 (미팅 피드백: "옅은 흰 선으로는 부족, 사각형/라벨로")
    for s, e in zip(boundaries, boundaries[1:]):
        n = e - s
        rect = patches.Rectangle((s - 0.5, s - 0.5), n, n,
                                 linewidth=1.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

    mids = [(s + e) / 2 - 0.5 for s, e in zip(boundaries, boundaries[1:])]
    labels = [f"Step {t}" for t in range(len(boundaries) - 1)]
    ax.set_xticks(mids); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(mids); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(hidden_dir: Path, task: str, level: str, status: str, seed: int | None = None,
        method: str = "full_sequence", ctx_tag: str = "with_prompt",
        k: int = K_EIG, scale: bool = SCALE, fix_sign: bool = FIX_SIGN,
        spectral_dir: Path | None = None, spectral_tag: str = "k8_scaled_signfix"):
    h_dir = hidden_dir / task / level / method / ctx_tag / status
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

    E, boundaries = episode["E"], episode["boundaries"]
    e_all, last_of_step = cumulative_within_step(E, boundaries, k, scale, fix_sign)

    if spectral_dir is not None:
        s_path = spectral_dir / task / level / method / ctx_tag / status / spectral_tag / chunk_name
        if s_path.exists():
            spectral_e = torch.load(s_path, map_location="cpu", weights_only=False)
            spectral_e = spectral_e["episodes"][seed]["e"]
            verify_against_spectral_states(last_of_step, spectral_e)

    sim = heatmap_matrix(e_all)
    return sim, boundaries, seed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--status", default="success", choices=["success", "failure"])
    ap.add_argument("--seed", type=int, default=None, help="episode env_seed. 안 주면 첫 episode")
    ap.add_argument("--methods", default="full_sequence")
    ap.add_argument("-k", type=int, default=K_EIG)
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--spectral-dir", type=Path, default=ROOT / "latent" / "spectral_states")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "visual" / "heatmap")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim, boundaries, seed = run(args.hidden_dir, args.task, args.level, args.status,
                               seed=args.seed, method=args.methods, k=args.k,
                               spectral_dir=args.spectral_dir)

    name = f"{args.task}_{args.level}_{args.status}_{seed}"
    plot_heatmap(sim, boundaries, args.out_dir / f"{name}.png",
                title=f"{args.task}/{args.level}/{args.status} seed={seed} (n_tok={sim.shape[0]})")
    print(f"n_tokens={sim.shape[0]} n_steps={len(boundaries) - 1}")
    print(f"saved -> {args.out_dir / name}.png")


if __name__ == "__main__":
    main()
