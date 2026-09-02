"""intra-step vs inter-step(스텝 거리별) 유사도 요약.

intra-step : 한 스텝 안 토큰들끼리의 평균 코사인 유사도 (hidden_states의 raw E 사용)
inter-step : 스텝 t와 t+d의 유사도, 대표 벡터 두 가지로 각각 계산
  - last_token : 그 스텝의 마지막 토큰 벡터 (attention으로 이미 그 스텝을 반영한, 모델이
                 자체적으로 만든 대표값)
  - e_t        : spectral_states의 e_t (우리가 SVD로 명시적으로 요약한 대표값)

새 추출 없이 기존 hidden_states/spectral_states만 읽는다.

    python visual/step_similarity.py --task decompose --level BabyAI-GoToObj-v0 --status success
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def load_chunk(pt_path: Path) -> dict:
    """청크 파일 하나 로드. {env_seed: {"E":..., "boundaries":..., ...}, ...} 반환."""
    return torch.load(pt_path, map_location="cpu", weights_only=False)["episodes"]


def intra_step_similarity(E: torch.Tensor, boundaries: list[int]) -> dict[int, float]:
    """스텝 t 안 토큰 쌍 코사인 유사도 평균 — 정규화된 토큰끼리 Un @ Un.T가 토큰
    Gram 행렬 그 자체. 토큰 1개짜리 스텝은 정의 안 됨(제외)."""
    Un = torch.nn.functional.normalize(E, dim=1)
    sims = {}
    for t, (s, e) in enumerate(zip(boundaries, boundaries[1:])):
        n = e - s
        if n < 2:
            continue
        S = Un[s:e] @ Un[s:e].T
        iu = torch.triu_indices(n, n, offset=1)
        sims[t] = S[iu[0], iu[1]].mean().item()
    return sims


def last_token_reps(E: torch.Tensor, boundaries: list[int]) -> dict[int, torch.Tensor]:
    return {t: E[e - 1] for t, (s, e) in enumerate(zip(boundaries, boundaries[1:])) if e > s}


def inter_step_similarity(reps: dict[int, torch.Tensor]) -> dict[int, list[float]]:
    """스텝 거리 d(=t2-t1)별 코사인 유사도 리스트."""
    steps = sorted(reps)
    un = {t: torch.nn.functional.normalize(reps[t].float(), dim=0) for t in steps}
    by_dist: dict[int, list[float]] = {}
    for i, t1 in enumerate(steps):
        for t2 in steps[i + 1:]:
            by_dist.setdefault(t2 - t1, []).append((un[t1] @ un[t2]).item())
    return by_dist


def run(hidden_dir: Path, spectral_dir: Path, task: str, level: str, status: str,
        method: str = "full_sequence", ctx_tag: str = "with_prompt",
        spectral_tag: str = "k8_scaled_signfix"):
    h_dir = hidden_dir / task / level / method / ctx_tag / status
    s_dir = spectral_dir / task / level / method / ctx_tag / status / spectral_tag
    chunk_files = sorted(h_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"no chunk_*.pt in {h_dir}")

    intra_all: list[float] = []
    inter_last: dict[int, list[float]] = {}
    inter_e: dict[int, list[float]] = {}
    n_episodes = 0
    n_no_spectral = 0

    for cf in chunk_files:
        hidden_episodes = load_chunk(cf)

        spectral_f = s_dir / cf.name
        spectral_episodes = {}
        if spectral_f.exists():
            spectral_episodes = torch.load(
                spectral_f, map_location="cpu", weights_only=False)["episodes"]

        for seed, ep in hidden_episodes.items():
            n_episodes += 1
            E, boundaries = ep["E"], ep["boundaries"]

            for v in intra_step_similarity(E, boundaries).values():
                intra_all.append(v)

            for d, sims in inter_step_similarity(last_token_reps(E, boundaries)).items():
                inter_last.setdefault(d, []).extend(sims)

            if seed not in spectral_episodes:
                n_no_spectral += 1
                continue
            e_reps = spectral_episodes[seed]["e"]
            for d, sims in inter_step_similarity(e_reps).items():
                inter_e.setdefault(d, []).extend(sims)

    def summarize(vals: list[float]) -> dict:
        return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "n": len(vals)}

    summary = {
        "task": task, "level": level, "status": status, "n_episodes": n_episodes,
        "n_episodes_missing_spectral": n_no_spectral,
        "intra_step": summarize(intra_all),
        "inter_step_last_token": {d: summarize(v) for d, v in sorted(inter_last.items())},
        "inter_step_e_t": {d: summarize(v) for d, v in sorted(inter_e.items())},
    }
    return summary


def plot(summary: dict, out_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    intra = summary["intra_step"]["mean"]
    ax.axhline(intra, color="gray", linestyle="--", label=f"intra-step (mean={intra:.3f})")

    for key, label, marker in [
        ("inter_step_last_token", "inter-step (last token)", "o"),
        ("inter_step_e_t", "inter-step (e_t, spectral)", "s"),
    ]:
        by_dist = summary[key]
        if not by_dist:
            continue
        ds = sorted(by_dist)
        means = [by_dist[d]["mean"] for d in ds]
        ax.plot(ds, means, marker=marker, label=label)

    ax.set_xlabel("step distance (d)")
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(f"{summary['task']}/{summary['level']}/{summary['status']} "
                f"(n={summary['n_episodes']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--status", default="success", choices=["success", "failure"])
    ap.add_argument("--methods", default="full_sequence")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--spectral-dir", type=Path, default=ROOT / "latent" / "spectral_states")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "visual" / "step_similarity")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.task}_{args.level}_{args.status}"

    summary = run(args.hidden_dir, args.spectral_dir, args.task, args.level, args.status,
                 method=args.methods)

    (args.out_dir / f"{name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(summary, args.out_dir / f"{name}.png")

    print(f"intra-step mean={summary['intra_step']['mean']:.3f} (n={summary['intra_step']['n']})")
    for d in sorted(summary["inter_step_last_token"]):
        lt = summary["inter_step_last_token"][d]
        et = summary["inter_step_e_t"].get(d)
        et_s = f"{et['mean']:.3f}" if et else "n/a"
        print(f"  d={d}: last_token={lt['mean']:.3f} (n={lt['n']})  e_t={et_s}")
    print(f"saved -> {args.out_dir / name}.json / .png")


if __name__ == "__main__":
    main()
