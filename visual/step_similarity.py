"""intra-step vs inter-step(스텝 거리별) 유사도 요약.

intra-step : 한 스텝 안 토큰들끼리의 평균 코사인 유사도 (hidden_states의 raw E 사용).
             스텝 인덱스별로 분리해서 저장(뭉개지 않음) + 참고용 전체 평균도 같이 냄.
inter-step : 스텝 t와 t+d의 유사도, 대표 벡터 두 가지로 각각 계산
  - last_token : 그 스텝의 마지막 토큰 벡터 (attention으로 이미 그 스텝을 반영한, 모델이
                 자체적으로 만든 대표값)
  - e_t        : spectral_states의 e_t (우리가 SVD로 명시적으로 요약한 대표값)

새 추출 없이 기존 hidden_states/spectral_states만 읽는다.
--seed 안 주면 레벨 전체 episode를 풀링해서 평균, 주면 그 episode 하나만(다른 episode랑 안 섞임).

    python visual/step_similarity.py --task decompose --level BabyAI-GoToObj-v0 --status success
    python visual/step_similarity.py --task decompose --level BabyAI-GoToObj-v0 --status success --seed 93
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
        seed: int | None = None, method: str = "full_sequence", ctx_tag: str = "with_prompt",
        spectral_tag: str = "k8_scaled_signfix"):
    """seed=None이면 레벨 전체 episode를 다 풀링해서 평균(기존 동작).
    seed를 주면 그 episode 하나만 갖고 계산 — 다른 episode랑 안 섞임."""
    h_dir = hidden_dir / task / level / method / ctx_tag / status
    s_dir = spectral_dir / task / level / method / ctx_tag / status / spectral_tag
    chunk_files = sorted(h_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"no chunk_*.pt in {h_dir}")

    intra_by_step: dict[int, list[float]] = {}   # 스텝 인덱스 t 별로 분리 유지 (뭉개지 않음)
    inter_last: dict[int, list[float]] = {}
    inter_e: dict[int, list[float]] = {}
    n_episodes = 0
    n_no_spectral = 0

    for cf in chunk_files:
        hidden_episodes = load_chunk(cf)
        if seed is not None:
            if seed not in hidden_episodes:
                continue
            hidden_episodes = {seed: hidden_episodes[seed]}

        spectral_f = s_dir / cf.name
        spectral_episodes = {}
        if spectral_f.exists():
            spectral_episodes = torch.load(
                spectral_f, map_location="cpu", weights_only=False)["episodes"]

        for sd, ep in hidden_episodes.items():
            n_episodes += 1
            E, boundaries = ep["E"], ep["boundaries"]

            for t, v in intra_step_similarity(E, boundaries).items():
                intra_by_step.setdefault(t, []).append(v)

            for d, sims in inter_step_similarity(last_token_reps(E, boundaries)).items():
                inter_last.setdefault(d, []).extend(sims)

            if sd not in spectral_episodes:
                n_no_spectral += 1
                continue
            e_reps = spectral_episodes[sd]["e"]
            for d, sims in inter_step_similarity(e_reps).items():
                inter_e.setdefault(d, []).extend(sims)

        if seed is not None and n_episodes:   # 이미 찾았으면 남은 청크 안 읽음
            break

    if seed is not None and n_episodes == 0:
        raise KeyError(f"seed {seed} not found under {h_dir}")

    def summarize(vals: list[float]) -> dict:
        return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "n": len(vals)}

    intra_pooled = [v for vals in intra_by_step.values() for v in vals]
    summary = {
        "task": task, "level": level, "status": status, "n_episodes": n_episodes,
        "n_episodes_missing_spectral": n_no_spectral,
        "intra_step_overall": summarize(intra_pooled),          # 참고용 기준선(스텝 구분 없이 전체 평균)
        "intra_step_by_index": {t: summarize(v) for t, v in sorted(intra_by_step.items())},
        "inter_step_last_token": {d: summarize(v) for d, v in sorted(inter_last.items())},
        "inter_step_e_t": {d: summarize(v) for d, v in sorted(inter_e.items())},
    }
    return summary


def plot(summary: dict, out_path: Path):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 왼쪽: inter-step vs 스텝 거리(d) — intra-step 전체 평균은 참고용 기준선으로만
    intra_ref = summary["intra_step_overall"]["mean"]
    ax1.axhline(intra_ref, color="gray", linestyle="--",
               label=f"intra-step overall (mean={intra_ref:.3f})")
    for key, label, marker in [
        ("inter_step_last_token", "inter-step (last token)", "o"),
        ("inter_step_e_t", "inter-step (e_t, spectral)", "s"),
    ]:
        by_dist = summary[key]
        if not by_dist:
            continue
        ds = sorted(by_dist)
        means = [by_dist[d]["mean"] for d in ds]
        ax1.plot(ds, means, marker=marker, label=label)
    ax1.set_xlabel("step distance (d)")
    ax1.set_ylabel("mean cosine similarity")
    ax1.set_title("inter-step (by distance)")
    ax1.legend(fontsize=8)

    # 오른쪽: intra-step을 스텝 인덱스(t)별로 — 뭉개지 않고 그대로
    by_idx = summary["intra_step_by_index"]
    ts = sorted(by_idx)
    if ts:
        means = [by_idx[t]["mean"] for t in ts]
        stds = [by_idx[t]["std"] for t in ts]
        ax2.errorbar(ts, means, yerr=stds, marker="o", capsize=3, color="gray")
    ax2.set_xlabel("step index (t)")
    ax2.set_ylabel("mean cosine similarity")
    ax2.set_title("intra-step (by step index)")

    fig.suptitle(f"{summary['task']}/{summary['level']}/{summary['status']} "
                f"(n={summary['n_episodes']})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--status", default="success", choices=["success", "failure"])
    ap.add_argument("--seed", type=int, default=None,
                    help="episode env_seed. 주면 그 episode 하나만 계산(다른 episode랑 안 섞임). "
                         "안 주면 레벨 전체 episode를 풀링해서 평균(기존 동작)")
    ap.add_argument("--methods", default="full_sequence")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--spectral-dir", type=Path, default=ROOT / "latent" / "spectral_states")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "visual" / "step_similarity")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.task}_{args.level}_{args.status}"
    if args.seed is not None:
        name += f"_{args.seed}"

    summary = run(args.hidden_dir, args.spectral_dir, args.task, args.level, args.status,
                 seed=args.seed, method=args.methods)

    (args.out_dir / f"{name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(summary, args.out_dir / f"{name}.png")

    ov = summary["intra_step_overall"]
    print(f"intra-step overall mean={ov['mean']:.3f} (n={ov['n']})")
    for t in sorted(summary["intra_step_by_index"]):
        it = summary["intra_step_by_index"][t]
        print(f"  step {t}: intra={it['mean']:.3f} (n={it['n']})")
    for d in sorted(summary["inter_step_last_token"]):
        lt = summary["inter_step_last_token"][d]
        et = summary["inter_step_e_t"].get(d)
        et_s = f"{et['mean']:.3f}" if et else "n/a"
        print(f"  d={d}: last_token={lt['mean']:.3f} (n={lt['n']})  e_t={et_s}")
    print(f"saved -> {args.out_dir / name}.json / .png")


if __name__ == "__main__":
    main()
