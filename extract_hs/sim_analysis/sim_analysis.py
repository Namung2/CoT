"""Sanity-check the diversity of extracted <|thinking|> latents.

Loads every shard_*.npz under a task-suite extraction output directory, reads
each record's episode/step directly from the shard's own columns (verified
against the record id via ``utils.verify_record_ids``), picks a handful of
episodes, and plots one cosine-similarity heatmap per
episode per thinking component (SUBTASK / BBOX / REASON — order follows the
``component_order`` used at extraction time). Episodes are independent tasks,
so each gets its own step-by-step matrix (own step count, own axis) rather
than being concatenated into a shared one.

Example:

    /home/hail/anaconda3/envs/lara-vla_/bin/python extract_hs/sim_analysis/sim_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SHARD_DIR = Path("/home/hail/HDD/lara_vla_dataset/cot/libero_goal_no_noops_1.0.0_lerobot")
DEFAULT_TASK_SUITE = "libero_goal_no_noops_1.0.0_lerobot"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "sim_heatmaps.png"

COMPONENT_NAMES = ("SUBTASK", "BBOX", "REASON")


def pick_episodes(labels: list[tuple[int, int]], num_episodes: int, seed: int):
    """Group record indices by episode, ordered by step_index within each episode."""
    by_episode: dict[int, list[int]] = {}
    for record_index, (episode_index, _step_index) in enumerate(labels):
        by_episode.setdefault(episode_index, []).append(record_index)

    available = sorted(by_episode)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(available, size=min(num_episodes, len(available)), replace=False).tolist())

    episode_record_indices = {
        episode_index: sorted(by_episode[episode_index], key=lambda i: labels[i][1])
        for episode_index in chosen
    }
    return chosen, episode_record_indices


def plot_similarity_heatmaps(hidden_states, labels, chosen_episodes, episode_record_indices, output_path):
    num_episodes = len(chosen_episodes)
    fig, axes = plt.subplots(num_episodes, 3, figsize=(19, 6 * num_episodes), squeeze=False)

    for row, episode_index in enumerate(chosen_episodes):
        record_indices = episode_record_indices[episode_index]
        sampled = hidden_states[record_indices]  # (n, 3, 2560), n = this episode's own step count
        n = sampled.shape[0]
        step_indices = [labels[i][1] for i in record_indices]

        tick_stride = max(1, n // 15)
        tick_positions = list(range(0, n, tick_stride))
        tick_labels = [str(step_indices[p]) for p in tick_positions]

        for col, component_name in enumerate(COMPONENT_NAMES):
            vectors = sampled[:, col, :].astype(np.float32)
            sim = utils.cosine_similarity_matrix(vectors)
            off_diag = sim[~np.eye(n, dtype=bool)]
            print(
                f"[ep{episode_index} {component_name}] cosine sim off-diagonal: "
                f"mean={off_diag.mean():.4f} std={off_diag.std():.4f} "
                f"min={off_diag.min():.4f} max={off_diag.max():.4f}"
            )

            ax = axes[row][col]
            im = ax.imshow(sim, vmin=-1.0, vmax=1.0, cmap="coolwarm", origin="lower")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=13)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=13)
            ax.set_xlabel("step_index", fontsize=13)
            ax.set_ylabel("step_index", fontsize=13)
            ax.set_title(f"ep{episode_index} — {component_name} (n={n} steps)", fontsize=15)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"[save] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--task-suite", default=DEFAULT_TASK_SUITE)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = utils.load_all_records(args.shard_dir)
    num_records = len(data["ids"])
    print(f"[data] loaded {num_records} records from {args.shard_dir}")

    utils.verify_record_ids(data, args.task_suite)
    labels = list(zip(data["episode_index"].tolist(), data["step_index"].tolist()))
    print("[labels] read from shard episode_index/step_index; id-consistency verified")

    chosen_episodes, episode_record_indices = pick_episodes(labels, args.num_episodes, args.seed)
    tasks = utils.load_episode_tasks(args.task_suite)
    print(f"[episodes] chosen ({len(chosen_episodes)}):")
    for episode_index in chosen_episodes:
        print(
            f"  ep{episode_index:03d} ({len(episode_record_indices[episode_index])} steps): "
            f"{tasks.get(episode_index, '?')}"
        )

    plot_similarity_heatmaps(
        data["thinking_hidden_states"], labels, chosen_episodes, episode_record_indices, args.output
    )


if __name__ == "__main__":
    main()
