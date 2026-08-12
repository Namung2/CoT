"""Leaf helpers for sim_analysis.py: shard loading, cheap id verification, and
cosine similarity. No plotting lives here.

episode_index/step_index are read directly from the shard's own embedded
columns (saved by main_extract.py) rather than reconstructed by replaying
dataset iteration — reconstruction assumed a 1:1 position match between a
fresh iteration and the saved records, which breaks now that main_extract.py
can skip unreadable frames (e.g. the corrupted episode 82 video).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so `from extract_hs.utils import record_id` resolves

from extract_hs.utils import load_feature_shards, record_id  # noqa: E402


def load_all_records(shard_dir: Path) -> dict[str, np.ndarray]:
    """Columnar dict (ids, episode_index, step_index, input_ids(+length),
    thinking_positions, thinking_hidden_states, num_reasoning_passes, ...)
    merged across every shard_*.npz under shard_dir. Pure numpy, no torch.
    """
    return load_feature_shards(shard_dir)


def verify_record_ids(data: dict[str, np.ndarray], task_suite: str) -> None:
    """Cheap by-construction check: id should equal record_id(task_suite, ep, step).
    Both were derived from the same (episode_index, step_index) at write time, so
    this only catches real corruption — no dataset/model reconstruction needed.
    """
    for i in range(len(data["ids"])):
        expected = record_id(task_suite, int(data["episode_index"][i]), int(data["step_index"][i]))
        actual = str(data["ids"][i])
        if expected != actual:
            raise ValueError(
                f"id mismatch at record {i}: stored id={actual} but recomputed "
                f"id={expected} from episode={data['episode_index'][i]} "
                f"step={data['step_index'][i]}"
            )


def load_episode_tasks(task_suite: str) -> dict[int, str]:
    path = REPO_ROOT / "data/libero_lerobot" / task_suite / "meta/episodes.jsonl"
    tasks: dict[int, str] = {}
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            tasks[row["episode_index"]] = ", ".join(row.get("tasks", []))
    return tasks


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-8, None)
    return normalized @ normalized.T
