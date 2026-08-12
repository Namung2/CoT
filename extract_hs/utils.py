"""Leaf helpers for main_extract.py: trajectory-safe index selection, record
building, JSONL/shard/video I/O. No model or dataset orchestration lives here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import imageio
import numpy as np
import torch


def batched(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def select_trajectory_indices(
    all_steps: list[tuple[int, int]],
    start_index: int,
    max_trajectories: int,
    stride: int,
    max_samples: int,
) -> list[int]:
    """Select flat dataset indices trajectory-by-trajectory, never crossing episode
    boundaries. ``stride`` subsamples steps within each trajectory; ``max_trajectories``
    caps how many whole episodes are included; ``max_samples`` is an optional hard cap
    on the total step count applied last (and may truncate the final trajectory).
    """
    trajectory_order: list[int] = []
    trajectory_steps: dict[int, list[int]] = {}
    for flat_index, (trajectory_id, _base_index) in enumerate(all_steps):
        steps = trajectory_steps.get(trajectory_id)
        if steps is None:
            steps = trajectory_steps[trajectory_id] = []
            trajectory_order.append(trajectory_id)
        steps.append(flat_index)

    if start_index > 0:
        trajectory_order = trajectory_order[start_index:]
    if max_trajectories > 0:
        trajectory_order = trajectory_order[:max_trajectories]

    selected: list[int] = []
    for trajectory_id in trajectory_order:
        selected.extend(trajectory_steps[trajectory_id][::stride])

    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def save_trajectory_preview_video(
    dataset, indices: list[int], output_path: Path, fps: int = 10
) -> None:
    """Render one trajectory's primary-camera frames to an mp4 for a quick sanity check."""
    frames = [np.asarray(dataset[index]["image"][0]) for index in indices]
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264")
    print(f"[video] {output_path} ({len(frames)} frames)")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def base_instruction(formatted_instruction: str) -> str:
    text = (formatted_instruction or "").strip()
    if ". @ " in text:
        return text.split(". @ ", 1)[0].strip()
    return text


def record_id(task_suite: str, episode_index: int, step_index: int) -> str:
    """Stable join key shared by train.jsonl, labels JSONL, and feature shards."""
    key = f"{task_suite}|{episode_index}|{step_index}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def make_feature_record(
    record_id_value: str,
    episode_index: int,
    step_index: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    thinking_positions: torch.Tensor,
    thinking_hidden_states: torch.Tensor,
    num_reasoning_passes: int,
    sample: dict[str, Any],
    save_images: bool,
) -> dict[str, Any]:
    """Binary feature payload. Ground-truth annotation fields must not enter here.

    ``episode_index``/``step_index`` are included so a shard (one episode; see
    ``save_shard``) is self-describing without cross-referencing train.jsonl.
    """
    valid_length = (
        int(attention_mask.sum().item())
        if attention_mask is not None
        else int(input_ids.numel())
    )
    record: dict[str, Any] = {
        "id": record_id_value,
        "episode_index": int(episode_index),
        "step_index": int(step_index),
        "input_ids": input_ids[:valid_length].detach().cpu(),
        "thinking_positions": thinking_positions.detach().cpu(),
        "thinking_hidden_states": thinking_hidden_states.detach().to(
            device="cpu", dtype=torch.float16
        ),
        "num_reasoning_passes": int(num_reasoning_passes),
    }
    if save_images:
        record["images"] = torch.stack(
            [torch.from_numpy(np.asarray(image).copy()) for image in sample["image"]],
            dim=0,
        )
    return record


def make_steps_record(
    record_id_value: str,
    task_suite: str,
    dataset_index: int,
    episode_index: int,
    step_index: int,
    sample: dict[str, Any],
    record_index: int,
    thinking_positions: torch.Tensor,
    thinking_hidden_states: torch.Tensor,
) -> dict[str, Any]:
    """Observable sample metadata plus a pointer into the binary feature shard.

    One shard == one episode (see ``save_shard``), so the shard filename is
    derived directly from ``episode_index``.
    """
    return {
        "id": record_id_value,
        "task_suite": task_suite,
        "dataset_index": int(dataset_index),
        "episode_index": int(episode_index),
        "step_index": int(step_index),
        "instruction": base_instruction(str(sample.get("language", ""))),
        "formatted_instruction": str(sample.get("language", "")),
        "feature": {
            "shard": f"shard_ep{episode_index:06d}.npz",
            "record_index": int(record_index),
            "thinking_count": int(thinking_positions.numel()),
            "hidden_size": int(thinking_hidden_states.shape[-1]),
            "dtype": "float16",
        },
    }


def make_labels_record(
    record_id_value: str,
    task_suite: str,
    dataset_index: int,
    episode_index: int,
    step_index: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Ground-truth-only sidecar. Join to train.jsonl explicitly by ``id``."""
    return {
        "id": record_id_value,
        "task_suite": task_suite,
        "dataset_index": int(dataset_index),
        "episode_index": int(episode_index),
        "step_index": int(step_index),
        "cot_subtask": str(sample.get("cot_subtask", "")),
        "cot_reasoning": str(sample.get("cot_reasoning", "")),
        "cot_gripper_state": sample.get("cot_gripper_state"),
        "bbox": np.asarray(sample.get("bbox", np.zeros(4)), dtype=np.float32),
        "bbox_valid": bool(sample.get("bbox_valid", False)),
        "action": np.asarray(sample["action"], dtype=np.float32),
    }


def write_jsonl_record(file, record: dict[str, Any]) -> None:
    file.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    """Preflight output collision and remove only known artifacts with --overwrite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_path = output_dir / "train.jsonl"
    labels_path = output_dir / "train.labels.jsonl"
    manifest_path = output_dir / "manifest.json"
    legacy_paths = [output_dir / "metadata.json", output_dir / "summary.json"]
    known = [steps_path, labels_path, manifest_path, *legacy_paths,
             *sorted(output_dir.glob("shard_*.npz"))]
    existing = [path for path in known if path.exists()]
    if existing and not overwrite:
        shown = "\n  ".join(str(path) for path in existing[:10])
        raise FileExistsError(
            f"Output artifacts already exist. Use --overwrite or a new --output-root:\n  {shown}"
        )
    if overwrite:
        for path in existing:
            path.unlink()
        for path in (steps_path.with_suffix(".jsonl.tmp"),
                     labels_path.with_suffix(".jsonl.tmp")):
            if path.exists():
                path.unlink()
    return steps_path, labels_path, manifest_path


def save_shard(
    output_dir: Path,
    episode_index: int,
    records: list[dict[str, Any]],
) -> Path:
    """Pack one episode's records into one compressed .npz shard: fixed-size
    numeric arrays only (ids as fixed-width unicode, input_ids padded + a length
    column) so it loads with plain ``np.load`` — no torch and no ``allow_pickle``
    required. One shard == one episode, so a training batch of N episodes is
    just N shard files, no cross-shard stitching or partial-episode boundaries.
    """
    output_path = output_dir / f"shard_ep{episode_index:06d}.npz"
    temporary_path = output_path.with_suffix(".npz.tmp")

    ids = np.array([r["id"] for r in records], dtype="<U12")
    episode_indices = np.array([r["episode_index"] for r in records], dtype=np.int64)
    step_indices = np.array([r["step_index"] for r in records], dtype=np.int64)
    max_len = max(int(r["input_ids"].numel()) for r in records)
    input_ids = np.zeros((len(records), max_len), dtype=np.int64)
    input_ids_length = np.zeros(len(records), dtype=np.int64)
    for i, r in enumerate(records):
        arr = r["input_ids"].numpy()
        input_ids[i, : arr.shape[0]] = arr
        input_ids_length[i] = arr.shape[0]
    thinking_positions = np.stack([r["thinking_positions"].numpy() for r in records])
    thinking_hidden_states = np.stack([r["thinking_hidden_states"].numpy() for r in records])
    num_reasoning_passes = np.array([r["num_reasoning_passes"] for r in records], dtype=np.int64)

    payload = dict(
        ids=ids,
        episode_index=episode_indices,
        step_index=step_indices,
        input_ids=input_ids,
        input_ids_length=input_ids_length,
        thinking_positions=thinking_positions,
        thinking_hidden_states=thinking_hidden_states,
        num_reasoning_passes=num_reasoning_passes,
    )
    if "images" in records[0]:
        payload["images"] = np.stack([r["images"].numpy() for r in records])

    with temporary_path.open("wb") as file:
        np.savez_compressed(file, **payload)
    os.replace(temporary_path, output_path)
    print(f"[save] {output_path} ({len(records)} feature records, ep{episode_index})")
    return output_path


def load_feature_shards(shard_dir: Path) -> dict[str, np.ndarray]:
    """Concatenate every shard_*.npz under shard_dir into one columnar dict,
    re-padding ``input_ids`` to the global max length across shards.
    """
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found under {shard_dir}")
    chunks: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path) as data:
            chunks.append({key: data[key] for key in data.files})

    max_input_len = max(chunk["input_ids"].shape[1] for chunk in chunks)
    merged: dict[str, np.ndarray] = {}
    for key in chunks[0]:
        if key == "input_ids":
            padded = []
            for chunk in chunks:
                arr = chunk[key]
                pad_width = max_input_len - arr.shape[1]
                if pad_width:
                    arr = np.pad(arr, ((0, 0), (0, pad_width)))
                padded.append(arr)
            merged[key] = np.concatenate(padded, axis=0)
        else:
            merged[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)
    return merged
