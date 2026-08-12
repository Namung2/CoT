"""Extract implicit-CoT hidden states from LaRA-VLA on LIBERO LeRobot data.

This is an offline extractor.  It doesn't start LIBERO, robosuite, or the
websocket policy server.  It loads the released ``Qwen_GR00T`` checkpoint,
rebuilds the Stage-4 prompt used during training, runs ``forward_latent``, and
saves only the final-layer vectors at ``<|thinking|>`` token positions.

Sampling is trajectory-safe: steps are selected trajectory-by-trajectory (never
crossing an episode boundary), ``--stride`` subsamples within each trajectory, and
``--max-trajectories`` caps how many whole episodes are included. Use
``--max-trajectories`` (not ``--max-samples``) for a quick debug run.

Example (debug, a few complete trajectories):

    /home/hail/anaconda3/envs/lara-vla_/bin/python extract_hs/main_extract.py \
      --checkpoint checkpoints/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt \
      --task-suite libero_goal_no_noops_1.0.0_lerobot \
      --max-trajectories 3

Example (full extraction, every trajectory in the task suite):

    /home/hail/anaconda3/envs/lara-vla_/bin/python extract_hs/main_extract.py \
      --checkpoint checkpoints/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt \
      --task-suite libero_goal_no_noops_1.0.0_lerobot

The default output is ``/home/hail/HDD/lara_vla_dataset/cot/<task-suite>/``.
Hidden-state features, labels, and generation provenance are stored separately:
``shard_ep<episode_index>.npz`` (one shard per episode, self-describing with its
own episode_index/step_index arrays — a training batch of N episodes is just N
shard files) + ``train.jsonl`` + ``train.labels.jsonl`` + ``manifest.json``.

Leaf helpers (index selection, record building, JSONL/shard/video I/O) live in
``extract_hs/utils.py``; this file is just dataset setup, model loading, and
the extraction/save loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import tqdm
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so `from extract_hs import utils` resolves when run directly

from extract_hs import utils
from extract_hs.parser import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_REPO,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    TASK_SUITES,
    build_argparser,
)
from laravla.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from laravla.model.framework.base_framework import baseframework
from laravla.model.framework.share_tools import read_mode_config


def ensure_dataset_suite(
    dataset_root: Path,
    repo_id: str,
    task_suite: str,
    offline: bool,
) -> Path:
    """Materialize one suite from the Hub cache when it isn't a local dataset yet."""
    dataset_root = dataset_root.expanduser().resolve()
    suite_path = dataset_root / task_suite
    required = (
        suite_path / "meta/info.json",
        suite_path / "meta/modality.json",
        suite_path / "meta/episodes.jsonl",
        suite_path / "meta/tasks.jsonl",
    )
    if all(path.exists() for path in required):
        return suite_path

    if offline:
        missing = "\n  ".join(str(path) for path in required if not path.exists())
        raise FileNotFoundError(
            f"Dataset suite isn't materialized and --offline was set. Missing:\n  {missing}"
        )

    dataset_root.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] materializing {repo_id}/{task_suite} into {dataset_root}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_root),
        allow_patterns=[f"{task_suite}/**"],
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset download finished but required files are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    return suite_path


def extraction_dataset_config(model_config: dict[str, Any], require_cot: bool) -> tuple[dict, dict]:
    """Copy checkpoint data config while removing machine-specific cache paths."""
    vla_cfg = ((model_config.get("datasets") or {}).get("vla_data") or {})
    annotations_cfg = copy.deepcopy(vla_cfg.get("bridge_annotations") or {})
    reasoning_cfg = copy.deepcopy(vla_cfg.get("bridge_reasoning") or {})

    # These paths point at the training machine and aren't needed for extraction.
    annotations_cfg["steps_cache_path"] = None
    annotations_cfg["write_steps_cache"] = False
    annotations_cfg["fast_tokenizer_name"] = None

    filters_cfg = copy.deepcopy(annotations_cfg.get("filters") or {})
    filters_cfg["require_cot_episode"] = bool(require_cot)
    annotations_cfg["filters"] = filters_cfg

    if not reasoning_cfg:
        reasoning_cfg = {
            "enable": True,
            "stage": 4,
            "include_bbox": True,
            "include_action_tokens": False,
            "include_img_next": True,
            "thinking_token": "<|thinking|>",
            "start_token": "<|start_of_thinking|>",
            "end_token": "<|end_of_thinking|>",
            "component_order": ["SUBTASK", "BBOX", "REASON"],
            "tag2think_count": {"SUBTASK": 1, "BBOX": 1, "REASON": 1},
        }
    reasoning_cfg["enable"] = True
    reasoning_cfg["include_action_tokens"] = False
    return annotations_cfg, reasoning_cfg


def load_model(checkpoint: Path, device: str) -> baseframework:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")

    print(f"[model] loading {checkpoint}")
    model = baseframework.from_pretrained(str(checkpoint))
    model = model.to(dtype=torch.bfloat16).to(device).eval()
    print(f"[model] framework={type(model).__name__}")
    return model


@torch.inference_mode()
def extract(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.stride <= 0:
        raise ValueError("--batch-size and --stride must be positive")
    if args.start_index < 0 or args.max_samples < 0 or args.max_trajectories < 0:
        raise ValueError(
            "--start-index, --max-samples, and --max-trajectories must be non-negative"
        )

    checkpoint = args.checkpoint.expanduser().resolve()
    model_config, _ = read_mode_config(checkpoint)
    suite_path = ensure_dataset_suite(
        dataset_root=args.dataset_root,
        repo_id=args.dataset_repo,
        task_suite=args.task_suite,
        offline=args.offline,
    )
    annotations_cfg, reasoning_cfg = extraction_dataset_config(
        model_config=model_config,
        require_cot=args.require_cot,
    )

    dataset = make_LeRobotSingleDataset(
        data_root_dir=suite_path.parent,
        data_name=suite_path.name,
        robot_type="libero_franka",
        delete_pause_frame=args.delete_pause_frames,
        bridge_annotation_cfg=annotations_cfg,
        bridge_filter_cfg=annotations_cfg.get("filters"),
        bridge_reasoning_cfg=reasoning_cfg,
    )
    print(f"[dataset] {dataset} at {suite_path}")

    all_indices = utils.select_trajectory_indices(
        dataset.all_steps,
        start_index=args.start_index,
        max_trajectories=args.max_trajectories,
        stride=args.stride,
        max_samples=args.max_samples,
    )
    if not all_indices:
        raise ValueError("No dataset samples selected")

    output_dir = args.output_root.expanduser().resolve() / args.task_suite
    steps_path, labels_path, manifest_path = utils.prepare_output_dir(
        output_dir, args.overwrite
    )
    steps_tmp = steps_path.with_suffix(".jsonl.tmp")
    labels_tmp = labels_path.with_suffix(".jsonl.tmp")

    if args.save_video:
        first_trajectory_id = dataset.all_steps[all_indices[0]][0]
        preview_indices = [
            index for index in all_indices
            if dataset.all_steps[index][0] == first_trajectory_id
        ]
        utils.save_trajectory_preview_video(
            dataset, preview_indices, output_dir / "preview_traj0.mp4"
        )

    model = load_model(checkpoint, args.device)
    interface = model.qwen_vl_interface
    thinking_token_id = getattr(interface, "thinking_token_id", None)
    if thinking_token_id is None:
        raise RuntimeError("The loaded model has no thinking_token_id; implicit reasoning isn't enabled")

    feature_records: list[dict[str, Any]] = []
    current_shard_episode: int | None = None
    shard_count = 0
    saved_count = 0
    skipped_without_cot = 0
    skipped_without_thinking = 0
    skipped_corrupt = 0

    try:
        with steps_tmp.open("w", encoding="utf-8") as steps_file, \
             labels_tmp.open("w", encoding="utf-8") as labels_file:
            num_batches = -(-len(all_indices) // args.batch_size)  # ceil div
            progress = tqdm.tqdm(
                utils.batched(all_indices, args.batch_size),
                total=num_batches,
                unit="batch",
                desc=args.task_suite,
            )
            for batch_indices in progress:
                samples = []
                readable_indices = []
                for index in batch_indices:
                    try:
                        samples.append(dataset[index])
                        readable_indices.append(index)
                    except Exception as exc:
                        skipped_corrupt += 1
                        tqdm.tqdm.write(
                            f"[warn] skipping unreadable sample at dataset_index={index}: {exc}"
                        )
                batch_indices = readable_indices
                if not samples:
                    continue
                kept = [
                    (index, sample)
                    for index, sample in zip(batch_indices, samples)
                    if (not args.require_cot) or bool(sample.get("cot_available", False))
                ]
                skipped_without_cot += len(samples) - len(kept)
                if not kept:
                    continue

                kept_indices = [item[0] for item in kept]
                kept_samples = [item[1] for item in kept]
                qwen_inputs = interface.build_qwenvl_inputs(
                    images=[sample["image"] for sample in kept_samples],
                    instructions=[str(sample["language"]) for sample in kept_samples],
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.device.startswith("cuda"),
                ):
                    outputs = interface.forward_latent(
                        input_ids=qwen_inputs["input_ids"],
                        attention_mask=qwen_inputs["attention_mask"],
                        pixel_values=qwen_inputs.get("pixel_values"),
                        image_grid_thw=qwen_inputs.get("image_grid_thw"),
                    )

                hidden_states = outputs["hidden_states"]
                num_reasoning_passes = int(outputs.get("num_reasoning_passes", 0))
                input_ids = qwen_inputs["input_ids"]
                attention_mask = qwen_inputs.get("attention_mask")

                for batch_index, (dataset_index, sample) in enumerate(
                    zip(kept_indices, kept_samples)
                ):
                    positions = torch.nonzero(
                        input_ids[batch_index] == int(thinking_token_id), as_tuple=False
                    ).squeeze(-1)
                    if positions.numel() == 0:
                        skipped_without_thinking += 1
                        continue
                    vectors = hidden_states[batch_index, positions, :]
                    episode_index, step_index = dataset.all_steps[dataset_index]

                    # One shard == one episode: flush the just-finished episode's
                    # buffer the moment we see the next episode start.
                    if (
                        current_shard_episode is not None
                        and episode_index != current_shard_episode
                        and feature_records
                    ):
                        utils.save_shard(output_dir, current_shard_episode, feature_records)
                        feature_records = []
                        shard_count += 1
                    current_shard_episode = episode_index

                    rid = utils.record_id(args.task_suite, int(episode_index), int(step_index))
                    record_index = len(feature_records)

                    feature_records.append(
                        utils.make_feature_record(
                            record_id_value=rid,
                            episode_index=int(episode_index),
                            step_index=int(step_index),
                            input_ids=input_ids[batch_index],
                            attention_mask=(attention_mask[batch_index]
                                            if attention_mask is not None else None),
                            thinking_positions=positions,
                            thinking_hidden_states=vectors,
                            num_reasoning_passes=num_reasoning_passes,
                            sample=sample,
                            save_images=args.save_images,
                        )
                    )
                    utils.write_jsonl_record(
                        steps_file,
                        utils.make_steps_record(
                            record_id_value=rid,
                            task_suite=args.task_suite,
                            dataset_index=dataset_index,
                            episode_index=int(episode_index),
                            step_index=int(step_index),
                            sample=sample,
                            record_index=record_index,
                            thinking_positions=positions,
                            thinking_hidden_states=vectors,
                        ),
                    )
                    utils.write_jsonl_record(
                        labels_file,
                        utils.make_labels_record(
                            record_id_value=rid,
                            task_suite=args.task_suite,
                            dataset_index=dataset_index,
                            episode_index=int(episode_index),
                            step_index=int(step_index),
                            sample=sample,
                        ),
                    )
                    saved_count += 1

                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                progress.set_postfix(
                    saved=saved_count,
                    skip_cot=skipped_without_cot,
                    skip_think=skipped_without_thinking,
                    skip_bad=skipped_corrupt,
                )

            if feature_records:
                utils.save_shard(output_dir, current_shard_episode, feature_records)
                shard_count += 1
    except Exception:
        for path in (steps_tmp, labels_tmp):
            if path.exists():
                path.unlink()
        raise

    os.replace(steps_tmp, steps_path)
    os.replace(labels_tmp, labels_path)

    stats = {
        "selected_samples": len(all_indices),
        "saved_records": saved_count,
        "feature_shards": shard_count,
        "skipped_without_cot": skipped_without_cot,
        "skipped_without_thinking": skipped_without_thinking,
        "skipped_corrupt": skipped_corrupt,
    }
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "config": {
            "checkpoint": str(checkpoint),
            "framework": model_config.get("framework", {}).get("name"),
            "dataset_repo": args.dataset_repo,
            "dataset_path": str(suite_path),
            "task_suite": args.task_suite,
            "start_index": args.start_index,
            "stride": args.stride,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "save_images": args.save_images,
            "require_cot": args.require_cot,
            "delete_pause_frames": args.delete_pause_frames,
            "reasoning": reasoning_cfg,
        },
        "schema": {
            "features": "shard_*.npz: ids, input_ids(+input_ids_length), thinking_positions, thinking_hidden_states, num_reasoning_passes",
            "samples": "train.jsonl: observable identity/instruction and feature shard pointer",
            "labels": "train.labels.jsonl: CoT annotations, bbox, and action; explicit id join required",
            "join_key": "id",
            "row_order": "train.jsonl and train.labels.jsonl are aligned",
        },
        "stats": stats,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(utils.to_jsonable(manifest), file, ensure_ascii=False, indent=2)
    print(json.dumps({**stats, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    extract(build_argparser(description=__doc__).parse_args())
