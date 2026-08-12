"""CLI argument parser and path defaults for main_extract.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/LaRA-VLA-libero/checkpoints/steps_25000_pytorch_model.pt"
)
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/libero_lerobot"
DEFAULT_OUTPUT_ROOT = Path("/home/hail/HDD/lara_vla_dataset/cot")
DEFAULT_DATASET_REPO = "lovejuly/libero_lerobot_all"

TASK_SUITES = (
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)


def build_argparser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--task-suite", choices=TASK_SUITES, default=TASK_SUITES[0])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N trajectories (episodes), in dataset order.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="Maximum number of trajectories (episodes) to include, keeping each one "
        "intact; 0 means all trajectories. Use this (not --max-samples) to take a "
        "trajectory-safe debug subset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Hard cap on total selected steps after trajectory/stride selection; 0 "
        "means no cap. Can still cut a trajectory short — prefer --max-trajectories "
        "for debug runs that should keep whole episodes.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Within each trajectory, take every Nth step. Use 8 to match the "
        "action-chunk cadence. Never skips across trajectory boundaries.",
    )
    parser.add_argument(
        "--require-cot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip steps without dense CoT annotations.",
    )
    parser.add_argument(
        "--delete-pause-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same pause-frame filtering as training.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also store the two input images in every record (substantially larger output).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never contact the Hub; require a materialized local task-suite directory.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default="debugpy" in sys.modules,
        help="Replace existing extractor artifacts instead of failing. "
        "Defaults to on when running under a debugger (debugpy attached).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Also render the first selected trajectory's primary-camera frames to "
        "<output_dir>/preview_traj0.mp4, as a quick sanity check.",
    )
    return parser
