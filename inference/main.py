from __future__ import annotations

from pathlib import Path

from extract import load_episodes, extract_run
from spectral import spectral_run

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "baby_ai" / "data"
HIDDEN_DIR = ROOT / "state" / "hidden_states"
SPECTRAL_DIR = ROOT / "state" / "spectral_states"

LEVEL = "gotoseq_10to50_by_step"
CASE = "c3"
METHODS = ("full",) # full: full_sequence | A: cumulative_prefix_A | B: cumulative_prefix_B

if __name__ == "__main__":
    episodes = load_episodes(DATA_DIR, level=LEVEL, case=CASE)
    print(f"{len(episodes)} episodes")
    extract_run(episodes, DATA_DIR, HIDDEN_DIR)

    # 원본 jsonl 파일 하나당 hidden state 서브트리 하나 (예: by_step/c3/30step)
    rels = sorted({path.relative_to(DATA_DIR).with_suffix("") for path, _ in episodes})
    for rel in rels:
        for method in METHODS:
            spectral_run(
                method=method,
                data_dir=HIDDEN_DIR / rel / method,
                dst=SPECTRAL_DIR / rel / f"{method}.pt",
            )
