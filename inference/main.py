from __future__ import annotations

from pathlib import Path

from extract import load_episodes, extract_run
from spectral import spectral_run

ROOT = Path(__file__).resolve().parent.parent

# 여기 세 경로만 바꾸면 데이터/hidden state/spectral 출력 위치가 전부 바뀝니다.
DATA_DIR = ROOT / "data"
HIDDEN_DIR = ROOT / "result" / "hidden_states"
SPECTRAL_DIR = ROOT / "result" / "spectral_states"

LEVEL = None   # "gotoseq_*", "*_step*", None(전부) 도 가능
CASE = "cases"
METHODS = ("full",)

if __name__ == "__main__":
    episodes = load_episodes(DATA_DIR, level=LEVEL, case=CASE)
    print(f"{len(episodes)} episodes")
    extract_run(episodes, DATA_DIR, HIDDEN_DIR)

    # 원본 jsonl 파일 하나당 hidden state 서브트리 하나 (예: gotoseq_step10/cases/c3)
    rels = sorted({path.relative_to(DATA_DIR).with_suffix("") for path, _ in episodes})
    for rel in rels:
        for method in METHODS:
            spectral_run(
                method=method,
                data_dir=HIDDEN_DIR / rel / method,
                dst=SPECTRAL_DIR / rel / f"{method}.pt",
            )