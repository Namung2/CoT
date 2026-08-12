from __future__ import annotations

from pathlib import Path

from extract import load_episodes, extract_run
from spectral import spectral_dir_run

ROOT = Path(__file__).resolve().parent.parent

# 여기 세 경로만 바꾸면 데이터/hidden state/spectral 출력 위치가 전부 바뀝니다.
DATA_DIR = ROOT / "data"
HIDDEN_DIR = ROOT / "result" / "hidden_states"
SPECTRAL_DIR = ROOT / "result" / "spectral_states"

LEVEL = None   # "gotoseq_*", "*_step*", None(전부) 도 가능
CASE = "cases"
METHODS = ("full",)

EXTRACT = False     # False면 hidden state 재사용, 스펙트럴만 다시 계산
K = 8
SCALES = (True, False)      # E_t를 sqrt(n_t)로 정규화할지
FIX_SIGNS = (True, False)   # 고유벡터 부호를 고정할지


def tag_of(k: int, scale: bool, fix_sign: bool):
    return f"k{k}{'_scaled' if scale else ''}{'_signfix' if fix_sign else ''}"

if __name__ == "__main__":
    episodes = load_episodes(DATA_DIR, level=LEVEL, case=CASE)
    if EXTRACT:
        extract_run(episodes, DATA_DIR, HIDDEN_DIR)

    # 원본 jsonl 파일 하나당 hidden state 서브트리 하나 (예: gotoseq_step10/cases/c3)
    rels = sorted({path.relative_to(DATA_DIR).with_suffix("") for path, _ in episodes})
    for rel in rels:
        for method in METHODS:
            for scale in SCALES:
                for fix_sign in FIX_SIGNS:
                    tag = tag_of(K, scale, fix_sign)
                    spectral_dir_run(
                        src_dir=HIDDEN_DIR / rel / method,
                        dst=SPECTRAL_DIR / rel / f"{method}_{tag}.pt",
                        method=method,
                        k=K, scale=scale, fix_sign=fix_sign,
                    )