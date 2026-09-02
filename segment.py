from __future__ import annotations

import re

STEP_PAT = re.compile(r"(?mi)^(?:#+\s*|\*+\s*)?Step\s*(\d+)\s*[.:]")


def split_steps(text: str) -> list[str]:
    """all_llm_output을 step 단위로 분할.

    계약: "".join(반환값) == text (strip 없음, 원문 보존).
    Step 1 헤더 앞에 서두가 있으면 그게 step0, 없으면 Step 1이 첫 원소.
    헤더가 하나도 없으면 통째로 원소 하나.
    """
    if not text.strip():
        return []

    starts = [m.start() for m in STEP_PAT.finditer(text)]
    if not starts:                      # 헤더 없음 → 통째로 하나
        return [text]
    if text[:starts[0]].strip():        # 서두 있음 → step0
        starts = [0] + starts
    else:                               # 헤더 앞이 공백뿐 → 첫 step에 흡수
        starts[0] = 0

    bounds = starts + [len(text)]
    return [text[s:e] for s, e in zip(bounds, bounds[1:])]