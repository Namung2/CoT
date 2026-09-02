from __future__ import annotations

import re

STEP_PAT = re.compile(r"(?mi)^(?:#+\s*|\*+\s*)?Step\s*(\d+)\s*[.:]")


def split_steps(text: str) -> list[str]:
    
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