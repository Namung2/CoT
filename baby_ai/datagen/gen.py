"""BabyAI 관측/지시 유틸. 데이터 생성은 gen_oneshot.py.

References
    BabyAI    arXiv 1810.08272   관측 인코딩, Baby Language 문법
    GLAM      arXiv 2302.02662   describe_view 는 BabyAI-Text describer 이식

설계 근거와 미결 항목은 AGENT_CONTEXT.md 참고.
"""
from __future__ import annotations

import numpy as np
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
from minigrid.envs.babyai.core.verifier import ActionInstr

DIR_NAME = {0: "east", 1: "south", 2: "west", 3: "north"}
ACT_NAME = {0: "left", 1: "right", 2: "forward",
            3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
ACT_ID = {v: k for k, v in ACT_NAME.items()}
DOOR_STATE = {0: "open", 1: "closed", 2: "locked"}

AGENT_I, AGENT_J = 3, 6

def front_cell(img: np.ndarray) -> tuple[str, str | None, str | None]:
    """정면 한 칸의 (물체종류, 색, 문상태)."""
    c = img[AGENT_I, AGENT_J - 1]
    obj = IDX_TO_OBJECT[c[0]]
    color = IDX_TO_COLOR[c[1]] if obj not in ("empty", "unseen", "wall") else None
    state = DOOR_STATE[int(c[2])] if obj == "door" else None
    return obj, color, state

def describe_view(img: np.ndarray, carrying) -> str:
    """7x7 egocentric 관측 -> 텍스트.

    GLAM (arXiv 2302.02662) 의 BabyAI-Text describer 를 그대로 옮긴 것이다.
    BALROG (arXiv 2411.13543) 도 이 패키지를 그대로 설치해 쓴다.
    """
    lines: list[str] = []
    if carrying is not None:
        lines.append(f"You carry a {carrying.color} {carrying.type}")

    ai, aj = AGENT_I, AGENT_J

    def cell(i, j):
        return int(img[i, j, 0])

    j = aj - 1
    while j >= 0:
        o = cell(ai, j)
        if o not in (0, 1):
            if o == 2:
                n = aj - j
                lines.append(f"You see a wall {n} step{'s' if n > 1 else ''} forward")
            break
        j -= 1
    i = ai - 1
    while i >= 0:
        o = cell(i, aj)
        if o not in (0, 1):
            if o == 2:
                n = ai - i
                lines.append(f"You see a wall {n} step{'s' if n > 1 else ''} left")
            break
        i -= 1
    i = ai + 1
    while i < img.shape[0]:
        o = cell(i, aj)
        if o not in (0, 1):
            if o == 2:
                n = i - ai
                lines.append(f"You see a wall {n} step{'s' if n > 1 else ''} right")
            break
        i += 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if cell(i, j) in (0, 1, 2):
                continue
            if i == ai and j == aj:
                continue
            dx, dy = i - ai, aj - j
            dists = []
            if dx == 0:
                dists.append((dy, "forward"))
            elif dy == 0:
                dists.append((abs(dx), "right" if dx > 0 else "left"))
            else:
                dists.append((abs(dx), "right" if dx > 0 else "left"))
                dists.append((dy, "forward"))

            o, c, st = cell(i, j), IDX_TO_COLOR[int(img[i, j, 1])], int(img[i, j, 2])
            name = IDX_TO_OBJECT[o]
            if o != 4:
                desc = f"You see a {c} {name} "
            elif DOOR_STATE[st] != "open":
                desc = f"You see a {DOOR_STATE[st]} {c} {name} "
            else:
                desc = f"You see an {DOOR_STATE[st]} {c} {name} "
            desc += " and ".join(
                f"{n} step{'s' if n > 1 else ''} {d}" for n, d in dists)
            lines.append(desc)

    return "\n".join(lines)

def instr_leaves(i) -> list:
    return [i] if isinstance(i, ActionInstr) else \
        instr_leaves(i.instr_a) + instr_leaves(i.instr_b)

def instr_done(i) -> int:
    """완료된 leaf instruction 개수. 중첩 트리를 재귀적으로 센다."""
    if isinstance(i, ActionInstr):
        return 0
    n = 0
    for child, flag in ((i.instr_a, "a_done"), (i.instr_b, "b_done")):
        if getattr(i, flag, None) == "success":
            n += len(instr_leaves(child))
        else:
            n += instr_done(child)
    return n