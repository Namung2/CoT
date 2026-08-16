"""
프롬프트 구성.

근거 (setting.md 2 "프롬프트 근거") 
  * CTRLS Appendix F: "We follow the prompt design in (Wei et al., 2022) to guide
    step-wise CoT generation."  -> 프롬프트는 **문제 진술만**. 해법 구조도 중간 상태
    라벨도 넣지 않는다.
  * 마커 강제 없음 (`Step k:` 금지). CTRLS 에 없고, TRACES 2604.21057 이
    특수 토큰 생성 강제 방식의 낮은 신뢰성을 지적한다. 마커를 안 쓰면 "숫자 k가
    텍스트에 t를 새긴다"는 위치 누출 위험이 사라진다.
  * 액션 규칙 문구는 LLM-BabyBench 2505.12135 를 따른다.
  * 추론 텍스트 안에서 액션명 금지 -> e_t -> a_t probe 가 자명해지는 것을 막는다.

액션 블록은 **말미에 한 번**. 이유: 성공 판정이 환경 실행에 의존하므로 파싱은
반드시 성공해야 하지만, step_spans 와 action_spans 사이에 어떤 대응도 요구하지
않기 때문에 추론과 분리해 두는 편이 파싱이 확실하다.
"""
from __future__ import annotations

from .babyai_env import ACTIONS

ACTION_RULES = """You control an agent in a grid world. Available actions:
- left: turn 90 degrees to your left (does not move you)
- right: turn 90 degrees to your right (does not move you)
- forward: move one cell in the direction you are facing
- pickup: pick up the object in the cell directly in front of you
- drop: drop what you are carrying into the cell directly in front of you

Rules:
- You can only carry one object at a time.
- You cannot move into a cell occupied by a wall or an object.
- pickup and drop only affect the cell directly in front of you."""

VIEW_NOTE = {
    "partial": (
        "You see only the cells in front of you. Distances are given relative to "
        "your current position and facing direction."
    ),
    "full": (
        "You are given the full state of the grid in absolute coordinates, where "
        "(0, 0) is the top-left corner."
    ),
}

REASONING_INSTRUCTION = """Think through the problem step by step before answering.
Do not name any action inside your reasoning.
After your reasoning, output the complete action sequence on a single line inside
<actions></actions> tags, separated by commas."""

ASSISTANT_PREFIX = "Let's think step by step."

TERMINAL_TEXT = "The mission is now complete."


def build_messages(mission: str, obs_text: str, obs_mode: str,
                   exemplars: list[tuple[str, str]] | None = None) -> list[dict]:
    """chat messages 를 만든다.

    exemplars : (user, assistant) 쌍의 리스트. P4 (3-shot) 전용, 기본 None.
    """
    system = "\n\n".join([
        f"Your mission: {mission}",
        ACTION_RULES,
        VIEW_NOTE[obs_mode],
        REASONING_INSTRUCTION,
    ])

    msgs = [{"role": "system", "content": system}]
    for u, a in (exemplars or []):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": obs_text})
    return msgs


def valid_actions() -> set[str]:
    return set(ACTIONS)