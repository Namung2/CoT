"""
프롬프트. 공개 함수 3개.

    build_messages(mission, obs_text, obs_mode, prompt) -> list[dict]
    answer_spec(prompt)                                 -> dict
    prompt_fingerprint(level, obs_mode, prompt)         -> str

축 두 개가 직교한다.

    obs_mode  partial(GLAM describer) / full(LLM-BabyBench Structured)   -> C1
    prompt    bb / balrog                                               -> 프롬프트 대조

두 축 모두 seed 구간이 같으므로 `id = level|seed` 로 조인된다.
디렉토리는 data/{cell}-{generator}-{prompt} 로 갈린다.

--- prompt="bb" : LLM-BabyBench 2505.12135 --------------------------------------
저장소 choukrani/llm-babybench, prompters/utils.py TEMPLATES["cot"][Task.PLAN].
환경 서술 산문 + Step 1~5 스캐폴드 + 상세한 답 형식 지시.
{description} 은 formatters/structured.py 의 Structured 형식
(formatter 비교: Narrative 56.78 / JSON 59.28 / Structured 61.08).
답 추출은 llms/utils.py 의 lower().find() 이므로 **첫 번째** match.

--- prompt="balrog" : BALROG 2411.13543 -----------------------------------------
저장소 balrog-ai/BALROG.
  environments/babyai_text/__init__.py :: get_instruction_prompt
  agents/chain_of_thought.py           :: cot_instructions
  prompt_builder/history.py            :: "Current Observation:" 라벨, 전부 user role
BabyAI 를 zero-shot CoT 로 돌린 유일한 벤치마크이고 putnext 가 5개 태스크 중 하나다.
답 추출은 `split("ACTION:")[-1]` 이므로 **마지막** match. bb 와 동점 규칙이 반대다.

--- 두 변형에 공통으로 적용한 이탈 ----------------------------------------------
  1. toggle 제외, 5 actions.
     PutNextLocal 계열에 문이 없고 minigrid Box.toggle 이 미션 대상 박스를 소멸시킨다.
     balrog 의 Tips 첫 줄(toggle 사용 조언)도 함께 뺐다.
  2. 단일 액션 -> 액션 시퀀스.
     balrog 는 매 턴 액션 하나를 뽑는 멀티턴 루프다. one-shot 시퀀스로 옮기려면
     "a single output action" 을 시퀀스로 바꿔야 한다.
  3. obs_text 를 두 변형에서 동일하게 쓴다.
     BALROG 의 _form_prompt 는 GLAM describer 출력에서 "You see " 를 떼고 개행으로
     잇는다. 그걸 따르면 prompt 축이 관측까지 바꿔 obs_mode 축과 교란된다.
  4. balrog x full 에는 좌표계 설명이 필요하다.
     BALROG 는 부분관측 전용이라 좌표 설명이 없다. 절대좌표를 주면서 규약을 안
     알려주면 셀이 성립하지 않으므로 LLM-BabyBench 의 문장을 빌려 쓴다.
"""
from __future__ import annotations

import hashlib

from .babyai_env import ACTIONS, make_episode


# =============================================================== 공통 조각

# 좌표계 설명 (LLM-BabyBench Structured 산문 컨텍스트). full 관측에만 필요하다.
COORD_NOTE = (
    "Using a coordinate system where the (0, 0) position is the top-left corner "
    "of the grid world, necessarily corresponding to a wall, the coordinates "
    "follow the format (x, y), with x denoting the horizontal position in the "
    "grid and y denoting the vertical position in the grid."
)

# =============================================================== bb 변형

BB_WORLD = (
    "An agent is in a grid world consisting of one square room. The room is "
    "bounded by walls and might contain objects such as keys, balls, and boxes "
    "of different colors. The agent can perform 5 actions:\n"
    + "".join(f"- {a},\n" for a in ACTIONS[:-1]) + f"- {ACTIONS[-1]}.\n"
    + "Only the go forward action changes the agent's position in the grid world. "
    "Turning left or right changes the agent's orientation only but not the "
    "position. The agent cannot move into a cell that is already occupied by an "
    "object, even if the object is one it is trying to interact with."
)

BB_SCAFFOLD = (
    "Let's plan step-by-step to solve this MiniGrid environment task.\n\n"
    "Step 1. Identify the mission goal\n"
    "Step 2. Map out the environment layout\n"
    "Step 3. Break down the mission into actions\n"
    "Step 4. For each action:\n"
    "   - Determine optimal path to relevant places\n"
    "   - Plan the necessary rotations\n"
    "Step 5. Verify the plan actually works\n\n"
)

BB_LEAD = "The LLM's action sequence is: "

BB_FORMAT = (
    "At the end of your response, write the action sequence in the following way "
    "action1, action2, action3 "
    f"The names of possible actions are again: {', '.join(ACTIONS)}. "
    f"Begin this section with the phrase: '{BB_LEAD}'. "
    f"Example: {BB_LEAD}go forward, go forward, turn left "
    "Please do not write a full stop mark at the end!"
)


def _build_bb(mission: str, obs_text: str, obs_mode: str) -> str:
    world = BB_WORLD + (" " + COORD_NOTE if obs_mode == "full" else "")
    return (
        "You are an intelligent agent. You are given the following environment "
        f"description:\n{world}\n\n"
        f"{obs_text}\n\n"
        f"The agent's mission is '{mission}'. "
        "So, your goal is to accomplish this mission, and you need to determine "
        "the sequence of actions to get there. "
        + BB_SCAFFOLD + BB_FORMAT
    )


# =============================================================== balrog 변형

# BALROG environments/babyai_text/__init__.py 의 ACTIONS. toggle 만 제외했다.
BALROG_GLOSS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object below you",
    "drop": "drop the object that you are holding",
}

BALROG_LEAD = "ACTION: "


def _build_balrog(mission: str, obs_text: str, obs_mode: str) -> str:
    action_strings = ",\n".join(f"{a}: {BALROG_GLOSS[a]}" for a in ACTIONS)
    instruction = (
        "You are an agent playing a simple navigation game. Your goal is to "
        f"{mission}. The following are the possible actions you can take in the "
        "game, followed by a short description of each action:\n\n"
        f"{action_strings}.\n\n"
        "In a moment I will present you an observation.\n\n"
        "Tips:\n"
        "- It doesn't make sense to repeat the same action over and over if the "
        "observation doesn't change.\n\n"
        "PLAY!"
    )
    if obs_mode == "full":
        instruction += "\n\n" + COORD_NOTE          # 이탈 4
    # history.py 의 마지막 관측 메시지 라벨. 이력이 없으므로 Observation 0 뿐이다.
    cot = (
        "First think about what's the best course of action step by step.\n"
        "Finally, provide the full sequence of actions at the end of the message "
        f"in the form of: {BALROG_LEAD}action1, action2, action3"   # 이탈 2
    )
    return f"{instruction}\n\nCurrent Observation:\n{obs_text}\n\n{cot}"


# =============================================================== 레지스트리

PROMPTS = {
    # tie: 답 블록이 여러 번 나올 때 어느 것을 취하는가. 각 레퍼런스의 추출 코드를 따른다.
    #   bb     llms/utils.py            lower().find(...)        -> 첫 번째
    #   balrog agents/chain_of_thought  split("ACTION:")[-1]     -> 마지막
    "bb": {"build": _build_bb, "lead": BB_LEAD, "tie": "first"},
    "balrog": {"build": _build_balrog, "lead": BALROG_LEAD, "tie": "last"},
}

# 스키마/예시 플레이스홀더. 모델이 그대로 베꼈는지 score.py 가 감사한다.
# score.py 의 정규화가 숫자를 지우므로 ("action1" -> "action") 번호 붙은 형태는
# 넣지 않는다 — 넣어도 도달하지 않고, "action" 하나로 다 잡힌다.
SCHEMA_TOKENS = {"action", "actions"}

PROBE_SEED = 0


def build_messages(mission: str, obs_text: str, obs_mode: str,
                   prompt: str = "bb") -> list[dict]:
    """단일 user 메시지.

    레퍼런스 어느 쪽도 system/user 를 나누지 않는다. LLM-BabyBench 는 문자열 하나를
    반환하고, BALROG 는 prompt_builder/history.py 에서 instruction 도 관측도 전부
    role="user" 로 넣는다. assistant prefill 도 쓰지 않는다 (Claude 4.6+ 미지원).
    """
    content = PROMPTS[prompt]["build"](mission, obs_text, obs_mode)
    return [{"role": "user", "content": content}]


def answer_spec(prompt: str = "bb") -> dict:
    """답 블록 추출 규약. score.py 가 쓴다.

    regex 는 lead 이후 개행 전까지를 잡는다. 답은 한 줄이라는 것이 두 레퍼런스의
    공통 전제다.
    """
    lead = PROMPTS[prompt]["lead"]
    return {
        "lead": lead,
        "regex": lead.strip().replace(" ", r"\s+") + r"[ \t]*([^\n]*)",
        "tie": PROMPTS[prompt]["tie"],
    }


def prompt_fingerprint(level: str, obs_mode: str, prompt: str = "bb") -> str:
    """**실제로 렌더된 프롬프트**의 지문. 템플릿 문자열만이 아니다.

    고정 probe seed 하나를 렌더해 해시하면 템플릿·미션 배치·관측 렌더러가 한 번에
    덮인다. 템플릿만 해시하면 describe_view / render_full 을 고쳐도 지문이 안 바뀐다
    (GLAM gen_graph() 이식 때 실제로 그랬다).

    덮지 못하는 것: 생성/인코딩 모델, tokenizer, decoding, level 자체.
    이것들은 manifest 값 비교로 따로 본다.

    가드와 축은 다르다. 이 지문은 같은 디렉토리 안에서 프롬프트가 모르는 새 바뀌는
    사고를 막는 가드다. 일부러 바꿔 대조하는 것(obs_mode / prompt / generator 축)은
    디렉토리를 갈라서 한다.
    """
    ep = make_episode(level, PROBE_SEED, obs_mode)
    msgs = build_messages(ep["mission"], ep["obs_text"], obs_mode, prompt)
    blob = "\n".join([msgs[0]["content"], ",".join(ACTIONS)])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]