"""
BabyAI 환경 — P1(partial) / P2(full) 공용. 최소 구성.

공개 함수 4개
    describe_view(img, ax, ay, carrying) -> list[str]   P1. GLAM gen_graph() 축자 이식
    render_full(env)                     -> str         P2. LLM-BabyBench Structured
    make_episode(level, seed, obs_mode)  -> dict        mission + 관측 텍스트  [생성]
    replay(level, seed, actions)         -> dict        이진 성공 + 가시성     [채점]

describer 출처
    flowersteam/Grounding_LLMs_with_online_RL
    babyai-text/gym-minigrid/gym_minigrid/minigrid.py :: MiniGridEnv.gen_graph()
    본문을 축자 이식했다. 그들의 포크는 구버전 gym-minigrid(gym 기반)라 import 는
    불가하지만, gen_graph 가 쓰는 것은 image / agent view coords / carrying /
    IDX_TO_* 뿐이고 상수 테이블이 최신 minigrid 와 완전히 동일하다.
    -> 재구현이 아니라 이식이므로 "GLAM describer 이식"으로 보고할 수 있다.

    이식하며 바꾼 것은 배관뿐:
      * self.gen_obs_grid()/encode() 대신 이미 계산된 img 를 인자로 받는다
      * self.get_view_coords() 결과를 인자로 받는다 (하드코딩 (3,6) 아님)
      * french 분기 제거
      * 반환은 원본과 같이 **리스트**. 문자열 결합은 make_episode 의 책임
    본문 로직은 손대지 않았다. 원본의 버그도 그대로 둔다 (아래 주석 참조).

주의
  * toggle / done 은 액션 공간에서 제외 (M6): PutNextLocal 계열에 문이 없고
    Box.toggle 이 미션 대상 박스를 소멸시킨다.
  * 좌표는 partial 서술에 절대 넣지 않는다.
  * n_targets_visible 은 분석 전용 공변량이며 프롬프트에 들어가지 않는다.
"""
from __future__ import annotations

import contextlib
import io
import re

import gymnasium as gym
import minigrid  # noqa: F401  (레벨 등록)
from minigrid.core.actions import Actions


IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
IDX_TO_STATE = {0: "open", 1: "closed", 2: "locked"}
# minigrid agent_dir: 0=east 1=south 2=west 3=north (y 는 아래로 증가)
DIR_NAME = {0: "east", 1: "south", 2: "west", 3: "north"}

# GLAM / BabyAI-Text (2302.02662) 의 텍스트 명령. 순서는 minigrid Actions 0..4 와 일치.
ACTIONS = ["turn left", "turn right", "go forward", "pick up", "drop"]
ACT_ID = {a: i for i, a in enumerate(ACTIONS)}

# 순서가 계약이다. ACT_ID 의 값이 그대로 env.step() 의 정수 액션이 된다. 어긋나면
# replay 가 조용히 다른 액션을 실행하고 성공률만 떨어진다 — 예외도 경고도 없어서
# "모델이 못 푼다" 로 오진하기 쉽다. import 시점에 못 박는다.
#
# **이름 -> enum** 을 비교해야 한다. ACT_ID 의 값만 보면 ([ACT_ID[a] for a in ACTIONS])
# ACTIONS 를 어떻게 뒤섞어도 항상 [0..4] 라 동어반복이 된다.
assert ACT_ID == {
    "turn left": Actions.left,
    "turn right": Actions.right,
    "go forward": Actions.forward,
    "pick up": Actions.pickup,
    "drop": Actions.drop,
}, f"ACTIONS 가 minigrid Actions 0..4 와 어긋났다: {ACT_ID}"


# 관측 항목 구분자. GLAM base_agent.generate_prompt 은 ", " 로 한 줄에 잇는다.
# 우리는 chat message 안에 넣으므로 개행이 더 읽기 쉽지만, 레퍼런스를 따른다.
OBS_JOIN = ", "

_MISSION_RE = re.compile(r"put the (\w+) (\w+) next to the (\w+) (\w+)")


# ---------------------------------------------------------------- 내부


@contextlib.contextmanager
def _env(level: str, seed: int):
    """reset 까지 마친 env 와 초기 obs. minigrid 의 stdout 출력은 삼킨다."""
    env = gym.make(level)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed)
        yield env, obs
    finally:
        env.close()


def _n_targets_visible(obs) -> int | None:
    """미션 대상 2개 중 초기 시야에 몇 개가 있는가 (0/1/2).

    프롬프트에서 복원 가능한 파생 통계이므로 특권 정보가 아니다.
    가시 판정 필터는 gen_graph 와 동일하게 idx not in (0,1,2) 를 쓴다.
    (type, color) 쌍으로 세므로 같은 색·타입 distractor 가 있으면 과대계수된다.
    """
    m = _MISSION_RE.match(obs["mission"])
    if not m:
        return None
    img = obs["image"]
    vis = {
        (IDX_TO_OBJECT[int(img[i, j, 0])], IDX_TO_COLOR[int(img[i, j, 1])])
        for i in range(img.shape[0])
        for j in range(img.shape[1])
        if int(img[i, j, 0]) not in (0, 1, 2)
    }
    tgt = {(m.group(2), m.group(1)), (m.group(4), m.group(3))}
    return len(tgt & vis)


# ---------------------------------------------------------------- 공개


def describe_view(img, agent_pos_vx: int, agent_pos_vy: int, carrying) -> list[str]:
    """GLAM `MiniGridEnv.gen_graph()` 축자 이식. 좌표 없음.

    원본: babyai-text/gym-minigrid/gym_minigrid/minigrid.py (english 분기)
    변수명과 제어 흐름을 원본 그대로 두어 upstream 대조가 가능하게 했다.

    항목 순서도 원본을 따른다:
        carrying -> 벽(전방/좌/우) -> 시야 안 물체(i 오름차순, j 오름차순)

    벽 서술 규칙 (원본 주석): "We describe a wall only if there is no objects
    between the agent and the wall in straight line." 시선 상 첫 비-빈칸이
    물체이면 그 축의 벽은 서술하지 않는다. S6N4 실측으로 벽 서술의 15.8% 가
    이 규칙에서 사라진다 — 재구현본과 원본의 실질적 차이가 여기였다.
    """
    list_textual_descriptions = []

    if carrying is not None:
        list_textual_descriptions.append(
            "You carry a {} {}".format(carrying.color, carrying.type))

    view_field_dictionary = dict()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] != 0 and img[i][j][0] != 1 and img[i][j][0] != 2:
                if i not in view_field_dictionary.keys():
                    view_field_dictionary[i] = dict()
                view_field_dictionary[i][j] = img[i][j]

    # --- 벽. 시선 상 첫 비-빈칸이 물체이면 그 축은 서술하지 않는다.
    j = agent_pos_vy - 1
    object_seen = False
    while j >= 0 and not object_seen:
        if img[agent_pos_vx][j][0] != 0 and img[agent_pos_vx][j][0] != 1:
            if img[agent_pos_vx][j][0] == 2:
                n = agent_pos_vy - j
                list_textual_descriptions.append(
                    f"You see a wall {n} step{'s' if n > 1 else ''} forward")
            object_seen = True
        j -= 1

    i = agent_pos_vx - 1
    object_seen = False
    while i >= 0 and not object_seen:
        if img[i][agent_pos_vy][0] != 0 and img[i][agent_pos_vy][0] != 1:
            if img[i][agent_pos_vy][0] == 2:
                n = agent_pos_vx - i
                list_textual_descriptions.append(
                    f"You see a wall {n} step{'s' if n > 1 else ''} left")
            object_seen = True
        i -= 1

    i = agent_pos_vx + 1
    object_seen = False
    while i < img.shape[0] and not object_seen:
        if img[i][agent_pos_vy][0] != 0 and img[i][agent_pos_vy][0] != 1:
            if img[i][agent_pos_vy][0] == 2:
                n = i - agent_pos_vx
                list_textual_descriptions.append(
                    f"You see a wall {n} step{'s' if n > 1 else ''} right")
            object_seen = True
        i += 1

    # --- 시야 안 물체의 상대 위치
    for i in view_field_dictionary.keys():
        for j in view_field_dictionary[i].keys():
            if i != agent_pos_vx or j != agent_pos_vy:
                obj = view_field_dictionary[i][j]
                relative_position = dict()

                if i - agent_pos_vx > 0:
                    relative_position["x_axis"] = ("right", i - agent_pos_vx)
                elif i - agent_pos_vx == 0:
                    relative_position["x_axis"] = ("face", 0)
                else:
                    relative_position["x_axis"] = ("left", agent_pos_vx - i)
                if agent_pos_vy - j >= 0:
                    relative_position["y_axis"] = ("forward", agent_pos_vy - j)

                distances = []
                if relative_position["x_axis"][0] == "face":
                    distances.append((relative_position["y_axis"][1],
                                      relative_position["y_axis"][0]))
                elif relative_position["y_axis"][1] == 0:
                    distances.append((relative_position["x_axis"][1],
                                      relative_position["x_axis"][0]))
                else:
                    distances.append((relative_position["x_axis"][1],
                                      relative_position["x_axis"][0]))
                    distances.append((relative_position["y_axis"][1],
                                      relative_position["y_axis"][0]))

                if obj[0] != 4:  # 문이 아님
                    description = (f"You see a {IDX_TO_COLOR[int(obj[1])]} "
                                   f"{IDX_TO_OBJECT[int(obj[0])]} ")
                else:
                    # 원본 그대로. `IDX_TO_STATE[...] != 0` 은 str 과 int 비교라
                    # 항상 True 이므로 "an open ..." 분기는 원본에서 dead 다.
                    # PutNextLocal 에는 문이 없어 이 경로 자체가 실행되지 않는다.
                    description = (f"You see a {IDX_TO_STATE[int(obj[2])]} "
                                   f"{IDX_TO_COLOR[int(obj[1])]} "
                                   f"{IDX_TO_OBJECT[int(obj[0])]} ")

                for _i, _distance in enumerate(distances):
                    if _i > 0:
                        description += " and "
                    description += (f"{_distance[0]} "
                                    f"step{'s' if _distance[0] > 1 else ''} "
                                    f"{_distance[1]}")
                list_textual_descriptions.append(description)

    return list_textual_descriptions


def render_full(env) -> str:
    """P2. LLM-BabyBench(2505.12135) Structured formatter 재현. 절대좌표 포함.

    그들의 formatter ablation(Predict, prompter=ToT):
        Narrative 56.78 / JSON 59.28 / **Structured 61.08** -> Structured 채택.
    구조는 "산문 컨텍스트 + bullet {key:value}" 이고, 여기서는 bullet 부분만 만든다.
    산문 컨텍스트(행동 목록·Key rule·좌표계 설명)는 prompt.py 가 담당하며
    **P1 과 완전히 동일한 문자열**이어야 C1 이 관측 인코딩만 대조하게 된다.

    원문과 의도적으로 다른 점
      * `- Mission:` bullet 을 넣지 않는다. prompt.py 가 관측 뒤에
        "The agent's mission is '...'" 로 한 번만 넣는다 (P1/P2 공통).
        여기에 또 넣으면 P2 만 미션을 두 번 보게 되어 C1 이 오염된다.
      * locked 플래그는 door 에만 붙인다 (원문과 동일). PutNextLocal 에는 door 가 없다.
    """
    u = env.unwrapped
    g = u.grid
    ax, ay = (int(v) for v in u.agent_pos)
    fx, fy = (int(v) for v in u.front_pos)
    rows = getattr(u, "num_rows", 1)
    cols = getattr(u, "num_cols", 1)
    rs = getattr(u, "room_size", g.width)

    lines = [
        f"- Number of rooms: {cols}x{rows}",
        f"- Size of each room (including walls): {rs}x{rs}",
        f"- Effective room size (excluding walls): {rs - 2}x{rs - 2}",
        f"- Total grid size: {g.width}x{g.height}",
        f"- Agent initial position: ({ax}, {ay})",
        f"- Agent facing direction: {DIR_NAME[int(u.agent_dir)]} (toward ({fx}, {fy}))",
        "- Objects in environment:",
    ]
    for idx, o in enumerate(g.grid):
        if o is None or o.type in ("wall", "empty"):
            continue
        x, y = idx % g.width, idx // g.width
        lock = f", locked={o.is_locked}" if o.type == "door" else ""
        lines.append(f"    * {o.type}, color={o.color}, position=({x}, {y}){lock}")
    lines.append(
        f"- Agent is carrying: {u.carrying.color} {u.carrying.type}" if u.carrying
        else "- Agent is carrying: nothing")
    return "\n".join(lines)


def make_episode(level: str, seed: int, obs_mode: str = "partial") -> dict:
    """생성 단계가 필요로 하는 전부. 라벨도 공변량도 여기서 만들지 않는다.

    partial 결합자는 GLAM base_agent.generate_prompt 을 따라 ", " 다.
    (원본은 항목마다 말미에 ", " 를 붙여 마지막에도 남지만, 그 하나는 재현하지 않는다.)
    """
    with _env(level, seed) as (env, obs):
        u = env.unwrapped
        if obs_mode == "partial":
            ax, ay = (int(v) for v in u.get_view_coords(*u.agent_pos))
            desc = describe_view(obs["image"], ax, ay, u.carrying)
            text, n = OBS_JOIN.join(desc), len(desc)
        elif obs_mode == "full":
            text = render_full(env)
            n = text.count("\n") + 1
        else:
            raise ValueError(obs_mode)
        return {"mission": obs["mission"], "obs_text": text, "n_descriptions": n}


def replay(level: str, seed: int, actions: list[str]) -> dict:
    """액션 시퀀스를 실행하고 **이진 성공**만 읽는다.

    outcome
        success     env reward > 0
        failed      종료했는데 보상 없음
        incomplete  액션이 소진되었으나 환경이 종료하지 않음
    """
    with _env(level, seed) as (env, obs):
        u = env.unwrapped
        # partial view 기준 공변량. P2(full) 에서도 **같은 seed 의 partial 가시성**을
        # 그대로 기록한다 -> P2 성공률을 P1 의 가시성 군으로 층화할 수 있고,
        # "P1 의 실패 중 얼마가 지각 탓이고 얼마가 추론 탓인가" 가 바로 나온다.
        n_vis = _n_targets_visible(obs)

        reward, n_exec, term, trunc = 0.0, 0, False, False
        for a in actions[: u.max_steps]:
            _, r, term, trunc, _ = env.step(ACT_ID[a])
            n_exec += 1
            reward = float(r)
            if term or trunc:
                break

        success = reward > 0
        outcome = "success" if success else ("failed" if (term or trunc) else "incomplete")
        return {
            "success": success,
            "reward": reward,
            "n_exec": n_exec,
            "n_over_max_steps": max(len(actions) - u.max_steps, 0),
            "outcome": outcome,
            "n_targets_visible": n_vis,
        }