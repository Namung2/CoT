"""
BabyAI 환경 래퍼.

- partial 관측 서술 : GLAM / BabyAI-Text describer (arXiv 2302.02662)
- full 관측 서술    : LLM-BabyBench StructuredFormatter (arXiv 2505.12135)
- 라벨              : 환경에서 직접 읽음 (pos, dir, carrying, front_obj)
- 공변량            : n_targets_visible, all_objects_visible  (분석 전용, 프롬프트 미포함)
- bot               : 같은 seed 독립 롤아웃 1회 -> bot_reference_len 만

주의
  * toggle / done 은 액션 공간에서 제외 (setting.md M6):
    PutNextLocal 계열에 문이 없고, Box.toggle 이 대상 박스를 소멸시킨다.
  * instr_done 은 PutNextLocal 계열에서 항상 0 (setting.md M5) -> 라벨에서 제외.
  * 좌표는 partial 서술에 절대 넣지 않는다.
"""
from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass, field, asdict

import gymnasium as gym
import minigrid  # noqa: F401  (레벨 등록)
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
from minigrid.utils.baby_ai_bot import BabyAIBot

# ---------------------------------------------------------------- 액션 공간

ACTIONS = ["left", "right", "forward", "pickup", "drop"]
ACT_ID = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4}
ID_ACT = {v: k for k, v in ACT_ID.items()}

DIR_NAME = {0: "east", 1: "south", 2: "west", 3: "north"}

# 7x7 자기중심 뷰에서 에이전트 위치
AGENT_I, AGENT_J = 3, 6

SKIP = {"unseen", "empty", "wall", "agent"}


# ---------------------------------------------------------------- 관측 서술


def _cell(img, i, j):
    return IDX_TO_OBJECT[int(img[i, j, 0])]


def _color(img, i, j):
    return IDX_TO_COLOR[int(img[i, j, 1])]


def _plural(n):
    return "step" if n == 1 else "steps"


def describe_view(img, carrying) -> str:
    """GLAM / BabyAI-Text 자기중심 서술. 좌표 없음.

    img[i][j] : i = 측면축(0..6, 에이전트 3), j = 깊이축(0..6, 에이전트 6)
    dx = i - 3  (음수 왼쪽 / 양수 오른쪽),  dy = 6 - j  (전방 거리)
    """
    lines = []

    # --- 가장 가까운 벽 (전방 / 좌 / 우)
    for dy in range(1, AGENT_J + 1):
        if _cell(img, AGENT_I, AGENT_J - dy) == "wall":
            lines.append(f"You see a wall {dy} {_plural(dy)} forward")
            break
    for dx in range(1, AGENT_I + 1):
        if _cell(img, AGENT_I - dx, AGENT_J) == "wall":
            lines.append(f"You see a wall {dx} {_plural(dx)} left")
            break
    for dx in range(1, img.shape[0] - AGENT_I):
        if _cell(img, AGENT_I + dx, AGENT_J) == "wall":
            lines.append(f"You see a wall {dx} {_plural(dx)} right")
            break

    # --- 시야 안 모든 물체
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (i, j) == (AGENT_I, AGENT_J):
                continue
            o = _cell(img, i, j)
            if o in SKIP:
                continue
            dx, dy = i - AGENT_I, AGENT_J - j
            desc = f"a {_color(img, i, j)} {o}"
            parts = []
            if dx < 0:
                parts.append(f"{-dx} {_plural(-dx)} left")
            elif dx > 0:
                parts.append(f"{dx} {_plural(dx)} right")
            if dy > 0:
                parts.append(f"{dy} {_plural(dy)} forward")
            loc = " and ".join(parts) if parts else "right here"
            lines.append(f"You see {desc} {loc}")

    if carrying is not None:
        lines.append(f"You carry a {carrying.color} {carrying.type}")

    return "\n".join(lines)


def render_full(env) -> str:
    """LLM-BabyBench StructuredFormatter. 절대 좌표 포함. (P2 전용)"""
    u = env.unwrapped
    g = u.grid
    ax, ay = int(u.agent_pos[0]), int(u.agent_pos[1])

    lines = [
        f"Grid size: {g.width}x{g.height}",
        f"Agent position: ({ax}, {ay})",
        f"Agent direction: {DIR_NAME[int(u.agent_dir)]}",
        "Objects:",
    ]
    for idx, obj in enumerate(g.grid):
        if obj is None or obj.type in ("wall", "empty"):
            continue
        x, y = idx % g.width, idx // g.width
        lines.append(f"  - {obj.color} {obj.type} at ({x}, {y})")
    lines.append(
        f"Carrying: {u.carrying.color} {u.carrying.type}" if u.carrying else "Carrying: nothing"
    )
    return "\n".join(lines)


def front_obj(img) -> str | None:
    """에이전트 정면 칸의 물체 타입 (라벨용)."""
    o = _cell(img, AGENT_I, AGENT_J - 1)
    return None if o in ("unseen", "empty") else o


# ---------------------------------------------------------------

_MISSION_RE = re.compile(r"put the (\w+) (\w+) next to the (\w+) (\w+)")


def _visible_objs(img) -> set[tuple[str, str]]:
    out = set()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            o = _cell(img, i, j)
            if o in SKIP:
                continue
            out.add((o, _color(img, i, j)))
    return out


def visibility_covariates(env, obs) -> dict:
    """분석 전용 공변량. 프롬프트에 절대 포함하지 않는다.

    n_targets_visible   : 프롬프트에서 복원 가능한 파생 통계 (0/1/2)
    all_objects_visible : 방 안 총 물체 수를 알아야 하므로 환경 특권 정보
    """
    u = env.unwrapped
    vis = _visible_objs(obs["image"])

    m = _MISSION_RE.match(obs["mission"])
    if m:
        tgt = {(m.group(2), m.group(1)), (m.group(4), m.group(3))}
        n_tgt = len(tgt & vis)
    else:
        n_tgt = None

    n_total = sum(
        1 for o in u.grid.grid if o is not None and o.type in ("ball", "box", "key")
    )
    return {
        "n_targets_visible": n_tgt,
        "all_objects_visible": len(vis) == n_total,
        "n_objects_total": n_total,
        "n_objects_visible": len(vis),
    }


# ---------------------------------------------------------------- bot 레퍼런스


def bot_reference_len(level: str, seed: int) -> int | None:
    """같은 seed 에 대한 규칙기반 bot 롤아웃 길이.

    용도는 Efficiency Ratio (LLM-BabyBench) 와 해당 seed 의 해결 가능성 확인뿐.
    bot 은 최적이 아니라 휴리스틱이므로 'optimal' 이 아니라 'reference' 로 부른다.
    """
    env = gym.make(level)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            env.reset(seed=seed)
        u = env.unwrapped
        bot = BabyAIBot(env)
        for t in range(u.max_steps):
            try:
                a = int(bot.replan())
            except Exception:
                return None
            _, r, term, trunc, _ = env.step(a)
            if term or trunc:
                return t + 1 if r > 0 else None
        return None
    finally:
        env.close()


# ---------------------------------------------------------------- 에피소드


@dataclass
class Episode:
    level: str
    seed: int
    mission: str
    obs_text: str
    covariates: dict
    max_steps: int


def make_episode(level: str, seed: int, obs_mode: str) -> Episode:
    """환경을 리셋하고 초기 관측 텍스트 + 공변량을 만든다."""
    env = gym.make(level)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        cov = visibility_covariates(env, obs)
        if obs_mode == "partial":
            text = describe_view(obs["image"], u.carrying)
        elif obs_mode == "full":
            text = render_full(env)
        else:
            raise ValueError(obs_mode)
        return Episode(
            level=level,
            seed=seed,
            mission=obs["mission"],
            obs_text=text,
            covariates=cov,
            max_steps=int(u.max_steps),
        )
    finally:
        env.close()


def replay_and_label(level: str, seed: int, actions: list[str]) -> dict:
    """LLM 이 낸 액션 시퀀스를 환경에서 실행하고 매 스텝 라벨을 읽는다.

    라벨은 **행동 직전** 상태를 기록한다:  label[t] = s_t,  action[t] = a_t.
    reasoning step 과의 정렬은 요구하지 않는다 (setting.md 4).
    """
    env = gym.make(level)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            obs, _ = env.reset(seed=seed)
        u = env.unwrapped

        lab = {k: [] for k in ("action", "pos", "dir", "carrying", "front_obj")}
        n_exec, reward, term, trunc = 0, 0.0, False, False

        for a in actions[: u.max_steps]:
            lab["action"].append(a)
            lab["pos"].append([int(u.agent_pos[0]), int(u.agent_pos[1])])
            lab["dir"].append(int(u.agent_dir))
            lab["carrying"].append(
                None if u.carrying is None else f"{u.carrying.color} {u.carrying.type}"
            )
            lab["front_obj"].append(front_obj(obs["image"]))

            obs, r, term, trunc, _ = env.step(ACT_ID[a])
            n_exec += 1
            reward = float(r)
            if term or trunc:
                break

        success = reward > 0
        if not term and not trunc:
            outcome = "incomplete"
        else:
            outcome = "success" if success else "failed"

        return {
            "labels": lab,
            "n_exec": n_exec,
            "reward": reward,
            "success": success,
            "outcome": outcome,
            "final_pos": [int(u.agent_pos[0]), int(u.agent_pos[1])],
            "final_dir": int(u.agent_dir),
        }
    finally:
        env.close()