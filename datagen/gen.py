"""BabyAI -> CoT JSONL dateset 생성 

    python datagen/gen.py configs/bosslevel.yaml
    python datagen/gen.py configs/bosslevel.yaml --n 500 --seed 99 --out data/val

레벨을 섞지 않는다. 레벨마다 미션 구조가 달라 step 이 무조건적으로 다르므로,필요시 레벨별로 따로 뽑고, 분석 단계에서 합친다. 현재는 2가지 env(mission + competecies)
    1. GoToSeq : OPEN(문 열기), UNLOCK(잠금 해제), PICKUP(물건 줍기), PUT(물건 놓기), UNBLOCK(장애물 치우기) competencies들 모두 제외
       ROOM + DISTR-BOX +  GOTO + SEQ + MAZE => 이동만 수행(장애물,경유 포함)
    2. BossLevel : 모든 competencies 체크
모든 competencies 정리는 docs/competencies.md 에 있음(참고)

정답 궤적은 BabyAI expert bot 이 만듦.(신버전 bot minigrid import 포함)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import gymnasium as gym
import minigrid 
import numpy as np
import yaml
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
from minigrid.envs.babyai.core.verifier import ActionInstr
from minigrid.utils.baby_ai_bot import BabyAIBot

# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------

DIR_NAME = {0: "east", 1: "south", 2: "west", 3: "north"}
ACT_NAME = {0: "left", 1: "right", 2: "forward",
            3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
ACT_ID = {v: k for k, v in ACT_NAME.items()}
DOOR_STATE = {0: "open", 1: "closed", 2: "locked"}

AGENT_I, AGENT_J = 3, 6      # 7x7 egocentric view 에서 agent 위치
SEED_SPACE = 2**31 - 1

MIN_STEPS = 3 # 2 or 3 으로 논의 필요..?

# 사용 레벨은 configs/*.yaml 참고. 아래는 검토 후 제외한 것들.
#   GoToRedBallGrey / GoToLocal / PickupLoc : subgoal 전환 0~1 회.
#   UnblockPickup : instr_kinds=["action"] 이라 미션이 항상 단일 PickupInstr.
#                   instruction 전환 0.00 회. 긴 시간축 구조가 없다.
#   SynthSeq      : BossLevel 과 implicit_unlock 하나만 다르다.

EXCLUDED_LEVELS = [
    "BabyAI-GoToRedBallGrey-v0", "BabyAI-GoToLocal-v0", "BabyAI-PickupLoc-v0",
    "BabyAI-UnblockPickup-v0", "BabyAI-SynthSeq-v0",
] # 비교적 쉬운 MISSION들 -> 요구하는 competencies 가 몇개 없음 일단 사용 X -> 그만큼 같은 subgoal 만 수행

# 모든 레코드에서 동일하므로 데이터에 저장하지 않는다.
# 프롬프트 구성은 llm loader가 결정. 필요하면 import 해서 쓴다:
#     from datagen import TASK_PREFIX
#     prompt = TASK_PREFIX + rec["mission"]
TASK_PREFIX = (
    "You are an agent in a grid world. You can turn left, turn right, move "
    "forward, pick up an object, drop an object, or interact with what is in "
    "front of you. Your mission: "
) # 있는데 안씀 -> 철친씨 파트


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

class Cfg(dict):
    """cfg.min_steps 처럼 접근 가능한 dict."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def load_cfg(path: str | Path) -> Cfg:
    with Path(path).open() as f:
        return Cfg(yaml.safe_load(f))


# --------------------------------------------------------------------------
# 관측 -> 텍스트
#
# 좌표를 쓰지 않는다. 좌표를 넣으면 e_t 가 위치 숫자를 인코딩하는 쪽으로 흘러서
# "state 를 잡았다" 와 "숫자를 읽었다" 가 구분되지 않는다.
# --------------------------------------------------------------------------

def front_cell(img: np.ndarray) -> tuple[str, str | None, str | None]:
    """정면 한 칸의 (물체종류, 색, 문상태)."""
    c = img[AGENT_I, AGENT_J - 1]
    obj = IDX_TO_OBJECT[c[0]]
    color = IDX_TO_COLOR[c[1]] if obj not in ("empty", "unseen", "wall") else None
    state = DOOR_STATE[int(c[2])] if obj == "door" else None
    return obj, color, state


def describe_view(img: np.ndarray, carrying) -> str:
    obj, color, state = front_cell(img)
    if obj == "wall":
        ahead = "A wall is directly ahead."
    elif obj in ("empty", "unseen"):
        ahead = "The way ahead is clear."
    elif obj == "door":
        art = "An" if state[0] in "aeiou" else "A"
        ahead = f"{art} {state} {color} door is directly ahead."
    else:
        ahead = f"A {color} {obj} is directly ahead."

    seen = []
    for i in range(7):
        for j in range(7):
            if (i, j) == (AGENT_I, AGENT_J):
                continue
            o = IDX_TO_OBJECT[img[i, j, 0]]
            if o in ("empty", "unseen", "wall"):
                continue
            depth, side = AGENT_J - j, i - AGENT_I
            where = "ahead" if depth > 0 else "beside you"
            if side < 0:
                where += " on the left"
            elif side > 0:
                where += " on the right"
            seen.append((depth * depth + side * side,
                         f"a {IDX_TO_COLOR[img[i, j, 1]]} {o} {where}"))
    seen.sort()
    vis = ("You can see " + ", ".join(s for _, s in seen[:4]) + "."
           if seen else "Nothing of interest is in view.")

    hold = (f"You are carrying a {carrying.color} {carrying.type}."
            if carrying is not None else "Your hands are empty.")
    return f"{ahead} {vis} {hold}"


# --------------------------------------------------------------------------
# instruction 트리 (verifier)
#
# 세 층위 중 가장 긴 시간축. 전환이 에피소드당 평균 0.39 회로 매우 sparse 하고
# 61% 의 에피소드는 전환이 0 회다. per-step state 라벨로는 쓸 수 없고,
# 긴 시간축 경계 확인용 / 에피소드 층화용으로 기록한다.
#
# 주의: instrs.verify() 를 직접 호출하면 안 된다. 부작용이 있고
# (a_done/b_done/lastStepMatch 갱신) env.step() 이 이미 호출한다. 읽기만 할 것.
# --------------------------------------------------------------------------

def instr_leaves(i) -> list:
    return [i] if isinstance(i, ActionInstr) else \
        instr_leaves(i.instr_a) + instr_leaves(i.instr_b)


def instr_done(i) -> int:
    """완료된 leaf instruction 개수. 중첩 트리를 재귀적으로 센다.

    root 의 a_done/b_done 만 보면 leaf 4개짜리 BeforeInstr 가 최대 1 밖에
    안 나온다. 하위 SeqInstr 의 진행 상태까지 내려가야 한다.
    부작용 없이 속성만 읽는다.
    """
    if isinstance(i, ActionInstr):
        return 0
    n = 0
    for child, flag in ((i.instr_a, "a_done"), (i.instr_b, "b_done")):
        if getattr(i, flag, None) == "success":
            n += len(instr_leaves(child))     # 하위 전체 완료
        else:
            n += instr_done(child)            # 부분 진행
    return n


# --------------------------------------------------------------------------
# 자료구조
# --------------------------------------------------------------------------

def subgoal_info(top) -> tuple[str | None, str | None, str | None]:
    """bot stack top 에서 (종류, 대상, 이유) 를 읽는다. 부작용 없음.

    GoNextToSubgoal 은 전체의 91% 를 차지하지만 datum/reason 으로 갈린다:
      datum='blue door', reason='Open'    문을 열러 가는 중
      datum=(6,9),       reason='Explore' 탐색 중
      datum='blue key',  reason=None      목표물로 가는 중
    종류만 쓰면 이 구분이 사라진다.
    """
    if top is None:
        return None, None, None
    kind = type(top).__name__
    d = getattr(top, "datum", None)
    if d is None:
        target = None
    elif hasattr(d, "color") or hasattr(d, "type"):        # ObjDesc
        target = " ".join(str(x) for x in (getattr(d, "color", None),
                                           getattr(d, "type", None)) if x)
    else:                                                   # 좌표
        target = "position"
    return kind, target or None, getattr(top, "reason", None)


@dataclass
class Step:
    text: str                 # 관측 서술. e_t 를 뽑을 대상.
    action: str
    subgoal: str | None       # bot stack top 의 종류
    subgoal_target: str | None
    subgoal_reason: str | None
    stack_depth: int
    front_obj: str
    front_state: str | None
    instr_done: int           # 완료된 leaf instruction 수. sparse.
    pos: tuple[int, int]
    dir: int
    carrying: str | None


@dataclass
class Episode:
    level: str
    seed: int
    mission: str
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    reward: float = 0.0
    final_pos: tuple[int, int] = (0, 0)
    final_dir: int = 0
    # 마지막 행동 이후의 관측. transition 모델 학습에 필요한 흡수 상태.
    # 상태 T+1 개 / 행동 T 개가 되도록 한다.
    terminal_text: str = ""
    terminal_front_obj: str = ""
    terminal_front_state: str | None = None
    terminal_carrying: str | None = None
    terminal_instr_done: int = 0
    instr_root: str = ""            # AndInstr / BeforeInstr / PickupInstr ...
    instr_types: list[str] = field(default_factory=list)  # leaf 유형 순서대로

    @property
    def id(self) -> str:
        return hashlib.md5(f"{self.level}|{self.seed}".encode()).hexdigest()[:12]

    @property
    def n(self) -> int:
        return len(self.steps)


# --------------------------------------------------------------------------
# 롤아웃
# --------------------------------------------------------------------------

def rollout(level: str, seed: int) -> Episode | None:
    """bot 으로 1 에피소드. bot 실패 / max_steps 초과 / 미성공이면 None.
    여기서의 None 은 unsolvable 인스턴스 제거에 해당한다.

    max_steps 는 env 자체 예산(레벨별 576~1152, room_size^2 x rows x cols
    공식)을 그대로 쓴다. 원본 babyai/scripts/make_agent_demos.py 도 별도
    캡을 두지 않고 이 값에만 의존한다.

    (level, seed) 만으로 완전히 결정적이다. 난수를 쓰지 않는다."""
    env = gym.make(level)
    try:
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        ep = Episode(level=level, seed=seed, mission=obs["mission"])
        ep.instr_root = type(u.instrs).__name__
        ep.instr_types = [type(x).__name__ for x in instr_leaves(u.instrs)]
        bot = BabyAIBot(env)

        for _ in range(u.max_steps):
            try:
                a = int(bot.replan())
            except Exception:
                return None
            act = ACT_NAME[a]
            fobj, _, fstate = front_cell(obs["image"])
            kind, tgt, rsn = subgoal_info(bot.stack[-1] if bot.stack else None)
            ep.steps.append(Step(
                text=describe_view(obs["image"], u.carrying),
                action=act,
                subgoal=kind, subgoal_target=tgt, subgoal_reason=rsn,
                stack_depth=len(bot.stack),
                front_obj=fobj, front_state=fstate,
                instr_done=instr_done(u.instrs),
                pos=(int(u.agent_pos[0]), int(u.agent_pos[1])),
                dir=int(u.agent_dir),
                carrying=(f"{u.carrying.color} {u.carrying.type}"
                          if u.carrying is not None else None),
            ))
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                ep.success, ep.reward = bool(r > 0), float(r)
                tobj, _, tstate = front_cell(obs["image"])
                ep.terminal_text = describe_view(obs["image"], u.carrying)
                ep.terminal_front_obj = tobj
                ep.terminal_front_state = tstate
                ep.terminal_carrying = (f"{u.carrying.color} {u.carrying.type}"
                                        if u.carrying is not None else None)
                ep.terminal_instr_done = instr_done(u.instrs)
                break
        else:
            return None

        ep.final_pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        ep.final_dir = int(u.agent_dir)
        return ep if ep.success else None
    finally:
        env.close()


# --------------------------------------------------------------------------
# 직렬화
#
# steps 파일과 labels 파일을 분리한다. 발견 단계에서 라벨을 보지 않기 위한 것.
# 두 파일은 id 로 join 되고 행 순서가 일치한다.
# --------------------------------------------------------------------------

def steps_record(ep: Episode) -> dict:
    return {
        "id": ep.id, "level": ep.level, "seed": ep.seed,
        "mission": ep.mission,
        "steps": [s.text for s in ep.steps],
        # 마지막 행동 이후의 관측. len(steps) == len(action_seq) 이고
        # terminal 을 붙이면 상태 T+1 개가 된다. transition 학습 시 사용.
        "terminal": ep.terminal_text,
        "answer": {
            "action_seq": [s.action for s in ep.steps],
            "final_pos": list(ep.final_pos),
            "final_dir": DIR_NAME[ep.final_dir],
            "success": ep.success, "reward": ep.reward,
        },
        "n_steps": ep.n,
    }


def labels_record(ep: Episode) -> dict:
    """gt 원재료. state 라벨을 여기서 확정하지 않는다."""
    return {
        "id": ep.id, "level": ep.level, "seed": ep.seed, "n_steps": ep.n,
        "action": [s.action for s in ep.steps],
        "subgoal": [s.subgoal for s in ep.steps],
        "subgoal_target": [s.subgoal_target for s in ep.steps],
        "subgoal_reason": [s.subgoal_reason for s in ep.steps],
        "stack_depth": [s.stack_depth for s in ep.steps],
        "front_obj": [s.front_obj for s in ep.steps],
        "front_state": [s.front_state for s in ep.steps],
        "instr_done": [s.instr_done for s in ep.steps],
        "terminal": {
            "front_obj": ep.terminal_front_obj,
            "front_state": ep.terminal_front_state,
            "carrying": ep.terminal_carrying,
            "instr_done": ep.terminal_instr_done,
            "pos": list(ep.final_pos),
            "dir": ep.final_dir,
            "subgoal": None,      # bot 은 종료 후 replan 하지 않는다
        },
        "instr_root": ep.instr_root,
        "instr_types": ep.instr_types,
        "pos": [list(s.pos) for s in ep.steps],
        "dir": [s.dir for s in ep.steps],
        "carrying": [s.carrying for s in ep.steps],
    }


# --------------------------------------------------------------------------
# 생성
# --------------------------------------------------------------------------

def seed_stream(master: int) -> Iterator[int]:
    """master 로 초기화된 RNG 에서 비복원 추출.
    흩어지면서도 재현된다. 같은 master 면 같은 데이터가 나온다."""
    rng, seen = random.Random(master), set()
    while True:
        s = rng.randrange(SEED_SPACE)
        if s not in seen:
            seen.add(s)
            yield s


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def generate(cfg: Cfg, out_dir: Path | None = None, verbose: bool = True) -> dict:
    """steps / labels / manifest 3 파일을 쓴다. 레벨 하나당 한 디렉토리."""
    out = Path(out_dir or cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = tried = 0
    drop = {"bot_fail": 0, "too_short": 0}
    last_seed = 0

    fs = (out / "train.jsonl").open("w")
    fl = (out / "train.labels.jsonl").open("w")
    try:
        for seed in seed_stream(cfg.seed):
            if kept >= cfg.n:
                break
            last_seed = seed
            ep = rollout(cfg.level, seed)
            tried += 1
            if ep is None:
                drop["bot_fail"] += 1
            elif ep.n < MIN_STEPS:
                drop["too_short"] += 1
            else:
                fs.write(json.dumps(steps_record(ep), ensure_ascii=False) + "\n")
                fl.write(json.dumps(labels_record(ep), ensure_ascii=False) + "\n")
                kept += 1
                if verbose and kept % 200 == 0:
                    print(f"  {kept}/{cfg.n}  (tried {tried})")
    finally:
        fs.close()
        fl.close()

    stats = {"kept": kept, "tried": tried, "yield": round(kept / max(tried, 1), 4),
             "dropped": drop, "next_seed": last_seed + 1}
    with (out / "manifest.json").open("w") as f:
        json.dump({
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": _git_hash(),
            "python": sys.version.split()[0],
            "minigrid": minigrid.__version__,
            "config": dict(cfg),
            "stats": stats,
        }, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"{kept} eps -> {out}  yield={stats['yield']}  dropped={drop}")
    return stats


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--n", type=int, help="config 덮어쓰기")
    p.add_argument("--level", help="config 덮어쓰기")
    p.add_argument("--out", help="out_dir 덮어쓰기")
    p.add_argument("--seed", type=int, help="마스터 시드 덮어쓰기")
    a = p.parse_args()
    cfg = load_cfg(a.config)
    if a.n:
        cfg["n"] = a.n
    if a.level:
        cfg["level"] = a.level
        cfg["out_dir"] = f"data/{a.level}"
    if a.seed is not None:
        cfg["seed"] = a.seed
    generate(cfg, Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()