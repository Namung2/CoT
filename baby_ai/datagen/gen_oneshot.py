"""원샷 계획으로 LLM CoT 를 생성한다.  --obs partial -> c1,  full -> c2

    python datagen/gen_oneshot.py --obs partial --n 20
    python datagen/gen_oneshot.py --obs full    --n 20
    python datagen/gen_oneshot.py --obs partial --model meta-llama/Llama-3.2-3B-Instruct
    python datagen/gen_oneshot.py --obs partial --backend openai --model gpt-4o

References
    CTRLS               arXiv 2507.08182   Assumption 5.2 (자기회귀 인수분해)
    LLM-BabyBench       arXiv 2505.12135   원샷 태스크, 시스템 프롬프트, StructuredFormatter
                        github.com/choukrani/llm-babybench
    GLAM                arXiv 2302.02662   BabyAI-Text describer (c1 관측)
    AgentGym-RL         arXiv 2509.08755   <think> / <action> 태그 분리
    PriorZero           arXiv 2605.12289   액션 누출 차단 지시
    ECoT                arXiv 2407.08693   전문가 사후 어노테이션으로 라벨 생성
    BALROG              arXiv 2411.13543   성능 대역 참고용. 멀티턴이라 프롬프트 출처 아님
    BabyAI              arXiv 1810.08272   레벨 및 bot

"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen import (  # noqa: E402   gen.py 재사용
    ACT_ID,
    DIR_NAME,
    describe_view,
    front_cell,
    instr_done,
    instr_leaves,
)

LEVEL = "BabyAI-PutNextLocal-v0"
CASE_OF = {"partial": "c1", "full": "c2"}

def render_partial(env) -> str:
    """c1. gen.py 의 describe_view. GLAM 의 BabyAI-Text describer 그대로."""
    u = env.unwrapped
    return describe_view(u.gen_obs()["image"], u.carrying)

def render_full(env) -> str:
    """c2. LLM-BabyBench (arXiv 2505.12135) 의 StructuredFormatter 를 이식."""
    from minigrid.core.world_object import Ball, Box, Door, Key

    u = env.unwrapped
    g = u.grid
    lines = []
    n_r, n_c = int(u.num_rows), int(u.num_cols)
    rs = int(u.room_size)
    lines.append(f"- Number of rooms: {n_r}x{n_c}" if (n_r > 1 or n_c > 1)
                 else "- Number of rooms: 1")
    lines.append(f"- Size of each room (including walls): {rs}x{rs}")
    lines.append(f"- Effective room size (excluding walls): {rs - 2}x{rs - 2}")
    lines.append(f"- Total grid size: {g.height}x{g.width}")
    lines.append(f"- Agent initial position: "
                 f"{tuple(int(x) for x in u.agent_pos)}")
    lines.append(f"- Agent facing direction: {DIR_NAME[int(u.agent_dir)]} "
                 f"(toward {tuple(int(x) for x in u.front_pos)})")

    objs = []
    for idx, obj in enumerate(g.grid):
        if isinstance(obj, (Door, Key, Ball, Box)):
            j, i = idx // g.width, idx % g.width
            lock = f", locked={obj.is_locked}" if isinstance(obj, Door) else ""
            objs.append(f"  - {obj.type}, color={obj.color}, "
                        f"position={(i, j)}{lock}")
    if objs:
        lines.append("- Objects in environment:")
        lines.extend(objs)
    else:
        lines.append("- Objects in environment: none")

    carry = (f"{u.carrying.color} {u.carrying.type}"
             if u.carrying is not None else "nothing")
    lines.append(f"- Agent is carrying: {carry}")

    header = ("These are the specifics regarding this environment: \n\n")
    return header + "\n".join(lines)

RENDER = {"partial": render_partial, "full": render_full}

# 프롬프트 출처
#   시스템 문장 / 액션 설명 / 관측 포맷   LLM-BabyBench arXiv 2505.12135
#                                        github.com/choukrani/llm-babybench
#   <think> / <action> 태그 분리          AgentGym-RL arXiv 2509.08755
#   "Do not reveal the action"            PriorZero arXiv 2605.12289 Appendix D
#   c1 관측 본문                          GLAM arXiv 2302.02662 (gen.py::describe_view)
# BALROG (arXiv 2411.13543) 는 멀티턴이고 액션 설명에 오류가 있어 제외.

ACTION_SENTENCE = (
    "The agent can perform 6 actions: left (turn left), right (turn right), "
    "forward (move forward), pickup (pickup an object), drop (drop an object), "
    "and toggle (open/close a door or a box)."
)
ACTIONS = ("left", "right", "forward", "pickup", "drop", "toggle")

RULES = (
    "The agent cannot move into a cell that is already occupied by an object, "
    "even if the object is one it is trying to interact with. "
    "Only the forward action changes the agent's position in the grid world. "
    "Turning left or right changes the agent's orientation only but not the "
    "position."
)

VIEW_NOTE = {
    "partial": "You are shown only what the agent can currently see. "
               "Parts of the room may be out of view.",
    "full": "You are shown the complete grid.",
}

SYSTEM = """
You are an intelligent agent in a grid world. Your mission is to {mission}.

{action_sentence}
{rules}

{view_note} Determine the full sequence of actions that completes the mission, reasoning one step at a time.

Output will be parsed by a strict program. For every step, output exactly two tags on one line:
<think>what holds at this point and what still needs doing</think><action>one action</action>

Do not reveal the action inside <think>. Do not write anything else.
""".strip()

def build_system(mission: str, obs_mode: str) -> str:
    return SYSTEM.format(mission=mission, action_sentence=ACTION_SENTENCE,
                         rules=RULES, view_note=VIEW_NOTE[obs_mode])

RE_PAIR = re.compile(r"<think>(.*?)</think>\s*<action>(.*?)</action>", re.S)

def parse_plan(out: str) -> tuple[list[str], list[str], int]:
    """(cots, actions, n_bad). 쌍이 안 맞는 항목은 버리고 개수만 센다."""
    cots: list[str] = []
    acts: list[str] = []
    bad = 0
    for m in RE_PAIR.finditer(out):
        think = m.group(1).strip()
        act = m.group(2).strip().lower().replace(" ", "_")
        if act == "pick_up":
            act = "pickup"
        if not think or act not in ACTIONS:
            bad += 1
            continue
        cots.append(think)
        acts.append(act)
    return cots, acts, bad

class Chat:
    """LLM 백엔드. 두 가지를 지원한다."""

    def __init__(self, backend: str, model: str, max_tokens: int,
                 retries: int = 5, timeout: int = 60, device: str = "auto"):
        self.backend, self.model = backend, model
        self.max_tokens, self.retries = max_tokens, retries
        if backend == "openai":
            from openai import OpenAI
            self.cli = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                              timeout=timeout)
        elif backend == "local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.tok = AutoTokenizer.from_pretrained(model)
            self.lm = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.bfloat16, device_map=device).eval()
            self.gpu_lock = threading.Lock()
        else:
            raise ValueError(f"unknown backend: {backend}")

    def __call__(self, system: str, user: str) -> tuple[str, str]:
        """(본문, finish_reason). finish_reason == 'length' 면 잘린 것."""
        if self.backend == "local":
            return self._local(system, user)
        return self._openai(system, user)

    def _openai(self, system: str, user: str) -> tuple[str, str]:
        last: Exception | None = None
        for i in range(self.retries):
            try:
                r = self.cli.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                c = r.choices[0]
                return (c.message.content or ""), (c.finish_reason or "")
            except Exception as e:
                last = e
                time.sleep(2 ** i)      # 지수 백오프. BALROG 의 delay=2 와 동일.
        raise last

    def _local(self, system: str, user: str) -> tuple[str, str]:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        with self.gpu_lock, self.torch.no_grad():
            enc = self.tok(text, return_tensors="pt").to(self.lm.device)
            out = self.lm.generate(
                **enc,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        new = out[0, enc["input_ids"].shape[1]:]
        finish = "length" if int(new.numel()) >= self.max_tokens else "stop"
        return self.tok.decode(new, skip_special_tokens=True), finish

LABEL_KEYS = ("pos", "dir", "carrying", "front_obj", "front_state", "instr_done")

def replay_and_label(seed: int, num_objs: int, acts: list[str]):
    """계획을 실행하며 스텝별 상태 라벨을 모은다.

    라벨은 환경에서 직접 읽는다. bot 을 쓰지 않는다.
    LLM-BabyBench (arXiv 2505.12135) 도 상태를 ((x, y), dir) 로 정의하고
    스텝별 subgoal 을 저장하지 않는다.
    """
    env = gym.make(LEVEL, num_objs=num_objs)
    try:
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        lab: dict[str, list] = {k: [] for k in LABEL_KEYS}
        outcome, reward, n_exec = "incomplete", 0.0, 0

        for act in acts[:u.max_steps]:
            fobj, _, fstate = front_cell(obs["image"])
            for k, v in (
                ("pos", [int(u.agent_pos[0]), int(u.agent_pos[1])]),
                ("dir", int(u.agent_dir)),
                ("carrying", f"{u.carrying.color} {u.carrying.type}"
                             if u.carrying is not None else None),
                ("front_obj", fobj), ("front_state", fstate),
                ("instr_done", instr_done(u.instrs)),
            ):
                lab[k].append(v)

            obs, r, term, trunc, _ = env.step(ACT_ID[act])
            n_exec += 1
            if term or trunc:
                outcome = "success" if r > 0 else "failed"
                reward = float(r)
                break

        tobj, _, tstate = front_cell(obs["image"])
        terminal = {
            "front_obj": tobj, "front_state": tstate,
            "carrying": (f"{u.carrying.color} {u.carrying.type}"
                         if u.carrying is not None else None),
            "instr_done": instr_done(u.instrs),
            "final_pos": [int(u.agent_pos[0]), int(u.agent_pos[1])],
            "final_dir": int(u.agent_dir),
        }
        return lab, outcome, reward, n_exec, terminal
    finally:
        env.close()


def rollout(chat: Chat, seed: int, num_objs: int, obs_mode: str, max_plan: int):
    """((steps_rec, labels_rec), outcome) 또는 (None, 사유)."""
    env = gym.make(LEVEL, num_objs=num_objs)
    try:
        env.reset(seed=seed)
        u = env.unwrapped
        mission = u.mission
        obs_txt = RENDER[obs_mode](env)
        instr_root = type(u.instrs).__name__
        instr_types = [type(x).__name__ for x in instr_leaves(u.instrs)]
    finally:
        env.close()

    try:
        raw, finish = chat(build_system(mission, obs_mode), obs_txt)
    except Exception as e:
        return None, f"error:{type(e).__name__}"

    cots, acts, n_bad = parse_plan(raw)
    if not cots:
        return None, "unparsed"
    cots, acts = cots[:max_plan], acts[:max_plan]

    lab, outcome, reward, n_exec, terminal = replay_and_label(seed, num_objs, acts)
    cots, acts = cots[:n_exec], acts[:n_exec]

    case = CASE_OF[obs_mode]
    ep_id = f"putnext_{seed}"
    steps_rec = {
        "id": ep_id, "level": LEVEL, "seed": seed, "mission": mission,
        "steps": cots,
        "terminal": f"The episode ended at ({terminal['final_pos'][0]}, "
                    f"{terminal['final_pos'][1]}) facing "
                    f"{DIR_NAME[terminal['final_dir']]}.",
        "answer": {
            "action_seq": acts,
            "final_pos": terminal["final_pos"], "final_dir": terminal["final_dir"],
            "success": outcome == "success", "reward": reward,
        },
        "n_steps": len(cots), "case": case, "outcome": outcome,
        "obs_mode": obs_mode,
        "instr_root": instr_root, "instr_types": instr_types,
        "n_bad_pairs": n_bad, "n_planned": len(acts),
        "truncated": finish == "length",
        "observation": obs_txt,
    }
    labels_rec = {
        "id": ep_id, "level": LEVEL, "seed": seed,
        "n_steps": len(cots), "outcome": outcome, "action": acts,
        **{k: v[:n_exec] for k, v in lab.items()},
        "terminal": terminal,
    }
    return (steps_rec, labels_rec), outcome

class Sink:
    """성공 / 실패를 각각 다른 디렉토리에 쓴다."""

    def __init__(self, root: Path, case: str):
        self.case = case
        self.ok_dir = root / "cases"
        self.ng_dir = root / "fail" / "cases"
        for d in (self.ok_dir, self.ng_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.f = {
            "ok_s": (self.ok_dir / f"{case}.jsonl").open("a"),
            "ok_l": (self.ok_dir / f"{case}.labels.jsonl").open("a"),
            "ng_s": (self.ng_dir / f"{case}.jsonl").open("a"),
            "ng_l": (self.ng_dir / f"{case}.labels.jsonl").open("a"),
        }

    def seen(self) -> set:
        out: set = set()
        for p in (self.ok_dir / f"{self.case}.jsonl",
                  self.ng_dir / f"{self.case}.jsonl"):
            if p.exists():
                with p.open() as f:
                    out |= {json.loads(l)["seed"] for l in f if l.strip()}
        return out

    def write(self, s_rec: dict, l_rec: dict, ok: bool) -> None:
        ks, kl = ("ok_s", "ok_l") if ok else ("ng_s", "ng_l")
        self.f[ks].write(json.dumps(s_rec, ensure_ascii=False) + "\n")
        self.f[kl].write(json.dumps(l_rec, ensure_ascii=False) + "\n")
        self.f[ks].flush()
        self.f[kl].flush()

    def close(self) -> None:
        for f in self.f.values():
            f.close()

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs", choices=("partial", "full"), default="partial",
                   help="partial -> c1, full -> c2")
    p.add_argument("--out", default="data/putnext")
    p.add_argument("--n", type=int, default=20, help="목표 성공 에피소드 수")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-limit", type=int, default=0,
                   help="시도할 시드 개수 상한. 0 이면 n*10")
    p.add_argument("--num-objs", type=int, default=8,
                   help="방 안 물체 총수. 2 개가 미션 대상이므로 실질 방해물은 "
                        "num_objs-2. minigrid PutNextLocal 기본값 8")
    p.add_argument("--max-plan", type=int, default=64,)
    p.add_argument("--backend", choices=("local", "openai"), default="local",
                   help="local 이면 CTRLS 백본과 일치. openai 는 상한 대조군")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="CTRLS 는 Qwen2.5-3B-Instruct / LLaMA-3.2-3B-Instruct 사용")
    p.add_argument("--device", default="auto", help="local 백엔드용")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    case = CASE_OF[a.obs]
    root = Path(a.out)
    sink = Sink(root, case)
    done = sink.seen()
    if done:
        print(f"[resume] {len(done)} episodes already saved")

    if a.backend == "local" and a.workers > 1:
        print(f"[info] local run")
    chat = Chat(a.backend, a.model, a.max_tokens, device=a.device)
    limit = a.seed_limit or a.n * 10
    seeds = [s for s in range(a.seed_start, a.seed_start + limit) if s not in done]

    lock = threading.Lock()
    stat: Counter = Counter()
    lens: dict[str, list] = defaultdict(list)
    n_ok = n_trunc = 0

    def work(seed: int) -> None:
        nonlocal n_ok, n_trunc
        with lock:
            if n_ok >= a.n:
                return
        res, why = rollout(chat, seed, a.num_objs, a.obs, a.max_plan)
        with lock:
            stat[why.split(":")[0]] += 1
            if res is None:
                return
            s_rec, l_rec = res
            ok = why == "success"
            sink.write(s_rec, l_rec, ok)
            lens[why].append(s_rec["n_steps"])
            n_trunc += bool(s_rec["truncated"])
            if ok:
                n_ok += 1
            print(f"  {'OK ' if ok else '-- '}[{n_ok}/{a.n}] seed={seed} "
                  f"T={s_rec['n_steps']} planned={s_rec['n_planned']} "
                  f"bad={s_rec['n_bad_pairs']} {why}")

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, seeds))
    finally:
        sink.close()

    tried = sum(stat.values())
    print(f"\n{'=' * 58}")
    print(f"case       {case}  ({a.obs} observation, one-shot)")
    print(f"model      {a.model}  [{a.backend}]")
    print(f"tried      {tried}")
    if tried:
        print(f"success    {stat['success']}  ({stat['success'] / tried:.1%})")
    for k in ("failed", "incomplete", "unparsed", "error"):
        if stat[k]:
            print(f"{k:10s} {stat[k]}")
    if n_trunc:
        print(f"truncated  {n_trunc}   <- max-tokens 를 올릴 것")
    if lens:
        print("\nT 분포 (성공/실패 e_t 비교 시 길이 통제 필요)")
        for k in ("success", "failed", "incomplete"):
            v = lens.get(k)
            if v:
                print(f"  {k:11s} n={len(v):3d}  mean {sum(v)/len(v):6.1f}  "
                      f"min {min(v):3d}  max {max(v):3d}")
    (root / f"{case}.manifest.json").write_text(json.dumps({
        "case": case, "level": LEVEL, "observability": a.obs,
        "model": a.model, "backend": a.backend,
        "num_objs": a.num_objs,
        "seed_range": [a.seed_start, a.seed_start + limit],
        "created": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2))
    print(f"\n-> {sink.ok_dir}/{case}.jsonl")
    print(f"-> {sink.ng_dir}/{case}.jsonl")

if __name__ == "__main__":
    main()