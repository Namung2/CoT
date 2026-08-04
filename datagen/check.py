"""데이터 검증 + 분포 진단.

    python datagen/check.py data/v1_hard
    python datagen/check.py data/v1_hard --replay 100

state 정의를 정하기 전에 반드시 볼 것.
subgoal 단독은 GoNextToSubgoal 로 90% 넘게 쏠린다.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent))
from gen import ACT_ID

# state 후보 조합. 확정이 아니라 탐색용.
CANDIDATES = {
    "subgoal": lambda r, i: r["subgoal"][i],
    "subgoal+carry": lambda r, i:
        f"{r['subgoal'][i]}|{'hold' if r['carrying'][i] else 'free'}",
    "subgoal+front": lambda r, i: f"{r['subgoal'][i]}|{r['front_obj'][i]}",
    "subgoal+carry+front": lambda r, i:
        f"{r['subgoal'][i]}|{'hold' if r['carrying'][i] else 'free'}|{r['front_obj'][i]}",
    # GoNextTo 91% 쏠림은 reason 으로 갈린다. 36% 까지 내려간다.
    "subgoal+reason": lambda r, i: f"{r['subgoal'][i]}|{r['subgoal_reason'][i]}",
    "subgoal+reason+target": lambda r, i:
        f"{r['subgoal'][i]}|{r['subgoal_reason'][i]}|{r['subgoal_target'][i]}",
    "action": lambda r, i: r["action"][i],
    # 가장 긴 시간축. 매우 sparse (전환 0.39회/ep, 61% 는 0회).
    # per-step state 라벨로는 부적합. 경계 확인용.
    "instr_done": lambda r, i: str(r["instr_done"][i]),
    "subgoal+instr": lambda r, i: f"{r['subgoal'][i]}|i{r['instr_done'][i]}",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def replay(level: str, seed: int, actions: list[str]) -> bool:
    """저장된 action_seq 를 환경에서 재실행.

    문자열 비교로 채점하면 안 된다. 같은 목표를 달성하는 유효 시퀀스가
    여럿 존재하므로 반드시 실행 검증해야 한다. (LLM 궤적 채점 시에도 동일)
    """
    env = gym.make(level)
    try:
        env.reset(seed=seed)
        r = 0.0
        for a in actions:
            _, r, term, trunc, _ = env.step(ACT_ID[a])
            if term or trunc:
                break
        return bool(r > 0)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dir")
    p.add_argument("--replay", type=int, default=0, help="N 개 재실행 검증")
    p.add_argument("--top", type=int, default=12)
    a = p.parse_args()
    d = Path(a.data_dir)

    steps = read_jsonl(d / "cases" / "c1.jsonl")
    labels = read_jsonl(d / "train.labels.jsonl")

    # 스키마
    assert len(steps) == len(labels), "steps / labels 길이 불일치"
    assert all(s["id"] == l["id"] for s, l in zip(steps, labels)), "id 정렬 불일치"
    ids = [s["id"] for s in steps]
    assert len(set(ids)) == len(ids), "중복 id"
    for r in steps:
        assert len(r["steps"]) == len(r["answer"]["action_seq"]) == r["n_steps"], \
            f"길이 불변식 위반: {r['id']}"
        assert r["terminal"], f"terminal 누락: {r['id']}"
        assert r["answer"]["success"], f"실패 에피소드 포함: {r['id']}"
    print(f"schema ok: {len(steps)} rows, ids unique, steps/labels aligned")
    print("  steps == action_seq == n_steps, terminal 존재, 전건 success")

    # 길이
    lens = [r["n_steps"] for r in labels]
    sw = st.mean(sum(1 for x, y in zip(r["subgoal"], r["subgoal"][1:]) if x != y)
                 for r in labels)
    print(f"steps: median {st.median(lens)} min {min(lens)} max {max(lens)} "
          f"total {sum(lens)}")
    lv = collections.Counter(r["level"] for r in steps)
    assert len(lv) == 1, f"레벨이 섞여 있다: {dict(lv)}"
    print(f"level: {next(iter(lv))}")
    isw = st.mean(sum(1 for x, y in zip(r["instr_done"], r["instr_done"][1:])
                      if x != y) for r in labels)
    flat = sum(1 for r in labels
               if len(set(r["instr_done"])) == 1) / len(labels)
    print(f"switches / episode: subgoal {sw:.1f}, instr {isw:.2f} "
          f"(instr 전환 0회인 ep {flat:.0%})")
    roots = collections.Counter(r["instr_root"] for r in labels)
    nleaf = collections.Counter(len(r["instr_types"]) for r in labels)
    print(f"instr root: {dict(roots.most_common(5))}")
    print(f"leaf count: {dict(sorted(nleaf.items()))}\n")

    # state 후보별 분포
    for name, fn in CANDIDATES.items():
        c = collections.Counter(fn(r, i) for r in labels for i in range(r["n_steps"]))
        tot = sum(c.values())
        top = c.most_common(1)[0]
        print(f"-- {name}: {len(c)} classes, top={top[1]/tot:.1%} ({top[0]}) --")
        for k, v in c.most_common(a.top):
            print(f"   {str(k):44} {v:7} {v/tot:6.1%}")
        if len(c) > a.top:
            print(f"   ... +{len(c)-a.top} more")
        print()

    if a.replay:
        n = min(a.replay, len(steps))
        bad = [r["id"] for r in steps[:n]
               if replay(r["level"], r["seed"], r["answer"]["action_seq"])
               != r["answer"]["success"]]
        print(f"replay {n}: {n - len(bad)} ok, {len(bad)} mismatch"
              + (f" {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    main()