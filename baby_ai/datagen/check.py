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


def _dup_rate(text_lists: list[list[str]]) -> tuple[int, int]:
    """(중복 step 수, 전체 step 수). 중복 = 같은 에피소드 안에서 2회 이상 나오는 텍스트."""
    dup = total = 0
    for texts in text_lists:
        counts = collections.Counter(texts)
        for t in texts:
            total += 1
            if counts[t] > 1:
                dup += 1
    return dup, total


def case_report(d: Path, top: int = 5):
    """조건(c1~c6)별 텍스트 중복/모호도 리포트. cases/ 없으면 조용히 건너뛴다.

    - 중복률: 에피소드 내 완전 중복 step 텍스트 비율
    - 고유 텍스트 수 / 전체 step 수: 조건 전체(pool)에서의 텍스트 다양성
    - txt->subgoal / txt->action 모호도: 같은 텍스트가 2개 이상의 subgoal/action
      에 대응하는 고유 텍스트 비율. 높으면 그 텍스트만으로는 라벨을 못 구분한다.
    - obs-dup: c1/c2/c6 은 관측 원문을 그대로 포함하므로, action/subgoal 서술이
      덧붙어 중복률이 낮아 보이는 걸 막기 위해 c1 원문 기준으로 따로 잰다.
    - 모호한 텍스트가 몰리는 subgoal/action: 모호한 텍스트에 해당하는 step
      인스턴스(전체 개수 기준)를 실제 subgoal/action 별로 집계한 top-N.
    """
    cases_dir = d / "cases"
    if not cases_dir.exists():
        return

    labels = {r["id"]: r for r in read_jsonl(d / "train.labels.jsonl")}
    c1_path = cases_dir / "c1.jsonl"
    obs_by_id = {r["id"]: r["steps"] for r in read_jsonl(c1_path)} \
        if c1_path.exists() else {}

    print("== case report (텍스트 중복/모호도) ==")
    header = f"{'cond':4} {'dup%':>7} {'unique/total':>16} " \
             f"{'sg-ambig%':>10} {'act-ambig%':>11} {'obs-dup%':>9}"
    print(header)
    confusion = {}   # c -> (sg_confuse Counter, act_confuse Counter)
    for c in ["c1", "c2", "c3", "c4", "c5", "c6"]:
        path = cases_dir / f"{c}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)

        text_lists = [r["steps"] for r in rows]
        dup, total = _dup_rate(text_lists)

        text_count: collections.Counter = collections.Counter()
        text_subgoals = collections.defaultdict(set)
        text_actions = collections.defaultdict(set)
        instances = []   # (text, subgoal, action) 모든 step 인스턴스
        for r in rows:
            lab = labels.get(r["id"])
            for i, t in enumerate(r["steps"]):
                sg = lab["subgoal"][i] if lab else None
                act = r["answer"]["action_seq"][i]
                text_count[t] += 1
                text_subgoals[t].add(sg)
                text_actions[t].add(act)
                instances.append((t, sg, act))

        n_unique = len(text_count)
        sg_ambig_texts = {t for t, s in text_subgoals.items() if len(s) > 1}
        act_ambig_texts = {t for t, s in text_actions.items() if len(s) > 1}
        sg_ambig = len(sg_ambig_texts) / n_unique
        act_ambig = len(act_ambig_texts) / n_unique

        # 모호한 텍스트에 해당하는 step 인스턴스가 실제로 어느 subgoal/action 인지
        sg_confuse = collections.Counter(sg for t, sg, act in instances
                                         if t in sg_ambig_texts)
        act_confuse = collections.Counter(act for t, sg, act in instances
                                          if t in act_ambig_texts)
        confusion[c] = (sg_confuse, act_confuse)

        obs_str = "   -"
        if c in ("c1", "c2", "c6") and obs_by_id:
            obs_lists = [obs_by_id[r["id"]] for r in rows if r["id"] in obs_by_id]
            odup, ototal = _dup_rate(obs_lists)
            obs_str = f"{odup/ototal:6.1%}" if ototal else "   -"

        print(f"{c:4} {dup/total:7.1%} {n_unique:>7}/{total:<8} "
              f"{sg_ambig:9.1%} {act_ambig:10.1%} {obs_str:>9}")
    print()

    print(f"== 모호한 텍스트가 몰리는 subgoal/action (top {top}) ==")
    for c, (sg_confuse, act_confuse) in confusion.items():
        if not sg_confuse and not act_confuse:
            continue
        print(f"-- {c} --")
        if sg_confuse:
            tot = sum(sg_confuse.values())
            top_sg = ", ".join(f"{k}={v/tot:.0%}" for k, v in sg_confuse.most_common(top))
            print(f"   subgoal ({tot} steps 영향): {top_sg}")
        if act_confuse:
            tot = sum(act_confuse.values())
            top_act = ", ".join(f"{k}={v/tot:.0%}" for k, v in act_confuse.most_common(top))
            print(f"   action  ({tot} steps 영향): {top_act}")
    print()


def mission_report(steps: list[dict]) -> None:
    """레벨 내 mission(자연어 지시어) 텍스트 중복 비율. 상황(seed)이 아니라 문자열 기준."""
    missions = [r["mission"] for r in steps]
    c = collections.Counter(missions)
    total = len(missions)
    uniq = len(c)
    dup = sum(v for v in c.values() if v > 1)
    print(f"mission: {total} eps, unique {uniq} ({uniq/total:.1%}), "
          f"dup {dup} ({dup/total:.1%})\n")


def step_text_dup_report(steps: list[dict]) -> None:
    """step(관측) 텍스트 중복: 어떤 텍스트가, 몇 %씩, 전체의 몇 %를 차지하는지.
    중복되는 텍스트 전부를 빈도순으로 출력한다 (터미널에 길게 나오면 파일로 리다이렉트)."""
    texts = [t for r in steps for t in r["steps"]]
    c = collections.Counter(texts)
    total = len(texts)
    dup_texts = {t: v for t, v in c.items() if v > 1}
    dup_total = sum(dup_texts.values())
    print(f"step text: {total} steps, {len(c)} unique, "
          f"dup {dup_total} ({dup_total/total:.1%} of all steps)")
    print(f"  duplicated texts ({len(dup_texts)}종류, 빈도순 전체):")
    for t, v in sorted(dup_texts.items(), key=lambda kv: -kv[1]):
        print(f"    {v:6} ({v/total:5.1%})  {t}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dir")
    p.add_argument("--replay", type=int, default=0, help="N 개 재실행 검증")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--cases", action="store_true",
                   help="조건(c1~c6)별 텍스트 중복/모호도 리포트")
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
    print("  steps == action_seq == n_steps, terminal 존재, 전건 success\n")

    mission_report(steps)
    step_text_dup_report(steps)

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

    if a.cases:
        case_report(d)


if __name__ == "__main__":
    main()