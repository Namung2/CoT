"""조건 c1~c7 후처리.  train.jsonl + labels -> cases/c1..c7.jsonl

    python datagen/case.py data/bosslevel
    python datagen/case.py data/bosslevel --cases c1 c2

    c7  행동만                              라벨 불필요. 하한(degenerate floor).
    c1  관측만                              라벨 불필요. 본 실험.
    c2  관측 + 행동                         라벨 불필요.
    c3  subgoal 종류 + 행동                 라벨 필요.
    c4  subgoal reason + 행동               라벨 필요.
    c5  subgoal 종류+reason+대상 + 행동      라벨 필요.
    c6  관측 + subgoal 전체(subgoal + datum + reason) + 행동           라벨 필요. 상한(positive control).

c3~c6 은 state 라벨이 텍스트에 직접 들어간다.
c7 은 action 종류 수(7가지)만큼만 고유 텍스트가 나오는 degenerate 조건이다.
c1 이 c7 대비 얼마나 나은지가 "관측이 기여하는 정보량"을 보여준다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = {
    "c1": "관측만",
    "c2": "관측 + 행동",
    "c3": "subgoal 종류 + 행동",
    "c4": "subgoal reason + 행동",
    "c5": "subgoal 종류+reason+대상 + 행동",
    "c6": "관측 + subgoal 전체 + 행동 (상한)",
    "c7": "행동만 (하한)",
}
NEEDS_LABELS = {"c3", "c4", "c5", "c6"}

# 행동은 사실 서술만 한다. "Nothing blocks me, so I advance" 같은 인과는
# bot 의 의도를 우리가 추측한 것이라 e_t 를 오염시킨다.
ACTION_PHRASE = {
    "left":    "I turn to my left.",
    "right":   "I turn to my right.",
    "forward": "I move ahead one square.",
    "pickup":  "I take what is in front of me.",
    "drop":    "I set down what I am holding.",
    "toggle":  "I operate what is in front of me.",
    "done":    "I stop here.",
}

KIND_PHRASE = {
    "GoNextToSubgoal": "I am moving toward something.",
    "OpenSubgoal": "I am about to open what is in front of me.",
    "PickupSubgoal": "I am about to take what is in front of me.",
    "DropSubgoal": "I am about to set down what I am holding.",
    "CloseSubgoal": "I am about to close what is in front of me.",
}
REASON_PHRASE = {
    "Explore": "I am exploring to find my way.",
    "Open": "I am on my way to open something.",
    "PutNext": "I am on my way to place something beside another.",
    "Pickup": "I am on my way to take something.",
}
DONE_PHRASE = "I have finished."


def full_subgoal_phrase(kind, target, reason) -> str:
    """종류 + 이유 + 대상. 가장 자세한 subgoal 서술."""
    if kind != "GoNextToSubgoal":
        return KIND_PHRASE.get(kind or "", DONE_PHRASE)
    if reason == "Explore":
        return REASON_PHRASE["Explore"]
    where = f"the {target}" if target and target != "position" else "my target"
    tail = {"Open": " so that I can open it.",
            "PutNext": " to place something beside it.",
            "Pickup": " to take it."}.get(reason or "", ".")
    return f"I am heading to {where}{tail}"

# json 파싱
def build_text(cond: str, obs: str, action: str | None,
               kind=None, target=None, reason=None) -> str:
    """조건에 맞는 step 텍스트. action 이 None 이면 terminal 이다."""
    act = ACTION_PHRASE[action] if action else ""
    if cond == "c1":
        return obs
    if cond == "c2":
        return f"{obs} {act}".strip()
    if cond == "c3":
        return f"{KIND_PHRASE.get(kind or '', DONE_PHRASE)} {act}".strip()
    if cond == "c4":
        return f"{REASON_PHRASE.get(reason or '', DONE_PHRASE) if kind else DONE_PHRASE} {act}".strip()
    if cond == "c5":
        return f"{full_subgoal_phrase(kind, target, reason)} {act}".strip()
    if cond == "c6":
        return f"{obs} {full_subgoal_phrase(kind, target, reason)} {act}".strip()
    if cond == "c7":
        return act
    raise ValueError(f"unknown condition: {cond}")


def read_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def split_by_step(rows: list[dict], out_dir: Path, verbose: bool = True) -> dict[int, int]:
    """rows 를 n_steps 값별로 <out_dir>/<n>step.jsonl 로 쪼갠다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[int, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r["n_steps"], []).append(r)

    counts = {}
    for n, group in sorted(buckets.items()):
        path = out_dir / f"{n}step.jsonl"
        with path.open("w") as f:
            for row in group:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[n] = len(group)
        if verbose:
            print(f"    {n}step: {len(group)} eps -> {path}")
    return counts


def build(data_dir: Path, cases: list[str], verbose: bool = True,
          split_steps: bool = False) -> dict:
    """cases/cN.jsonl 을 쓴다. 스키마는 train.jsonl 과 동일.
    split_steps 면 cases/by_step/cN/<n>step.jsonl 로도 추가로 쪼갠다."""
    steps = read_jsonl(data_dir / "train.jsonl")
    need = any(c in NEEDS_LABELS for c in cases)
    labels = {r["id"]: r for r in read_jsonl(data_dir / "train.labels.jsonl")} \
        if need else {}
    out_dir = data_dir / "cases"
    out_dir.mkdir(exist_ok=True)

    written = {}
    for c in cases:
        rows = []
        for r in steps:
            acts = r["answer"]["action_seq"]
            assert len(acts) == len(r["steps"]), f"정렬 불일치: {r['id']}"
            lab = labels.get(r["id"])
            new = []
            for t, obs in enumerate(r["steps"]):
                if lab:
                    new.append(build_text(c, obs, acts[t], lab["subgoal"][t],
                                          lab["subgoal_target"][t],
                                          lab["subgoal_reason"][t]))
                else:
                    new.append(build_text(c, obs, acts[t]))
            # terminal 은 행동 이후 상태다. action 도 subgoal 도 없다.
            term = build_text(c, r["terminal"], None) if c in ("c1", "c2", "c6") \
                else DONE_PHRASE
            rows.append({**r, "steps": new, "terminal": term, "case": c})
        path = out_dir / f"{c}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[c] = len(rows)
        if verbose:
            print(f"  {c}: {len(rows)} eps -> {path}  ({CONDITIONS[c]})")
        if split_steps:
            split_by_step(rows, out_dir / "by_step" / c, verbose)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir")
    p.add_argument("--cases", nargs="+", default=list(CONDITIONS),
                   choices=list(CONDITIONS))
    p.add_argument("--split-steps", action="store_true",
                   help="cases/cN.jsonl 을 n_steps 별로 cases/by_step/cN/ 밑에 추가로 쪼갠다")
    a = p.parse_args()
    build(Path(a.data_dir), a.cases, split_steps=a.split_steps)


if __name__ == "__main__":
    main()