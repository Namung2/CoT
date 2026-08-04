"""조건 c1~c6 후처리.  train.jsonl + labels -> cases/c1..c6.jsonl

    python datagen/case.py data/bosslevel
    python datagen/case.py data/bosslevel --cases c1 c2

정보량이 단조 증가하는 여섯 조건으로 step 텍스트를 다시 쓴다.
출력 스키마는 train.jsonl 과 동일하므로 소비자(latent/) 는 조건에 무관하게
같은 로더를 쓴다.

    c1  관측만                              라벨 불필요. 본 실험.
    c2  관측 + 행동                         라벨 불필요.
    c3  subgoal 종류 + 행동                 라벨 필요.
    c4  subgoal reason + 행동               라벨 필요.
    c5  subgoal 종류+reason+대상 + 행동      라벨 필요.
    c6  관측 + subgoal 전체(subgoal + datum + reason) + 행동           라벨 필요. 상한(positive control).

c3~c6 은 state 라벨이 텍스트에 직접 들어간다.
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
    raise ValueError(f"unknown condition: {cond}")


def read_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def build(data_dir: Path, cases: list[str], verbose: bool = True) -> dict:
    """cases/cN.jsonl 을 쓴다. 스키마는 train.jsonl 과 동일."""
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
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir")
    p.add_argument("--cases", nargs="+", default=list(CONDITIONS),
                   choices=list(CONDITIONS))
    a = p.parse_args()
    build(Path(a.data_dir), a.cases)


if __name__ == "__main__":
    main()