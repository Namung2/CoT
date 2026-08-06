"""JSONL 에피소드를 사람이 읽기 좋게 출력.

    python view_epi.py data/bosslevel/cases/c1.jsonl
    python view_epi.py data/bosslevel/cases/c1.jsonl --idx 3
    python view_epi.py data/bosslevel/cases/c1.jsonl --id 3419984b5597
    python view_epi.py data/bosslevel/cases/c1.jsonl --n 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def show(r: dict) -> None:
    print(f"id: {r['id']}  level: {r.get('level')}  seed: {r.get('seed')}")
    print(f"mission: {r['mission']}")
    for i, (t, act) in enumerate(zip(r["steps"], r["answer"]["action_seq"])):
        print(f"step {i+1}: {t}  [{act}]")
    print(f"terminal: {r['terminal']}")
    ans = r["answer"]
    print(f"answer: success={ans['success']} reward={ans['reward']:.3f} "
          f"final_pos={ans['final_pos']} final_dir={ans['final_dir']}")
    print(f"n_steps: {r['n_steps']}")
    print("-" * 60)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--idx", type=int, default=None, help="N 번째 에피소드 (0-based)")
    p.add_argument("--id", default=None, help="특정 id 하나만")
    p.add_argument("--n", type=int, default=1, help="처음부터 N개 출력")
    a = p.parse_args()

    rows = read_jsonl(a.path)
    if a.id:
        rows = [r for r in rows if r["id"] == a.id]
    elif a.idx is not None:
        rows = [rows[a.idx]]
    else:
        rows = rows[:a.n]

    for r in rows:
        show(r)


if __name__ == "__main__":
    main()
