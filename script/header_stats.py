"""Step 헤더 줄이 스텝 번호를 얼마나 드러내는지 센다 (프로빙 치팅 점검용).

extract.py 의 STEP_PAT 과 같은 기준으로 헤더를 찾고, 헤더 줄 전체에서
"Step N" 부분과 마크다운 장식(#, *, 공백)을 벗긴 "제목"을 뽑아

  - 헤더 줄 형태 분포          ("Step N." / "### Step N:" / "**Step N:**" ...)
  - 제목이 있는 헤더 비율       (제목 없으면 "Step N" 만 지워도 줄이 사라진다)
  - 스텝별 제목 다양성           (고유 제목 수, 최빈 제목과 그 점유율)
  - 제목만 보고 스텝 번호를 맞힐 수 있는 비율 (제목→최빈 스텝 매핑의 정확도)

를 태스크별로 찍는다. 마지막 수치가 높으면 "Step N" 만 지우는 건 의미가 없고
헤더 줄 전체를 지워야 한다.

    python script/header_stats.py --data-dir /home/hail/HDD/cot/generation/trajectory
    python script/header_stats.py --task plan --level BabyAI-GoToObj-v0
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

from extract import STEP_PAT, load_episodes, step_char_bounds  # noqa: E402

_DECOR = re.compile(r"^[\s#*_]+|[\s#*_:.]+$")


def header_shape(line: str, m: re.Match) -> str:
    """헤더 줄을 형태 문자열로 — 숫자는 N, 제목은 T 로 치환."""
    head = line[: m.end() - m.start()]
    head = re.sub(r"\d+", "N", head)
    rest = line[m.end() - m.start():].strip()
    return head + (" T" if _DECOR.sub("", rest) else "") + \
        ("**" if rest.endswith("**") else "")


def header_title(line: str, m: re.Match) -> str:
    rest = line[m.end() - m.start():]
    return _DECOR.sub("", rest).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "generation" / "trajectory")
    ap.add_argument("--task", choices=["decompose", "plan", "predict"], default=None)
    ap.add_argument("--level", default=None, help="env_name 으로 필터 (기본: 전체)")
    ap.add_argument("--mode", default="no_thinking")
    ap.add_argument("--clean-only", action="store_true",
                    help="step_char_bounds 를 통과하는 (extract 가 실제로 쓰는) 궤적만")
    ap.add_argument("--top", type=int, default=3, help="스텝별로 보여줄 최빈 제목 수")
    args = ap.parse_args()

    tasks = [args.task] if args.task else ["decompose", "plan", "predict"]
    for task in tasks:
        src = args.data_dir / f"{task}_{args.mode}.jsonl"
        if not src.is_file():
            print(f"== {task}: {src} 없음"); continue
        eps = [e for e in load_episodes(src)
               if e.get("task", task) == task
               and (args.level is None or e.get("env_name") == args.level)]

        shapes = Counter()
        titles_by_step: dict[int, Counter] = defaultdict(Counter)
        n_traj = n_head = n_titled = 0
        for e in eps:
            out = e.get("all_llm_output") or ""
            if args.clean_only and step_char_bounds(out, task)[0] is None:
                continue
            heads = list(STEP_PAT.finditer(out))
            if not heads:
                continue
            n_traj += 1
            for m in heads:
                ls = out.rfind("\n", 0, m.start()) + 1
                le = out.find("\n", m.end())
                line = out[ls: le if le >= 0 else len(out)]
                m2 = STEP_PAT.search(line)
                shapes[header_shape(line, m2)] += 1
                t = header_title(line, m2)
                n_head += 1
                if t:
                    n_titled += 1
                titles_by_step[int(m.group(1))][t] += 1

        print(f"\n== {task}  trajectories={n_traj} (of {len(eps)} loaded)  headers={n_head}")
        if not n_head:
            continue
        print(f"  헤더에 제목이 붙은 비율: {n_titled}/{n_head} = {n_titled / n_head:.1%}")
        print("  헤더 줄 형태:")
        for s, n in shapes.most_common(10):
            print(f"    {n:7d}  {n / n_head:6.1%}  {s!r}")

        # 제목 → 최빈 스텝 매핑으로 스텝 번호를 맞히면 몇 %?
        title_step = defaultdict(Counter)
        for step, c in titles_by_step.items():
            for t, n in c.items():
                if t:
                    title_step[t][step] += n
        correct = sum(c.most_common(1)[0][1] for c in title_step.values())
        print(f"  제목만으로 스텝 번호 식별 가능: {correct}/{n_titled} = "
              f"{correct / max(n_titled, 1):.1%}  (고유 제목 {len(title_step)}개)")

        print("  스텝별 제목:")
        for step in sorted(titles_by_step):
            c = titles_by_step[step]
            tot = sum(c.values())
            uniq = sum(1 for t in c if t)
            print(f"    Step {step}: n={tot}  고유제목={uniq}")
            for t, n in c.most_common(args.top):
                print(f"        {n / tot:6.1%}  {t or '(제목 없음)'!r}")


if __name__ == "__main__":
    main()
