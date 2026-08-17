"""
raw.jsonl 뷰어. id(또는 seed 번호)로 에피소드 하나를 찾아 사람이 읽기 좋게 출력한다.
채점/분석은 하지 않는다 -- 순수 보기용.

id 는 `level|seed` 형태라 셀/생성기/prompt 변형을 구분 못 한다. 그래서 기본은
`data/*/raw.jsonl` 전부를 뒤져서 매치되는 걸 다 보여준다 -- 같은 seed 가 여러
디렉토리에 있으면 전부 나온다. --dir 로 좁힐 수 있다.

사용
    python -m datagen.view 12                          # 모든 data/*/raw.jsonl 에서 seed=12
    python -m datagen.view BabyAI-PutNextLocalS6N4-v0|12
    python -m datagen.view 12 --dir data/P2-claude-bb   # 그 디렉토리만
    python -m datagen.view 12 --prompt                  # 관측/프롬프트도 같이
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _matches(rec_id: str, query: str) -> bool:
    """query 가 숫자면 seed 로, 아니면 id 부분 문자열로 매치한다."""
    if query.isdigit():
        return rec_id.rsplit("|", 1)[-1] == query
    return query in rec_id


def _user_content(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return "(user 메시지 없음)"


def _show(dir_: Path, r: dict, show_prompt: bool) -> None:
    print(f"===== {dir_}  id={r['id']} =====")
    print(f"generator={r.get('generator')}  obs_mode={r.get('obs_mode')}  "
          f"prompt_variant={r.get('prompt_variant')}  n_chars={r.get('n_chars')}")
    print(f"mission: {r['mission']}")
    if show_prompt:
        print("--- 관측/프롬프트 (user 메시지) ---")
        print(_user_content(r["messages"]))
    print("--- assistant_text ---")
    print(r["assistant_text"])
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="id 전체 또는 seed 번호 (예: 12)")
    ap.add_argument("--dir", default=None,
                    help="지정 안 하면 data/*/raw.jsonl 전부 검색")
    ap.add_argument("--prompt", action="store_true",
                    help="관측/프롬프트(user 메시지)도 같이 출력")
    args = ap.parse_args()

    dirs = ([Path(args.dir)] if args.dir
            else sorted({p.parent for p in Path("data").glob("*/raw.jsonl")}))

    n = 0
    for d in dirs:
        path = d / "raw.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if _matches(r["id"], args.query):
                    _show(d, r, args.prompt)
                    n += 1

    if n == 0:
        print(f"(해당 없음: {args.query})")


if __name__ == "__main__":
    main()
