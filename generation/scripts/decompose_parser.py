"""Decompose 출력에서 <START>...<END> 블록을 꺼내는 파서.

llm-babybench의 llms.utils.parser 는 "<start>\\n" 을 문자 그대로 찾기 때문에
모델이 마크다운 줄바꿈용 공백을 붙여 "<START>  \\n" 로 쓰면 빈 문자열을 돌려준다
(decompose_no_thinking.jsonl 기준 45% 가 이 이유로 eval_error 처리됐다).
여기서는 마커 앞뒤 공백 / **볼드** / ``` / <<START>> 같은 장식을 허용하고,
본문의 불릿(-, *, 1.)과 볼드 표시를 걷어낸 뒤 줄 단위 서브골 문자열로 돌려준다.
evaluator 에 넘기는 형식(줄바꿈으로 구분된 "(GoNextToSubgoal, (x, y))")은 원본과 같다.
"""
from __future__ import annotations

import re

_BLOCK = re.compile(r"(?is)<+\s*START\s*>+\**\s*\n(.*?)\n\s*[*`]*<+\s*END\s*>+")
_BULLET = re.compile(r"(?m)^\s*(?:[-*]\s*|\d+\.\s*)")


def parse_decompose_output(text: str) -> str:
    """<START>..<END> 사이 서브골을 줄바꿈 구분 문자열로. 블록이 없으면 ""."""
    m = _BLOCK.search(text)
    if not m:
        return ""
    body = _BULLET.sub("", m.group(1))
    body = re.sub(r"[*`]", "", body)
    return "\n".join(line.strip() for line in body.splitlines() if line.strip())
