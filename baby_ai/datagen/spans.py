"""
사후 분절 + 토큰 span 매핑 + 정합 검증.

분절 근거 (setting.md 2 "분절 근거")
  TRACES 2604.21057 : 단락 수준(`\n\n`)이 "현재 문헌에서 가장 널리 채택된 정의".
  Local Causal Attribution 2606.21821, PUMA 2605.17672 도 같은 계보.

  위험: TRACES 의 `\n\n` 관찰은 DeepSeek-R1 / QwQ / GPT 같은 reasoning 모델 기준이고
  Qwen2.5-3B-Instruct 는 reasoning 모델이 아니다. -> `\n\n` 을 충분히 안 낼 수 있다.
  폴백: MuCRASP 2605.25842 의 marker taxonomy, 그것도 없으면 equal-interval 분할.

  **자동 폴백은 하지 않는다.** 분절 방식은 통제된 실험 축(`--segment`)이고,
  에피소드마다 방식이 달라지면 셀 내부 비교 가능성이 깨진다. "step 중앙값이 1이면
  폴백" 은 n=1 로는 판정할 수 없는 STEP 2 집계 게이트다 (setting.md 6 STEP 2).
  판정은 check.py 의 `steps_per_episode.median` / `frac_single` 이 한다.

span 정합 (setting.md 4)
  토큰 경계가 한 칸만 밀려도 E_t 가 옆 블록 토큰을 물고 들어와서 Δ_hist 가
  "history 통합" 대신 "경계 오염"을 재게 된다. 다만 BPE 는 선행 공백을 다음 토큰에
  붙이므로(`"\n\nThe"` 가 한 토큰), 트림된 문자 구간을 토큰 경계로 확장하면
  **구조적으로 거의 항상** 선행 공백이 되돌아온다. 따라서 불일치를 두 종류로 나눈다:
      whitespace : strip 후 동일 -> 무해. 게이트 통과
      content    : 옆 블록 내용이 새어듦 -> 치명. 게이트 차단
  `exact` 는 진단용으로 따로 보고한다.
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right

# ---------------------------------------------------------------- 분절

# 개행 사이에 공백이 끼는 경우("\n \n")도 잡는다. 모델 출력에서 드물지 않다.
_BLANK = re.compile(r"\n[ \t]*\n[ \t\n]*")

# MuCRASP 2605.25842 계열 폴백: 구조적 delimiter + 논리 커넥티브.
# 줄머리뿐 아니라 문장 경계(마침표 뒤)에서도 잡는다. `\n\n` 이 실패했다는 것은
# 대개 한 단락/한 줄이라는 뜻이라, 줄머리만 보면 폴백이 무력화된다.
_MARKERS = re.compile(
    r"(?:(?<=\A)|(?<=\n)|(?<=[.!?]\s)|(?<=[.!?]\n))"
    r"[ \t]*(?:"
    r"\d+[.)]\s"                                          # "1. " "2) "
    r"|[-*]\s"                                            # "- " "* "
    r"|(?:First|Second|Third|Next|Then|Now|Finally|Once|"
    r"After that|Therefore|Thus|So|Since|Because|"
    r"However|But|Wait)\b"
    r")"
)

_SENT_END = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|\Z)")


def _trim(text: str, spans) -> list[tuple[int, int]]:
    """앞뒤 공백을 뺀 실제 내용 구간만 남긴다. 빈 구간은 버린다."""
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def _from_cuts(text: str, cuts) -> list[tuple[int, int]]:
    cuts = sorted({c for c in cuts if 0 <= c <= len(text)})
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
    if cuts[-1] != len(text):
        cuts.append(len(text))
    return _trim(text, list(zip(cuts[:-1], cuts[1:])))


def split_blank(text: str) -> list[tuple[int, int]]:
    """`\n\n` 기준 단락 분절. TRACES 최다 채택 정의."""
    spans, pos = [], 0
    for m in _BLANK.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    return _trim(text, spans)


def split_marker(text: str) -> list[tuple[int, int]]:
    """구조적 delimiter / 논리 커넥티브 기준 분절."""
    return _from_cuts(text, [m.start() for m in _MARKERS.finditer(text)])


def split_equal(text: str, n: int = 3) -> list[tuple[int, int]]:
    """균등 분할. 단, 문장 경계에 스냅한다.

    문자 인덱스로 그냥 자르면 단어 중간이 잘려 그 블록의 첫/마지막 토큰이
    반쪽이 되고, 정확히 우리가 막으려는 경계 오염이 된다.
    """
    if not text.strip():
        return []
    ends = [m.end() for m in _SENT_END.finditer(text) if 0 < m.end() < len(text)]

    n = max(1, n)
    targets = [round(i * len(text) / n) for i in range(1, n)]
    if ends:
        cuts = [min(ends, key=lambda x: abs(x - t)) for t in targets]
    else:
        # 문장 경계가 아예 없으면 공백 경계로라도 스냅한다.
        ws = [m.end() for m in re.finditer(r"\s+", text) if 0 < m.end() < len(text)]
        cuts = [min(ws, key=lambda x: abs(x - t)) for t in targets] if ws else targets
    return _from_cuts(text, cuts)


_SPLITTERS = {"nl": split_blank, "marker": split_marker, "equal": split_equal}


def segment(text: str, mode: str = "nl") -> tuple[list[tuple[int, int]], str]:
    """분절 진입점. (char_spans, 사용된 mode) 반환.

    **자동 폴백 없음.** 시킨 방식만 적용한다. 그 방식이 1블록만 만들어도 그대로
    1블록을 돌려주고, 폴백 여부는 STEP 2 집계 게이트가 판정한다.
    """
    try:
        fn = _SPLITTERS[mode]
    except KeyError:
        raise ValueError(f"unknown segment mode: {mode!r} "
                         f"(expected one of {sorted(_SPLITTERS)})") from None
    return fn(text), mode


# ---------------------------------------------------------------- char -> token


def char_offsets(tok, ids: list[int]) -> list[int]:
    """cum[i] = len(tok.decode(ids[:i])). 길이 len(ids)+1.

    토큰 i 는 문자 구간 [cum[i], cum[i+1]) 을 덮는다.
    개별 토큰을 따로 디코드해 이어붙이면 BPE 병합 경계에서 원문과 달라질 수
    있으므로, 실제 생성에 쓰인 접두어 디코드를 그대로 누적한다.
    재토크나이즈를 하지 않으므로 원본 토큰 경계가 보존된다.
    """
    cum = [0]
    for i in range(1, len(ids) + 1):
        cum.append(len(tok.decode(ids[:i], skip_special_tokens=False)))
    return cum


def char_to_token_span(cum: list[int], a: int, b: int) -> tuple[int, int]:
    """문자 구간 [a,b) 를 덮는 최소 토큰 구간 [s,e). 토큰 경계까지 확장된다.

    s : cum[s] <= a 인 최대 s   (a 를 포함하는 토큰)
    e : cum[e] >= b 인 최소 e   (exclusive end)
    """
    n = len(cum) - 1                      # 토큰 개수
    if n <= 0:
        return 0, 0
    s = max(bisect_right(cum, a) - 1, 0)
    s = min(s, n - 1)
    e = bisect_left(cum, b)
    if e <= s:
        e = s + 1
    return s, min(e, n)


# ---------------------------------------------------------------- 정합 검증


def verify_spans(tok, ids: list[int], spans, texts: list[str]) -> dict:
    """토큰 span 이 원문 블록을 정확히 덮는지 검증. 크래시 대신 리포트를 반환.

    ok    : 게이트용. 내용 오염과 겹침만 차단하고, 선행 공백 차이는 통과시킨다.
    exact : 진단용. 토큰 경계가 문자 경계와 정확히 일치하는지.

    STEP 0 에서는 `bad` 의 expected/got 원문을 보고 프롬프트 포맷을 고친다.
    STEP 2 부터 check.py 가 `ok` 를 하드 게이트로 쓴다.
    """
    spans = [tuple(x) for x in spans]

    bad, ws_only = [], 0
    for i, ((s, e), expected) in enumerate(zip(spans, texts)):
        got = tok.decode(ids[s:e], skip_special_tokens=False)
        if got == expected:
            continue
        kind = "whitespace" if got.strip() == expected.strip() else "content"
        if kind == "whitespace":
            ws_only += 1
        bad.append({"i": i, "s": s, "e": e, "kind": kind,
                    "expected": expected, "got": got})

    overlaps = [
        {"i": i, "end_i": spans[i][1], "start_next": spans[i + 1][0]}
        for i in range(len(spans) - 1)
        if spans[i][1] > spans[i + 1][0]
    ]

    n_content = len(bad) - ws_only
    return {
        "ok": n_content == 0 and len(overlaps) == 0,
        "exact": len(bad) == 0 and len(overlaps) == 0,
        "n": len(spans),
        "n_bad": len(bad),
        "n_whitespace_only": ws_only,
        "n_content_mismatch": n_content,
        "n_overlap": len(overlaps),
        "bad": bad,
        "overlaps": overlaps,
    }