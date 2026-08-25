"""free-run CoT 출력(cot_output)을 step 단위 char span으로 분할.

설계 원칙
- 문자열을 자르지 않고 "문자 offset span"만 계산한다.
  → 원문 재조립 문제가 원천적으로 없음 (span들이 전체를 빈틈없이 덮는지 assert).
- step0 = Step 1 헤더가 나오기 전까지의 출력 (없으면 span 길이 0으로 생략).
- terminal = 최종 답 시작 지점("The LLM's action sequence is:" 또는 "<START>")부터 끝까지.
- 토큰화는 여기서 하지 않는다. extract.py가 full text를 1회 토큰화한 뒤
  offset_mapping으로 char span → token span 변환을 담당한다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# 줄 시작의 "Step N." / "Step N:" / "### Step N." / "**Step N.**" 변형 허용
STEP_PAT = re.compile(r"(?m)^(?:#+\s*|\*+\s*)?Step\s*(\d+)\s*[.:]")

# 최종 답 블록 시작 (먼저 등장하는 것 사용).
# 업스트림 llms/utils.py 의 parser() 는 text.lower() 로 찾으므로 대소문자를 안 가린다.
# 여기서만 구분하면 모델이 "The Agent's Final State Is:" 처럼 쓸 때 업스트림은
# 답을 뽑아내는데 우리만 no_terminal 로 버리게 된다 — re.I 로 맞춘다.
TERMINAL_PATS = [
    re.compile(r"The LLM's action sequence is:", re.I),   # plan
    re.compile(r"The agent's final state is:", re.I),     # predict
    re.compile(r"(?m)^<START>", re.I),                    # decompose
]

# 프롬프트 본문에 박혀 있는 액션 목록. cot_predict.py 는 이 값을 따로 저장하지
# 않지만 prompt 안에는 그대로 있어 복원할 수 있다.
#     ... executing the following action sequence:
#     left, forward, forward
# 주의: OmniBot 이 미션을 못 풀면 스텝 상한(1000)까지 "done" 으로 채운 시퀀스가
# 나온다. 정규식이 잘린 게 아니라 업스트림 데이터의 성질이다.
ACTION_SEQ_PAT = re.compile(r"action sequence:\s*\n([a-z, ]+)")

# 액션 블록 = CoT 안에서 액션 하나에 해당하는 추론 덩어리. 헤더 줄 + 그 아래
# 설명 줄들, 다음 액션 헤더 직전까지.
#
# 헤더 표기가 25종 넘게 갈린다 ("- **First action: `left`**", "1. **right**:",
# "#### Action 3: forward", "- Action 'left':" ...). 하나의 정규식으로 묶으면
# 백트래킹 순서가 어긋나 "1. **right**" 같은 형태를 통째로 놓치므로, 앞머리
# 잡음을 먼저 벗기고 남은 부분이 액션 이름으로 시작하는지 보는 2단계로 간다.
ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")

_ORDINALS = ("first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
             "eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
             "seventeenth|eighteenth|nineteenth|twentieth")

# 헤딩(#) · 불릿(-*+) · 볼드(**) · 번호(3.) · 라벨(Action/Step/서수) · 따옴표를
# 임의 순서로 흡수한다. 전부 선택적이라 항상 매치된다 (길이 0 가능).
_HEAD_NOISE = re.compile(
    rf"^[ \t]*(?:[#>*+\-]+[ \t]*)*(?:\*{{1,2}})?"
    rf"(?:(?:{_ORDINALS}|step|action)\b[ \t]*)?"
    r"(?:\*{1,2})?[ \t]*(?:\d+[ \t]*[.:)][ \t]*)?(?:\*{1,2})?[ \t]*"
    rf"(?:(?:{_ORDINALS}|step|action)\b[ \t]*[:.]?[ \t]*)?(?:\*{{1,2}})?[`'\"]?",
    re.I)
_ACTION_HEAD = re.compile("(" + "|".join(ACTION_NAMES) + r")\b", re.I)

def _action_starts(body: str, base: int) -> list[int]:
    """body 안에서 액션 블록이 시작하는 절대 char offset.

    줄 위치는 finditer 가 준 lm.start() 를 그대로 쓴다. split 후 len(line)+1 을
    누적하는 방식은 "구분자는 항상 1글자"를 가정하는데, \\r\\n 이 섞이면 offset 이
    누적으로 밀린다. 그렇게 밀려도 split_cot 의 덮개 assert 는 span 끼리의 연속성만
    보므로 통과한다 — 전부 어긋난 세그먼트가 조용히 나온다.
    """
    starts = []
    for lm in re.finditer(r"(?m)^[^\n]*", body):
        line = lm.group()
        m = _HEAD_NOISE.match(line)
        if _ACTION_HEAD.match(line[m.end():]):
            starts.append(base + lm.start())
    return starts


def split_cot(cot_output: str, by_action: bool = True) -> tuple[list[dict], dict]:
    """cot_output → (segments, info)

    by_action=False 면 3단계(액션 재분할)를 건너뛰고 Step N. 단위에서 멈춘다.
    두 분절을 같은 에피소드에서 대조하려고 두었다. info 의 액션 블록 수는
    플래그와 무관하게 항상 채워진다 — 두 호출의 진단값이 갈리면 헷갈린다.

    segments: [{"name": ..., "start": int, "end": int}, ...]
              name 은 "step0" | "step{N}" | "step{N}.a{K}" | "terminal".
              .a{K} 는 step{N} 안을 액션 블록 단위로 다시 쪼갠 것이고,
              앞의 "step{N}" 은 첫 블록 전까지의 머리말("Step 2. Process each
              action in sequence:")이다.
              cot_output[start:end]가 해당 세그먼트. 전체를 빈틈없이 덮음.
    info:     파싱 진단 정보.
    """
    text = cot_output

    # 1) terminal 경계
    term_start = len(text)
    for pat in TERMINAL_PATS:
        m = pat.search(text)
        if m:
            term_start = min(term_start, m.start())

    body = text[:term_start]

    # 2) step 헤더 위치 (terminal 이전 구간에서만)
    matches = list(STEP_PAT.finditer(body))
    starts = [m.start() for m in matches]
    labels = [int(m.group(1)) for m in matches]

    info = {
        "n_step_headers": len(starts),
        "step_labels": labels,
        # 순증가만 통과. sorted() 비교는 중복을 허용해서 [1,2,2,2,3] 을 놓치는데,
        # 같은 라벨이 여러 번 나오면 세그먼트 이름이 겹쳐(step2 가 5개) 이름으로
        # 세그먼트를 지목할 수 없게 된다. 번호를 다시 1부터 매기는 경우
        # ([1,2,1,2,3,...] — 스캐폴드 번호와 모델 자체 액션 번호가 충돌)도 함께
        # 걸러진다. 다른 궤적과 구조가 달라 섞으면 분석이 흐려진다.
        "labels_monotonic": all(a < b for a, b in zip(labels, labels[1:])),
        "parse_ok": len(starts) > 0,
        "has_terminal": term_start < len(text),
    }

    segments: list[dict] = []

    if not starts:
        # step 헤더가 하나도 없으면 body 전체를 step0으로
        if body:
            segments.append({"name": "step0", "start": 0, "end": term_start})
    else:
        if starts[0] > 0:  # Step 1 이전의 출력 = step0
            segments.append({"name": "step0", "start": 0, "end": starts[0]})
        bounds = starts + [term_start]
        for lab, s, e in zip(labels, bounds, bounds[1:]):
            segments.append({"name": f"step{lab}", "start": s, "end": e})

    if term_start < len(text):
        segments.append({"name": "terminal", "start": term_start, "end": len(text)})

    # 3) 액션 단위 재분할. Step 헤더는 국면 단위라("Step 2. Process each action
    #    in sequence") 액션별 추론이 한 세그먼트 안에 통째로 들어간다. 액션 블록이
    #    시작하는 지점마다 더 쪼개 stepN.aK 를 만든다.
    #    terminal 은 제외한다 — 최종 답에도 액션 이름이 들어가 잘못 잘린다.
    #    블록이 없으면(모델이 산문으로 서술) 재분할 없이 그대로 둔다.
    refined: list[dict] = []
    n_blocks = 0
    for seg in segments:
        if seg["name"] == "terminal":
            refined.append(seg)
            continue

        starts = _action_starts(text[seg["start"]:seg["end"]], seg["start"])
        n_blocks += len(starts)

        # 세그먼트 첫 글자부터가 블록이면 자를 데가 없다 (세그먼트 = 블록 하나).
        cuts = [s for s in starts if s > seg["start"]]
        if not cuts:
            refined.append(seg)
            continue

        bounds = [seg["start"], *cuts, seg["end"]]
        refined.append({"name": seg["name"], "start": bounds[0], "end": bounds[1]})
        for k, (s, e) in enumerate(zip(bounds[1:], bounds[2:]), 1):
            refined.append({"name": f"{seg['name']}.a{k}", "start": s, "end": e})

    info["n_action_blocks"] = n_blocks
    info["has_action_blocks"] = n_blocks > 0
    if by_action:
        segments = refined

    # 4) 무결성: span들이 전체를 빈틈없이 덮는가
    if segments:
        cover_ok = segments[0]["start"] == 0 and segments[-1]["end"] == len(text)
        for a, b in zip(segments, segments[1:]):
            cover_ok = cover_ok and (a["end"] == b["start"])
    else:
        cover_ok = len(text) == 0
    assert cover_ok, "segments do not cover cot_output without gaps/overlaps"

    return segments, info


def _load_records(in_path: Path) -> list[dict]:
    """JSONL 또는 JSON 배열 파일 모두 지원."""
    text = in_path.read_text(encoding="utf-8")
    head = text.lstrip()[:1]
    if head == "[":
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def predict_input_action(ep: dict) -> str | None:
    """predict 프롬프트에 주어진 액션 시퀀스 — 모델이 시뮬레이션할 대상.

    predict 전용이다. 태스크마다 액션 시퀀스가 놓인 자리가 반대라서 한 함수로
    묶으면 엉뚱한 값을 집는다.
        predict : 시퀀스가 입력  -> 프롬프트에서 복원
        plan    : 시퀀스가 답    -> parsed_llm_output
        decompose : 액션이 아니라 subgoal -> 해당 없음
    """
    if ep.get("task") != "predict":
        return None

    # TODO(plan): plan 의 분절 검증 기준은 모델이 스스로 낸 시퀀스다.
    #   if ep.get("task") == "plan":
    #       return ep.get("parsed_llm_output")
    # cot_paln.py 가 남기는 optimal_action_seq 를 쓰면 안 된다 — 봇의 정답이고
    # 모델은 본 적이 없어서, 대조하면 "분절이 맞는가" 가 아니라 "봇과 같은
    # 계획을 세웠나" 를 재게 된다. llm_efficiency != 1 인 성공 케이스가 전부
    # 분절 실패로 잘못 찍힌다. plan CoT 실물을 보고 확정할 것.

    if ep.get("action_seq"):          # cot_predict.py 가 나중에 저장하면 그걸 쓴다
        return ep["action_seq"]
    m = ACTION_SEQ_PAT.search(ep.get("prompt") or "")
    return m.group(1).strip() if m else None


def _get_success(ep: dict) -> bool | None:
    ev = ep.get("eval_result") or {}
    if "success" in ev:          # predict 스키마
        return bool(ev["success"])
    if "CR" in ev:               # plan/decompose 스키마
        return ev["CR"] == 1.0
    return None


def _get_mission(ep: dict) -> str | None:
    if ep.get("mission"):
        return ep["mission"]
    m = re.search(r"Mission:\s*'([^']+)'", ep.get("prompt") or "")
    return m.group(1) if m else None


def _fail_reason(ep: dict, info: dict) -> str | None:
    if ep.get("truncated", False):
        return "truncated"
    if not info["parse_ok"]:
        return "no_step_header"
    if not info["has_terminal"]:
        return "no_terminal"
    if not info["labels_monotonic"]:
        return "labels_not_monotonic"
    return None


def convert_file(in_path: Path, out_dir: Path) -> dict:
    """free-run 파일 → level × (success|fail) 로 나눠 JSONL 저장.

    출력 구조:
      out_dir/{level}/success.jsonl
      out_dir/{level}/fail.jsonl
      out_dir/_bad.jsonl              (파싱 불가/truncated, reason 포함)
    """
    records = _load_records(in_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"in": len(records), "out": 0, "bad": 0, "bad_reasons": {}, "groups": {}}
    handles: dict[tuple[str, str], object] = {}
    bad_f = (out_dir / "_bad.jsonl").open("w", encoding="utf-8")

    try:
        for ep in records:
            # plan 은 빌드 실패한 시드도 skip 행으로 남긴다 (prompt/출력이 전부
            # None). 파싱할 것이 없으므로 사유만 남기고 넘긴다.
            if ep.get("skipped"):
                stats["bad"] += 1
                r = f"skipped:{ep['skipped']}"
                stats["bad_reasons"][r] = stats["bad_reasons"].get(r, 0) + 1
                bad_f.write(json.dumps({**ep, "fail_reason": r},
                                       ensure_ascii=False) + "\n")
                continue

            cot = ep.get("all_llm_output") or ""
            segments, info = split_cot(cot)
            # Step N. 단위 분절도 같이 남긴다. 액션 분절과 어느 쪽이 latent 궤적을
            # 잘 잡는지 같은 에피소드에서 대조해야 해서 한 레코드에 둘 다 넣는다.
            # 파일을 나누면 cot_output 원문까지 복제돼 용량이 두 배가 된다.
            step_segments, _ = split_cot(cot, by_action=False)

            reason = _fail_reason(ep, info)
            success = _get_success(ep)

            rec = {
                "id": f"{ep['env_name']}|{ep['env_seed']}",
                "env_name": ep["env_name"],
                "env_seed": ep["env_seed"],
                "task": ep.get("task"),
                "mission": _get_mission(ep),
                # 분절 검증용 기준. segments_action 의 블록이 실제 액션과
                # 1:1로 맞는지 대조할 때 쓴다. 계산에는 관여하지 않는다.
                "check_action": predict_input_action(ep),
                "prompt": ep["prompt"],
                "all_llm_output": cot,
                # 태스크의 답 자체. predict 는 "((x, y), dir)", plan 은 액션
                # 시퀀스. eval_result 에는 점수만 있고 답이 없어서 따로 남긴다.
                "parsed_llm_output": ep.get("parsed_llm_output"),
                "segments_action": segments,      # 액션 단위 (stepN.aK 포함)
                "segments_step": step_segments,   # Step N. 단위
                "seg_info": info,
                "success": success,
                "eval_result": ep.get("eval_result"),
                "truncated": ep.get("truncated", False),
            }

            if reason is not None or success is None:
                reason = reason or "no_success_label"
                rec["fail_reason"] = reason
                stats["bad"] += 1
                stats["bad_reasons"][reason] = stats["bad_reasons"].get(reason, 0) + 1
                bad_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            group = "success" if success else "fail"
            key = (ep["env_name"], group)
            if key not in handles:
                gdir = out_dir / ep["env_name"]
                gdir.mkdir(parents=True, exist_ok=True)
                handles[key] = (gdir / f"{group}.jsonl").open("w", encoding="utf-8")
            handles[key].write(json.dumps(rec, ensure_ascii=False) + "\n")

            stats["out"] += 1
            gk = f"{ep['env_name']}/{group}"
            stats["groups"][gk] = stats["groups"].get(gk, 0) + 1
    finally:
        bad_f.close()
        for h in handles.values():
            h.close()

    print(f"{in_path.name}: in={stats['in']} out={stats['out']} bad={stats['bad']}")
    for gk in sorted(stats["groups"]):
        print(f"  {gk}: {stats['groups'][gk]}")
    if stats["bad_reasons"]:
        print("  bad_reasons:", stats["bad_reasons"])
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="raw jsonl 또는 json 배열 파일")
    ap.add_argument("out_dir", type=Path, help="level×success 그룹별 jsonl이 저장될 디렉토리")
    args = ap.parse_args()
    convert_file(args.input, args.out_dir)