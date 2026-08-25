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

# 최종 답 블록 시작 (먼저 등장하는 것 사용)
TERMINAL_PATS = [
    re.compile(r"The LLM's action sequence is:"),   # plan
    re.compile(r"The agent's final state is:"),     # predict
    re.compile(r"(?m)^<START>"),                    # decompose
]


def split_cot(cot_output: str) -> tuple[list[dict], dict]:
    """cot_output → (segments, info)

    segments: [{"name": "step0"|"step{N}"|"terminal", "start": int, "end": int}, ...]
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
        "labels_monotonic": labels == sorted(labels),
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

    # 3) 무결성: span들이 전체를 빈틈없이 덮는가
    if segments:
        cover_ok = segments[0]["start"] == 0 and segments[-1]["end"] == len(text)
        for a, b in zip(segments, segments[1:]):
            cover_ok = cover_ok and (a["end"] == b["start"])
    else:
        cover_ok = len(text) == 0
    info["cover_ok"] = cover_ok
    assert cover_ok, "segments do not cover cot_output without gaps/overlaps"

    return segments, info


def _load_records(in_path: Path) -> list[dict]:
    """JSONL 또는 JSON 배열 파일 모두 지원."""
    text = in_path.read_text(encoding="utf-8")
    head = text.lstrip()[:1]
    if head == "[":
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _get_success(ep: dict) -> bool | None:
    ev = ep.get("eval") or {}
    if "success" in ev:          # predict 스키마
        return bool(ev["success"])
    if "CR" in ev:               # plan/decompose 스키마
        return ev["CR"] == 1.0
    return None


def _get_mission(ep: dict) -> str | None:
    if ep.get("mission"):
        return ep["mission"]
    m = re.search(r"Mission:\s*'([^']+)'", ep.get("prompt", ""))
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
            cot = ep.get("cot_output") or ep.get("assistant_text") or ""
            segments, info = split_cot(cot)

            reason = _fail_reason(ep, info)
            success = _get_success(ep)

            rec = {
                "id": ep.get("id") or f"{ep['level']}|{ep['seed']}",
                "level": ep["level"],
                "seed": ep.get("seed"),
                "mission": _get_mission(ep),
                "action_seq": ep.get("action_seq") or ep.get("optimal_action_seq"),
                "prompt": ep["prompt"],
                "cot_output": cot,
                "segments": segments,
                "seg_info": info,
                "success": success,
                "eval": ep.get("eval"),
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
            key = (ep["level"], group)
            if key not in handles:
                gdir = out_dir / ep["level"]
                gdir.mkdir(parents=True, exist_ok=True)
                handles[key] = (gdir / f"{group}.jsonl").open("w", encoding="utf-8")
            handles[key].write(json.dumps(rec, ensure_ascii=False) + "\n")

            stats["out"] += 1
            gk = f"{ep['level']}/{group}"
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