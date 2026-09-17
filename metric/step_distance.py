"""step 마지막 토큰(raw hidden state) 사이 거리를 success / fail 로 비교한다.

Sun et al. 2026 (arXiv:2604.05655) Figure 2(a) "Between-step activation distances" 재현.
논문은 step 안의 모든 토큰이 아니라 두 종류 토큰만 쓴다 (§3.2, Appendix F):

    h_{t(Step k)-1}  : Step k 마커 직전 토큰  = 직전 step 의 마지막 토큰
    h_{t(term)-1}    : 최종답 마커 직전 토큰

우리 step 구간을 s_0..s_{T-1} (extract.step_char_bounds), last(s_t) = E[b[t+1]-1] 이라 하면

    논문 Step 1 -> 2         : d(last(s_0), last(s_1))   -- s_0 이 preamble 이면 정확히 일치,
                                                            아니면 Step 1 직전 토큰이 프롬프트 쪽이라
                                                            저장돼 있지 않으므로 "첫 가용 전이"로 대체
    논문 2nd-last -> Last    : d(last(s_{T-3}), last(s_{T-2}))
    논문 Last -> Ans. Marker : d(last(s_{T-2}), E[idx_ans-1])   idx_ans = 답 마커 첫 토큰 인덱스

답 마커는 boundaries 에 없으므로(마지막 step 구간에 흡수됨) jsonl 텍스트 + 토크나이저 offset 으로
extract.tokenize_episode 와 같은 규약으로 다시 구한다. 모델은 안 올리고 토크나이저만 연다.

거리: Euclidean, cosine distance(1 - cos). 논문 §4. 우리 E 는 최종 RMSNorm 이후 값이라
논문의 절대값과는 비교 불가, Δ 부호와 † 패턴만 비교한다.
†: success / fail 의 95% CI(평균 ± 1.96·SE)가 겹치지 않음 (논문 Fig.2 캡션 규칙).

    python metric/step_distance.py --task predict --level BabyAI-GoToObj-v0
    python metric/step_distance.py --task decompose --level BabyAI-GoToObj-v0 --max-episodes 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

from extract import (MODEL, load_episodes,                        # noqa: E402
                     step_char_bounds, char_to_token_bounds)

TRANSITIONS = ("first", "second_last_to_last", "last_to_answer")
METRICS = ("euclid", "cosine")
LABELS = {"success": "success", "failure": "fail"}   # 디렉토리명 -> 표기

# 논문의 termination marker(####)에 해당하는 것: llm-babybench cot 프롬프트가 강제한 답 구문.
# 마지막 출현을 쓴다. decompose 는 파서 버그(<START> 뒤 공백)와 무관하게 위치만 잡으면 되므로 관대하게.
ANSWER_MARKER = {
    "decompose": re.compile(r"<START>", re.IGNORECASE),
    "plan":      re.compile(r"The LLM[’']s action sequence is\s*:", re.IGNORECASE),
    "predict":   re.compile(r"The agent[’']s final state is\s*:", re.IGNORECASE),
}


# ------------------------------------------------------------------ 텍스트 -> 인덱스

def answer_marker_char(output: str, task: str) -> int | None:
    """출력 텍스트에서 답 마커의 (마지막 출현) 시작 char 위치. 없으면 None."""
    hits = list(ANSWER_MARKER[task].finditer(output))
    return hits[-1].start() if hits else None


def char_to_token(char_pos: int, offsets) -> int:
    """char_pos 를 포함하는 토큰 인덱스 (offset[k][0] <= char_pos < offset[k][1]).
    공백이 앞 토큰에 붙어 마커 첫 글자가 offset 사이 틈에 있으면 그 다음 토큰."""
    for k, (s, e) in enumerate(offsets):
        if e > char_pos:
            return k
    raise ValueError(f"char {char_pos} beyond last offset")


def episode_token_info(prompt: str, output: str, task: str, tok) -> dict:
    """extract.tokenize_episode 와 동일한 토크나이즈로 boundaries 를 다시 만들고,
    답 마커 토큰 인덱스(idx_ans, 출력 기준)와 preamble 유무를 함께 돌려준다.
    boundaries 가 청크에 저장된 것과 같은지 호출측에서 검증할 것."""
    enc = tok(prompt + output, add_special_tokens=False, return_offsets_mapping=True)
    char_bounds = [len(prompt) + b for b in step_char_bounds(output)]
    tok_bounds = char_to_token_bounds(char_bounds, enc.offset_mapping)
    ctx = tok_bounds[0]
    boundaries = [b - ctx for b in tok_bounds]

    starts = [m.start() for m in re.finditer(r"(?mi)^(?:#+\s*|\*+\s*)?Step\s*(\d+)\s*[.:]", output)]
    has_preamble = bool(starts) and bool(output[:starts[0]].strip())

    c = answer_marker_char(output, task)
    idx_ans = None
    if c is not None:
        idx_ans = char_to_token(len(prompt) + c, enc.offset_mapping) - ctx
    return {"boundaries": boundaries, "idx_ans": idx_ans, "has_preamble": has_preamble}


# ------------------------------------------------------------------ 거리

def _dist(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float(), b.float()
    return {"euclid": torch.dist(a, b).item(),
            "cosine": 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0).item()}


def transition_distances(E: torch.Tensor, boundaries: list[int],
                         idx_ans: int | None) -> dict:
    """episode 하나의 세 전이 거리. 정의 안 되는 전이는 None + reason.

    반환 예: {"first": {"euclid":..,"cosine":..}, "second_last_to_last": None, ...,
             "n_steps": T, "ans_reason": "ok" | "no_marker" | "not_in_last_step",
             "rep_norms": [..]}"""
    T = len(boundaries) - 1
    last = {t: E[e - 1] for t, (s, e) in enumerate(zip(boundaries, boundaries[1:])) if e > s}
    out = {k: None for k in TRANSITIONS}
    out["n_steps"] = T

    if T >= 2 and 0 in last and 1 in last:
        out["first"] = _dist(last[0], last[1])
    if T >= 3 and (T - 3) in last and (T - 2) in last:
        out["second_last_to_last"] = _dist(last[T - 3], last[T - 2])

    reason = "ok"
    if idx_ans is None:
        reason = "no_marker"
    elif not (boundaries[-2] < idx_ans <= boundaries[-1] - 1) or T < 2:
        # E[idx_ans-1] 이 마지막 step 구간 안에 있어야 논문의 h_{t(term)-1} 에 대응
        reason = "not_in_last_step" if T >= 2 else "too_few_steps"
    elif (T - 2) in last:
        out["last_to_answer"] = _dist(last[T - 2], E[idx_ans - 1])
    out["ans_reason"] = reason

    norms = [last[t].float().norm().item() for t in sorted(last)]
    if reason == "ok":
        norms.append(E[idx_ans - 1].float().norm().item())
    out["rep_norms"] = norms
    return out


# ------------------------------------------------------------------ 집계

def _ci(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"n": 0}
    m = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "std": sd, "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se}


def aggregate(rows: dict[str, list[dict]]) -> dict:
    """rows: {"success": [transition_distances 결과...], "fail": [...]}
    -> table[transition][metric] = {success: ci, fail: ci, delta_f_minus_s, dagger}"""
    table = {}
    for tr in TRANSITIONS:
        table[tr] = {}
        for me in METRICS:
            cell = {}
            for lab, rs in rows.items():
                cell[lab] = _ci([r[tr][me] for r in rs if r[tr] is not None])
            s, f = cell.get("success", {}), cell.get("fail", {})
            if s.get("n") and f.get("n"):
                cell["delta_f_minus_s"] = f["mean"] - s["mean"]
                cell["dagger"] = bool(f["ci_lo"] > s["ci_hi"] or s["ci_lo"] > f["ci_hi"])
            table[tr][me] = cell
    return table


def format_table(table: dict, title: str) -> str:
    head = {"first": "First (Step1→2*)", "second_last_to_last": "2nd-last→Last",
            "last_to_answer": "Last→Ans. Marker"}
    lines = [f"### {title}", "",
             "| Distance | Group | " + " | ".join(head[t] for t in TRANSITIONS) + " |",
             "|---|---|" + "---|" * len(TRANSITIONS)]
    for me, name in (("euclid", "Euclidean"), ("cosine", "Cosine")):
        for lab in ("success", "fail"):
            cells = []
            for tr in TRANSITIONS:
                c = table[tr][me].get(lab, {})
                cells.append(f"{c['mean']:.4g} (n={c['n']})" if c.get("n") else "–")
            lines.append(f"| {name if lab == 'success' else ''} | {lab} | " + " | ".join(cells) + " |")
        cells = []
        for tr in TRANSITIONS:
            c = table[tr][me]
            if "delta_f_minus_s" in c:
                cells.append(f"{c['delta_f_minus_s']:+.4g}" + ("†" if c["dagger"] else ""))
            else:
                cells.append("–")
        lines.append("| | Δ(F–S) | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("*First = 가장 이른 가용 전이. s_0 이 preamble 인 episode 에서만 논문의 Step1→2 와 정확히 일치.")
    return "\n".join(lines)


# ------------------------------------------------------------------ 실행

def resolve_status_dir(hidden_dir: Path, task: str, level: str, status: str, ctx_tag: str) -> Path:
    """현재 extract.py 레이아웃(task/level/<ctx_tag>/status)과 서버의 옛 레이아웃(task/level/status)
    둘 다 받는다. ctx_tag 를 빈 문자열로 주면 옛 레이아웃만 본다."""
    cands = []
    if ctx_tag:
        cands.append(hidden_dir / task / level / ctx_tag / status)
    cands.append(hidden_dir / task / level / status)
    for d in cands:
        if any(d.glob("chunk_*.pt")):
            return d
    raise FileNotFoundError("no chunk_*.pt in any of: " + ", ".join(map(str, cands)))


def resolve_data_file(data_dir: Path, task: str, mode: str) -> Path:
    """generation/trajectory/ (현재 기본값) 에 없으면 옛 위치 data/ 도 본다."""
    cands = [data_dir / f"{task}_{mode}.jsonl", ROOT / "data" / f"{task}_{mode}.jsonl"]
    for f in cands:
        if f.is_file():
            return f
    raise FileNotFoundError("no jsonl in any of: " + ", ".join(map(str, cands)))


def load_text_index(data_dir: Path, task: str, level: str, mode: str) -> dict[int, dict]:
    eps = load_episodes(resolve_data_file(data_dir, task, mode))
    return {e["env_seed"]: e for e in eps
            if e.get("task") == task and e.get("env_name") == level and not e.get("skipped")}


def run(hidden_dir: Path, data_dir: Path, task: str, level: str, mode: str = "no_thinking",
        ctx_tag: str = "with_prompt", max_episodes: int | None = None, tok=None) -> dict:
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
    texts = load_text_index(data_dir, task, level, mode)
    if not texts:
        raise ValueError(f"no episodes for {task}/{level} in {data_dir}")

    rows: dict[str, list[dict]] = {"success": [], "fail": []}
    stats = {"n_seen": 0, "n_missing_text": 0, "n_sha_mismatch": 0,
             "n_boundary_mismatch": 0, "n_preamble": 0,
             "ans_reason": {"ok": 0, "no_marker": 0, "not_in_last_step": 0, "too_few_steps": 0}}
    use_prompt = None   # 청크 헤더(use_prompt_context)에서 읽음. 없으면 ctx_tag 로 추정
    for status, lab in LABELS.items():
        d = resolve_status_dir(hidden_dir, task, level, status, ctx_tag)
        files = sorted(d.glob("chunk_*.pt"))
        done = False
        for cf in files:
            blob = torch.load(cf, map_location="cpu", weights_only=False)
            hdr = blob.get("use_prompt_context")
            if use_prompt is None:
                # 헤더가 있으면 그 값, 없으면(옛 청크) --ctx-tag 로 결정. boundaries/idx_ans 는
                # 출력 기준(ctx 만큼 shift)이라 프롬프트 포함 여부는 프롬프트/출력 경계에서 토큰이
                # 합쳐지는 경우에만 영향을 주고, 그 경우는 n_boundary_mismatch 로 드러난다.
                use_prompt = bool(hdr) if hdr is not None else (ctx_tag == "with_prompt")
                stats["use_prompt_context"] = use_prompt
                stats["hidden_dir_used"] = str(d.parent)
            for seed, ep in blob["episodes"].items():
                stats["n_seen"] += 1
                src = texts.get(seed)
                if src is None:
                    stats["n_missing_text"] += 1
                    continue
                out = src["all_llm_output"]
                if ep.get("output_sha1") and hashlib.sha1(out.encode()).hexdigest() != ep["output_sha1"]:
                    stats["n_sha_mismatch"] += 1
                    continue
                info = episode_token_info(src["prompt"] if use_prompt else "", out, task, tok)
                if info["boundaries"] != list(ep["boundaries"]):
                    stats["n_boundary_mismatch"] += 1
                    continue
                r = transition_distances(ep["E"], ep["boundaries"], info["idx_ans"])
                r["seed"] = seed
                stats["ans_reason"][r["ans_reason"]] += 1
                stats["n_preamble"] += int(info["has_preamble"])
                rows[lab].append(r)
                if max_episodes and len(rows[lab]) >= max_episodes:
                    done = True
                    break
            if done:
                break

    norms = [v for rs in rows.values() for r in rs for v in r["rep_norms"]]
    by_steps = {}
    for T in sorted({r["n_steps"] for rs in rows.values() for r in rs}):
        sub = {lab: [r for r in rs if r["n_steps"] == T] for lab, rs in rows.items()}
        by_steps[T] = aggregate(sub)

    return {"task": task, "level": level, "model": MODEL, "ctx_tag": ctx_tag,
            "n": {lab: len(rs) for lab, rs in rows.items()}, "stats": stats,
            "rep_norm": {"mean": statistics.fmean(norms), "std": statistics.pstdev(norms)} if norms else None,
            "pooled": aggregate(rows), "by_n_steps": by_steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(ANSWER_MARKER))
    ap.add_argument("--level", required=True)
    ap.add_argument("--mode", default="no_thinking", choices=["no_thinking", "thinking"])
    ap.add_argument("--ctx-tag", default="with_prompt",
                    help="hidden_states/task/level/<ctx_tag>/status. 그 층이 없으면 자동으로 "
                         "task/level/status(옛 레이아웃)를 본다. 빈 문자열이면 옛 레이아웃만")
    ap.add_argument("--max-episodes", type=int, default=None, help="status 당 최대 episode 수 (빠른 확인용)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "generation" / "trajectory")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "latent" / "step_distance")
    args = ap.parse_args()

    res = run(args.hidden_dir, args.data_dir, args.task, args.level, args.mode,
              args.ctx_tag, args.max_episodes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.out_dir / f"{args.task}_{args.level}"
    base.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [format_table(res["pooled"], f"{args.task}/{args.level} pooled  "
                       f"(success n={res['n']['success']}, fail n={res['n']['fail']})")]
    for T, tb in res["by_n_steps"].items():
        md.append(format_table(tb, f"n_steps={T}"))
    st = res["stats"]
    md.append(f"stats: {json.dumps(st, ensure_ascii=False)}")
    if res["rep_norm"]:
        rn = res["rep_norm"]
        md.append(f"rep-token norm: mean={rn['mean']:.3f} std={rn['std']:.3f} "
                  f"(std/mean={rn['std'] / rn['mean']:.3f}; 작으면 Euclid ≈ cosine 의 단조함수)")
    text = "\n\n".join(md)
    base.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved -> {base}.json / .md")


if __name__ == "__main__":
    main()