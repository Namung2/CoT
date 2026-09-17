"""step 마지막 토큰(raw hidden state) 사이 거리를 success / fail 로 비교한다.

Sun et al. 2026 (arXiv:2604.05655) Figure 2(a) "Between-step activation distances" 재현.
논문은 step 안의 모든 토큰이 아니라 두 종류 토큰만 쓴다 (§3.2, Appendix F):

    h_{t(Step k)-1}  : Step k 마커 직전 토큰  (= 직전 구간의 마지막 토큰)
    h_{t(term)-1}    : 최종답 마커 직전 토큰   (= 마지막 step 의 마지막 토큰)

서버 extract.py 의 boundaries 규약 (길이 N+3):
    b = [0, 프롬프트 끝, step1 끝, ..., stepN 끝, 전체 끝]
    구간 = [프롬프트][Step 1]...[Step N][터미널(정답 문장)]
프롬프트 토큰이 E 에 들어 있고 터미널 구간이 따로 갈라져 있으므로 논문의 세 전이가 전부
boundaries 만으로 정해진다 (텍스트·토크나이저 불필요):

    Step 1 -> 2          : d(E[b[1]-1], E[b[2]-1])        b[1]-1 = 프롬프트 마지막 토큰 = Step 1 직전
    2nd-last -> Last     : d(E[b[N-1]-1], E[b[N]-1])
    Last -> Ans. Marker  : d(E[b[N]-1], E[b[N+1]-1])      b[N+1]-1 = 터미널 직전 토큰

거리: Euclidean, cosine distance(1 - cos) — 논문 §4. 우리 E 는 최종 RMSNorm 이후 값이라
논문의 절대값과는 비교 불가, Δ 부호와 † 패턴만 비교한다.
†: success / fail 의 95% CI(평균 ± 1.96·SE)가 겹치지 않음 (논문 Fig.2 캡션 규칙).

    python metric/step_distance.py --task predict --level BabyAI-GoToObj-v0
    python metric/step_distance.py --task plan --level CustomBabyAI-GoToRedBall-Small-4Dists-v0 --max-episodes 500
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

TRANSITIONS = ("step1_to_2", "second_last_to_last", "last_to_answer")
METRICS = ("euclid", "cosine")
LABELS = {"success": "success", "failure": "fail"}   # 디렉토리명 -> 표기
LAYOUT = "prompt|steps|terminal"                     # extract.chunk_header()["boundary_layout"]


# ------------------------------------------------------------------ 거리

def _dist(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float(), b.float()
    return {"euclid": torch.dist(a, b).item(),
            "cosine": 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0).item()}


def step_end_tokens(E: torch.Tensor, boundaries: list[int]) -> tuple[list[torch.Tensor], int]:
    """h_k := 구간 k 의 마지막 토큰 (k=0 프롬프트, 1..N step). 반환 (h[0..N], N).

    h[0] = E[b[1]-1] 은 Step 1 직전 토큰(논문 h_{t(Step 1)-1}),
    h[k] = E[b[k+1]-1] 은 Step k+1 직전 토큰, h[N] = E[b[N+1]-1] 은 터미널 직전 토큰(h_{t(term)-1})."""
    N = len(boundaries) - 3
    if N < 1:
        raise ValueError(f"boundaries too short for layout {LAYOUT!r}: {boundaries}")
    return [E[boundaries[k + 1] - 1] for k in range(N + 1)], N


def transition_distances(E: torch.Tensor, boundaries: list[int]) -> dict:
    """episode 하나의 세 전이 거리. 정의 안 되는 전이(N<2)는 None."""
    h, N = step_end_tokens(E, boundaries)
    out = {k: None for k in TRANSITIONS}
    out["n_steps"] = N
    if N >= 2:
        out["step1_to_2"] = _dist(h[0], h[1])
        out["second_last_to_last"] = _dist(h[N - 2], h[N - 1])
    out["last_to_answer"] = _dist(h[N - 1], h[N])
    out["rep_norms"] = [v.float().norm().item() for v in h]
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
    """rows: {"success": [...], "fail": [...]} -> table[transition][metric]
    = {success: ci, fail: ci, delta_f_minus_s, dagger}"""
    table = {}
    for tr in TRANSITIONS:
        table[tr] = {}
        for me in METRICS:
            cell = {lab: _ci([r[tr][me] for r in rs if r[tr] is not None])
                    for lab, rs in rows.items()}
            s, f = cell.get("success", {}), cell.get("fail", {})
            if s.get("n") and f.get("n"):
                cell["delta_f_minus_s"] = f["mean"] - s["mean"]
                cell["dagger"] = bool(f["ci_lo"] > s["ci_hi"] or s["ci_lo"] > f["ci_hi"])
            table[tr][me] = cell
    return table


def format_table(table: dict, title: str) -> str:
    head = {"step1_to_2": "Step 1→2", "second_last_to_last": "2nd-last→Last",
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
            cells.append(f"{c['delta_f_minus_s']:+.4g}" + ("†" if c["dagger"] else "")
                         if "delta_f_minus_s" in c else "–")
        lines.append("| | Δ(F–S) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ 실행

def status_dir(hidden_dir: Path, task: str, level: str, status: str) -> Path:
    d = hidden_dir / task / level / status
    if not any(d.glob("chunk_*.pt")):
        raise FileNotFoundError(f"no chunk_*.pt in {d}")
    return d


def load_blob(cf: Path) -> dict:
    """청크 하나가 ~4 GB(256 ep × ~1500 tok × 5120 × bf16)인데 쓰는 건 episode 당 N+1 행뿐이라
    mmap 으로 열어 인덱싱한 행만 디스크에서 읽는다. (구형 저장 포맷이면 일반 로드로 폴백.)"""
    try:
        return torch.load(cf, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(cf, map_location="cpu", weights_only=False)


def run(hidden_dir: Path, task: str, level: str, max_episodes: int | None = None) -> dict:
    rows: dict[str, list[dict]] = {"success": [], "fail": []}
    stats = {"n_seen": 0, "n_bad_boundaries": 0, "layout": None, "label_priority": None}

    for status, lab in LABELS.items():
        d = status_dir(hidden_dir, task, level, status)
        done = False
        for cf in tqdm(sorted(d.glob("chunk_*.pt")), desc=f"{task}/{level}/{status}", unit="chunk"):
            blob = load_blob(cf)
            layout = blob.get("boundary_layout")
            if layout != LAYOUT:
                raise ValueError(f"{cf}: boundary_layout={layout!r}, expected {LAYOUT!r} — "
                                 f"이 스크립트는 서버 extract.py(프롬프트|스텝|터미널) 청크 전용")
            stats["layout"] = layout
            stats["label_priority"] = blob.get("label_priority")
            for seed, ep in blob["episodes"].items():
                stats["n_seen"] += 1
                b = list(ep["boundaries"])
                if len(b) < 4 or b[0] != 0 or b[-1] != ep["E"].shape[0] \
                        or any(x >= y for x, y in zip(b, b[1:])):
                    stats["n_bad_boundaries"] += 1
                    continue
                r = transition_distances(ep["E"], b)
                r["seed"] = seed
                rows[lab].append(r)
                if max_episodes and len(rows[lab]) >= max_episodes:
                    done = True
                    break
            if done:
                break

    norms = [v for rs in rows.values() for r in rs for v in r["rep_norms"]]
    by_steps = {}
    for N in sorted({r["n_steps"] for rs in rows.values() for r in rs}):
        by_steps[N] = aggregate({lab: [r for r in rs if r["n_steps"] == N]
                                 for lab, rs in rows.items()})
    return {"task": task, "level": level, "n": {lab: len(rs) for lab, rs in rows.items()},
            "stats": stats,
            "rep_norm": {"mean": statistics.fmean(norms), "std": statistics.pstdev(norms)} if norms else None,
            "pooled": aggregate(rows), "by_n_steps": by_steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--max-episodes", type=int, default=None, help="status 당 최대 episode 수 (빠른 확인용)")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "latent" / "step_distance")
    args = ap.parse_args()

    res = run(args.hidden_dir, args.task, args.level, args.max_episodes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.out_dir / f"{args.task}_{args.level}"
    base.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [format_table(res["pooled"], f"{args.task}/{args.level} pooled  "
                       f"(success n={res['n']['success']}, fail n={res['n']['fail']})")]
    for N, tb in res["by_n_steps"].items():
        md.append(format_table(tb, f"n_steps={N}"))
    md.append(f"stats: {json.dumps(res['stats'], ensure_ascii=False)}")
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