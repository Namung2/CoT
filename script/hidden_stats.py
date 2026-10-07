#!/usr/bin/env python
"""latent/hidden_states 통계: task·level 마다 성공/실패 개수, 스텝 개수, 스텝 당 토큰 개수.

입력 (extract.py 출력):
    <root>/<task>/<level>/meta.jsonl            궤적 전체 (추출 안 된 것도 포함, extract_skipped 사유)
    <root>/<task>/<level>/<status>/chunk_*.pt   {"episodes": {seed: {"E", "boundaries", ...}}}

boundaries 규약 (길이 N+3): [0, 프롬프트끝, step1끝, ..., stepN끝, 전체끝]
    → n_steps = len(b) - 3, step k 토큰 수 = b[k+1] - b[k], 터미널 = b[-1] - b[-2]

.pt 는 mmap 으로 열어 boundaries 만 읽는다 (E 는 건드리지 않아 청크당 수십 ms).

출력 (level 마다):
    [1] 성공/실패 개수   meta.jsonl 기준 전체 + .pt 에 실제 추출된 개수 + extract_skipped 사유
    [2] 스텝 개수        status 별 n_steps 분포·평균
    [3] 스텝 당 토큰 수  status 별 prompt / step_1..N / answer(터미널) 구간 토큰 수 (mean±std, median, min–max)

    python script/hidden_stats.py
    python script/hidden_stats.py --task plan --level CustomBabyAI-GoToRedBall-Small-4Dists-v0
    python script/hidden_stats.py --by-nsteps            # 스텝 수가 같은 에피소드끼리 나눠서 [3]
    python script/hidden_stats.py --csv out/hidden_stats.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HIDDEN = ROOT / "latent" / "hidden_states"
STATUSES = ("success", "failure")


# ───────────────────────── 읽기 ─────────────────────────

def read_meta(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_boundaries(level_dir: Path) -> dict[str, list[list[int]]]:
    """status → [boundaries, ...]. 규약에 안 맞는 에피소드는 세고 뺀다."""
    out, bad = {}, Counter()
    for status in STATUSES:
        bs = []
        for p in sorted(glob.glob(str(level_dir / status / "chunk_*.pt"))):
            d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            for ep in d["episodes"].values():
                b = [int(x) for x in ep["boundaries"]]
                if len(b) < 4 or b[0] != 0:
                    bad["bad_boundaries"] += 1
                    continue
                if any(x >= y for x, y in zip(b, b[1:])):
                    bad["nonmonotonic"] += 1
                    continue
                bs.append(b)
        out[status] = bs
    if bad:
        print(f"  ! 제외된 에피소드: {dict(bad)}")
    return out


# ───────────────────────── 통계 ─────────────────────────

def describe(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def fmt(d: dict) -> str:
    if d["n"] == 0:
        return "-"
    return f"{d['mean']:7.1f} ± {d['std']:5.1f}  med {d['median']:6.1f}  [{d['min']:.0f}, {d['max']:.0f}]  (n={d['n']})"


def segment_tokens(bs: list[list[int]]) -> dict[str, list[int]]:
    """구간 이름 → 토큰 수 리스트. step_k 는 그 스텝이 있는 에피소드만 모은다."""
    seg = defaultdict(list)
    for b in bs:
        seg["prompt"].append(b[1] - b[0])
        for k in range(1, len(b) - 2):
            seg[f"step_{k}"].append(b[k + 1] - b[k])
        seg["answer"].append(b[-1] - b[-2])
        seg["steps_only"].append(b[-2] - b[1])
        seg["output(steps+answer)"].append(b[-1] - b[1])
    return seg


def seg_order(names) -> list[str]:
    steps = sorted((n for n in names if n.startswith("step_")), key=lambda s: int(s.split("_")[1]))
    return ["prompt", *steps, "answer", "steps_only", "output(steps+answer)"]


# ───────────────────────── 레벨 하나 ─────────────────────────

def level_report(task: str, level: str, level_dir: Path, by_nsteps: bool, rows: list[dict]):
    print(f"\n{'=' * 100}\n{task} / {level}\n{'=' * 100}")

    # [1] 성공/실패 개수
    meta = read_meta(level_dir / "meta.jsonl") if (level_dir / "meta.jsonl").exists() else []
    bs_by_status = read_boundaries(level_dir)

    print("[1] 성공/실패 개수")
    print(f"    {'':12s} {'meta 전체':>10s} {'추출됨(.pt)':>12s} {'skip':>8s}")
    all_status = Counter(m.get("status") for m in meta)
    extracted_meta = Counter(m.get("status") for m in meta if "chunk" in m and not m.get("extract_skipped"))
    for status in (*STATUSES, None):
        tot = all_status.get(status, 0)
        if tot == 0 and status is None:
            continue
        ext = len(bs_by_status.get(status, [])) if status else 0
        name = status or "(no status)"
        print(f"    {name:12s} {tot:10d} {ext:12d} {tot - ext:8d}")
        rows.append(dict(task=task, level=level, status=name, kind="count",
                         segment="", n_total=tot, n_extracted=ext))
    n_tot = sum(all_status.values())
    n_ok = sum(all_status.get(s, 0) for s in STATUSES)
    if n_ok:
        print(f"    success rate: {all_status.get('success', 0) / n_ok:.3f} (status 있는 {n_ok}개 중)"
              + (f", {len(bs_by_status['success']) / max(1, sum(len(v) for v in bs_by_status.values())):.3f} (추출됨 기준)"))
    for status in STATUSES:   # meta 의 추출 수와 .pt 에피소드 수가 다르면 알린다
        if extracted_meta.get(status, 0) != len(bs_by_status[status]):
            print(f"    ! {status}: meta 추출 {extracted_meta.get(status, 0)} ≠ .pt {len(bs_by_status[status])}")
    skipped = Counter(m.get("extract_skipped") for m in meta if m.get("extract_skipped"))
    if skipped:
        print("    extract_skipped:", ", ".join(f"{k}={v}" for k, v in skipped.most_common()))

    # [2] 스텝 개수
    print("[2] 스텝 개수 (n_steps)")
    for status in STATUSES:
        ns = [len(b) - 3 for b in bs_by_status[status]]
        d = describe(ns)
        dist = dict(sorted(Counter(ns).items()))
        print(f"    {status:8s} {fmt(d)}")
        print(f"    {'':8s} 분포 {dist}")
        rows.append(dict(task=task, level=level, status=status, kind="n_steps", segment="", **d,
                         dist=json.dumps(dist)))

    # [3] 스텝 당 토큰 수
    print("[3] 구간 당 토큰 수 (step_k = k번째 스텝 구간, answer = 터미널 구간)")
    for status in STATUSES:
        groups = {"all": bs_by_status[status]}
        if by_nsteps:
            g = defaultdict(list)
            for b in bs_by_status[status]:
                g[len(b) - 3].append(b)
            groups = {f"n_steps={k}": v for k, v in sorted(g.items())}
        for gname, bs in groups.items():
            print(f"  - {status} / {gname} ({len(bs)} episodes)")
            seg = segment_tokens(bs)
            for name in seg_order(seg):
                d = describe(seg[name])
                print(f"    {name:22s} {fmt(d)}")
                rows.append(dict(task=task, level=level, status=status, kind="tokens",
                                 group=gname, segment=name, **d))


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_HIDDEN)
    ap.add_argument("--task", action="append", help="decompose/plan/predict (여러 번 가능, 기본 전체)")
    ap.add_argument("--level", action="append", help="env_name (여러 번 가능, 기본 전체)")
    ap.add_argument("--by-nsteps", action="store_true", help="[3] 을 n_steps 가 같은 에피소드끼리 나눠서")
    ap.add_argument("--csv", type=Path, default=None, help="모든 수치를 CSV 로도 저장")
    args = ap.parse_args()

    tasks = args.task or sorted(p.name for p in args.root.iterdir() if p.is_dir())
    rows: list[dict] = []
    for task in tasks:
        task_dir = args.root / task
        if not task_dir.is_dir():
            print(f"! no such task dir: {task_dir}", file=sys.stderr)
            continue
        levels = args.level or sorted(p.name for p in task_dir.iterdir() if p.is_dir())
        for level in levels:
            level_dir = task_dir / level
            if not level_dir.is_dir():
                continue
            level_report(task, level, level_dir, args.by_nsteps, rows)

    if args.csv and rows:
        cols = ["task", "level", "status", "kind", "group", "segment",
                "n_total", "n_extracted", "n", "mean", "std", "median", "min", "max", "dist"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved {len(rows)} rows → {args.csv}")


if __name__ == "__main__":
    main()
