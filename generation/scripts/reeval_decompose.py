"""decompose jsonl 에서 eval_error 난 행을 고친 파서로 다시 채점해 제자리 갱신한다.

배경: llm-babybench 파서가 "<START>  \\n"(뒤 공백) 을 못 읽어 전체의 45% 가
빈 문자열 → "Input string format not recognized: ''" 로 failure 처리돼 있었다.
decompose_parser.parse_decompose_output 으로 다시 파싱해 evaluator 를 돌리고,
결과가 나온 행만 parsed_llm_output / eval_result / eval_error 를 덮어쓴다.
갱신된 행에는 eval_reparsed=true 를 단다. 원본은 <jsonl>.bak 으로 남긴다.

    python generation/scripts/reeval_decompose.py                       # 전체 재채점 (~10분, 32 proc)
    python generation/scripts/reeval_decompose.py --reuse results.jsonl # 이미 돌린 결과 재사용
    python generation/scripts/reeval_decompose.py --dry-run             # 갱신 안 하고 집계만
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import signal
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _v in (ROOT / "generation" / "third_party" / "llm-babybench", ROOT / "third_party" / "llm-babybench"):
    if (_v / "evaluators").is_dir():          # 서브모듈이 비어 있으면 루트 third_party/ 로 폴백
        sys.path.insert(0, str(_v))
        break
else:
    sys.exit("llm-babybench 를 찾을 수 없음 (generation/third_party/ 또는 third_party/)")

from decompose_parser import parse_decompose_output  # noqa: E402

LEVELS = ["BabyAI-GoToObj-v0", "BabyAI-GoTo-v0", "BabyAI-Synth-v0", "BabyAI-BossLevel-v0"]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


_ev = None


def _init():
    global _ev
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from evaluators import get_evaluator
        _ev = get_evaluator("decompose")
    signal.signal(signal.SIGALRM, _alarm)


def _work(job):
    level, seed, text, timeout = job
    from runner.env_loader import make_env
    pred = parse_decompose_output(text)
    env, res, err = None, None, None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            env = make_env(level, seed)
            signal.alarm(timeout)
            try:
                res = _ev.evaluate(env, pred)
            finally:
                signal.alarm(0)
    except _Timeout:
        err = "_EvalTimeout: evaluator.evaluate() timed out"
    except Exception as e:  # 형식 위반(Unknown subgoal 등), 봇 예외
        err = f"{type(e).__name__}: {e}"
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
    return {"env_name": level, "env_seed": seed, "parsed": pred, "eval_result": res, "eval_error": err}


def reevaluate(rows, timeout, procs):
    """apply_async + get(timeout): 봇이 워커를 세그폴트로 죽이면 imap 은 영원히 기다리므로
    (실측: 56,410건 중 41건) 결과마다 대기 상한을 두고, 안 오면 worker_crashed 로 기록한다."""
    jobs = [(r["env_name"], r["env_seed"], r["all_llm_output"], timeout) for r in rows]
    out, t0 = {}, time.time()
    with Pool(procs, initializer=_init) as pool:
        pending = [(j, pool.apply_async(_work, (j,))) for j in jobs]
        for i, (j, ar) in enumerate(pending, 1):
            try:
                r = ar.get(timeout=timeout * 3)
            except Exception as e:                       # multiprocessing.TimeoutError 등
                r = {"env_name": j[0], "env_seed": j[1], "parsed": parse_decompose_output(j[2]),
                     "eval_result": None, "eval_error": f"worker_crashed_or_hung: {type(e).__name__}"}
            out[(r["env_name"], r["env_seed"])] = r
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, default=ROOT / "generation" / "trajectory" / "decompose_no_thinking.jsonl")
    ap.add_argument("--reuse", type=Path, default=None, help="이전 재채점 결과 jsonl (env_name, env_seed, parsed, eval_result, eval_error)")
    ap.add_argument("--save-results", type=Path, default=None, help="재채점 결과를 따로 저장 (--reuse 용)")
    ap.add_argument("--eval-timeout", type=int, default=60)
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.jsonl.open(encoding="utf-8") if l.strip()]
    targets = [r for r in rows if not r.get("skipped") and not r.get("eval_result")]
    print(f"{len(rows)} rows, {len(targets)} with eval_error -> re-evaluate")

    cached = {}
    if args.reuse:
        for l in args.reuse.open(encoding="utf-8"):
            r = json.loads(l)
            cached[(r["env_name"], r["env_seed"])] = r
        # 파서가 바뀌었으면 캐시가 낡은 것 — parsed 가 같은 것만 재사용
        stale = [r for r in targets if (c := cached.get((r["env_name"], r["env_seed"]))) is None
                 or c["parsed"] != parse_decompose_output(r["all_llm_output"])]
        print(f"reuse: {len(targets) - len(stale)} cached, {len(stale)} to run")
        todo = stale
    else:
        todo = targets
    results = {k: v for k, v in cached.items()}
    if todo:
        results.update(reevaluate(todo, args.eval_timeout, args.procs))
    if args.save_results:
        with args.save_results.open("w", encoding="utf-8") as f:
            for r in targets:
                k = (r["env_name"], r["env_seed"])
                if k in results:
                    f.write(json.dumps(results[k]) + "\n")

    # 병합 + 집계
    stat = defaultdict(Counter)
    remaining_err = Counter()
    n_changed = 0
    for r in rows:
        lv = r["env_name"]
        if r.get("skipped"):
            continue
        stat[lv]["n"] += 1
        res = r.get("eval_result")
        if res:
            stat[lv]["old_CR"] += res["CR"] == 1; stat[lv]["old_PR"] += res["PR"] == 1
            stat[lv]["new_CR"] += res["CR"] == 1; stat[lv]["new_PR"] += res["PR"] == 1
            continue
        rr = results.get((lv, r["env_seed"]))
        if rr is None or rr["eval_result"] is None:
            remaining_err[(rr or {}).get("eval_error") or r.get("eval_error") or "?"] += 1
            continue
        stat[lv]["new_CR"] += rr["eval_result"]["CR"] == 1
        stat[lv]["new_PR"] += rr["eval_result"]["PR"] == 1
        if not args.dry_run:
            r["parsed_llm_output"] = rr["parsed"]
            r["eval_result"] = rr["eval_result"]
            r["eval_error"] = None
            r["eval_reparsed"] = True
        n_changed += 1

    print(f"\n{'level':22} {'n':>6} | {'CR old -> new':>18} | {'PR old -> new':>18}")
    for lv in LEVELS:
        s = stat[lv]
        if not s["n"]:
            continue
        print(f"{lv:22} {s['n']:6d} | {s['old_CR']/s['n']*100:7.2f}% -> {s['new_CR']/s['n']*100:7.2f}% "
              f"| {s['old_PR']/s['n']*100:7.2f}% -> {s['new_PR']/s['n']*100:7.2f}%")
    print(f"\nrows to update: {n_changed}, still eval_error: {sum(remaining_err.values())}")
    for k, v in remaining_err.most_common(6):
        print(f"  {v:6d} {k[:80]!r}")

    if args.dry_run:
        print("dry-run: jsonl 미변경")
        return
    bak = args.jsonl.with_suffix(args.jsonl.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(args.jsonl, bak)
        print(f"backup -> {bak}")
    tmp = args.jsonl.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, args.jsonl)
    print(f"updated {args.jsonl} ({n_changed} rows)")


if __name__ == "__main__":
    main()
