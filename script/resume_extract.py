"""중단된 extract 를 이어서 완성한다 — 이미 디스크에 있는 청크는 건드리지 않는다.

extract_run 은 256개가 모일 때마다 청크를 쓰므로, 도중에 죽으면
  (a) 버퍼에 있던 에피소드 (meta 에는 chunk 번호가 적혔지만 파일이 없음)
  (b) 아직 순회 안 한 에피소드
가 빠진다. 이 스크립트는 meta.jsonl 과 실제 chunk_*.pt 를 대조해 (a)+(b) 만
다시 뽑고, 새 청크를 기존 번호 다음부터 이어 쓰고, meta.jsonl 을 에피소드
순서대로 다시 만든다. 처음부터 다시 돌리는 것과 결과가 같아야 한다
(같은 process_episode, 같은 순서). 단 청크 경계 위치만 다르다.

    CUDA_VISIBLE_DEVICES=0 python script/resume_extract.py --task decompose \\
        --level BabyAI-GoToObj-v0 --limit 10000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

from extract import chunk_header, load_episodes, process_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--mode", default="no_thinking")
    ap.add_argument("--limit", type=int, default=None, help="원래 실행에 준 --limit 과 같아야 한다")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "generation" / "trajectory")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    args = ap.parse_args()

    src = args.data_dir / f"{args.task}_{args.mode}.jsonl"
    episodes = [e for e in load_episodes(src)
                if e.get("task") == args.task and e.get("env_name") == args.level]
    if args.limit is not None:
        episodes = episodes[:args.limit]
    run_dir = args.hidden_dir / args.task / args.level
    meta_path = run_dir / "meta.jsonl"
    if not meta_path.is_file():
        sys.exit(f"no meta.jsonl under {run_dir} — 처음부터 돌리는 경우는 inference/main.py 를 쓴다")

    # 디스크에 실제로 있는 청크 번호
    have = {s: {int(p.stem.split("_")[1]) for p in (run_dir / s).glob("chunk_*.pt")}
            for s in ("success", "failure")}
    next_idx = {s: (max(have[s]) + 1 if have[s] else 0) for s in have}

    # 기존 meta 중 "청크 파일이 실제로 있는" 에피소드만 완료로 인정
    old_meta, done = {}, {}
    for line in meta_path.open(encoding="utf-8"):
        m = json.loads(line)
        if m.get("extract_skipped") is None and m.get("chunk") in have[m["status"]]:
            old_meta[m["env_seed"]] = m
            done[m["env_seed"]] = m["status"]
    print(f"{len(episodes)} episodes in scope; {len(done)} already on disk "
          f"(success chunks {sorted(have['success'])[:1]}..{max(have['success'], default=-1)}, "
          f"failure chunks {len(have['failure'])}); next chunk idx {next_idx}")

    todo = [e for e in episodes if e["env_seed"] not in done]
    print(f"re-extracting {len(todo)} episodes")

    buffers = {"success": {}, "failure": {}}
    new_meta = {}
    reasons = Counter()

    def flush(status):
        if not buffers[status]:
            return
        out = run_dir / status / f"chunk_{next_idx[status]:04d}.pt"
        torch.save({"episodes": buffers[status], **chunk_header()}, out)
        print(f"  wrote {out.relative_to(ROOT)} ({len(buffers[status])} episodes)")
        buffers[status] = {}
        next_idx[status] += 1

    for e in tqdm(todo, desc="resume", unit="episode"):
        meta, payload = process_episode(e, args.task, src.name)
        if payload is None:
            reasons[meta["extract_skipped"]] += 1
        else:
            status = meta["status"]
            buffers[status][e["env_seed"]] = payload
            meta["chunk"] = next_idx[status]
            if len(buffers[status]) >= args.chunk:
                flush(status)
        new_meta[e["env_seed"]] = meta
    for s in buffers:
        flush(s)

    # meta.jsonl 을 에피소드 순서대로 재작성 (완료분은 옛 줄, 나머지는 새 줄)
    tmp = meta_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for e in episodes:
            m = old_meta.get(e["env_seed"]) or new_meta[e["env_seed"]]
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(tmp, meta_path)

    saved = Counter(m["status"] for m in list(old_meta.values()) + list(new_meta.values())
                    if m.get("extract_skipped") is None)
    print(f"done: success {saved['success']} / failure {saved['failure']}, "
          f"skipped this run {sum(reasons.values())}: {dict(reasons)}")


if __name__ == "__main__":
    main()
