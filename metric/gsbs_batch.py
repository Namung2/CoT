"""데이터셋 전체에 step 안 GSBS 를 돌려 에피소드마다 한 줄씩 jsonl 로 쌓는다.

metric/gsbs.py 는 에피소드 하나를 골라 그림까지 그리는 도구다. 여기서는 그림을
빼고 "스텝 안에서 몇 개로 나뉘고 어디서 나뉘는지"만 남긴다. 집계는
metric/gsbs_summary.py 가 한다.

    출력: <out_dir>/<task>/<level>/<status>/<config>.jsonl
    config: spectral_k8_data | spectral_k8_max | tokens ...

한 줄 = 에피소드 하나:
    {"seed": 0, "n_tokens": 566, "n_segments": 7,
     "segments": [{"t": 0, "name": "Step 1", "n_tokens": 53, "skipped": false,
                   "n_states": 2, "kmax": 26, "kmax_saturated": false,
                   "splits": [12], "rel_pos": [0.226], "seconds": 0.14}, ...]}

같은 (에피소드, config) 가 이미 파일에 있으면 건너뛴다 → 중간에 끊겨도 다시 돌리면 이어서 한다.

k 절약: 같은 sign 모드에서 k=4, 8 의 e 는 k=16 e 의 앞 블록을 자른 것과 정확히
같다 (e = [σ₁v₁; σ₂v₂; …] 순서). 그래서 sign 모드마다 최대 k 로 한 번만 누적
스펙트럴을 만들고 슬라이스한다. GSBS 는 full-rank 축소 뒤에 돌아서 k 와 무관하게
같은 시간이 걸린다.

    python metric/gsbs_batch.py --task decompose --level BabyAI-GoToObj-v0
    python metric/gsbs_batch.py --task decompose --level BabyAI-GoToObj-v0 --status success -k 8 --sign-mode data
    python metric/gsbs_batch.py --task decompose --level BabyAI-GoToObj-v0 --limit 3   # 연습
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "visual"))
sys.path.insert(0, str(ROOT / "metric"))

from extract import gen_view, load_chunk, seg_labels                       # noqa: E402
from spectral import (K_EIG, SCALE, SIGN_MODE, SIGN_MODES, load_hidden_states,  # noqa: E402
                      CumulativeSpectral, tokens_cumulative)
from gsbs import run_gsbs_per_segment, substep_splits                       # noqa: E402

DEFAULT_KS = (4, 8, 16)
DEFAULT_SIGNS = ("data", "max")


def config_tag(inp: str, k: int | None = None, sign_mode: str | None = None,
               kmax: int | None = None) -> str:
    """jsonl 파일명. gsbs.py 의 itag 와 같은 규칙 (reduce/step 접미사만 뺌).

    kmax 를 제한하면 결과가 달라지므로 접미사 _kmax{N} 을 붙여 기본 실행과 섞이지 않게 한다."""
    tag = "tokens" if inp == "tokens" else f"spectral_k{k}_{sign_mode}"
    if kmax is not None:
        tag += f"_kmax{kmax}"
    return tag


def parse_config_tag(tag: str) -> dict:
    """config_tag 의 역함수. summary 가 파일명에서 설정을 되읽을 때 쓴다."""
    kmax = None
    if "_kmax" in tag:
        tag, km = tag.rsplit("_kmax", 1)
        kmax = int(km)
    if tag == "tokens":
        return {"input": "tokens", "k": None, "sign_mode": None, "kmax": kmax}
    _, k, sign = tag.split("_", 2)
    return {"input": "spectral", "k": int(k[1:]), "sign_mode": sign, "kmax": kmax}


def done_seeds(path: Path) -> set[int]:
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(int(json.loads(line)["seed"]))
    return out


def segment_records(X_in: torch.Tensor, seg: list[int], kmax: int | None,
                    reduce: int | None, min_tokens: int,
                    max_tokens: int | None = None) -> list[dict]:
    """구간마다 GSBS → 구간별 레코드. gsbs.py 의 per_segment + substep_splits 를 합친 모양.

    max_tokens 보다 긴 구간은 GSBS 를 안 돌리고 skipped=True, skip_reason="too_long" 으로
    남긴다. kmax 가 구간 토큰수/2 로 잡히면 GSBS 비용이 대략 토큰수⁴ 으로 늘어서
    1600 토큰짜리 구간 하나에 시간 단위가 걸린다."""
    long_segs = set()
    if max_tokens is not None:
        long_segs = {t for t, (a, b) in enumerate(zip(seg, seg[1:])) if b - a > max_tokens}
    if long_segs:
        # 긴 구간을 뺀 경계로 돌리면 인덱스가 어긋나므로, 구간을 하나씩 따로 돌린다.
        per, all_b = [], []
        names = seg_labels(seg)
        for t, (a, b) in enumerate(zip(seg, seg[1:])):
            if t in long_segs:
                per.append({"segment": t, "name": names[t], "start": int(a), "end": int(b),
                            "n_tokens": int(b - a), "skipped": True, "kmax": 0,
                            "kmax_saturated": False, "n_states": 1, "bounds": [],
                            "seconds": 0.0, "skip_reason": "too_long"})
                continue
            b_arr, one = run_gsbs_per_segment(X_in[a:b], [0, b - a], kmax, reduce,
                                              statewise=False, min_tokens=min_tokens)
            r = one[0]
            r.update({"segment": t, "name": names[t], "start": int(a), "end": int(b),
                      "bounds": [int(a + x) for x in r["bounds"]]})
            per.append(r)
            all_b.extend(r["bounds"])
        pred_b = np.asarray(sorted(all_b), dtype=int)
    else:
        pred_b, per = run_gsbs_per_segment(X_in, seg, kmax, reduce, statewise=False,
                                           min_tokens=min_tokens)
    splits = substep_splits(pred_b, seg)
    out = []
    for r, sp in zip(per, splits):
        out.append({"t": r["segment"], "name": r["name"], "n_tokens": r["n_tokens"],
                    "skipped": r["skipped"],
                    "skip_reason": r.get("skip_reason", "too_short" if r["skipped"] else None),
                    "n_states": r["n_states"], "kmax": r["kmax"],
                    "kmax_saturated": r["kmax_saturated"],
                    "splits": sp["splits"], "rel_pos": sp["rel_pos"],
                    "seconds": r["seconds"]})
    return out


def run(hidden_dir: Path, out_dir: Path, task: str, level: str, status: str,
        ks: list[int], sign_modes: list[str], with_tokens: bool,
        kmax: int | None, reduce: int | None, min_tokens: int,
        max_tokens: int | None = None, limit: int | None = None):
    chunk_files = load_hidden_states(hidden_dir.resolve(), task, level, status)
    o_dir = out_dir / task / level / status
    o_dir.mkdir(parents=True, exist_ok=True)

    # config → (파일, 이미 끝난 seed)
    configs: list[tuple[str, dict]] = []
    if with_tokens:
        configs.append((config_tag("tokens", kmax=kmax), {"input": "tokens"}))
    for sm in sign_modes:
        for k in ks:
            configs.append((config_tag("spectral", k, sm, kmax=kmax),
                            {"input": "spectral", "k": k, "sign_mode": sm}))
    files = {tag: o_dir / f"{tag}.jsonl" for tag, _ in configs}
    done = {tag: done_seeds(p) for tag, p in files.items()}
    k_top = max(ks) if ks else 0

    header = f"{task}/{level}/{status}"
    print(f"[{header}] {len(chunk_files)} chunks, configs={[t for t, _ in configs]}")
    n_ep = n_skip = n_seen = 0
    t_all = time.perf_counter()
    cs = {sm: CumulativeSpectral(k_top, SCALE, sm) for sm in sign_modes} if ks else {}

    for cf in chunk_files:
        chunk = load_chunk(cf)
        for seed, ep in chunk.items():
            if limit is not None and n_seen >= limit:
                break
            n_seen += 1
            todo = [(tag, c) for tag, c in configs if seed not in done[tag]]
            if not todo:
                n_skip += 1
                continue
            n_ep += 1
            E, seg = gen_view(ep)
            d = E.shape[1]
            t0 = time.perf_counter()

            # sign 모드마다 최대 k 로 한 번만 누적 → 작은 k 는 슬라이스
            e_by_sign: dict[str, torch.Tensor] = {}
            need_signs = {c["sign_mode"] for _, c in todo if c["input"] == "spectral"}
            for sm in need_signs:
                e_by_sign[sm], _ = tokens_cumulative(E, seg, k_top, SCALE, sm, cs=cs[sm])

            for tag, c in todo:
                if c["input"] == "tokens":
                    X_in = E.float()
                else:
                    X_in = e_by_sign[c["sign_mode"]][:, : c["k"] * d]
                segs = segment_records(X_in, seg, kmax, reduce, min_tokens, max_tokens)
                rec = {"seed": int(seed), "n_tokens": int(E.shape[0]),
                       "n_segments": len(seg) - 1, "segments": segs}
                with files[tag].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[tag].add(seed)

            n_split = sum(len(s["splits"]) for s in segs)   # 마지막 config 기준 요약
            print(f"  seed={seed:<6} tok={E.shape[0]:<5} configs={len(todo)} "
                  f"({time.perf_counter() - t0:.1f}s)  last cfg splits={n_split}")
        del chunk
        if limit is not None and n_seen >= limit:
            break

    print(f"[{header}] {n_ep} episodes run, {n_skip} already done, "
          f"{time.perf_counter() - t_all:.0f}s -> {o_dir}")


def main():
    sys.stdout.reconfigure(line_buffering=True)   # 리다이렉트돼도 에피소드마다 로그가 바로 보이게
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--status", nargs="+", default=["success", "failure"],
                    choices=["success", "failure"])
    ap.add_argument("-k", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--sign-mode", nargs="+", default=list(DEFAULT_SIGNS), choices=list(SIGN_MODES))
    ap.add_argument("--no-tokens", dest="tokens", action="store_false",
                    help="원본 E(tokens) 기준선 config 를 빼고 spectral 만")
    ap.add_argument("--kmax", type=int, default=None, help="gsbs.py 와 동일. 기본 구간 토큰수/2")
    ap.add_argument("--reduce", type=int, default=0, help="gsbs.py 와 동일. 0=full-rank 무손실")
    ap.add_argument("--min-tokens", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="이보다 긴 구간은 GSBS 를 건너뛴다 (skip_reason=too_long). "
                         "kmax 기본값이면 600 토큰 구간 하나에 config 당 수 분, 1600 토큰이면 시간 단위")
    ap.add_argument("--limit", type=int, default=None,
                    help="status 마다 데이터셋 앞에서 이 개수만 본다 (이미 끝난 것도 센다)")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "latent" / "gsbs" / "batch")
    a = ap.parse_args()

    reduce = None if a.reduce < 0 else a.reduce
    for status in dict.fromkeys(a.status):
        run(a.hidden_dir, a.out_dir, a.task, a.level, status,
            ks=sorted(set(a.k)), sign_modes=list(dict.fromkeys(a.sign_mode)),
            with_tokens=a.tokens, kmax=a.kmax, reduce=reduce,
            min_tokens=a.min_tokens, max_tokens=a.max_tokens, limit=a.limit)


if __name__ == "__main__":
    main()
