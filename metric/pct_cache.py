"""hidden state 청크에서 구간별 q% 위치의 토큰 벡터만 뽑아 작은 캐시로 저장한다.
거리 계산에 실제로 필요한 건 구간마다 몇 개 위치의 벡터뿐이므로 (구간 6개 x 퍼센트 8개 = 48행) 
그 행만 한 번 뽑아 두고 metric/pct_distance.py 가 이 캐시만 읽는다. 

입력  <hidden_dir>/<task>/<level>/<status>/chunk_XXXX.pt         (inference/extract.py 출력)
출력  <out_dir>/<task>/<level>/<status>/<tag>/chunk_XXXX.pt      (입력 청크와 1:1, 같은 번호)
      tag = "p" + "-".join(pct)   예: p5-10-30-50-70-90-95-100
      퍼센트 목록을 바꾸면 tag 가 달라져 옆에 따로 쌓인다. 저장하지 않은 % 는 다시 못 만든다
      (전체 E 를 다시 읽어야 한다).

구간과 위치
    구간 = extract.gen_view 기준 [Step 1]...[Step N][answer]  (프롬프트 제외, N+1 개).
    위치 = predict/probing.py 의 pct_position(q, n): 구간 길이 n 의 q% 지점 토큰 (t = ⌈q·n/100⌉, 0-based t-1).
    q=100 이 구간 마지막 토큰이고 probing 의 "구간 마지막 토큰" 과 같은 행이다. 짧은 구간에서는
    서로 다른 q 가 같은 토큰을 가리킬 수 있어 (예: n=15 면 95% 와 100% 가 같은 토큰) 위치 인덱스를
    "pos" 에 그대로 남긴다 — 거리 계산 쪽에서 같은 토큰 쌍(거리 0)을 걸러낼 수 있어야 한다.
    프롬프트 구간은 마지막 토큰 하나만 "prompt_last" 에 따로 둔다 (prompt→Step 1 전이용).

출력 청크 내용
    {"pct": [q...], "tag", "model", "step_headers", "hidden_dir", "src_chunk", "seg_names",
     "seeds":       [seed...]                        (n_ep,)
     "V":           (n_ep, n_seg, n_pct, d) float16  구간 t 의 q 번째 위치 벡터
     "prompt_last": (n_ep, d) float16                프롬프트 마지막 토큰
     "seg_len":     (n_ep, n_seg) int32              구간 토큰 수
     "pos":         (n_ep, n_seg, n_pct) int32       구간 안 0-based 위치 (V 가 어느 토큰인지)
     "n_tokens_prompt": (n_ep,) int32}
    구간 수(n_seg)는 파일 안에서 같다. 다른 n_seg 의 에피소드는 건너뛰고 사유를 센다 (n_seg_mismatch).
    에피소드당 6x8x5120x2B ≈ 0.5MB, 청크(256) 당 ≈ 125MB.

이미 있는 출력 청크는 건너뛴다 (중단 후 재실행 가능). --overwrite 로 다시 쓴다.

Usage :
    python metric/pct_cache.py --task decompose --level BabyAI-GoTo-v0 --limit 20          # 파일럿
    python metric/pct_cache.py --task decompose --level BabyAI-GoTo-v0
    python metric/pct_cache.py --task plan --level CustomBabyAI-GoToRedBall-Small-4Dists-v0
    python metric/pct_cache.py --task plan --level ... --pct 10 20 40 60 80 100            # 다른 % 조합
    python metric/pct_cache.py --task plan --level ... --hidden-dir latent/hidden_states    # 마커만 뗀 버전
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "predict"))

from extract import gen_view, seg_labels          # noqa: E402
from probing import pct_position                  # noqa: E402  (probing/spectral 과 같은 위치 규칙)

STATUSES = ("success", "failure")
DEFAULT_PCT = (5, 10, 30, 50, 70, 90, 95, 100)
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def make_tag(pct: list[int]) -> str:
    return "p" + "-".join(str(q) for q in pct)


def check_pct(pct: list[int]) -> list[int]:
    if sorted(set(pct)) != list(pct) or not all(1 <= q <= 100 for q in pct):
        raise SystemExit(f"--pct 는 1..100 사이 오름차순, 중복 없이: {pct}")
    return list(pct)


def pick_episode(ep: dict, pct: list[int], n_seg_expect: int | None, dtype: torch.dtype):
    """에피소드 하나 → (V, prompt_last, seg_len, pos, n_prompt) 또는 (None, 사유)."""
    b = [int(x) for x in ep["boundaries"]]
    E = ep["E"]
    if len(b) < 4 or b[0] != 0 or b[-1] != E.shape[0]:
        return None, "bad_boundaries"
    if any(x >= y for x, y in zip(b, b[1:])):
        return None, "nonmonotonic_boundaries"

    E_gen, seg = gen_view(ep)                      # 프롬프트 제외, seg = [0, step1 끝, ..., 전체 끝]
    n_seg = len(seg) - 1
    if n_seg_expect is not None and n_seg != n_seg_expect:
        return None, "n_seg_mismatch"

    seg_len = torch.tensor([seg[t + 1] - seg[t] for t in range(n_seg)], dtype=torch.int32)
    pos = torch.tensor([[pct_position(q, int(seg_len[t])) for q in pct] for t in range(n_seg)],
                       dtype=torch.int32)                                   # (n_seg, n_pct)
    idx = torch.tensor([[seg[t] + int(pos[t, j]) for j in range(len(pct))] for t in range(n_seg)])
    V = E_gen[idx.reshape(-1)].to(dtype).reshape(n_seg, len(pct), -1).clone()   # 인덱싱 = 복사 → E 는 해제됨
    prompt_last = E[b[1] - 1].to(dtype).clone()
    return (V, prompt_last, seg_len, pos, b[1]), None


def cache_status(task: str, level: str, status: str, pct: list[int], hidden_dir: Path,
                 out_dir: Path, dtype: torch.dtype, limit: int | None, overwrite: bool) -> Counter:
    src_dir = hidden_dir / task / level / status
    chunks = sorted(src_dir.glob("chunk_*.pt"))
    if not chunks:
        raise SystemExit(f"no chunks: {src_dir}")
    tag = make_tag(pct)
    dst_dir = out_dir / task / level / status / tag
    dst_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    n_seg_expect = None
    n_done = 0
    for src in chunks:
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            stats["chunks_skipped_existing"] += 1
            continue
        t0 = time.time()
        d = torch.load(src, map_location="cpu", weights_only=False)
        header = {k: v for k, v in d.items() if k != "episodes"}

        seeds, Vs, Ps, Ls, Qs, NPs = [], [], [], [], [], []
        for seed, ep in d["episodes"].items():
            if limit is not None and n_done >= limit:
                break
            got, reason = pick_episode(ep, pct, n_seg_expect, dtype)
            if got is None:
                stats[reason] += 1
                continue
            V, pl, sl, pos, n_prompt = got
            if n_seg_expect is None:
                n_seg_expect = V.shape[0]      # 첫 에피소드의 구간 수를 파일 전체 기준으로
            seeds.append(seed); Vs.append(V); Ps.append(pl); Ls.append(sl); Qs.append(pos); NPs.append(n_prompt)
            n_done += 1
        del d

        if Vs:
            seg_names = seg_labels(list(range(n_seg_expect + 1)))     # 길이만 맞으면 된다
            torch.save({
                "pct": pct, "tag": tag,
                "model": header.get("model"), "step_headers": header.get("step_headers"),
                "hidden_dir": str(hidden_dir), "src_chunk": str(src),
                "seg_names": seg_names,
                "seeds": seeds,
                "V": torch.stack(Vs),                      # (n_ep, n_seg, n_pct, d)
                "prompt_last": torch.stack(Ps),            # (n_ep, d)
                "seg_len": torch.stack(Ls),                # (n_ep, n_seg)
                "pos": torch.stack(Qs),                    # (n_ep, n_seg, n_pct)
                "n_tokens_prompt": torch.tensor(NPs, dtype=torch.int32),
            }, dst)
            stats["episodes"] += len(Vs)
            stats["chunks_written"] += 1
        print(f"  {status}/{src.name}: {len(Vs)} ep  ({time.time() - t0:.1f}s)", flush=True)
        if limit is not None and n_done >= limit:
            break
    return stats


def main():
    ap = argparse.ArgumentParser(description="구간별 q% 위치 hidden state 캐시")
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True, help="env_name 그대로")
    ap.add_argument("--status", nargs="+", default=list(STATUSES), choices=STATUSES)
    ap.add_argument("--pct", type=int, nargs="+", default=list(DEFAULT_PCT), metavar="Q")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states_notitle",
                    help="extract.py 출력 루트 (기본: 헤더 줄 전체를 뗀 notitle)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "latent" / "pct_vectors")
    ap.add_argument("--dtype", default="float16", choices=list(DTYPES))
    ap.add_argument("--limit", type=int, default=None, help="status 마다 앞에서 이 개수만 (파일럿)")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 출력 청크도 다시 쓴다")
    a = ap.parse_args()

    pct = check_pct(a.pct)
    print(f"task={a.task} level={a.level} pct={pct} tag={make_tag(pct)}\n"
          f"  hidden_dir={a.hidden_dir}\n  out_dir={a.out_dir}", flush=True)
    for status in a.status:
        stats = cache_status(a.task, a.level, status, pct, a.hidden_dir, a.out_dir,
                             DTYPES[a.dtype], a.limit, a.overwrite)
        print(f"{status}: {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
