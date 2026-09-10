"""hidden_states 청크에서 episode 몇 개만 뽑아 같은 디렉토리 구조로 저장한다.

서버에서 실행해 작은 청크를 만든 뒤 로컬로 가져오면, 로컬에서 모델 없이
spectral / heatmap / step_similarity 를 그대로 돌릴 수 있다 (--no-extract).

    # 서버: success/failure 각각 10개씩 → latent_small/ 아래에 저장
    python script/subset_hidden.py --task decompose --level BabyAI-GoToObj-v0 --n 10 \
        --out latent_small/hidden_states
    # 로컬로 복사 (같은 상대경로에 두면 기본 인자로 바로 돌아감)
    scp -r server:/home/hail/HDD/cot/latent_small/hidden_states latent/
    python inference/main.py --task decompose --level BabyAI-GoToObj-v0 --status success --no-extract
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

from extract import load_chunk  # noqa: E402


def subset(src_dir: Path, dst_dir: Path, n: int | None, seeds: list[int] | None):
    files = sorted(src_dir.glob("chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"no chunk_*.pt in {src_dir}")

    picked, header = {}, None
    for cf in files:
        blob = torch.load(cf, map_location="cpu", weights_only=False)
        header = header or {k: v for k, v in blob.items() if k != "episodes"}
        for seed, ep in blob["episodes"].items():
            if seeds is not None and seed not in seeds:
                continue
            picked[seed] = ep
            if n is not None and len(picked) >= n:
                break
        if n is not None and len(picked) >= n:
            break

    if not picked:
        raise ValueError(f"no episodes picked from {src_dir} (seeds={seeds})")

    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / "chunk_0000.pt"
    torch.save({**header, "episodes": picked}, out)
    size_mb = out.stat().st_size / 1e6
    print(f"{src_dir.relative_to(src_dir.parents[4])}: {len(picked)} episodes "
          f"(seeds {sorted(picked)[:5]}{'...' if len(picked) > 5 else ''}) -> {out} ({size_mb:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["decompose", "plan", "predict"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--status", nargs="+", default=["success", "failure"])
    ap.add_argument("--methods", default="full_sequence")
    ap.add_argument("--ctx-tag", default="with_prompt")
    ap.add_argument("--n", type=int, default=10, help="status 당 뽑을 episode 수 (--seeds 주면 무시)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="특정 env_seed 만")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--out", type=Path, default=ROOT / "latent_small" / "hidden_states")
    args = ap.parse_args()

    for status in args.status:
        rel = Path(args.task) / args.level / args.methods / args.ctx_tag / status
        subset(args.hidden_dir / rel, args.out / rel,
               None if args.seeds else args.n, args.seeds)


if __name__ == "__main__":
    main()
