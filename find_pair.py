"""성공/실패 에피소드에서 구조가 맞는 쌍을 찾는다.

왜 필요한가
    성공 궤적과 실패 궤적의 유사도 구조를 비교하려면 둘의 크기가 맞아야 한다.
    블록이 작을수록 그 안의 토큰이 물리적으로 가까워 유사도가 높게 나오므로,
    크기가 다른 둘을 그냥 비교하면 "실패 궤적이 더 응집력 있다" 같은 엉뚱한
    결론이 나온다. 통계로 보정하는 대신 애초에 맞는 쌍을 고른다 — GoToObj 는
    조건을 만족하는 조합이 수만 개라 고르는 쪽이 싸고 깨끗하다.

토큰 수가 아니라 글자 수로 맞추는 이유
    토큰 수를 재려면 토크나이저를 올려야 한다. 같은 레벨·같은 프롬프트 형식이라
    글자 수가 토큰 수의 충분히 좋은 대리값이고, 실제 N 은 probe_gram.py 가
    찍어주므로 거기서 확인하면 된다.

    python find_pair.py data/segmented/BabyAI-GoToObj-v0
    python find_pair.py data/segmented/BabyAI-GoToObj-v0 --top 10
    python find_pair.py data/segmented/BabyAI-GoToObj-v0 --quiet   # "3080 10" 만
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def load(path: Path):
    """액션 블록이 있는 에피소드만. 없으면 액션 단위 비교 자체가 불가능하다."""
    out = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if not r["seg_info"]["has_action_blocks"]:
                continue
            sa = r["segments_action"]
            out.append({
                "index": i, "id": r["id"],
                "nchar": len(r["all_llm_output"]),
                "nblocks": len(sa),
                "size": median(s["end"] - s["start"] for s in sa),
            })
    return out


def find(succ, fail, char_tol, size_tol, top):
    pairs = []
    for f in fail:
        for s in succ:
            if s["nblocks"] != f["nblocks"]:
                continue
            dc = abs(s["nchar"] - f["nchar"]) / max(s["nchar"], f["nchar"])
            if dc > char_tol:
                continue
            ds = abs(s["size"] - f["size"]) / max(s["size"], f["size"])
            if ds > size_tol:
                continue
            pairs.append((dc + ds, s, f))          # 둘 다 작을수록 좋은 쌍
    pairs.sort(key=lambda x: x[0])
    return pairs[:top], len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level_dir", type=Path,
                    help="success.jsonl / fail.jsonl 이 있는 디렉토리")
    ap.add_argument("--char-tol", type=float, default=0.05, help="글자 수 허용 오차")
    ap.add_argument("--size-tol", type=float, default=0.10, help="블록 크기 허용 오차")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--quiet", action="store_true",
                    help='최적 쌍의 인덱스만 "<success> <fail>" 로 출력 (쉘에서 쓰기)')
    args = ap.parse_args()

    succ = load(args.level_dir / "success.jsonl")
    fail = load(args.level_dir / "fail.jsonl")
    best, total = find(succ, fail, args.char_tol, args.size_tol, args.top)

    if not best:
        raise SystemExit(f"조건을 만족하는 쌍이 없습니다 "
                         f"(success {len(succ)} / fail {len(fail)}). "
                         f"--char-tol / --size-tol 을 키우세요")

    if args.quiet:
        print(best[0][1]["index"], best[0][2]["index"])
        return

    print(f"후보 success {len(succ)}  fail {len(fail)}   조건 만족 쌍 {total}\n")
    print(f"{'':4}{'success':>38}  |{'fail':>38}")
    for _, s, f in best:
        print(f"    idx {s['index']:5} blocks {s['nblocks']:2} char {s['nchar']:5} "
              f"size {s['size']:5.0f}  | idx {f['index']:5} blocks {f['nblocks']:2} "
              f"char {f['nchar']:5} size {f['size']:5.0f}")
    s, f = best[0][1], best[0][2]
    print(f"\n최적 쌍:")
    print(f"  python probe_gram.py {args.level_dir}/success.jsonl "
          f"--index {s['index']} --out visual/pair_success")
    print(f"  python probe_gram.py {args.level_dir}/fail.jsonl "
          f"--index {f['index']} --out visual/pair_fail")
    print(f"  python visual/heatmap.py visual/pair_*.pt")


if __name__ == "__main__":
    main()
