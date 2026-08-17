"""이미 추출해둔 두 hidden state가 같은지 확인.

usage:
    python check_same.py a.pt b.pt
    python check_same.py a.pt b.pt --key h        # 딕셔너리 안의 특정 키
    python check_same.py a.pt b.pt --verbose      # 다른 항목 전부 나열
"""
from __future__ import annotations

import argparse
import sys

import torch


def flatten(obj, prefix=""):
    """중첩 dict/list 안의 텐서를 {경로: 텐서} 로 펼침."""
    out = {}
    if torch.is_tensor(obj):
        out[prefix or "<tensor>"] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    return out


def load(path, key=None):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if key is not None:
        for part in key.split("/"):
            obj = obj[part]
    return flatten(obj)


def stats(a, b):
    """두 텐서의 차이 요약. 비교는 float64로."""
    x, y = a.double(), b.double()
    d = (x - y).abs()
    na = x.pow(2).sum().sqrt()
    return dict(
        bitwise=torch.equal(a, b),
        max_abs=d.max().item(),
        mean_abs=d.mean().item(),
        # ‖a-b‖ / ‖a‖ : 신호 크기 대비 오차. 이게 핵심 지표
        rel_fro=(d.pow(2).sum().sqrt() / na.clamp_min(1e-12)).item(),
        n_diff=int((d != 0).sum()),
        n_total=d.numel(),
    )


def verdict(s):
    if s["bitwise"]:
        return "IDENTICAL (비트 동일)"
    r = s["rel_fro"]
    if r < 1e-6:
        return f"EQUAL (fp32 잡음, rel={r:.1e})"
    if r < 5e-3:
        return f"EQUAL~ (bf16/tf32 잡음, rel={r:.1e})"
    return f"DIFFERENT (rel={r:.1e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--key", default=None,
                    help="딕셔너리 안의 경로. 예: e 또는 h/sid123")
    ap.add_argument("--verbose", action="store_true",
                    help="일치하는 항목까지 전부 출력")
    args = ap.parse_args()

    A, B = load(args.a, args.key), load(args.b, args.key)

    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a:
        print(f"[!] a 에만 있는 항목 {len(only_a)}개: {only_a[:5]}")
    if only_b:
        print(f"[!] b 에만 있는 항목 {len(only_b)}개: {only_b[:5]}")

    common = sorted(set(A) & set(B))
    if not common:
        print("공통 항목이 없음. --key 로 경로를 지정해 보세요.")
        print(f"  a 의 키: {sorted(A)[:10]}")
        print(f"  b 의 키: {sorted(B)[:10]}")
        sys.exit(1)

    n_same = n_equal = n_diff = 0
    worst = None

    for k in common:
        ta, tb = A[k], B[k]
        if ta.shape != tb.shape:
            print(f"  {k}: SHAPE MISMATCH {tuple(ta.shape)} vs {tuple(tb.shape)}")
            n_diff += 1
            continue
        s = stats(ta, tb)
        v = verdict(s)
        if s["bitwise"]:
            n_same += 1
        elif v.startswith("EQUAL"):
            n_equal += 1
        else:
            n_diff += 1
        if worst is None or s["rel_fro"] > worst[1]["rel_fro"]:
            worst = (k, s)
        if args.verbose or not v.startswith(("IDENTICAL", "EQUAL")):
            print(f"  {k}: {v}  max_abs={s['max_abs']:.3e}  "
                  f"diff {s['n_diff']}/{s['n_total']}")

    print(f"\n항목 {len(common)}개  |  비트동일 {n_same}  "
          f"수치동등 {n_equal}  다름 {n_diff}")
    if worst:
        k, s = worst
        print(f"오차 최대: {k}  rel_fro={s['rel_fro']:.3e}  "
              f"max_abs={s['max_abs']:.3e}  dtype={A[k].dtype}")
        print(f"판정: {verdict(s)}")


if __name__ == "__main__":
    main()