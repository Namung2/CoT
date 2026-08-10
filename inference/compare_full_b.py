from __future__ import annotations

import csv
from pathlib import Path

import torch

HID_DIR = Path("result/hidden_states")


def compare_all_steps(level, case, step_folder, sample_id):
    base = HID_DIR / level / "by_step" / case / step_folder
    full = torch.load(base / "full" / f"{sample_id}.pt", map_location="cpu")["E"]
    B = torch.load(base / "B" / f"{sample_id}.pt", map_location="cpu")["E"]

    for t in sorted(full):
        a, b = full[t].float(), B[t].float()
        if a.shape != b.shape:
            print(f"t={t:2d}  SHAPE MISMATCH full={tuple(a.shape)} B={tuple(b.shape)}")
            continue
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        rel = (a - b).norm(dim=1) / a.norm(dim=1)
        print(f"t={t:2d} n={a.shape[0]:3d}  cos_min={cos.min():.6f}  rel_mean={rel.mean()*100:.3f}%  rel_max={rel.max()*100:.3f}%")

    diff = (a - b).abs()                     # (n_tok, d)
    row_max = diff.max(dim=1).values         # 토큰별 최대 절댓값 차이
    row_mean = diff.mean(dim=1)               # 토큰별 평균 절댓값 차이
    row_l2 = diff.norm(dim=1)                 # 토큰별 L2 거리
    row_a_norm = a.norm(dim=1)                # 토큰별 원본(full) 벡터 크기
    row_rel = row_l2 / row_a_norm             # 상대 오차 (원본 크기 대비 %)
    row_cos = torch.nn.functional.cosine_similarity(a, b, dim=1)  # 방향(의미) 유사도

    print(f"t={t}: n_tokens={a.shape[0]} d={a.shape[1]}")
    print(f"  overall max_abs_diff={diff.max().item():.6f} mean_abs_diff={diff.mean().item():.6f}")
    print(f"  overall relative_diff mean={row_rel.mean().item()*100:.2f}% max={row_rel.max().item()*100:.2f}%")
    print(f"  overall cos_sim mean={row_cos.mean().item():.6f} min={row_cos.min().item():.6f}")

    if save_path is None:
        for i in range(a.shape[0]):
            print(
                f"  tok={i:3d} max={row_max[i].item():.6f} mean={row_mean[i].item():.6f} "
                f"l2={row_l2[i].item():.6f} rel={row_rel[i].item()*100:.2f}% cos={row_cos[i].item():.6f}"
            )
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_idx", "max_abs_diff", "mean_abs_diff", "l2_diff", "a_norm", "relative_diff_pct", "cos_sim"])
        for i in range(a.shape[0]):
            writer.writerow([
                i, row_max[i].item(), row_mean[i].item(), row_l2[i].item(),
                row_a_norm[i].item(), row_rel[i].item() * 100, row_cos[i].item(),
            ])
    print(f"  saved per-token diff to {save_path}")


if __name__ == "__main__":
    compare_one_step(
        level="gotoseq_10to50_by_step",
        case="c3",
        step_folder="10step",
        sample_id="00194924af64",
        t=0,
        save_path="analyze/full_vs_B_diff/c3_10step_00194924af64_t0.csv",
    )
