#!/usr/bin/env python
"""구간별 대표 벡터를 t-SNE 로 2차원에 찍는다 (마지막 레이어). 논문 Figure 1a 대응.

probing.py 의 load_pt 를 그대로 쓴다 — 같은 벡터, 같은 라벨 체계.
    step_num:  1..N = step 구간, -1 = 터미널(정답 문장), 0 = 프롬프트(--with-prompt)

논문과 같은 점:
  - 정답/오답을 섞는다. 3장의 질문은 "step 별 구조가 있나"이지 정답성과 무관하다.
    (Figure 1a 캡션도 GSM8K test split 전체이고 correctness 필터가 없다)
  - 색은 step 번호로만 칠한다.
  - TSNE(n_components=2, random_state=42, perplexity=30)

논문과 다른 점:
  - 우리는 에피소드가 훨씬 많다 (3만 × 6구간 = 18만 점). 그대로는 t-SNE 가 안 끝나므로
    에피소드 단위로 서브샘플링한다 (한 궤적의 구간들이 같이 들어가도록).
  - 5,120차원을 바로 넣지 않고 PCA 50 으로 줄인 뒤 t-SNE (sklearn 권장 관례).

Usage:
    python tsne.py --pt 'latent/hidden_states/plan/*/*/chunk_*.pt' --output out/plan_tsne
    python tsne.py --pt 'latent/hidden_states/decompose/BabyAI-GoTo-v0/*/chunk_*.pt' \
        --output out/decompose_goto_tsne --max-episodes 3000
"""
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from probing import load_pt, target_name, PROMPT_LABEL, ANSWER_LABEL

# 논문 visualize_act_tsne.py 의 STEP_COLORS 를 그대로 가져왔다.
STEP_COLORS = {
    1: "#1f77b4",   # blue
    2: "#2ca02c",   # green
    3: "#ff7f0e",   # orange
    4: "#9467bd",   # purple
    5: "#8c564b",   # brown
    6: "#e377c2",   # pink
    7: "#17becf",   # cyan
    8: "#bcbd22",   # yellow-green
    9: "#d62728",   # red
    10: "#7f7f7f",  # gray
}
ANSWER_COLOR = "#ffb3ba"   # light red — 논문의 HASH_COLOR
PROMPT_COLOR = "#cccccc"


def subsample_by_episode(data, max_episodes, seed):
    """에피소드(group) 단위로 뽑는다. 점 단위로 뽑으면 한 궤적의 구간이 흩어진다."""
    groups = np.unique(data["group"])
    if max_episodes is None or len(groups) <= max_episodes:
        return data, len(groups)
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(groups, size=max_episodes, replace=False).tolist())
    m = np.array([g in keep for g in data["group"]])
    return {k: v[m] for k, v in data.items()}, max_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", nargs="+", required=True,
                    help="chunk_*.pt glob. success/failure 양쪽을 다 넣는다 "
                         "(논문도 정답/오답을 섞는다)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="구간 마지막 토큰에서 몇 칸 앞 (probing.py 와 같은 의미)")
    ap.add_argument("--with-prompt", action="store_true",
                    help="프롬프트 구간 마지막 토큰도 포함")
    ap.add_argument("--max-episodes", type=int, default=2000,
                    help="t-SNE 에 넣을 에피소드 수. None 이면 전부")
    ap.add_argument("--pca", type=int, default=0,
                    help="t-SNE 전에 줄일 차원. 0 이면 PCA 생략 (논문은 생략)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--point-size", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)

    data = load_pt(a.pt, a.offset, a.with_prompt)
    data, n_ep = subsample_by_episode(data, a.max_episodes, a.seed)
    X, labels = data["X"], data["step_num"]
    print(f"t-SNE 입력: {X.shape}  에피소드 {n_ep}개  "
          f"구간별 {dict(sorted(Counter(labels.tolist()).items()))}")

    if a.pca:
        k = min(a.pca, X.shape[0], X.shape[1])
        pca = PCA(n_components=k, random_state=a.seed)
        X = pca.fit_transform(X)
        print(f"PCA {k}차원, 설명 분산 {pca.explained_variance_ratio_.sum():.3f}")

    perp = min(a.perplexity, len(X) - 1)
    print(f"TSNE(perplexity={perp}) 시작 — 점 {len(X)}개")
    emb = TSNE(n_components=2, random_state=a.seed, perplexity=perp).fit_transform(X)

    # 보기 좋은 순서: prompt → step 1..N → answer. 나중에 그린 게 위로 올라온다.
    uniq = sorted(np.unique(labels).tolist())
    order = ([PROMPT_LABEL] if PROMPT_LABEL in uniq else []) \
            + [l for l in uniq if l > 0] \
            + ([ANSWER_LABEL] if ANSWER_LABEL in uniq else [])

    fig, ax = plt.subplots(figsize=(8, 8))
    for l in order:
        m = labels == l
        if l == PROMPT_LABEL:
            c = PROMPT_COLOR
        elif l == ANSWER_LABEL:
            c = ANSWER_COLOR
        else:
            c = STEP_COLORS.get(l, "#000000")
        ax.scatter(emb[m, 0], emb[m, 1], c=c, s=a.point_size, alpha=a.alpha,
                   linewidths=0, label=f"{target_name(l)} (n={m.sum()})")

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"Step-wise activations, last layer "
                 f"(episodes={n_ep}, offset={a.offset})")
    leg = ax.legend(markerscale=3, fontsize=9, loc="best", framealpha=0.9)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(a.output / "tsne.png", dpi=150)
    plt.close(fig)

    np.savez_compressed(a.output / "tsne.npz",
                        embedding=emb, step_num=labels,
                        group=data["group"].astype(str))
    print(f"saved → {a.output}")


if __name__ == "__main__":
    main()