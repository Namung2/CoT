# analyze.py — 최종 통합 분석: local/cum 임베딩을 같은 파이프라인으로 기술·정량화
#
# 사용법:
#   python analyze.py --emb e --method A --tag A
#   python analyze.py --emb e --method B --tag B
#
# 케이스별 산출물 (analysis_<tag>/):
#   tsne_action_<case>.png / tsne_progress_<case>.png / tsne_traj_<case>.png
#   heatmap_<case>_{short,median,long}.png     (percentile 색 스케일)
#   sim_heatmap_<case>.png                      (진행도 bin×bin 평균)
#   adjacent_<case>.png / decay_<case>.png
#   tsne_cache_<case>.npz
# 전체 요약: summary.csv (케이스별 지표 표)

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from openTSNE import TSNE

LEVEL = "gotoseq_10to50"
RAW_DIR = Path("data") / LEVEL / "cases"

N_CLUSTERS = 5
N_PC = 50
N_BINS = 20          # 평균 유사도 히트맵의 진행도 bin
MAX_LAG = 40
N_BASELINE = 20000   # 무관-쌍(궤적 간) 기준선 샘플 수
SEED = 0


# ================= 데이터 로드 =================

def load_meta(raw_path: Path) -> dict:
    meta = {}
    with open(raw_path) as f:
        for line in f:
            s = json.loads(line)
            meta[s["id"]] = s["answer"]["action_seq"]
    return meta


def build_pool(emb_path: Path, meta: dict):
    data = torch.load(emb_path, map_location="cpu")
    X, labels, traj_ids, ts = [], [], [], []
    for sid, e_dict in data["e"].items():
        actions = meta.get(sid)
        if actions is None:
            continue
        for t in sorted(e_dict):
            X.append(e_dict[t].numpy())
            if t == 0:
                labels.append("mission")
            elif t - 1 < len(actions):
                labels.append(actions[t - 1])
            else:
                labels.append("terminal")
            traj_ids.append(sid)
            ts.append(t)
    return (np.stack(X), np.array(labels),
            np.array(traj_ids), np.array(ts))


def compute_progress(traj_ids, ts):
    prog = np.zeros(len(ts), dtype=float)
    for sid in np.unique(traj_ids):
        mm = traj_ids == sid
        prog[mm] = ts[mm] / max(int(ts[mm].max()), 1)
    return prog


def traj_matrices(X, traj_ids, ts):
    """궤적별 (정규화된 (T,dim) 행렬, 스텝순) — mission 제외"""
    out = {}
    for sid in np.unique(traj_ids):
        mm = (traj_ids == sid) & (ts > 0)
        if mm.sum() < 2:
            continue
        order = np.argsort(ts[mm])
        E = X[mm][order]
        out[sid] = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return out


# ================= A. 전역 구조 =================

def tsne_block(Xp, labels, prog, ts, traj_ids, case, out_dir):
    Z = np.asarray(TSNE(n_components=2, perplexity=30, initialization="pca",
                        n_jobs=-1, random_state=SEED).fit(Xp))
    np.savez(out_dir / f"tsne_cache_{case}.npz",
             Z=Z, labels=labels, ts=ts, traj_ids=traj_ids)
    m = ts > 0

    # ① action
    plt.figure(figsize=(8, 7))
    for a in sorted(set(labels)):
        mm = labels == a
        plt.scatter(Z[mm, 0], Z[mm, 1], s=5, alpha=0.5,
                    label=f"{a} ({int(mm.sum())})")
    plt.legend(markerscale=3, fontsize=8)
    plt.title(f"t-SNE by action  [{case}]")
    plt.tight_layout(); plt.savefig(out_dir / f"tsne_action_{case}.png", dpi=150)
    plt.close()

    # ② progress
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(Z[m, 0], Z[m, 1], c=prog[m], s=3, cmap="viridis", alpha=0.6)
    plt.colorbar(sc, label="progress t/T")
    plt.title(f"t-SNE by progress  [{case}]")
    plt.tight_layout(); plt.savefig(out_dir / f"tsne_progress_{case}.png", dpi=150)
    plt.close()

    # ③ trajectory highlight (20개)
    rng = np.random.RandomState(SEED)
    sids = np.unique(traj_ids)
    picked = set(rng.choice(sids, size=min(20, len(sids)), replace=False).tolist())
    plt.figure(figsize=(8, 7))
    bg = ~np.isin(traj_ids, list(picked))
    plt.scatter(Z[bg & m, 0], Z[bg & m, 1], s=2, color="lightgray", alpha=0.3)
    cmap = plt.get_cmap("tab20")
    for i, sid in enumerate(sorted(picked)):
        mm = (traj_ids == sid) & m
        plt.scatter(Z[mm, 0], Z[mm, 1], s=6, color=cmap(i % 20), alpha=0.9)
    plt.title(f"t-SNE, 20 trajectories highlighted  [{case}]")
    plt.tight_layout(); plt.savefig(out_dir / f"tsne_traj_{case}.png", dpi=150)
    plt.close()


def kmeans_metrics(Xp, labels):
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(Xp)
    sil = silhouette_score(Xp, km.labels_,
                           sample_size=min(5000, len(Xp)), random_state=SEED)
    ari_all = adjusted_rand_score(labels, km.labels_)
    mm = labels != "mission"
    km2 = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(Xp[mm])
    ari_wo = adjusted_rand_score(labels[mm], km2.labels_)
    return sil, ari_all, ari_wo


# ================= B. 궤적 내 구조 =================

def rep_heatmaps(trajs, case, out_dir):
    by_len = sorted(trajs, key=lambda s: len(trajs[s]))
    picks = {"short": by_len[0], "median": by_len[len(by_len) // 2],
             "long": by_len[-1]}
    for tag, sid in picks.items():
        S = trajs[sid] @ trajs[sid].T
        off = S[np.triu_indices_from(S, k=1)]
        vmin, vmax = np.percentile(off, [1, 99])       # percentile 스케일
        plt.figure(figsize=(6.5, 5.5))
        plt.imshow(S, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        plt.colorbar(label=f"cosine (range [{vmin:.2f},{vmax:.2f}])")
        plt.xlabel("step"); plt.ylabel("step")
        plt.title(f"step-step sim [{case} {tag}] T={len(S)} ({sid[:8]})")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_{case}_{tag}.png", dpi=150)
        plt.close()


def avg_sim_heatmap(trajs, case, out_dir):
    S_sum = np.zeros((N_BINS, N_BINS)); cnt = np.zeros((N_BINS, N_BINS))
    for sid, E in trajs.items():
        T = len(E)
        S = E @ E.T
        p = np.minimum(((np.arange(T) / max(T - 1, 1))
                        * (N_BINS - 1)).astype(int), N_BINS - 1)
        np.add.at(S_sum, (p[:, None].repeat(T, 1), p[None, :].repeat(T, 0)), S)
        np.add.at(cnt, (p[:, None].repeat(T, 1), p[None, :].repeat(T, 0)), 1.0)
    S_avg = S_sum / np.maximum(cnt, 1)
    off = S_avg[~np.eye(N_BINS, dtype=bool)]
    vmin, vmax = off.min(), off.max()
    plt.figure(figsize=(7, 6))
    im = plt.imshow(S_avg, origin="lower", cmap="RdBu_r",
                    vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
    plt.colorbar(im, label="mean cosine")
    plt.xlabel("progress t'/T"); plt.ylabel("progress t/T")
    plt.title(f"trajectory-averaged similarity  [{case}]")
    plt.tight_layout(); plt.savefig(out_dir / f"sim_heatmap_{case}.png", dpi=150)
    plt.close()


def adjacent_decay(trajs, case, out_dir):
    B = 30
    adj_sum = np.zeros(B); adj_cnt = np.zeros(B); adj_all = []
    lag_sum = np.zeros(MAX_LAG + 1); lag_cnt = np.zeros(MAX_LAG + 1)
    for sid, E in trajs.items():
        T = len(E)
        sims = (E[:-1] * E[1:]).sum(axis=1)
        adj_all.append(sims)
        pos = (np.arange(T - 1) / max(T - 2, 1) * (B - 1)).astype(int)
        np.add.at(adj_sum, pos, sims); np.add.at(adj_cnt, pos, 1.0)
        S = E @ E.T
        for lag in range(1, min(MAX_LAG, T - 1) + 1):
            v = np.diagonal(S, offset=lag)
            lag_sum[lag] += v.sum(); lag_cnt[lag] += len(v)

    adj_curve = adj_sum / np.maximum(adj_cnt, 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.linspace(0, 1, B), adj_curve)
    plt.xlabel("progress t/T"); plt.ylabel("mean cos(e_t, e_{t+1})")
    plt.title(f"adjacent-step similarity  [{case}]")
    plt.tight_layout(); plt.savefig(out_dir / f"adjacent_{case}.png", dpi=150)
    plt.close()

    lag_curve = lag_sum[1:] / np.maximum(lag_cnt[1:], 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, MAX_LAG + 1), lag_curve, marker=".")
    plt.xlabel("|t - t'|"); plt.ylabel("mean cosine")
    plt.title(f"similarity decay  [{case}]")
    plt.tight_layout(); plt.savefig(out_dir / f"decay_{case}.png", dpi=150)
    plt.close()

    adj_flat = np.concatenate(adj_all)
    return float(adj_flat.mean()), float(lag_curve[0]), float(lag_curve[-1])


# ================= C. 정량 지표 =================

def cross_traj_baseline(trajs):
    """무관-쌍 기준선: 서로 다른 궤적의 스텝 쌍 평균 코사인 (anisotropy 눈금)"""
    rng = np.random.RandomState(SEED)
    sids = list(trajs)
    vals = []
    for _ in range(N_BASELINE):
        a, b = rng.choice(len(sids), 2, replace=False)
        Ea, Eb = trajs[sids[a]], trajs[sids[b]]
        vals.append(float(Ea[rng.randint(len(Ea))] @ Eb[rng.randint(len(Eb))]))
    v = np.array(vals)
    return float(v.mean()), float(v.std())


def pr_df_si(trajs):
    """궤적별 PR(유효차원), DF(표류비율), SI(이웃 식별) → 중앙값 요약"""
    prs, dfs, sis = [], [], []
    for sid, E in trajs.items():
        T = len(E)
        if T < 5:
            continue
        # PR: 중심화 후 공분산 고유값 — T×T Gram 경유 (kd차원 우회)
        Ec = E - E.mean(axis=0, keepdims=True)
        lam = np.linalg.eigvalsh(Ec @ Ec.T)
        lam = np.clip(lam, 0, None)
        s1, s2 = lam.sum(), (lam ** 2).sum()
        if s2 > 0:
            prs.append(float(s1 ** 2 / s2))
        # DF: 평균 변화 방향이 설명하는 비율
        D = E[1:] - E[:-1]
        num = float(np.linalg.norm(D.mean(axis=0)) ** 2)
        den = float((np.linalg.norm(D, axis=1) ** 2).mean())
        if den > 0:
            dfs.append(num / den)
        # SI: 각 스텝의 최근접(자기 제외)이 t±1인 비율
        S = E @ E.T
        np.fill_diagonal(S, -np.inf)
        nn = S.argmax(axis=1)
        hits = np.abs(nn - np.arange(T)) == 1
        sis.append(float(hits.mean()))
    med = lambda a: float(np.median(a)) if a else float("nan")
    return med(prs), med(dfs), med(sis)


# ================= 케이스 루프 =================

def run_case(case, method, emb_dir: Path, out_dir: Path):
    print(f"\n=== {case} ({method}) ===")
    meta = load_meta(RAW_DIR / f"{case}.jsonl")
    X, labels, traj_ids, ts = build_pool(emb_dir / LEVEL / f"{case}_{method}.pt", meta)
    print(f"  pool: {X.shape[0]} steps x {X.shape[1]} dims, "
          f"{len(np.unique(traj_ids))} trajs")

    pca = PCA(n_components=min(N_PC, *X.shape), random_state=SEED)
    Xp = pca.fit_transform(X)
    ev = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA -> {Xp.shape[1]} dims (explained var {ev:.3f})")

    prog = compute_progress(traj_ids, ts)
    trajs = traj_matrices(X, traj_ids, ts)

    tsne_block(Xp, labels, prog, ts, traj_ids, case, out_dir)
    sil, ari_all, ari_wo = kmeans_metrics(Xp, labels)
    rep_heatmaps(trajs, case, out_dir)
    avg_sim_heatmap(trajs, case, out_dir)
    adj_mean, lag1, lag_max = adjacent_decay(trajs, case, out_dir)
    base_mean, base_std = cross_traj_baseline(trajs)
    pr, df, si = pr_df_si(trajs)

    row = dict(case=case, n_steps=len(X), n_trajs=len(trajs),
               explained_var=round(ev, 4), silhouette=round(sil, 4),
               ari_all=round(ari_all, 4), ari_wo_mission=round(ari_wo, 4),
               adj_mean=round(adj_mean, 4), decay_lag1=round(lag1, 4),
               decay_lagmax=round(lag_max, 4),
               baseline_mean=round(base_mean, 4),
               baseline_std=round(base_std, 4),
               PR_median=round(pr, 2), DF_median=round(df, 4),
               SI_median=round(si, 4))
    print("  " + "  ".join(f"{k}={v}" for k, v in row.items()
                           if k not in ("case",)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="e", help="spectral.py 출력 루트 (e.g. e)")
    ap.add_argument("--method", default="B", choices=["A", "B"])
    ap.add_argument("--tag", default=None, help="기본값: --method 값")
    args = ap.parse_args()
    tag = args.tag or args.method

    out_dir = Path(f"analysis_{tag}")
    out_dir.mkdir(exist_ok=True)

    emb_dir = Path(args.emb)
    cases = sorted(p.stem[: -len(f"_{args.method}")]
                   for p in (emb_dir / LEVEL).glob(f"*_{args.method}.pt"))
    if not cases:
        raise SystemExit(f"no *_{args.method}.pt found under {emb_dir / LEVEL}")
    print(f"cases: {cases}")

    rows = [run_case(c, args.method, emb_dir, out_dir) for c in cases]

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nsummary -> {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()