from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from openTSNE import TSNE

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = Path(__file__).resolve().parent

N_PC = 50
N_BINS = 20          # 평균 유사도 히트맵의 진행도 bin
MAX_LAG = 40          # decay 집계 상한 (레벨별로 관측 가능한 최대 lag로 자동 축소됨)
N_BASELINE = 20000    # 무관-쌍(궤적 간) 기준선 샘플 수
SEED = 0

LEVEL_RE = re.compile(r".*_step\d+$")


# ================= 레벨/케이스 탐색 =================

def discover_levels(data_dir: Path) -> list[str]:
    levels = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and LEVEL_RE.match(p.name) and (p / "cases").is_dir():
            levels.append(p.name)
    return levels


def discover_cases(data_dir: Path, level: str) -> list[str]:
    return sorted(p.stem for p in (data_dir / level / "cases").glob("*.jsonl"))


# ================= 데이터 로드 =================

def load_meta(raw_path: Path) -> dict:
    meta = {}
    with open(raw_path) as f:
        for line in f:
            s = json.loads(line)
            meta[s["id"]] = s["answer"]["action_seq"]
    return meta


def build_pool(emb_path: Path, meta: dict, level: str, case: str):
    data = torch.load(emb_path, map_location="cpu")
    X, labels, traj_ids, ts = [], [], [], []
    for sid, e_dict in data["e"].items():
        actions = meta.get(sid)
        if actions is None:
            continue
        assert len(e_dict) == len(actions) + 2, (
            f"[{level}/{case}] sid={sid}: len(e_dict)={len(e_dict)} != "
            f"len(action_seq)+2={len(actions) + 2} — step 인덱스가 action_seq와 "
            f"어긋났을 가능성 (steps 필터링 등)"
        )
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
    n_clusters = len(set(labels))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED).fit(Xp)
    sil = silhouette_score(Xp, km.labels_,
                           sample_size=min(5000, len(Xp)), random_state=SEED)
    ari_all = adjusted_rand_score(labels, km.labels_)
    mm = labels != "mission"
    n_clusters_wo = len(set(labels[mm]))
    km2 = KMeans(n_clusters=n_clusters_wo, n_init=10, random_state=SEED).fit(Xp[mm])
    ari_wo = adjusted_rand_score(labels[mm], km2.labels_)
    return sil, ari_all, ari_wo, n_clusters, n_clusters_wo


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
    """진행도 bin×bin 평균 코사인. 자기 자신과의 쌍(S[i,i]=1)은 제외하고 집계."""
    S_sum = np.zeros((N_BINS, N_BINS)); cnt = np.zeros((N_BINS, N_BINS))
    for sid, E in trajs.items():
        T = len(E)
        S = E @ E.T
        W = np.ones((T, T))
        np.fill_diagonal(S, 0.0)   # 자기유사도(=1) 제외
        np.fill_diagonal(W, 0.0)   # 해당 쌍의 카운트도 제외
        p = np.minimum(((np.arange(T) / max(T - 1, 1))
                        * (N_BINS - 1)).astype(int), N_BINS - 1)
        pi = p[:, None].repeat(T, 1)
        pj = p[None, :].repeat(T, 0)
        np.add.at(S_sum, (pi, pj), S)
        np.add.at(cnt, (pi, pj), W)
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

    # 이 케이스에서 실제로 관측 가능한 최대 lag로 상한을 낮춰서
    # (없는 lag를 0/1=0.0으로 잘못 표시하는 것을 방지)
    max_T = max(len(E) for E in trajs.values())
    max_lag = min(MAX_LAG, max_T - 1)
    lag_sum = np.zeros(max_lag + 1); lag_cnt = np.zeros(max_lag + 1)

    for sid, E in trajs.items():
        T = len(E)
        sims = (E[:-1] * E[1:]).sum(axis=1)
        adj_all.append(sims)
        pos = (np.arange(T - 1) / max(T - 2, 1) * (B - 1)).astype(int)
        np.add.at(adj_sum, pos, sims); np.add.at(adj_cnt, pos, 1.0)
        S = E @ E.T
        for lag in range(1, min(max_lag, T - 1) + 1):
            v = np.diagonal(S, offset=lag)
            lag_sum[lag] += v.sum(); lag_cnt[lag] += len(v)

    adj_curve = adj_sum / np.maximum(adj_cnt, 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.linspace(0, 1, B), adj_curve)
    plt.xlabel("progress t/T"); plt.ylabel("mean cos(e_t, e_{t+1})")
    plt.title(f"adjacent-step similarity  [{case}]")
    plt.tight_layout(); plt.savefig(out_dir / f"adjacent_{case}.png", dpi=150)
    plt.close()

    lag_curve = np.where(lag_cnt[1:] > 0, lag_sum[1:] / np.maximum(lag_cnt[1:], 1), np.nan)
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, max_lag + 1), lag_curve, marker=".")
    plt.xlabel("|t - t'|"); plt.ylabel("mean cosine")
    plt.title(f"similarity decay  [{case}]  (max observed lag={max_lag})")
    plt.tight_layout(); plt.savefig(out_dir / f"decay_{case}.png", dpi=150)
    plt.close()

    adj_flat = np.concatenate(adj_all)
    valid = ~np.isnan(lag_curve)
    lag_max_val = float(lag_curve[valid][-1]) if valid.any() else float("nan")
    return float(adj_flat.mean()), float(lag_curve[0]), lag_max_val, max_lag


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


# ================= 케이스/레벨 루프 =================

def run_case(level, case, method, data_dir: Path, emb_dir: Path, out_dir: Path):
    print(f"\n=== {level} / {case} ({method}) ===")
    meta = load_meta(data_dir / level / "cases" / f"{case}.jsonl")
    emb_path = emb_dir / level / "cases" / case / f"{method}.pt"
    X, labels, traj_ids, ts = build_pool(emb_path, meta, level, case)
    print(f"  pool: {X.shape[0]} steps x {X.shape[1]} dims, "
          f"{len(np.unique(traj_ids))} trajs")

    pca = PCA(n_components=min(N_PC, *X.shape), svd_solver="randomized",
              random_state=SEED)
    Xp = pca.fit_transform(X)
    ev = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA -> {Xp.shape[1]} dims (explained var {ev:.3f})")

    prog = compute_progress(traj_ids, ts)
    trajs = traj_matrices(X, traj_ids, ts)

    tsne_block(Xp, labels, prog, ts, traj_ids, case, out_dir)
    sil, ari_all, ari_wo, n_clusters, n_clusters_wo = kmeans_metrics(Xp, labels)
    rep_heatmaps(trajs, case, out_dir)
    avg_sim_heatmap(trajs, case, out_dir)
    adj_mean, lag1, lag_max, max_lag_used = adjacent_decay(trajs, case, out_dir)
    base_mean, base_std = cross_traj_baseline(trajs)
    pr, df, si = pr_df_si(trajs)

    row = dict(level=level, case=case, n_steps=len(X), n_trajs=len(trajs),
               explained_var=round(ev, 4),
               n_clusters=n_clusters, n_clusters_wo_mission=n_clusters_wo,
               silhouette=round(sil, 4),
               ari_all=round(ari_all, 4), ari_wo_mission=round(ari_wo, 4),
               adj_mean=round(adj_mean, 4), decay_lag1=round(lag1, 4),
               decay_lagmax=round(lag_max, 4), decay_maxlag_used=max_lag_used,
               baseline_mean=round(base_mean, 4),
               baseline_std=round(base_std, 4),
               PR_median=round(pr, 2), DF_median=round(df, 4),
               SI_median=round(si, 4))
    print("  " + "  ".join(f"{k}={v}" for k, v in row.items()
                           if k not in ("level", "case")))
    return row


def run_level(level, method, data_dir: Path, emb_dir: Path, out_root: Path):
    out_dir = out_root / f"analysis2_{level}"
    out_dir.mkdir(exist_ok=True)
    cases = discover_cases(data_dir, level)
    if not cases:
        raise SystemExit(f"no *.jsonl found under {data_dir / level / 'cases'}")

    rows = [run_case(level, c, method, data_dir, emb_dir, out_dir) for c in cases]

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  summary -> {out_dir / 'summary.csv'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"),
                    help="레벨 디렉토리들이 들어있는 루트 (기본: <repo>/data)")
    ap.add_argument("--emb-dir", default=str(ROOT / "result" / "spectral_states"),
                    help="스펙트럴 임베딩 루트 (기본: <repo>/result/spectral_states)")
    ap.add_argument("--method", default="full")
    ap.add_argument("--levels", default=None,
                    help="쉼표로 구분한 레벨 이름 목록. 생략 시 data-dir에서 "
                         "*_step<N> 패턴을 자동 탐색")
    ap.add_argument("--out", default=str(OUT_ROOT),
                    help="analysis2_<level>/ 들이 생성될 상위 디렉토리 (기본: analyze/)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    emb_dir = Path(args.emb_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    levels = (args.levels.split(",") if args.levels
              else discover_levels(data_dir))
    if not levels:
        raise SystemExit(f"no <name>_step<N> level dirs found under {data_dir}")
    print(f"levels: {levels}")

    all_rows = []
    for level in levels:
        all_rows.extend(run_level(level, args.method, data_dir, emb_dir, out_root))

    combined_path = out_root / "analysis2_summary.csv"
    with open(combined_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\ncombined summary -> {combined_path}")


if __name__ == "__main__":
    main()
