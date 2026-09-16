#!/usr/bin/env python
"""마지막 레이어 하나에 대한 step-wise one-vs-rest 선형 probe.

slhleosun/reasoning-trajectory 의 train_stepwise_probes.py 를 기반으로:
  제거 - 레이어 루프, 레이어별 PCA 결정경계 그리드, 레이어 축 요약 플롯
  추가 - 질문 단위 split (GroupShuffleSplit), 다중 시드 mean±std, 선택적 C 튜닝

입력 npz (num_layers=1 로 포장한 것):
  step_activations   : 길이 1 리스트, 원소 [n_step, hidden_dim]
  hash_activations   : 길이 1 리스트, 원소 [n_hash, hidden_dim]
  step_numbers       : 길이 1 리스트, 원소 [n_step] int
  question_ids_step  : (선택) 길이 1 리스트, 원소 [n_step]   ← 없으면 행 단위 split 으로 대체
  question_ids_hash  : (선택) 길이 1 리스트, 원소 [n_hash]

Usage:
  python train_last_layer_probes.py --data last_layer.npz --output out/last_layer
  python train_last_layer_probes.py --data last_layer.npz --output out/last_layer_cv --cv
"""
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ---------------------------------------------------------------- data

def load(npz_path: Path, layer: int) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    out = dict(
        step=np.asarray(d["step_activations"][layer], dtype=np.float32),
        hash=np.asarray(d["hash_activations"][layer], dtype=np.float32),
        step_num=np.asarray(d["step_numbers"][layer], dtype=np.int32),
        q_step=None, q_hash=None,
    )
    if "question_ids_step" in d.files and "question_ids_hash" in d.files:
        out["q_step"] = np.asarray(d["question_ids_step"][layer])
        out["q_hash"] = np.asarray(d["question_ids_hash"][layer])
    print(f"loaded: step={out['step'].shape} hash={out['hash'].shape} "
          f"steps={dict(zip(*np.unique(out['step_num'], return_counts=True)))} "
          f"question_ids={'yes' if out['q_step'] is not None else 'no'}")
    return out


def make_binary(data: dict, target: str):
    """target 을 positive, 나머지(step 전부 or 다른 step + hash) 를 negative 로."""
    if target == "hash":
        Xp, Xn = data["hash"], data["step"]
        gp, gn = data["q_hash"], data["q_step"]
    else:
        n = int(target.split("_")[1])
        m = data["step_num"] == n
        Xp = data["step"][m]
        Xn = np.vstack([data["step"][~m], data["hash"]])
        if data["q_step"] is not None:
            gp = data["q_step"][m]
            gn = np.concatenate([data["q_step"][~m], data["q_hash"]])
        else:
            gp = gn = None
    X = np.vstack([Xp, Xn])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))].astype(int)
    g = None if gp is None else np.concatenate([gp, gn])
    return X, y, g


def split(X, y, g, test_size, seed, group):
    if group and g is not None:
        return next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=seed).split(X, y, g))
    return train_test_split(np.arange(len(y)), test_size=test_size,
                            random_state=seed, stratify=y)


# ---------------------------------------------------------------- model

def fit(Xtr, ytr, seed, cv):
    if cv:
        clf = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, scoring="roc_auc",
                                   class_weight="balanced", max_iter=2000,
                                   random_state=seed, n_jobs=-1)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                 random_state=seed)
    return clf.fit(Xtr, ytr)


def evaluate(clf, X, y) -> dict:
    p = clf.predict_proba(X)[:, 1]
    yhat = (p > 0.5).astype(int)          # 원 코드와 동일하게 0.5 고정
    return dict(acc=accuracy_score(y, yhat),
                f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0,
                    help="npz 안의 레이어 인덱스 (num_layers=1 이면 0)")
    ap.add_argument("--max-step", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-group-split", action="store_true",
                    help="질문 단위 split 끄고 행 단위 무작위 split")
    ap.add_argument("--cv", action="store_true", help="LogisticRegressionCV 로 C 튜닝")
    a = ap.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "classifiers").mkdir(exist_ok=True)

    data = load(a.data, a.layer)
    group = not a.no_group_split
    if group and data["q_step"] is None:
        print("[warn] question_ids 없음 → 행 단위 split 으로 대체")
        group = False

    targets = [f"step_{i}" for i in range(1, a.max_step + 1)] + ["hash"]
    rows = []

    for t in targets:
        X, y, g = make_binary(data, t)
        if y.sum() < 2 or (1 - y).sum() < 2:
            print(f"[skip] {t}: 샘플 부족")
            continue
        for s in a.seeds:
            tr, te = split(X, y, g, a.test_size, s, group)
            if len(np.unique(y[te])) < 2:
                print(f"[skip] {t} seed={s}: test 에 한 클래스만 있음")
                continue
            clf = fit(X[tr], y[tr], s, a.cv)
            m_tr, m_te = evaluate(clf, X[tr], y[tr]), evaluate(clf, X[te], y[te])
            rows.append(dict(
                target=t, seed=s, n_train=int(len(tr)), n_test=int(len(te)),
                n_pos_test=int(y[te].sum()),
                C=float(np.atleast_1d(getattr(clf, "C_", clf.C))[0]),
                **{f"train_{k}": float(v) for k, v in m_tr.items()},
                **{f"test_{k}": float(v) for k, v in m_te.items()},
            ))
            with open(a.output / "classifiers" / f"{t}_seed{s}.pkl", "wb") as f:
                pickle.dump(clf, f)
            print(f"{t:7s} seed={s:5d}  AUC={m_te['auc']:.3f} "
                  f"Acc={m_te['acc']:.3f} F1={m_te['f1']:.3f}")

    with open(a.output / "results_all.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ---- 시드 집계
    summary = {}
    for t in targets:
        r = [x for x in rows if x["target"] == t]
        if not r:
            continue
        summary[t] = {k: dict(mean=float(np.mean([x[k] for x in r])),
                              std=float(np.std([x[k] for x in r])))
                      for k in ("test_auc", "test_acc", "test_f1")}
        summary[t]["n_seeds"] = len(r)
    with open(a.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 62)
    print(f"{'target':8s} {'AUC':>15s} {'Acc':>15s} {'F1':>15s}")
    for t, s in summary.items():
        fmt = lambda k: f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}"
        print(f"{t:8s} {fmt('test_auc'):>15s} {fmt('test_acc'):>15s} {fmt('test_f1'):>15s}")

    # ---- 플롯: x 축 = target
    ts = list(summary)
    x, w = np.arange(len(ts)), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (k, lab) in enumerate([("test_auc", "AUC"), ("test_acc", "Acc"), ("test_f1", "F1")]):
        ax.bar(x + (i - 1) * w, [summary[t][k]["mean"] for t in ts], w,
               yerr=[summary[t][k]["std"] for t in ts], capsize=3, label=lab)
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["####" if t == "hash" else f"Step {t[5:]}" for t in ts])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Last-layer probes  (seeds={len(a.seeds)}, group_split={group}, cv={a.cv})")
    fig.tight_layout()
    fig.savefig(a.output / "summary.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved → {a.output}")


if __name__ == "__main__":
    main()

