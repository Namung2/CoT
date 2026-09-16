#!/usr/bin/env python
"""chunk_*.pt 를 직접 읽어 step-wise one-vs-rest 선형 probe 학습 (마지막 레이어).

.pt 형식 (extract.py 출력):
    {"model": ..., "boundary_layout": "prompt|steps|terminal", "label_priority": ...,
     "episodes": {env_seed: {"E": bf16[n_tok, D],
                             "boundaries": [0, b_prompt, b_step1, ..., b_stepN, n_tok],
                             "output_sha1": str}}}

boundaries 규약 (길이 N+3):
    구간 = [프롬프트][step 1]...[step N][터미널]
    b[0]=0, b[1]=프롬프트 끝, b[1+k]=step k 끝, b[-2]=step N 끝(=터미널 시작), b[-1]=전체 끝

경로: <hidden_root>/<task>/<level>/<status>/chunk_*.pt
    → group id = "<task>/<level>/<status>/<seed>"  (같은 level 이 여러 task 에 있으므로
      task 까지 넣어야 glob 을 섞었을 때 충돌하지 않는다)

대표 벡터: 구간의 마지막 토큰 E[end-1-offset].
    step k 의 마지막 토큰 = "Step k+1" 마커 직전 토큰. 논문(Sun et al.)은 같은 벡터를
    "Step k+1 activation" 이라 부른다 — 여기서는 구간 기준으로 step_k 라 부른다.
    터미널 구간의 마지막 토큰은 "answer" 클래스.

    --with-prompt 를 주면 프롬프트 구간의 마지막 토큰도 "prompt" 클래스로 넣는다.
    (논문의 "Step 1" 이 이 자리. 토큰 종류부터 달라서 거의 항상 분리되므로 기본은 제외)

Usage:
    python probing.py --pt 'latent/hidden_states/plan/*/*/chunk_*.pt' --output out/plan_all
    python probing.py --pt 'latent/hidden_states/plan/BabyAI-GoTo-v0/*/chunk_*.pt' \
        --output out/plan_goto --offset 1 --cv
"""
import json
import glob
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROMPT_LABEL = 0          # step_num 0 = 프롬프트 끝
ANSWER_LABEL = -1         # step_num -1 = 터미널(정답 문장)


# ---------------------------------------------------------------- .pt 로딩

def load_pt(patterns, offset, with_prompt=False, n_steps=None):
    """구간별 대표 벡터를 모은다.

    반환: dict(X, step_num, group) — step_num 은 1..N 이 step, 0 이 프롬프트,
    -1 이 터미널. 라벨링은 make_binary 가 담당한다.

    n_steps: 지정하면 그 step 수인 에피소드만 쓴다. 프롬프트가 고정 목차를 주지만
    모델이 가끔 다르게 쓰는데, step 수가 섞이면 step_N probe 의 positive 를 못 내는
    에피소드가 생겨서 probe 가 "N번째 구간인가"가 아니라 "N step 까지 쓴
    에피소드인가"를 학습할 여지가 있다.
    실측: decompose 6 step 99.9% / plan 5 step 99.1%.
    """
    Xs, Ns, Gs = [], [], []
    stats = Counter()

    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    if not paths:
        raise SystemExit(f"no files: {patterns}")

    for p in paths:
        p = Path(p)
        # .../<task>/<level>/<status>/chunk_XXXX.pt
        status, level, task = p.parent.name, p.parents[1].name, p.parents[2].name

        d = torch.load(p, map_location="cpu", weights_only=False)

        for seed, ep in d["episodes"].items():
            gid = f"{task}/{level}/{status}/{seed}"
            E = ep["E"].float().numpy()
            b = [int(x) for x in ep["boundaries"]]
            n_tok = E.shape[0]

            # 규약: [0, prompt, step1..stepN, terminal 끝] → 최소 길이 4 (step 1개)
            if len(b) < 4 or b[0] != 0 or b[-1] != n_tok:
                stats["bad_boundaries"] += 1
                continue
            if any(x >= y for x, y in zip(b, b[1:])):
                stats["nonmonotonic_boundaries"] += 1
                continue
            if n_steps is not None and len(b) - 3 != n_steps:
                stats["wrong_n_steps"] += 1
                continue
            stats["episodes"] += 1

            def take(start, end, label):
                i = end - 1 - offset
                if i < start:
                    stats["segment_too_short"] += 1
                    return
                Xs.append(E[i]); Ns.append(label); Gs.append(gid)

            if with_prompt:
                take(b[0], b[1], PROMPT_LABEL)
            for k, (s, e) in enumerate(zip(b[1:-2], b[2:-1]), start=1):   # step 1..N
                take(s, e, k)
            take(b[-2], b[-1], ANSWER_LABEL)                              # 터미널

    if not Xs:
        raise SystemExit("벡터 없음 — boundaries 규약 확인")

    data = dict(X=np.stack(Xs).astype(np.float32),
                step_num=np.array(Ns, np.int32),
                group=np.array(Gs, dtype=object))
    print(f"loaded: {dict(stats)}")
    print(f"  X={data['X'].shape}  구간별 개수={dict(sorted(Counter(Ns).items()))}  "
          f"groups={len(set(Gs))}")
    return data


# ---------------------------------------------------------------- 데이터셋 / 모델

def target_name(label: int) -> str:
    if label == PROMPT_LABEL:
        return "prompt"
    if label == ANSWER_LABEL:
        return "answer"
    return f"step_{label}"


def make_binary(data, label):
    """one-vs-rest: 해당 구간이 positive, 나머지 구간 전부가 negative."""
    m = data["step_num"] == label
    X = np.vstack([data["X"][m], data["X"][~m]])
    y = np.r_[np.ones(m.sum()), np.zeros((~m).sum())].astype(int)
    g = np.concatenate([data["group"][m], data["group"][~m]])
    return X, y, g


def split(X, y, g, test_size, seed, group):
    if group:
        return next(GroupShuffleSplit(1, test_size=test_size,
                                      random_state=seed).split(X, y, g))
    return train_test_split(np.arange(len(y)), test_size=test_size,
                            random_state=seed, stratify=y)


def fit(Xtr, ytr, seed, cv, scale):
    if cv:
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, scoring="roc_auc",
                                  class_weight="balanced", max_iter=2000,
                                  random_state=seed, n_jobs=-1)
    else:
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    return (make_pipeline(StandardScaler(), lr) if scale else make_pipeline(lr)).fit(Xtr, ytr)


def evaluate(clf, X, y):
    p = clf.predict_proba(X)[:, 1]
    yhat = (p > 0.5).astype(int)
    return dict(acc=accuracy_score(y, yhat), f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", nargs="+", required=True, help="chunk_*.pt glob (여러 개 가능)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="구간 마지막 토큰에서 몇 칸 앞을 대표로 쓸지 (ablation)")
    ap.add_argument("--with-prompt", action="store_true",
                    help="프롬프트 구간 마지막 토큰도 클래스로 포함")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="이 step 수인 에피소드만 사용. 모델이 목차를 다르게 쓴 소수를 "
                         "제외한다 (decompose=6, plan=5)")
    ap.add_argument("--max-step", type=int, default=None,
                    help="이 번호를 넘는 step 은 probe 대상에서 제외 (negative 로는 남음)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-group-split", action="store_true",
                    help="에피소드 단위 분할을 끈다 (같은 에피소드가 train/test 로 쪼개짐)")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--cv", action="store_true")
    a = ap.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "classifiers").mkdir(exist_ok=True)

    data = load_pt(a.pt, a.offset, a.with_prompt, a.n_steps)
    group, scale = not a.no_group_split, not a.no_scale
    if group and len(np.unique(data["group"])) < 2:
        print("[warn] 그룹이 1개뿐 → 행 단위 split 으로 대체")
        group = False

    labels = sorted(np.unique(data["step_num"]).tolist())
    if a.max_step is not None:
        labels = [l for l in labels if l <= a.max_step]
    # 보기 좋은 순서: prompt, step 1..N, answer
    labels = ([PROMPT_LABEL] if PROMPT_LABEL in labels else []) \
             + [l for l in labels if l > 0] \
             + ([ANSWER_LABEL] if ANSWER_LABEL in labels else [])

    print(f"targets={[target_name(l) for l in labels]} "
          f"group={group} scale={scale} cv={a.cv} offset={a.offset}")

    rows = []
    for label in labels:
        t = target_name(label)
        X, y, g = make_binary(data, label)
        if y.sum() < 2 or (1 - y).sum() < 2:
            print(f"[skip] {t}: 샘플 부족")
            continue
        for s in a.seeds:
            tr, te = split(X, y, g, a.test_size, s, group)
            if len(np.unique(y[te])) < 2:
                print(f"[skip] {t} seed={s}: test 에 한 클래스만")
                continue
            clf = fit(X[tr], y[tr], s, a.cv, scale)
            m_tr, m_te = evaluate(clf, X[tr], y[tr]), evaluate(clf, X[te], y[te])
            lr = clf[-1]
            rows.append(dict(target=t, seed=s, n_train=int(len(tr)), n_test=int(len(te)),
                             n_pos_test=int(y[te].sum()),
                             C=float(np.atleast_1d(getattr(lr, "C_", lr.C))[0]),
                             **{f"train_{k}": float(v) for k, v in m_tr.items()},
                             **{f"test_{k}": float(v) for k, v in m_te.items()}))
            with open(a.output / "classifiers" / f"{t}_seed{s}.pkl", "wb") as f:
                pickle.dump(clf, f)
            print(f"{t:8s} seed={s:5d}  AUC={m_te['auc']:.3f} "
                  f"Acc={m_te['acc']:.3f} F1={m_te['f1']:.3f}")

    if not rows:
        raise SystemExit("학습된 probe 가 없다")

    with open(a.output / "results_all.json", "w") as f:
        json.dump(rows, f, indent=2)

    summary = {}
    for label in labels:
        t = target_name(label)
        r = [x for x in rows if x["target"] == t]
        if r:
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

    ts = list(summary)
    x, w = np.arange(len(ts)), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (k, lab) in enumerate([("test_auc", "AUC"), ("test_acc", "Acc"), ("test_f1", "F1")]):
        ax.bar(x + (i - 1) * w, [summary[t][k]["mean"] for t in ts], w,
               yerr=[summary[t][k]["std"] for t in ts], capsize=3, label=lab)
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ts)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Last-layer probes (seeds={len(a.seeds)}, group={group}, offset={a.offset})")
    fig.tight_layout()
    fig.savefig(a.output / "summary.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved → {a.output}")


if __name__ == "__main__":
    main()