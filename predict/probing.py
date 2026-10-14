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

입력 종류 (--source). 라벨은 모두 같다: 해당 구간이면 참, 다른 구간이면 거짓.
    hidden   [1] hidden_states 의 구간 마지막 토큰 (위 설명)
                 --pct 5 10 30 … 100 을 주면 구간마다 q% 지점 토큰 E[s + pct_position(q, n)] 을 q 마다 하나씩
                 샘플로 넣는다 (누적 없음, 그 위치의 마지막 레이어 벡터 그대로; 100 = 구간 마지막 토큰).
                 학습은 모든 q 를 섞어 probe 하나를 만들고, 평가는 test 에피소드에서 q 마다 따로 잰다
                 (summary.json 의 by_pct, summary.png 는 q 축 AUC 곡선). 짧은 구간에서 여러 q 가 같은
                 토큰을 가리켜도 q 마다 행을 따로 둔다. --offset 은 --pct 와 같이 못 쓴다.
                 프롬프트(--with-prompt)는 항상 마지막 토큰 하나.
    spectral [2] latent/spectral 저장본의 구간 마지막 e_t (e[t][-1], 구간마다 kd 벡터 하나).
                 n_back ≥ 1 이거나 --all 로 저장한 파일이면 어느 것이든 된다. --with-prompt/--offset 은 무시된다.
    edges    [3-5] 같은 저장본의 구간 안 토큰별 누적 e_i (inference/spectral.py --n-front 5 --n-back 5).
                 누적은 구간 시작에서 리셋되고 e_i 하나하나가 샘플이다. --all 로 저장한 파일은 --part both 만 된다.
                 --part both  [3] "Step N" 헤더 토큰을 뺀 앞 5개 + 마지막 5개
                 --part front [4] 헤더를 뺀 앞 5개
                 --part back  [5] 마지막 5개 (맨 끝 하나 = [2] 의 e_t)
                 터미널 구간은 정답 앞 형식 문구("<START>" 등)를 헤더처럼 뺀다.
                 형식 문구 + 앞 5 + 뒤 5 보다 짧은 구간은 3/4/5 모두에서 뺀다 (--keep-short 로 끔).
                 --k / --sign-mode 로 spectral 설정을 고른다.
    --max-step M 은 step M 까지만 probe 한다. --drop-above-max 를 같이 주면 그 뒤 step 은
    음성 샘플에서도 빠진다.
    분할은 어느 입력이든 에피소드 단위라 같은 에피소드의 샘플이 train/test 에 섞이지 않는다.

교차 평가 (--test-pt). 주면 --pt 전체로 학습하고 --test-pt 전체로 평가한다 (split 없음, seed 는 첫 것 하나만).
    예: decompose 로 학습한 probe 를 plan 에서 평가. 두 쪽 모두 같은 --source/--pct 옵션으로 읽는다.
    출력 형식은 그대로 (q 별 AUC 포함). --max-step/--drop-above-max 는 양쪽에 같이 적용된다.

학습 장치 (--device). cpu 는 sklearn (StandardScaler + LogisticRegression lbfgs, balanced).
    cuda / cuda:N 은 같은 목적함수 — Σ_i s_i·logloss_i / n + ‖w‖² / (2·C·n), s_i = balanced 가중치,
    표준화는 train 평균/표준편차 — 를 torch LBFGS 로 GPU 에서 푼다. X 전체를 GPU 에 한 번 올리고
    타깃·seed 마다 train 행만 모아 fit 한다. 저장되는 pkl 은 두 경로 모두 sklearn Pipeline 이라
    predict_proba 로 똑같이 읽힌다. --cv 는 cpu 에서만 된다.

Usage:
    python probing.py --pt 'latent/hidden_states/plan/*/*/chunk_*.pt' --output out/plan_all
    python probing.py --source spectral \
        --pt 'latent/spectral/*/*/*/k8_scaled_sign-data_f0_b1/chunk_*.pt' --output out/spec_all
    python probing.py --source edges --part front \
        --pt 'latent/spectral/*/*/*/k8_scaled_sign-data_f5_b5/chunk_*.pt' --output out/edge_front
    python probing.py --pt 'latent/hidden_states/plan/BabyAI-GoTo-v0/*/chunk_*.pt' \
        --output out/plan_goto --offset 1 --cv
    python probing.py --pt 'latent/hidden_states/decompose/BabyAI-GoTo-v0/*/chunk_*.pt' \
        --pct 5 10 30 50 70 90 95 --output out/hidden_pct_goto     # 학습 전체 q, 평가 q 별
    python probing.py --device cuda:0 --pt '...' --pct 1 2 3 4 5 10 50 90 95 100 --output out/x   # GPU fit
    python probing.py --device cuda:0 --pct 5 10 50 90 95 100 --output out/dec2plan \
        --pt 'latent/hidden_states/decompose/*/*/chunk_*.pt' --test-pt 'latent/hidden_states/plan/*/*/chunk_*.pt'
"""
import sys
import json
import glob
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROMPT_LABEL = 0          # step_num 0 = 프롬프트 끝
ANSWER_LABEL = -1         # step_num -1 = 터미널(정답 문장)


# ---------------------------------------------------------------- .pt 로딩

SOURCES = ("hidden", "spectral", "edges")


def pct_position(q: int, n: int) -> int:
    """구간 길이 n 의 q% 지점 (0-based 위치). 누적 토큰 수 t = ⌈q·n/100⌉ 이고 위치는 t-1.
    q=100 → n-1 (구간 마지막), 작은 q 는 최소 0. inference/spectral.py 의 pct_position 과 같은 규칙
    (spectral --pct 저장본과 위치가 일치해야 하므로 바꾸면 둘 다 바꿀 것)."""
    t = -(-q * n // 100)
    return min(max(t, 1), n) - 1


def _paths(patterns):
    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    if not paths:
        raise SystemExit(f"no files: {patterns}")
    return [Path(p) for p in paths]


def _gid(p: Path, source: str, seed) -> str:
    """hidden: .../<task>/<level>/<status>/chunk.pt
       spectral/edges: .../<task>/<level>/<status>/<tag>/chunk.pt"""
    base = p.parent if source == "hidden" else p.parents[1]
    status, level, task = base.name, base.parent.name, base.parents[1].name
    return f"{task}/{level}/{status}/{seed}"


def load_pt(patterns, offset, with_prompt=False, pct=None):
    """[1] hidden_states: 구간 마지막 토큰 E[end-1-offset] 하나.
    pct 가 있으면 구간마다 q% 지점 토큰을 q 마다 하나씩 (offset 은 0 이어야 한다). 행마다 q 를 Ps 에 남긴다."""
    Xs, Ns, Gs, Ps = [], [], [], []
    stats = Counter()
    if pct and offset:
        raise SystemExit("--pct 와 --offset 은 같이 못 쓴다")

    for p in _paths(patterns):
        d = torch.load(p, map_location="cpu", weights_only=False)

        for seed, ep in d["episodes"].items():
            gid = _gid(p, "hidden", seed)
            E = ep["E"]                                   # bf16 그대로 두고 뽑는 행만 fp32 로
            b = [int(x) for x in ep["boundaries"]]
            n_tok = E.shape[0]

            # 규약: [0, prompt, step1..stepN, terminal 끝] → 최소 길이 4 (step 1개)
            if len(b) < 4 or b[0] != 0 or b[-1] != n_tok:
                stats["bad_boundaries"] += 1
                continue
            if any(x >= y for x, y in zip(b, b[1:])):
                stats["nonmonotonic_boundaries"] += 1
                continue
            stats["episodes"] += 1

            def take(start, end, label):
                if pct:                                       # q% 지점 토큰, q 마다 한 행
                    for q in pct:
                        i = start + pct_position(q, end - start)
                        Xs.append(E[i].float().numpy()); Ns.append(label); Gs.append(gid); Ps.append(q)
                    return
                i = end - 1 - offset
                if i < start:
                    stats["segment_too_short"] += 1
                    return
                Xs.append(E[i].float().numpy()); Ns.append(label); Gs.append(gid); Ps.append(100)   # view 금지: E 전체가 살아남는다

            if with_prompt:                                   # 프롬프트는 항상 마지막 토큰 하나
                i = b[1] - 1 - offset
                Xs.append(E[i].float().numpy()); Ns.append(PROMPT_LABEL); Gs.append(gid); Ps.append(100)
            for k, (s, e) in enumerate(zip(b[1:-2], b[2:-1]), start=1):   # step 1..N
                take(s, e, k)
            take(b[-2], b[-1], ANSWER_LABEL)                              # 터미널

    return _pack(Xs, Ns, Gs, stats, Ps)


def _seg_label(t: int, n_seg: int) -> int:
    """gen_view 구간 번호 t (0..N-1 = step 1..N, N = 터미널) → step_num."""
    return ANSWER_LABEL if t == n_seg - 1 else t + 1


def _config_ok(d, p, k, sign_mode, seen, stats):
    """헤더의 k / sign_mode 로 파일을 거른다. 서로 다른 설정이 섞이면 차원이 달라 멈춘다."""
    cfg = (d.get("k"), d.get("sign_mode"), d.get("scale"))
    if (k is not None and cfg[0] != k) or (sign_mode is not None and cfg[1] != sign_mode):
        stats["file_config_skipped"] += 1
        return False
    seen.add(cfg)
    if len(seen) > 1:
        raise SystemExit(f"서로 다른 spectral 설정이 섞였다 {sorted(seen, key=str)} — "
                         f"--k / --sign-mode 로 하나만 고르거나 glob 을 좁힐 것 ({p})")
    return True


def load_spectral(patterns, k=None, sign_mode=None):
    """[2] spectral: 구간 전체를 누적한 e_t = e[t][-1] (구간마다 하나, 마지막 위치가 저장된 구간만)."""
    Xs, Ns, Gs = [], [], []
    stats, seen = Counter(), set()
    for p in _paths(patterns):
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not _config_ok(d, p, k, sign_mode, seen, stats):
            continue
        for seed, ep in d["episodes"].items():
            gid = _gid(p, "spectral", seed)
            seg, n_seg = ep["seg"], len(ep["e"])
            stats["episodes"] += 1
            for t in sorted(ep["e"]):
                ps = ep["pos"][t]
                if not ps or ps[-1] != seg[t + 1] - seg[t] - 1:      # 마지막 토큰이 저장돼 있어야 e_t
                    stats["no_last_token"] += 1
                    continue
                Xs.append(ep["e"][t][-1].float().numpy()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
    return _pack(Xs, Ns, Gs, stats)


def load_edges(patterns, part, k=None, sign_mode=None, keep_short=False):
    """[3-5] edges: 구간 안 토큰별 누적 e_i 중 앞(형식 문구 제외)/뒤 가장자리.

    part: both (3) | front (4) | back (5). e_i 하나하나가 샘플이고 라벨은 그 구간.
    저장본은 위치 pos 와 e 만 갖고 있으므로 front/back 은 헤더의 n_front/n_back 과 marker 로 가른다:
    front = marker ≤ p < marker+n_front, back = p ≥ n-n_back. --all 저장본은 part both 만 된다.

    짧은 구간 제외: 구간 길이 < marker + n_front + n_back 이면 (앞/뒤가 겹치거나 모자람)
    그 구간은 통째로 뺀다 — 양성으로도 음성으로도 안 쓴다. part 와 상관없이 같은 기준이라
    3/4/5 가 같은 구간 집합 위에서 비교된다. keep_short=True 면 끈다 (all 저장본은 해당 없음).
    """
    Xs, Ns, Gs = [], [], []
    stats, seen = Counter(), set()
    for p in _paths(patterns):
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not _config_ok(d, p, k, sign_mode, seen, stats):
            continue
        n_front, n_back, is_all = d["n_front"], d["n_back"], d.get("all", False)
        if is_all and part != "both":
            raise SystemExit(f"--all 로 저장한 파일은 --part both 만 가능 ({p})")
        for seed, ep in d["episodes"].items():
            gid = _gid(p, "edges", seed)
            n_seg = len(ep["e"])
            stats["episodes"] += 1
            stats[f"marker_{ep.get('marker_src', 'text')}"] += 1
            seg = ep["seg"]
            for t in range(n_seg):
                n, m = seg[t + 1] - seg[t], ep["marker"][t]
                if not is_all and n < m + n_front + n_back:
                    stats[f"short_{target_name(_seg_label(t, n_seg))}"] += 1
                    if not keep_short:
                        continue
                for pos, row in zip(ep["pos"][t], ep["e"][t]):
                    if part == "front" and not (m <= pos < m + n_front):
                        continue
                    if part == "back" and pos < n - n_back:
                        continue
                    Xs.append(row.float().numpy()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
    return _pack(Xs, Ns, Gs, stats)


def _pack(Xs, Ns, Gs, stats, Ps=None):
    """pct: 행마다 "구간의 몇 % 지점인가" (Ps 가 없으면 전부 100 = 구간 끝). 평가를 % 별로 나눌 때 쓴다."""
    if not Xs:
        raise SystemExit("벡터 없음 — 입력 경로/규약 확인")
    P = np.array(Ps if Ps is not None else [100] * len(Ns), np.int32)
    data = dict(X=np.stack(Xs).astype(np.float32),
                step_num=np.array(Ns, np.int32),
                group=np.array(Gs, dtype=object),
                pct=P,
                domain=np.zeros(len(Ns), np.int8))        # 0 = 학습 데이터, 1 = --test-pt (concat_data)
    print(f"loaded: {dict(stats)}")
    print(f"  X={data['X'].shape}  구간별 개수={dict(sorted(Counter(Ns).items()))}  "
          f"groups={len(set(Gs))}"
          + (f"  pct별 개수={dict(sorted(Counter(P.tolist()).items()))}" if len(set(P.tolist())) > 1 else ""))
    return data


def concat_data(train, test):
    """--test-pt: 두 데이터를 이어 붙이고 domain 으로 구분한다 (0 = train, 1 = test)."""
    test = dict(test, domain=np.ones(len(test["step_num"]), np.int8))
    return {k: np.concatenate([train[k], test[k]]) for k in train}


# ---------------------------------------------------------------- 데이터셋 / 모델

def target_name(label: int) -> str:
    if label == PROMPT_LABEL:
        return "prompt"
    if label == ANSWER_LABEL:
        return "answer"
    return f"step_{label}"


def make_binary(data, label):
    """one-vs-rest: 해당 구간이 positive, 나머지 구간 전부가 negative.
    X 는 복사하지 않고 행 순서 order (positive 먼저, 그 다음 negative) 만 돌려준다 — X[order] 가 예전의 X.
    split 이 주는 tr/te 는 order 안의 위치이므로 실제 행은 order[tr], order[te]."""
    m = data["step_num"] == label
    order = np.r_[np.flatnonzero(m), np.flatnonzero(~m)]
    y = np.r_[np.ones(m.sum()), np.zeros((~m).sum())].astype(int)
    return order, y, data["group"][order], data["pct"][order], data["domain"][order]


def split(y, g, test_size, seed, group):
    if group:
        return next(GroupShuffleSplit(1, test_size=test_size,
                                      random_state=seed).split(np.zeros((len(y), 1)), y, g))
    return train_test_split(np.arange(len(y)), test_size=test_size,
                            random_state=seed, stratify=y)


def fit_cpu(Xtr, ytr, seed, cv, scale):
    """sklearn 경로. 반환: (Pipeline, C)."""
    if cv:
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, scoring="roc_auc",
                                  class_weight="balanced", max_iter=2000,
                                  random_state=seed, n_jobs=-1)
    else:
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf = (make_pipeline(StandardScaler(), lr) if scale else make_pipeline(lr)).fit(Xtr, ytr)
    return clf, float(np.atleast_1d(getattr(lr, "C_", lr.C))[0])


class GpuProbe:
    """X 전체를 GPU 에 올려 두고, 행 인덱스로 fit / 확률 계산을 한다.

    목적함수는 sklearn LogisticRegression(class_weight="balanced", penalty=l2, C) + StandardScaler 와 같다:
        (1/n) Σ_i s_i · log(1 + exp(-y_i z_i)) + ‖w‖² / (2 C n),  z = ((x - μ)/σ)·w + b,
        s_i = n / (2 n_class(i)),  μ/σ = train 행의 평균/모표준편차 (σ=0 이면 1).
    표준화는 X 를 바꾸지 않고 w/σ, b - (μ/σ)·w 로 접어 넣어 매 iteration 행렬곱 한 번이다.
    """

    def __init__(self, X: np.ndarray, device):
        self.device = torch.device(device)
        self.X = torch.from_numpy(X).to(self.device)              # fp32 (n, d)
        self.n, self.d = self.X.shape
        self.mean = self.std = self.w = self.b = None

    def fit(self, rows, y, scale=True, C=1.0, max_iter=2000, tol=1e-4):
        dev, d = self.device, self.d
        Xtr = self.X[torch.as_tensor(np.asarray(rows), device=dev)]
        n = Xtr.shape[0]
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=dev)
        n_pos = yt.sum()
        sw = torch.where(yt > 0.5, n / (2 * n_pos), n / (2 * (n - n_pos)))      # balanced
        sign = 2 * yt - 1
        if scale:
            mean = Xtr.mean(0)
            std = ((Xtr - mean).square().mean(0)).sqrt()
            std = torch.where(std > 0, std, torch.ones_like(std))
        else:
            mean, std = torch.zeros(d, device=dev), torch.ones(d, device=dev)
        beta = torch.zeros(d + 1, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([beta], lr=1.0, max_iter=max_iter, tolerance_grad=tol,
                                tolerance_change=1e-9, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            w, b = beta[:d], beta[d]
            ws = w / std
            z = Xtr @ ws + (b - mean @ ws)
            loss = (sw * F.softplus(-sign * z)).sum() / n + (w @ w) / (2 * C * n)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            self.w, self.b = beta[:d].clone(), beta[d].clone()
        self.mean, self.std, self.C = mean, std, C
        del Xtr
        torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def proba_all(self) -> np.ndarray:
        """X 의 모든 행에 대한 P(y=1) (원래 행 순서)."""
        ws = self.w / self.std
        z = self.X @ ws + (self.b - self.mean @ ws)
        return torch.sigmoid(z).cpu().numpy()

    def to_sklearn(self):
        """같은 μ/σ/w/b 를 가진 sklearn Pipeline (StandardScaler → LogisticRegression). pkl 호환용."""
        mean, std = self.mean.cpu().numpy().astype(np.float64), self.std.cpu().numpy().astype(np.float64)
        sc = StandardScaler()
        sc.mean_, sc.scale_, sc.var_ = mean, std, std ** 2
        sc.n_features_in_, sc.n_samples_seen_ = self.d, self.n
        lr = LogisticRegression(class_weight="balanced", C=self.C)
        lr.coef_ = self.w.cpu().numpy().astype(np.float64)[None]
        lr.intercept_ = np.array([float(self.b)])
        lr.classes_, lr.n_features_in_ = np.array([0, 1]), self.d
        return make_pipeline(sc, lr)


def evaluate(p, y):
    """p = P(y=1) 배열."""
    yhat = (p > 0.5).astype(int)
    return dict(acc=accuracy_score(y, yhat), f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", nargs="+", required=True, help="chunk_*.pt glob (여러 개 가능)")
    ap.add_argument("--test-pt", nargs="+", default=None, metavar="GLOB",
                    help="주면 --pt 전체로 학습, 이 glob 전체로 평가 (교차 평가; split 없음, seed 하나)")
    ap.add_argument("--source", choices=SOURCES, default="hidden",
                    help="hidden: 구간 마지막 토큰 hidden (1) | spectral: 구간 전체 누적 e_t (2) | "
                         "edges: 구간 앞/뒤 토큰별 누적 e_i (3-5, --part)")
    ap.add_argument("--part", choices=("both", "front", "back"), default="both",
                    help="--source edges 일 때: both=앞5+뒤5 (3), front=헤더 뺀 앞5 (4), back=뒤5 (5)")
    ap.add_argument("--k", type=int, default=None,
                    help="spectral/edges: 이 k 인 파일만 쓴다 (기본: glob 이 잡은 그대로)")
    ap.add_argument("--sign-mode", default=None, choices=("none", "first", "max", "data"),
                    help="spectral/edges: 이 부호 규칙인 파일만 쓴다")
    ap.add_argument("--keep-short", action="store_true",
                    help="edges: marker+앞+뒤 보다 짧은 구간도 넣는다 (기본은 제외)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="구간 마지막 토큰에서 몇 칸 앞을 대표로 쓸지 (ablation)")
    ap.add_argument("--pct", type=int, nargs="+", default=None, metavar="Q",
                    help="hidden 소스: 구간 길이의 Q%% 지점 토큰을 q 마다 하나씩 샘플로 (예: 5 10 30 50 70 90 95; "
                         "학습은 전체 q, 평가는 q 별. 100 = 구간 마지막 토큰)")
    ap.add_argument("--with-prompt", action="store_true",
                    help="프롬프트 구간 마지막 토큰도 클래스로 포함")
    ap.add_argument("--max-step", type=int, default=None,
                    help="이 번호를 넘는 step 은 probe 대상에서 제외 (기본: negative 로는 남음)")
    ap.add_argument("--drop-above-max", action="store_true",
                    help="--max-step 을 넘는 step 샘플을 데이터에서 아예 뺀다 (negative 로도 안 씀)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-group-split", action="store_true",
                    help="에피소드 단위 분할을 끈다 (같은 에피소드가 train/test 로 쪼개짐)")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="cpu = sklearn | cuda / cuda:N = torch LBFGS 로 GPU fit (같은 목적함수, --cv 불가)")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)                 # 로그 파일로 보내도 바로 보이게

    use_gpu = a.device != "cpu"
    if use_gpu:
        if a.cv:
            raise SystemExit("--cv 는 --device cpu 에서만 된다")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA 를 쓸 수 없다 — --device cpu 로")

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "classifiers").mkdir(exist_ok=True)

    if a.pct is not None:
        if a.source != "hidden":
            raise SystemExit("--pct 는 --source hidden 에서만 쓴다")
        if any(not (0 < q <= 100) for q in a.pct):
            raise SystemExit(f"--pct 는 0 < Q ≤ 100 (got {a.pct})")
    def load(patterns):
        if a.source == "hidden":
            return load_pt(patterns, a.offset, a.with_prompt, sorted(set(a.pct)) if a.pct else None)
        if a.source == "spectral":
            return load_spectral(patterns, a.k, a.sign_mode)
        return load_edges(patterns, a.part, a.k, a.sign_mode, a.keep_short)

    data = load(a.pt)
    transfer = a.test_pt is not None
    if transfer:
        print("--test-pt: 평가 데이터 로딩")
        data = concat_data(data, load(a.test_pt))
        if len(a.seeds) > 1:
            print(f"--test-pt 는 split 이 없어 seed 는 {a.seeds[0]} 하나만 쓴다")
            a.seeds = a.seeds[:1]
    if a.max_step is not None and a.drop_above_max:
        keep = data["step_num"] <= a.max_step          # answer(-1)·prompt(0) 는 남는다
        print(f"--drop-above-max: step>{a.max_step} 샘플 {int((~keep).sum())}개 제거")
        data = {k_: v[keep] for k_, v in data.items()}
    group, scale = not a.no_group_split, not a.no_scale
    X = data["X"]
    gpu = GpuProbe(X, a.device) if use_gpu else None
    if gpu is not None:
        print(f"X → {a.device}: {tuple(X.shape)} {X.nbytes / 2**30:.1f} GB")
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

    pcts = sorted(set(data["pct"].tolist()))                 # 평가를 나눌 % 지점 (하나뿐이면 전체와 같다)
    if len(pcts) == 1:
        pcts = []
    print(f"targets={[target_name(l) for l in labels]} "
          f"source={a.source} part={a.part} pct={a.pct} eval_pcts={pcts or '-'} "
          f"group={group} scale={scale} cv={a.cv} offset={a.offset} device={a.device} transfer={transfer}")

    rows = []
    for label in labels:
        t = target_name(label)
        order, y, g, q_of, dom = make_binary(data, label)
        if y.sum() < 2 or (1 - y).sum() < 2:
            print(f"[skip] {t}: 샘플 부족")
            continue
        for s in a.seeds:
            if transfer:
                tr, te = np.flatnonzero(dom == 0), np.flatnonzero(dom == 1)
            else:
                tr, te = split(y, g, a.test_size, s, group)
            if len(np.unique(y[te])) < 2:
                print(f"[skip] {t} seed={s}: test 에 한 클래스만")
                continue
            if gpu is not None:
                gpu.fit(order[tr], y[tr], scale=scale)
                p_all, clf, C = gpu.proba_all(), gpu.to_sklearn(), gpu.C
            else:
                clf, C = fit_cpu(X[order[tr]], y[tr], s, a.cv, scale)
                p_all = clf.predict_proba(X)[:, 1]
            p = p_all[order]                                   # order 기준 확률 → tr/te 로 바로 인덱싱
            m_tr, m_te = evaluate(p[tr], y[tr]), evaluate(p[te], y[te])
            rows.append(dict(target=t, seed=s, eval_pct="all", n_train=int(len(tr)), n_test=int(len(te)),
                             n_pos_test=int(y[te].sum()), C=C,
                             **{f"train_{k}": float(v) for k, v in m_tr.items()},
                             **{f"test_{k}": float(v) for k, v in m_te.items()}))
            per_pct = ""
            for q in pcts:                                   # 같은 probe 를 test 의 q% 지점 행에만
                te_q = te[q_of[te] == q]
                if len(te_q) == 0 or len(np.unique(y[te_q])) < 2:
                    continue
                m_q = evaluate(p[te_q], y[te_q])
                rows.append(dict(target=t, seed=s, eval_pct=int(q), n_train=int(len(tr)), n_test=int(len(te_q)),
                                 n_pos_test=int(y[te_q].sum()), C=C,
                                 **{f"test_{k}": float(v) for k, v in m_q.items()}))
                per_pct += f" {q}%={m_q['auc']:.3f}"
            with open(a.output / "classifiers" / f"{t}_seed{s}.pkl", "wb") as f:
                pickle.dump(clf, f)
            print(f"{t:8s} seed={s:5d}  AUC={m_te['auc']:.3f} "
                  f"Acc={m_te['acc']:.3f} F1={m_te['f1']:.3f}" + (f"  | AUC by pct:{per_pct}" if per_pct else ""))

    if not rows:
        raise SystemExit("학습된 probe 가 없다")

    with open(a.output / "results_all.json", "w") as f:
        json.dump(rows, f, indent=2)

    def agg(r):
        return {k: dict(mean=float(np.mean([x[k] for x in r])), std=float(np.std([x[k] for x in r])))
                for k in ("test_auc", "test_acc", "test_f1")} | {"n_seeds": len(r)}

    summary = {}
    for label in labels:
        t = target_name(label)
        r = [x for x in rows if x["target"] == t and x["eval_pct"] == "all"]
        if r:
            summary[t] = agg(r)
            if pcts:
                summary[t]["by_pct"] = {str(q): agg(rq) for q in pcts
                                        if (rq := [x for x in rows if x["target"] == t and x["eval_pct"] == q])}
    with open(a.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 62)
    print(f"{'target':8s} {'AUC':>15s} {'Acc':>15s} {'F1':>15s}")
    for t, s in summary.items():
        fmt = lambda k: f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}"
        print(f"{t:8s} {fmt('test_auc'):>15s} {fmt('test_acc'):>15s} {fmt('test_f1'):>15s}")
    if pcts:
        print("\nAUC by eval pct (학습은 전체 %, 평가는 그 % 지점만)")
        print(f"{'target':8s}" + "".join(f"{q:>9d}%" for q in pcts))
        for t, s in summary.items():
            print(f"{t:8s}" + "".join(f"{s['by_pct'][str(q)]['test_auc']['mean']:10.3f}"
                                       if str(q) in s.get("by_pct", {}) else f"{'-':>10s}" for q in pcts))

    ts = list(summary)
    src = {"hidden": f"hidden offset={a.offset}" + (f" pct={a.pct}" if a.pct else ""),
           "spectral": "spectral e_t", "edges": f"spectral edges ({a.part})"}[a.source]
    fig, ax = plt.subplots(figsize=(9, 5))
    if pcts:                                                  # q 축 AUC 곡선, 타깃마다 선 하나
        for t in ts:
            bp = summary[t].get("by_pct", {})
            qs = [q for q in pcts if str(q) in bp]
            ax.errorbar(qs, [bp[str(q)]["test_auc"]["mean"] for q in qs],
                        yerr=[bp[str(q)]["test_auc"]["std"] for q in qs], marker="o", capsize=3, label=t)
        ax.set_xticks(pcts)
        ax.set_xlabel("eval position (% of segment)")
        ax.set_ylabel("test AUC")
    else:
        x, w = np.arange(len(ts)), 0.25
        for i, (k, lab) in enumerate([("test_auc", "AUC"), ("test_acc", "Acc"), ("test_f1", "F1")]):
            ax.bar(x + (i - 1) * w, [summary[t][k]["mean"] for t in ts], w,
                   yerr=[summary[t][k]["std"] for t in ts], capsize=3, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(ts)
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Last-layer probes: {src} (seeds={len(a.seeds)}, group={group}"
                 + (", train=--pt → test=--test-pt" if transfer else "") + ")")
    fig.tight_layout()
    fig.savefig(a.output / "summary.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved → {a.output}")


if __name__ == "__main__":
    main()