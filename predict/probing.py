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
    spectral [2] latent/spectral 저장본의 구간 마지막 e_t (e[t][-1], 구간마다 kd 벡터 하나).
                 n_back ≥ 1 이거나 --all 로 저장한 파일이면 어느 것이든 된다. --with-prompt/--offset 은 무시된다.
                 옛 latent/spectral_states 포맷 (e[t] 가 (kd,) 하나, seg/pos 없음) 도 그대로 읽는다.
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
    --n-episodes N 을 주면 로딩 뒤 에피소드 N 개만 남긴다 (success 먼저, 모자라면 failure 로 채움,
    --sample-seed 로 고정). 남긴 에피소드 목록은 <output>/episodes.json 에 쓴다.

Usage:
    python probing.py --pt 'latent/hidden_states/plan/*/*/chunk_*.pt' --output out/plan_all
    python probing.py --source spectral \
        --pt 'latent/spectral/*/*/*/k8_scaled_sign-data_f0_b1/chunk_*.pt' --output out/spec_all
    python probing.py --source edges --part front \
        --pt 'latent/spectral/*/*/*/k8_scaled_sign-data_f5_b5/chunk_*.pt' --output out/edge_front
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

SOURCES = ("hidden", "spectral", "edges")


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


def load_pt(patterns, offset, with_prompt=False):
    """[1] hidden_states: 구간 마지막 토큰 E[end-1-offset] 하나."""
    Xs, Ns, Gs = [], [], []
    stats = Counter()

    for p in _paths(patterns):
        d = torch.load(p, map_location="cpu", weights_only=False)

        for seed, ep in d["episodes"].items():
            gid = _gid(p, "hidden", seed)
            E = ep["E"]                                   # bf16 torch 그대로 (GPU 경로가 그대로 올린다)
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
                i = end - 1 - offset
                if i < start:
                    stats["segment_too_short"] += 1
                    return
                Xs.append(E[i].clone()); Ns.append(label); Gs.append(gid)  # view 면 E 전체가 살아남아 300GB 까지 감

            if with_prompt:
                take(b[0], b[1], PROMPT_LABEL)
            for k, (s, e) in enumerate(zip(b[1:-2], b[2:-1]), start=1):   # step 1..N
                take(s, e, k)
            take(b[-2], b[-1], ANSWER_LABEL)                              # 터미널

    return _pack(Xs, Ns, Gs, stats)


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
            seg, n_seg = ep.get("seg"), len(ep["e"])
            stats["episodes"] += 1
            for t in sorted(ep["e"]):
                row = ep["e"][t]
                if row.dim() == 1:                                    # 옛 spectral_states: 구간마다 e_t 하나
                    Xs.append(row.clone()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
                    continue
                ps = ep["pos"][t]
                if not ps or ps[-1] != seg[t + 1] - seg[t] - 1:      # 마지막 토큰이 저장돼 있어야 e_t
                    stats["no_last_token"] += 1
                    continue
                Xs.append(row[-1].clone()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
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
                    Xs.append(row.clone()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
    return _pack(Xs, Ns, Gs, stats)


def _pack(Xs, Ns, Gs, stats):
    """행 목록 → data. X 는 torch 텐서로, 저장 dtype 그대로 둔다 (bf16 저장본이면 bf16 — fp32 로 바꾸면
    edges 는 수십 GB 가 두 배가 된다). CPU(sklearn) 경로는 main 에서 한 번 float32 numpy 로 바꾼다."""
    if not Xs:
        raise SystemExit("벡터 없음 — 입력 경로/규약 확인")
    X = torch.stack(Xs)
    Xs.clear()
    data = dict(X=X, step_num=np.array(Ns, np.int32), group=np.array(Gs, dtype=object))
    print(f"loaded: {dict(stats)}")
    print(f"  X={tuple(X.shape)} {str(X.dtype).removeprefix('torch.')}  "
          f"구간별 개수={dict(sorted(Counter(Ns).items()))}  groups={len(set(Gs))}")
    return data


def _take(data, idx):
    """행 인덱스로 data 를 줄인다 (X 는 torch, 나머지는 numpy)."""
    idx = np.asarray(idx)
    return {k: (v[torch.as_tensor(idx)] if torch.is_tensor(v) else v[idx]) for k, v in data.items()}


# ---------------------------------------------------------------- 데이터셋 / 모델

def target_name(label: int) -> str:
    if label == PROMPT_LABEL:
        return "prompt"
    if label == ANSWER_LABEL:
        return "answer"
    return f"step_{label}"


def subsample_episodes(data, n, seed):
    """에피소드(group) 단위로 n 개만 남긴다. success 를 먼저 채우고 모자라면 failure 로.

    group id 는 "<task>/<level>/<status>/<seed>" 라 status 를 여기서 읽는다. success 가 n 보다
    많으면 그 안에서 n 개를 무작위로 고른다. 반환: (줄인 data, 남긴 group 목록 정렬)."""
    groups = np.unique(data["group"])
    status = np.array([g.split("/")[2] for g in groups])
    rng = np.random.default_rng(seed)
    succ = rng.permutation(groups[status == "success"])
    fail = rng.permutation(groups[status != "success"])
    keep = np.concatenate([succ[:n], fail[:max(0, n - len(succ))]])
    mask = np.isin(data["group"], keep)
    print(f"--n-episodes {n}: success {min(n, len(succ))}/{len(succ)} + failure "
          f"{max(0, n - len(succ))}/{len(fail)} → {len(keep)} episodes, X={int(mask.sum())} rows")
    if not mask.all():
        data = _take(data, np.flatnonzero(mask))
    return data, sorted(keep.tolist())


def make_binary(data, label):
    """one-vs-rest 라벨: 해당 구간이 1, 나머지 구간 전부 0. X 는 복사하지 않는다 (edges 는 수십 GB)."""
    y = (data["step_num"] == label).astype(int)
    return data["X"], y, data["group"]


def split(n, y, g, test_size, seed, group):
    """행 인덱스 (train, test). group=True 면 에피소드 단위."""
    dummy = np.empty((n, 0))
    if group:
        return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(dummy, y, g))
    return train_test_split(np.arange(n), test_size=test_size, random_state=seed, stratify=y)


class GpuData:
    """X 전체를 GPU 에 한 번만 올려 두고 (저장 dtype 그대로 — bf16 이면 fp32 의 절반), fit / predict 가
    행 인덱스를 청크씩 꺼내 fp32 로 바꿔 쓴다. 70만 x 40960 bf16 ≈ 57GB 가 GPU 에 상주한다."""

    def __init__(self, X: torch.Tensor, device, chunk: int = 32768):
        self.device = torch.device(device)
        self.X = X.to(self.device)
        self.chunk = chunk

    def batches(self, idx):
        """idx 순서대로 (m x D) fp32 청크를 낸다."""
        idx_t = torch.as_tensor(np.asarray(idx), device=self.device)
        for i in range(0, len(idx_t), self.chunk):
            yield self.X[idx_t[i:i + self.chunk]].float()


class TorchLogReg:
    """GPU 로지스틱 회귀. sklearn LogisticRegression(C, L2, class_weight="balanced") 과 같은 목적함수

        C · Σ_i w_i · logloss_i(β)  +  ½‖β_w‖²      (w_i = n / (2·n_class(i)), 절편은 벌점 없음)

    를 torch L-BFGS(strong Wolfe) 로 푼다. 손실과 기울기는 GpuData 청크를 돌며 해석적으로 누적하므로
    autograd 그래프도, X 의 fp32 복사본도 만들지 않는다. scale=True 면 StandardScaler 와 같은 표준화
    (학습 행 기준 평균/표준편차, 2-pass) 를 내장한다. 학습 뒤 파라미터는 CPU 에 둬서 pickle 가능."""

    def __init__(self, C=1.0, scale=True, max_iter=500, tol=1e-5):
        self.C, self.scale, self.max_iter, self.tol = C, scale, max_iter, tol

    def fit(self, gd: GpuData, idx, y):
        dev, d = gd.device, gd.X.shape[1]
        n, n_pos = len(idx), float(np.sum(y))
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=dev)
        w_t = torch.where(y_t > 0.5, torch.tensor(n / (2 * n_pos), device=dev),
                          torch.tensor(n / (2 * (n - n_pos)), device=dev))        # balanced

        if self.scale:                                                           # 2-pass 평균/표준편차
            mean = torch.zeros(d, device=dev)
            for xb in gd.batches(idx):
                mean += xb.sum(0)
            mean /= n
            var = torch.zeros(d, device=dev)
            for xb in gd.batches(idx):
                xb -= mean
                var += (xb * xb).sum(0)
            std = (var / n).sqrt()
            std = torch.where(std > 0, std, torch.ones_like(std))
        else:
            mean, std = torch.zeros(d, device=dev), torch.ones(d, device=dev)

        beta = torch.zeros(d + 1, device=dev, requires_grad=True)
        bce = torch.nn.functional.binary_cross_entropy_with_logits

        def closure():
            with torch.no_grad():
                wv, b = beta[:-1], beta[-1]
                loss = torch.zeros((), device=dev)
                grad = torch.zeros(d + 1, device=dev)
                off = 0
                for xb in gd.batches(idx):
                    m = xb.shape[0]
                    yb, wb = y_t[off:off + m], w_t[off:off + m]
                    off += m
                    xb = (xb - mean) / std
                    z = xb @ wv + b
                    loss += self.C * (wb * bce(z, yb, reduction="none")).sum()
                    r = self.C * wb * (torch.sigmoid(z) - yb)
                    grad[:-1] += xb.T @ r
                    grad[-1] += r.sum()
                loss += 0.5 * (wv * wv).sum()
                grad[:-1] += wv
            beta.grad = grad
            return loss

        opt = torch.optim.LBFGS([beta], lr=1.0, max_iter=self.max_iter, history_size=20,
                                tolerance_grad=self.tol, tolerance_change=1e-9,
                                line_search_fn="strong_wolfe")
        opt.step(closure)
        self.n_iter_ = opt.state[opt._params[0]].get("n_iter", -1)
        self.mean_, self.std_, self.beta_ = mean.cpu(), std.cpu(), beta.detach().cpu()
        return self

    @torch.no_grad()
    def predict_proba(self, gd: GpuData, idx):
        dev = gd.device
        mean, std, beta = self.mean_.to(dev), self.std_.to(dev), self.beta_.to(dev)
        out = [torch.sigmoid(((xb - mean) / std) @ beta[:-1] + beta[-1]).cpu() for xb in gd.batches(idx)]
        p = torch.cat(out).double().numpy()
        return np.stack([1 - p, p], 1)


def fit(Xnp, gd, y, idx, seed, cv, scale):
    """Xnp: CPU 경로용 float32 numpy (GPU 경로면 None). gd: GpuData (CPU 경로면 None)."""
    if gd is not None:
        if cv:
            raise SystemExit("--cv 는 --device cpu 에서만 된다")
        return TorchLogReg(C=1.0, scale=scale).fit(gd, idx, y[idx])
    if cv:
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, scoring="roc_auc",
                                  class_weight="balanced", max_iter=2000,
                                  random_state=seed, n_jobs=-1)
    else:
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    return (make_pipeline(StandardScaler(), lr) if scale else make_pipeline(lr)).fit(Xnp[idx], y[idx])


def evaluate(clf, Xnp, gd, idx, y):
    p = clf.predict_proba(gd, idx)[:, 1] if gd is not None else clf.predict_proba(Xnp[idx])[:, 1]
    yhat = (p > 0.5).astype(int)
    return dict(acc=accuracy_score(y, yhat), f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", nargs="+", required=True, help="chunk_*.pt glob (여러 개 가능)")
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
    ap.add_argument("--with-prompt", action="store_true",
                    help="프롬프트 구간 마지막 토큰도 클래스로 포함")
    ap.add_argument("--max-step", type=int, default=None,
                    help="이 번호를 넘는 step 은 probe 대상에서 제외 (기본: negative 로는 남음)")
    ap.add_argument("--drop-above-max", action="store_true",
                    help="--max-step 을 넘는 step 샘플을 데이터에서 아예 뺀다 (negative 로도 안 씀)")
    ap.add_argument("--n-episodes", type=int, default=None,
                    help="로딩 뒤 에피소드 이 개수만 사용 (success 우선, 모자라면 failure 로 채움)")
    ap.add_argument("--sample-seed", type=int, default=0, help="--n-episodes 추출 시드")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-group-split", action="store_true",
                    help="에피소드 단위 분할을 끈다 (같은 에피소드가 train/test 로 쪼개짐)")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="cpu: sklearn LogisticRegression | cuda[:i]: 같은 목적함수를 torch L-BFGS 로 (TorchLogReg)")
    a = ap.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "classifiers").mkdir(exist_ok=True)

    if a.source == "hidden":
        data = load_pt(a.pt, a.offset, a.with_prompt)
    elif a.source == "spectral":
        data = load_spectral(a.pt, a.k, a.sign_mode)
    else:
        data = load_edges(a.pt, a.part, a.k, a.sign_mode, a.keep_short)
    if a.n_episodes is not None:
        data, kept = subsample_episodes(data, a.n_episodes, a.sample_seed)
        with open(a.output / "episodes.json", "w") as f:
            json.dump({"n_episodes": a.n_episodes, "sample_seed": a.sample_seed, "groups": kept}, f)
    if a.max_step is not None and a.drop_above_max:
        keep = data["step_num"] <= a.max_step          # answer(-1)·prompt(0) 는 남는다
        print(f"--drop-above-max: step>{a.max_step} 샘플 {int((~keep).sum())}개 제거")
        data = _take(data, np.flatnonzero(keep))
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
          f"source={a.source} part={a.part} group={group} scale={scale} cv={a.cv} offset={a.offset} device={a.device}")

    if a.device == "cpu":
        Xnp, gd = data["X"].float().numpy(), None            # float32 저장본이면 복사 없음
    else:
        Xnp, gd = None, GpuData(data["X"], a.device)
        print(f"X on {a.device}: {tuple(gd.X.shape)} {str(gd.X.dtype).removeprefix('torch.')} "
              f"({gd.X.numel() * gd.X.element_size() / 2**30:.1f} GiB)")

    rows = []
    for label in labels:
        t = target_name(label)
        X, y, g = make_binary(data, label)
        if y.sum() < 2 or (1 - y).sum() < 2:
            print(f"[skip] {t}: 샘플 부족")
            continue
        for s in a.seeds:
            tr, te = split(len(y), y, g, a.test_size, s, group)
            if len(np.unique(y[te])) < 2:
                print(f"[skip] {t} seed={s}: test 에 한 클래스만")
                continue
            clf = fit(Xnp, gd, y, tr, s, a.cv, scale)
            m_tr, m_te = evaluate(clf, Xnp, gd, tr, y[tr]), evaluate(clf, Xnp, gd, te, y[te])
            lr = clf if isinstance(clf, TorchLogReg) else clf[-1]
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
    src = {"hidden": f"hidden offset={a.offset}", "spectral": "spectral e_t",
           "edges": f"spectral edges ({a.part})"}[a.source]
    ax.set_title(f"Last-layer probes: {src} (seeds={len(a.seeds)}, group={group})")
    fig.tight_layout()
    fig.savefig(a.output / "summary.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved → {a.output}")


if __name__ == "__main__":
    main()