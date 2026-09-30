#!/usr/bin/env python
"""hidden state → spectral embedding e. 구간(step) 단위, 토큰 단위, 가장자리 저장까지 한 파일.

    E_t (n x d): 구간 t 의 토큰 hidden state.  e_t = [σ₁v₁; …; σ_k v_k]  (kd,)
    v_k 는 E_t 의 우특이벡터, σ_k² = λ_k 는 그람 G_t = E_t E_tᵀ 의 고유값.

세 가지 계산 경로가 있고 결과는 (같은 행렬이면) 서로 같다.

  1. 구간 단위 (spectral_embedding / spectral_run)
     E_t 를 통째로 SVD 해서 e_t 하나. 결과는 <spectral_dir>/<task>/<level>/<status>/<tag>/chunk_*.pt
     에 저장된다. inference/main.py 가 부른다.

  2. 토큰 단위 (CumulativeSpectral / stream_* / tokens_cumulative / stream_dataset)
     토큰이 하나 쌓일 때마다 누적 그람 → eigh → e_i. 리셋은 구간 경계에서만.
         토큰 1     → e_1,  토큰 1,2 → e_2,  …  [구간 경계] → 리셋
     각 구간의 마지막 e_i 는 1 의 e_t 와 같다. e_i 하나가 k·d 실수(k=8: 160KB)라
     전부 저장하면 에피소드당 ~100MB 가 되므로 파일을 만들지 않고 제너레이터로 흘려보낸다.
     heatmap / gsbs 가 이걸 받아서 필요한 만큼만 소비한다.

     수학 (왜 그람으로 하는가): G = E Eᵀ (n x n) 을 eigh 하면 λ_k = σ_k², u_k 는 좌특이벡터,
     σ_k v_k = Eᵀ u_k 가 나눗셈 없이 나온다 (σ_k = 0 이면 0 벡터). d=5120 ≫ n 이라
     (n x d) SVD 를 토큰마다 다시 도는 것보다 (n x n) eigh 가 훨씬 싸고, 그람은 토큰 하나에
     행/열 하나만 더 계산하면 된다. 부호 보정도 u 와 Eᵀu 만으로 같은 값이 나온다 (fix_sign).

  3. 가장자리 (episode_edges / CLI `edges`)
     구간마다 토큰별 누적 e_i 중 "앞 n_front 개"와 "뒤 n_back 개"만 뽑아
     <out_dir>/<task>/<level>/<status>/<tag>/chunk_*.pt 에 저장한다 (predict/probing.py 입력).
         marker = 구간 앞 형식 문구가 차지하는 토큰 수
                  step 구간: "Step N:" 헤더 / 터미널 구간: 정답 앞 문구 (TERMINAL_PAT)
         front  = 위치 marker .. marker+n_front-1   (형식 토큰의 e 는 버림, 누적에는 포함)
         back   = 위치 n-n_back .. n-1              (마지막 = 1 의 e_t)
     marker 는 원본 jsonl 을 추출 때와 똑같이 토크나이즈해서 STEP_PAT 매치가 끝나는 문자
     위치까지 걸친 토큰 수로 센다. 토큰 수/경계가 hidden_states 와 다르면 그 에피소드는 버린다.
     원본 텍스트가 없으면 --fallback-marker TASK=S:T 로 태스크별 고정 길이를 쓸 수 있다
     (Qwen3 토크나이저 샘플값: decompose 4:3, plan 4:9, predict 5:7). 어느 쪽인지는
     에피소드마다 "marker_src" ("text" | "fixed") 에 남는다. 짧은 구간 제외는 여기서 하지 않는다.

     출력 레코드: {"k", "scale", "sign_mode", "n_front", "n_back", "src", "model",
                   "episodes": {seed: {"seg", "labels", "marker": [m_0..m_N], "marker_src",
                                       "front": {t: (nf, kd)}, "front_pos": {t: [...]},
                                       "back":  {t: (nb, kd)}, "back_pos":  {t: [...]}}}}

구간 번호 t 는 전부 extract.gen_view 기준: 0..N-1 = Step 1..N, N = 터미널 (프롬프트 제외).
heatmap.py / gsbs.py 도 같은 gen_view 를 쓰므로 구간 인덱스가 맞물린다.

Usage:
    # 1. 구간 단위 저장 → inference/main.py
    # 2. 토큰 단위 스트리밍 확인 / SVD 와 대조
    python inference/spectral.py stream --task decompose --level BabyAI-GoToObj-v0 --status success --verify
    # 3. 가장자리 저장
    python inference/spectral.py edges                        # 전체, k8 data
    python inference/spectral.py edges -k 4 8 16 --sign-mode data max
    python inference/spectral.py edges --task decompose --verify
    python inference/spectral.py edges --fallback-marker decompose=4:3 plan=4:9 predict=5:7
"""
from __future__ import annotations

import time
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from typing import Iterator

import torch
from tqdm import tqdm

import extract
from extract import (MODEL, STEP_PAT, TERMINAL_PAT, load_episodes, load_chunk, gen_view, gen_views,
                     seg_labels, step_char_bounds, char_to_token_bounds)

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("decompose", "plan", "predict")

K_EIG = 8
SCALE = True        # E_t를 sqrt(n_t)로 나눠 토큰 수에 따른 고유값 증가를 방지
SIGN_MODE = "data"  # "none" | "first" | "max" | "data"
SIGN_MODES = ("none", "first", "max", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12


def make_tag(k: int, scale: bool, sign_mode: str) -> str:
    """저장 디렉토리 이름. 예: k8_scaled_sign-data, k8_scaled (none).

    읽는 쪽(visual/heatmap.py, visual/step_similarity.py, predict/probing.py)도 이 함수를 써야 한다 —
    문자열을 손으로 베끼면 규칙이 바뀔 때 조용히 어긋난다."""
    if sign_mode not in SIGN_MODES:
        raise ValueError(f"unknown sign_mode: {sign_mode!r} (choose from {SIGN_MODES})")
    tag = f"k{k}" + ("_scaled" if scale else "")
    if sign_mode != "none":
        tag += f"_sign-{sign_mode}"
    return tag


# ------------------------------------------------------------------ 부호 보정

def fix_sign(R: torch.Tensor, proj: torch.Tensor | None, mode: str):
    """고유벡터 부호 보정. SVD 경로와 그람 경로가 같은 함수를 쓴다.

    R    : (r, d)  부호를 정할 벡터들. v_k (SVD) 또는 σ_k v_k = Eᵀu_k (그람) — 부호 규칙은
           양수 배율에 무관하므로 어느 쪽이든 같은 부호가 나온다.
    proj : (n, r)  데이터 투영 v_k·x_i. SVD 경로는 E v_k, 그람 경로는 σ_k u_k. "data" 에서만 쓴다.

    - "none":  보정 안 함
    - "first": 각 벡터의 첫 성분이 양수가 되도록
    - "max":   각 벡터의 최대 절댓값 성분이 양수가 되도록
    - "data":  Bro, Acar & Kolda (2007)의 부호-가중 내적 점수
               s_k = Σ_i sign(v_k·x_i)(v_k·x_i)²  가 양수가 되도록. 대칭(Gram) 케이스 단순화 버전.

    반환: (부호 보정된 R, score 또는 None). "data" 의 score 는 (r,) — |score|/λ ∈ [0,1]이 부호 신뢰도.
    """
    if mode == "none":
        return R, None

    if mode == "first":
        sign = torch.sign(R[:, :1])                       # (r, 1)
        sign[sign == 0] = 1.0
        return R * sign, None

    if mode == "max":
        idx = R.abs().argmax(dim=1)                       # (r,)
        sign = torch.sign(R.gather(1, idx[:, None]))      # (r, 1)
        sign[sign == 0] = 1.0
        return R * sign, None

    if mode == "data":
        if proj is None:
            raise ValueError("sign_mode 'data' needs proj")
        score = (torch.sign(proj) * proj**2).sum(dim=0)   # (r,)
        sign = torch.sign(score)
        sign[sign == 0] = 1.0                             # 완전 대칭이면 그대로 둠
        return R * sign[:, None], score

    raise ValueError(f"unknown sign_mode: {mode!r} (choose from {SIGN_MODES})")


def _pad_k(R: torch.Tensor, sig: torch.Tensor, score: torch.Tensor | None, k: int):
    """rank r < k 이면 부족분을 0 으로 채운다 (σ=0 ⇒ σv = 0 이므로 정확한 값)."""
    r, d = R.shape
    if r < k:
        R = torch.cat([R, R.new_zeros(k - r, d)])
        sig = torch.cat([sig, sig.new_zeros(k - r)])
        if score is not None:
            score = torch.cat([score, score.new_zeros(k - r)])
    return R, sig, score


# ------------------------------------------------------------ 1. 구간 단위 SVD

@torch.no_grad()
def spectral_embedding(Et: torch.Tensor, k: int, scale: bool, sign_mode: str):
    """E_t (n x d) → (e (kd,), lam (k,), V (k x d), score (k,)|None). 전부 cpu."""
    Et = Et.float()  # torch.linalg.svd는 bf16 미지원

    if scale:
        Et = Et / (Et.shape[0] ** 0.5)  # Et.shape[0] = n_t (부호에는 영향 없음)

    _, S, Vh = torch.linalg.svd(Et, full_matrices=False)   # S:(r,), Vh:(r,d)
    r = min(k, S.shape[0])                                 # rank(G_t) ≤ n_t
    S_k, V_k = S[:r], Vh[:r]                               # 내림차순 보장됨

    V_k, score = fix_sign(V_k, Et @ V_k.T, sign_mode)
    V_k, S_k, score = _pad_k(V_k, S_k, score, k)

    e_t = (S_k[:, None] * V_k).reshape(-1)   # [√λ₁q₁; ...; √λ_k q_k], (kd,)
    lam = S_k ** 2                           # λ_i = σ_i²
    score = score.cpu() if score is not None else None
    return e_t.cpu(), lam.cpu(), V_k.cpu(), score


@torch.no_grad()
def episode_embeddings(views: list[torch.Tensor], k: int, scale: bool, sign_mode: str):
    e, lam, V, sc = {}, {}, {}, {}
    for t, Et in enumerate(views):
        e[t], lam[t], V[t], s = spectral_embedding(Et.to(DEVICE), k, scale, sign_mode)
        if s is not None:
            sc[t] = s
    return e, lam, V, sc


def load_hidden_states(data_dir: Path, task: str, level: str, status: str):
    """extract.py 출력 경로: <data_dir>/<task>/<level>/<status>/chunk_*.pt"""
    target = data_dir / task / level / status
    if not target.is_dir():
        raise FileNotFoundError(f"no such directory: {target}")

    files = sorted(target.glob("chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"no chunk_*.pt in {target}")
    return files


@torch.no_grad()
def spectral_run(data_root: Path, out_root: Path, task: str, level: str,
                 status: str, k: int = K_EIG, scale: bool = SCALE,
                 sign_mode: str = SIGN_MODE):
    """구간별 spectral embedding 을 만들어 저장한다 (inference/main.py 가 부른다).

    gen_views 를 쓰므로 프롬프트는 빠지고 step 1..N + 터미널(정답 문장)이 들어간다.
    딕셔너리 키 0..N-1 이 step 1..N, 키 N 이 터미널이다 (extract.seg_labels 와 동일).
    """
    data_root = data_root.resolve()
    chunk_files = load_hidden_states(data_root, task, level, status)

    tag = make_tag(k, scale, sign_mode)
    rel = Path(task) / level
    out_dir = out_root / rel / status / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.pt"):   # hidden_states 청크 개수가 줄었을 때 낡은 파일 안 남게
        old.unlink()

    n_episodes = 0
    for cf in tqdm(chunk_files, desc=f"{rel}/{status}/{tag}", unit="chunk"):
        chunk = load_chunk(cf)
        out = {}
        for seed, episode in chunk.items():
            _, _, views = gen_views(episode)
            e, lam, V, sc = episode_embeddings(views, k, scale, sign_mode)
            rec = {"e": e, "eigvals": lam, "V": V}
            if sc:
                rec["sign_score"] = sc
            out[seed] = rec
            n_episodes += 1

        torch.save({"k": k, "scale": scale, "sign_mode": sign_mode,
                    "src": str(cf), "model": MODEL, "episodes": out},
                   out_dir / cf.name)

    print(f"saved {n_episodes} episodes ({len(chunk_files)} chunks) under {out_dir}")


# ------------------------------------------------------- 2. 토큰 단위 누적 그람

class CumulativeSpectral:
    """구간 하나 안에서 토큰을 하나씩 받아 누적 그람 → e_i 를 낸다.

    push(x) 마다 (kd,) e_i 를 돌려주고, 구간 경계에서 reset() 을 부른다.
    입력 x 는 (d,) hidden state (bf16 이어도 됨, 내부에서 float32 로 올림).

        cs = CumulativeSpectral(k=8)
        for x in tokens_of_step:  e_i, lam_i, score_i = cs.push(x)
        cs.reset()

    내부 상태:
        X (cap x d) float32  — 지금 구간의 토큰들 (행 0..n-1)
        G (cap x cap) float32 — X Xᵀ 의 좌상단 n x n 블록만 유효
    cap 은 모자라면 두 배로 늘린다.
    """

    def __init__(self, k: int = K_EIG, scale: bool = SCALE, sign_mode: str = SIGN_MODE,
                 device: str | torch.device = DEVICE, cap: int = 64):
        if sign_mode not in SIGN_MODES:
            raise ValueError(f"unknown sign_mode: {sign_mode!r} (choose from {SIGN_MODES})")
        self.k, self.scale, self.sign_mode = k, scale, sign_mode
        self.device = torch.device(device)
        self.cap0 = cap
        self.X: torch.Tensor | None = None
        self.G: torch.Tensor | None = None
        self.n = 0

    # -- 버퍼 관리 --------------------------------------------------------

    def reset(self):
        """구간 경계. 버퍼는 그대로 두고 길이만 0 으로 (재할당 없음)."""
        self.n = 0

    def _ensure(self, d: int, need: int):
        if self.X is None:
            cap = max(self.cap0, need)
            self.X = torch.empty(cap, d, dtype=torch.float32, device=self.device)
            self.G = torch.empty(cap, cap, dtype=torch.float32, device=self.device)
            return
        cap = self.X.shape[0]
        if need <= cap:
            return
        while cap < need:
            cap *= 2
        X = torch.empty(cap, d, dtype=torch.float32, device=self.device)
        G = torch.empty(cap, cap, dtype=torch.float32, device=self.device)
        X[: self.n] = self.X[: self.n]
        G[: self.n, : self.n] = self.G[: self.n, : self.n]
        self.X, self.G = X, G

    # -- 입력 --------------------------------------------------------------

    @torch.no_grad()
    def push(self, x: torch.Tensor):
        """토큰 하나 추가 후 현재 누적의 (e, lam, score). e:(kd,), lam:(k,), score:(k,)|None"""
        x = x.to(self.device, torch.float32).reshape(-1)
        self._ensure(x.shape[0], self.n + 1)
        i = self.n
        self.X[i] = x
        g = self.X[: i + 1] @ x                  # (i+1,) — 새 행/열
        self.G[i, : i + 1] = g
        self.G[: i + 1, i] = g
        self.n = i + 1
        return self._embed()

    @torch.no_grad()
    def feed(self, Xseg: torch.Tensor) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """구간 전체 (n x d) 가 이미 손에 있을 때. 그람을 matmul 한 번으로 만들고
        토큰마다 좌상단 블록만 분해한다 — push 를 n 번 부른 것과 같은 결과."""
        Xseg = Xseg.to(self.device, torch.float32)
        n, d = Xseg.shape
        self._ensure(d, self.n + n)
        s = self.n
        self.X[s: s + n] = Xseg
        # 기존 n0 행과의 교차 블록 + 새 블록. 보통 reset 직후라 s == 0.
        self.G[s: s + n, : s + n] = Xseg @ self.X[: s + n].T
        self.G[: s, s: s + n] = self.G[s: s + n, : s].T
        for i in range(n):
            self.n = s + i + 1
            yield self._embed()

    @torch.no_grad()
    def feed_at(self, Xseg: torch.Tensor, positions) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """구간 전체 (n x d) 를 받아 지정한 위치(0-based 토큰 번호)에서만 e 를 낸다.

        feed 와 같은 누적 결과지만 필요 없는 위치의 eigh 를 건너뛴다. 위치 p 의 결과는
        토큰 0..p 를 누적한 것 (= feed 가 p 번째로 내는 값). 호출 전에 reset 된 상태여야
        하고, 호출 뒤에는 구간 전체가 들어간 상태로 남는다.
        """
        Xseg = Xseg.to(self.device, torch.float32)
        n, d = Xseg.shape
        if self.n != 0:
            raise RuntimeError("feed_at expects a reset accumulator")
        if any(p < 0 or p >= n for p in positions):
            raise ValueError(f"positions {list(positions)} out of range for {n} tokens")
        self._ensure(d, n)
        self.X[:n] = Xseg
        self.G[:n, :n] = Xseg @ Xseg.T
        out = []
        for p in positions:
            self.n = p + 1
            out.append(self._embed())
        self.n = n
        return out

    # -- 분해 --------------------------------------------------------------

    @torch.no_grad()
    def _embed(self):
        n, k = self.n, self.k
        G = self.G[:n, :n].double()             # 그람은 조건수가 σ² 이라 eigh 는 float64
        if self.scale:
            G = G / n                            # E/√n  ⇔  G/n
        w, U = torch.linalg.eigh(G)              # 오름차순
        r = min(k, n)
        idx = torch.arange(n - 1, n - 1 - r, -1, device=U.device)
        w = w[idx].clamp_min(0.0)                # 수치 오차로 생긴 음수 제거
        U = U[:, idx].float()                    # (n, r)

        Xs = self.X[:n]
        R = U.T @ Xs                             # (r, d) = σ_k v_k  (= Eᵀu_k)
        if self.scale:
            R = R / (n ** 0.5)
        sig = w.sqrt().float()                   # (r,)

        # proj_k = E v_k = G u_k / σ_k = σ_k u_k  → fix_sign 의 "data" 점수가 SVD 경로와 같아진다
        R, score = fix_sign(R, U * sig, self.sign_mode)
        R, sig, score = _pad_k(R, sig, score, k)

        e = R.reshape(-1)                        # [σ₁v₁; …; σ_k v_k]  (kd,)
        lam = sig ** 2
        return e, lam, score

    # -- 부가 --------------------------------------------------------------

    @staticmethod
    def eigvecs(e: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """e (kd,) 와 lam (k,) 에서 단위 고유벡터 V (k x d) 복원. λ=0 행은 0."""
        k = lam.shape[0]
        R = e.reshape(k, -1)
        sig = lam.sqrt()
        V = torch.where(sig[:, None] > EPS, R / sig.clamp_min(EPS)[:, None], torch.zeros_like(R))
        return V


# ------------------------------------------------------------- 에피소드 단위

@dataclass
class TokenRecord:
    """토큰 하나의 누적 스펙트럴 결과. 인덱스는 전부 extract.gen_view 기준 (프롬프트 제외)."""
    idx: int            # 생성 구간 안에서의 토큰 번호 (0..n_gen-1)
    seg: int            # 구간 번호 (0..N-1 = step 1..N, N = 터미널)
    pos: int            # 구간 안에서의 위치 (0 = 리셋 직후 첫 토큰)
    reset: bool         # pos == 0
    last: bool          # 구간의 마지막 토큰 (이 e 가 spectral_run 의 e_t 와 같다)
    e: torch.Tensor     # (kd,)
    lam: torch.Tensor   # (k,)
    score: torch.Tensor | None   # (k,)  sign_mode == "data" 일 때만

    @property
    def is_last_of_seg(self) -> bool:
        return self.last


@torch.no_grad()
def stream_tokens(E: torch.Tensor, seg: list[int], k: int = K_EIG, scale: bool = SCALE,
                  sign_mode: str = SIGN_MODE, device: str | torch.device = DEVICE,
                  cs: CumulativeSpectral | None = None,
                  out_device: str | torch.device = "cpu") -> Iterator[TokenRecord]:
    """토큰열 E (n x d) 를 seg 경계마다 리셋하며 토큰 단위로 흘려보낸다.

    seg = [0, b1, ..., n] (extract.gen_view 의 seg 와 같은 규약). 어떤 경계든
    받는다 — 텍스트 step 경계 대신 GSBS 경계를 넣어 리셋 지점을 바꿔 볼 수도 있다
    (metric/gsbs.py 의 reset 그림).

    cs 를 넘기면 그 톱니를 재사용한다 (버퍼 재할당 없이 여러 에피소드를 돌릴 때).
    """
    if seg[0] != 0 or seg[-1] != E.shape[0] or any(a >= b for a, b in zip(seg, seg[1:])):
        raise ValueError(f"bad seg {seg} for E of {E.shape[0]} tokens")
    if cs is None:
        cs = CumulativeSpectral(k, scale, sign_mode, device)
    idx = 0
    for t, (s, en) in enumerate(zip(seg, seg[1:])):
        cs.reset()
        n = en - s
        for pos, (e, lam, score) in enumerate(cs.feed(E[s:en])):
            yield TokenRecord(idx=idx, seg=t, pos=pos, reset=(pos == 0), last=(pos == n - 1),
                              e=e.to(out_device), lam=lam.to(out_device),
                              score=None if score is None else score.to(out_device))
            idx += 1


@torch.no_grad()
def stream_episode(episode: dict, k: int = K_EIG, scale: bool = SCALE,
                   sign_mode: str = SIGN_MODE, device: str | torch.device = DEVICE,
                   cs: CumulativeSpectral | None = None,
                   out_device: str | torch.device = "cpu") -> Iterator[TokenRecord]:
    """에피소드 하나(extract 청크의 dict)를 토큰 단위로. 프롬프트는 빼고 step 경계에서 리셋."""
    E, seg = gen_view(episode)
    yield from stream_tokens(E, seg, k, scale, sign_mode, device, cs, out_device)


@torch.no_grad()
def tokens_cumulative(E: torch.Tensor, seg: list[int], k: int = K_EIG, scale: bool = SCALE,
                      sign_mode: str = SIGN_MODE, device: str | torch.device = DEVICE,
                      cs: CumulativeSpectral | None = None,
                      out_dtype: torch.dtype = torch.float32):
    """stream_tokens 를 (n x kd) 텐서로 모은다. 반환 (e_all, lam_all).

    heatmap.py / gsbs.py 처럼 토큰 전체 행렬이 필요한 곳용. RSSM 학습 배치도 여기서
    나온 e_all 을 시퀀스로 쓰면 된다. 메모리: n·k·d·(dtype 바이트). k=8, 600
    토큰이면 float32 ≈ 98MB, bf16 ≈ 49MB — 에피소드 하나 단위로만 들고 있을 것.
    """
    n, d = E.shape
    e_all = torch.empty(n, k * d, dtype=out_dtype)
    lam_all = torch.empty(n, k, dtype=torch.float32)
    for rec in stream_tokens(E, seg, k, scale, sign_mode, device, cs):
        e_all[rec.idx] = rec.e.to(out_dtype)
        lam_all[rec.idx] = rec.lam
    return e_all, lam_all


@torch.no_grad()
def episode_cumulative(episode: dict, k: int = K_EIG, scale: bool = SCALE,
                       sign_mode: str = SIGN_MODE, device: str | torch.device = DEVICE,
                       cs: CumulativeSpectral | None = None,
                       out_dtype: torch.dtype = torch.float32):
    """에피소드 하나를 통째로 (n_gen x kd) 텐서로. 반환 (e_all, lam_all, seg)."""
    E, seg = gen_view(episode)
    e_all, lam_all = tokens_cumulative(E, seg, k, scale, sign_mode, device, cs, out_dtype)
    return e_all, lam_all, seg


# --------------------------------------------------------------- 데이터셋 단위

@dataclass
class EpisodeItem:
    task: str
    level: str
    status: str
    seed: int
    chunk: str          # chunk 파일명
    seg: list[int]      # gen_view 경계 [0, step1 끝, …, stepN 끝, 전체 끝]
    labels: list[str]   # seg_labels(seg)
    e: torch.Tensor     # (n_gen x kd)
    lam: torch.Tensor   # (n_gen x k)


@torch.no_grad()
def stream_dataset(hidden_dir: Path, task: str, level: str, status: str,
                   k: int = K_EIG, scale: bool = SCALE, sign_mode: str = SIGN_MODE,
                   device: str | torch.device = DEVICE, seeds: set[int] | None = None,
                   limit: int | None = None, out_dtype: torch.dtype = torch.float32,
                   per_token: bool = False):
    """<hidden_dir>/<task>/<level>/<status>/chunk_*.pt 를 읽어 에피소드마다 e 를 흘려보낸다.

    per_token=False (기본): EpisodeItem 을 yield — e (n_gen x kd) 한 덩어리.
    per_token=True        : (seed, TokenRecord) 를 토큰마다 yield — 메모리 최소.

    파일을 만들지 않는다. chunk 는 하나씩만 메모리에 올리고, 다음 chunk 로 넘어가면 버린다.
    """
    hidden_dir = Path(hidden_dir).resolve()
    chunk_files = load_hidden_states(hidden_dir, task, level, status)
    cs = CumulativeSpectral(k, scale, sign_mode, device)
    n_done = 0
    for cf in chunk_files:
        chunk = load_chunk(cf)
        for seed, episode in chunk.items():
            if seeds is not None and seed not in seeds:
                continue
            if limit is not None and n_done >= limit:
                return
            if per_token:
                for rec in stream_episode(episode, k, scale, sign_mode, device, cs):
                    yield seed, rec
            else:
                e_all, lam_all, seg = episode_cumulative(
                    episode, k, scale, sign_mode, device, cs, out_dtype)
                yield EpisodeItem(task=task, level=level, status=status, seed=seed,
                                  chunk=cf.name, seg=seg, labels=seg_labels(seg),
                                  e=e_all, lam=lam_all)
            n_done += 1
        del chunk


# ------------------------------------------------------------- 3. 가장자리 e

def load_sources(traj_dir: Path, task: str) -> dict:
    """(env_name, env_seed, output_sha1) → jsonl 에피소드."""
    out = {}
    for ep in load_episodes(traj_dir / f"{task}_no_thinking.jsonl"):
        text = ep.get("all_llm_output") or ""
        sha = hashlib.sha1(text.encode()).hexdigest()
        out[(ep["env_name"], int(ep["env_seed"]), sha)] = ep
    return out


def marker_lengths(src_ep: dict, task: str, boundaries: list[int]):
    """구간별 형식 토큰 수 [m_step1, ..., m_stepN, m_terminal]. 실패 시 (None, 사유).

    extract.tok 이 세팅되어 있어야 한다 (CLI `edges` 가 AutoTokenizer 를 넣어 준다)."""
    prompt = extract.render_prompt(src_ep)
    text = src_ep["all_llm_output"]
    enc = extract.tok(prompt + text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc.offset_mapping

    if len(enc.input_ids) != boundaries[-1]:
        return None, "token_count_mismatch"
    char_bounds, reason = step_char_bounds(text, task)
    if reason is not None:
        return None, f"bounds:{reason}"
    tb = char_to_token_bounds([0] + [len(prompt) + c for c in char_bounds], offs)
    if tb != list(boundaries):
        return None, "boundary_mismatch"

    def count(s_tok, e_tok, char_end):                          # char_end 까지 걸친 토큰 수
        m = 0
        while s_tok + m < e_tok and offs[s_tok + m][0] < char_end:
            m += 1
        return m

    heads = list(STEP_PAT.finditer(text))
    term = list(TERMINAL_PAT[task].finditer(text))[-1]          # step_char_bounds 와 같은 마지막 매치
    n_steps = len(boundaries) - 3
    marks = [count(boundaries[1 + t], boundaries[2 + t], len(prompt) + heads[t].end())
             for t in range(n_steps)]
    marks.append(count(boundaries[-2], boundaries[-1], len(prompt) + term.end()))
    return marks, None


def edge_positions(n: int, marker: int, n_front: int, n_back: int):
    """구간 길이 n 에서 front/back 위치. 짧으면 겹칠 수 있고, 모자라면 있는 만큼만."""
    front = list(range(min(marker, n), min(marker + n_front, n)))
    back = list(range(max(n - n_back, 0), n))
    return front, back


@torch.no_grad()
def episode_edges(episode: dict, marks: list[int], cs: CumulativeSpectral,
                  n_front: int, n_back: int, marker_src: str = "text") -> dict:
    E, seg = gen_view(episode)
    rec = {"seg": seg, "labels": seg_labels(seg), "marker": marks, "marker_src": marker_src,
           "front": {}, "front_pos": {}, "back": {}, "back_pos": {}}
    kd = cs.k * E.shape[1]
    for t, (s, e) in enumerate(zip(seg, seg[1:])):
        n = e - s
        fpos, bpos = edge_positions(n, marks[t], n_front, n_back)
        need = sorted(set(fpos) | set(bpos))
        cs.reset()
        got = dict(zip(need, (r[0].cpu() for r in cs.feed_at(E[s:e], need))))
        cs.reset()
        stack = lambda ps: torch.stack([got[p] for p in ps]) if ps else torch.empty(0, kd)
        rec["front"][t], rec["front_pos"][t] = stack(fpos), fpos
        rec["back"][t], rec["back_pos"][t] = stack(bpos), bpos
    return rec


# ------------------------------------------------------------------- 검증

def _close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    """‖a − b‖ ≤ atol + rtol‖b‖ (b 가 기준)."""
    return (a - b).norm().item() <= atol + rtol * b.norm().item()


def verify_edges(rec: dict, spec_path: Path, seed, atol=1e-3, rtol=1e-3) -> tuple[int, int]:
    """episode_edges 의 back 마지막 e == spectral_run 이 저장한 구간 e_t 인지. (일치, 불일치)"""
    if not spec_path.is_file():
        return 0, 0
    ref = torch.load(spec_path, map_location="cpu", weights_only=False)["episodes"].get(seed)
    if ref is None:
        return 0, 0
    ok = bad = 0
    for t, B in rec["back"].items():
        a, b = B[-1].float(), ref["e"][t].float()
        k = len(ref["eigvals"][t])
        # 성분별로 비교 — 부호가 임의인 성분(고유값이 거의 같은 쌍)은 뒤집혀도 허용
        for A_j, B_j in zip(a.reshape(k, -1), b.reshape(k, -1)):
            if _close(A_j, B_j, atol, rtol) or _close(-A_j, B_j, atol, rtol):
                ok += 1
            else:
                bad += 1
    return ok, bad


@torch.no_grad()
def verify_episode(episode: dict, k: int, scale: bool, sign_mode: str,
                   device=DEVICE, rtol: float = 1e-3, atol: float = 1e-3,
                   every_token: bool = True) -> dict:
    """스트리밍(그람) 결과를 spectral_embedding(SVD) 과 맞춰 본다.

    - 구간 마지막 토큰의 e_i == spectral_embedding(E_t)  (spectral_run 과 동일해야 함)
    - every_token 이면 토큰마다 spectral_embedding(E[s:i+1]) 을 다시 돌려서 전부 비교
      (느리다: 토큰마다 (i x d) SVD)

    비교는 e = σv 에 대해 상대 오차로 한다. 부호 보정이 애매한 성분(score≈0,
    또는 σ 가 거의 같은 두 고유값)은 원래 부호가 임의라 따로 센다.
    """
    E, seg = gen_view(episode)
    n_ok = n_bad = n_amb = 0
    worst = 0.0
    for rec in stream_episode(episode, k, scale, sign_mode, device):
        s = seg[rec.seg]
        if not (every_token or rec.last):
            continue
        e_ref, lam_ref, _, _ = spectral_embedding(E[s: s + rec.pos + 1].to(device), k, scale, sign_mode)
        R, Rr = rec.e.reshape(k, -1), e_ref.reshape(k, -1)
        for j in range(k):
            rel = (R[j] - Rr[j]).norm().item() / max(Rr[j].norm().item(), EPS)
            if _close(R[j], Rr[j], atol, rtol):
                n_ok += 1
                worst = max(worst, rel)
                continue
            # 부호만 다른가? (부호 결정이 임의였던 성분)
            close_eig = j + 1 < k and abs(lam_ref[j] - lam_ref[j + 1]) < 1e-3 * max(lam_ref[j].item(), EPS)
            if _close(-R[j], Rr[j], atol, rtol) or close_eig:
                n_amb += 1
            else:
                n_bad += 1
                worst = max(worst, rel)
    return {"n_ok": n_ok, "n_sign_ambiguous": n_amb, "n_bad": n_bad, "worst_rel": worst}


# ------------------------------------------------------------------------ CLI

def _parse_bool(s: str) -> bool:
    return s == "true"


def main_stream(a):
    """토큰 단위 스트리밍을 한 (task, level, status) 에 돌려 보고, --verify 면 SVD 와 대조."""
    scale = _parse_bool(a.scale)
    t0 = time.time()
    n_ep = n_tok = 0
    for item in stream_dataset(a.hidden_dir, a.task, a.level, a.status,
                               k=a.k, scale=scale, sign_mode=a.sign_mode, limit=a.limit):
        n_ep += 1
        n_tok += item.e.shape[0]
        print(f"seed={item.seed:<6} n_gen={item.e.shape[0]:<5} segs={len(item.seg) - 1} "
              f"e={tuple(item.e.shape)} top-λ(last tok)={item.lam[-1, 0]:.3g}")
    dt = time.time() - t0
    print(f"{n_ep} episodes, {n_tok} tokens, {dt:.1f}s  ({n_tok / max(dt, 1e-9):.0f} tok/s, device={DEVICE})")

    if a.verify:
        chunk_files = load_hidden_states(a.hidden_dir.resolve(), a.task, a.level, a.status)
        n_done = 0
        for cf in chunk_files:
            for seed, ep in load_chunk(cf).items():
                if a.limit is not None and n_done >= a.limit:
                    break
                t1 = time.time()
                r = verify_episode(ep, a.k, scale, a.sign_mode,
                                   every_token=not a.verify_last_only)
                print(f"verify seed={seed}: {r}  ({time.time() - t1:.1f}s)")
                if r["n_bad"]:
                    raise SystemExit(f"MISMATCH seed={seed}: {r}")
                n_done += 1
        print("verify OK")


def main_edges(a):
    """모든 (task, level, status, chunk) 에 대해 가장자리 e 를 저장한다."""
    from transformers import AutoTokenizer

    scale = _parse_bool(a.scale)
    fallback = {}
    for kv in a.fallback_marker:                                 # TASK=S:T
        task_, v = kv.split("=")
        st, te = v.split(":")
        fallback[task_] = (int(st), int(te))

    extract.tok = AutoTokenizer.from_pretrained(MODEL)
    configs = [(k, sm) for k in a.k for sm in a.sign_mode]
    stats = Counter()
    marker_hist, term_hist = Counter(), Counter()
    t0 = time.time()

    for task in a.task:
        sources = load_sources(a.traj_dir, task)
        levels = a.level or sorted(p.name for p in (a.hidden_dir / task).iterdir() if p.is_dir())
        for level in levels:
            for status in a.status:
                hdir = a.hidden_dir / task / level / status
                if not hdir.is_dir():
                    continue
                for cf in sorted(hdir.glob("chunk_*.pt")):
                    todo = [(k, sm) for k, sm in configs
                            if a.overwrite or not (a.out_dir / task / level / status
                                                   / make_tag(k, scale, sm) / cf.name).exists()]
                    if not todo:
                        print(f"[skip] {task}/{level}/{status}/{cf.name}: 이미 있음")
                        continue
                    d = torch.load(cf, map_location="cpu", weights_only=False)

                    marks_of = {}
                    for seed, ep in d["episodes"].items():
                        src = sources.get((level, int(seed), ep["output_sha1"]))
                        if src is None:
                            if task not in fallback:
                                stats["no_source"] += 1
                                continue
                            n_steps = len(ep["boundaries"]) - 3
                            st, te = fallback[task]
                            marks_of[seed] = ([st] * n_steps + [te], "fixed")
                            stats["marker_fixed"] += 1
                            continue
                        marks, reason = marker_lengths(src, task, ep["boundaries"])
                        if reason:
                            stats[reason] += 1
                            continue
                        marks_of[seed] = (marks, "text")
                        stats["marker_text"] += 1
                        marker_hist.update(marks[:-1])
                        term_hist.update(marks[-1:])

                    for k, sm in todo:
                        tag = make_tag(k, scale, sm)
                        cs = CumulativeSpectral(k, scale, sm, a.device)
                        eps = {}
                        for seed, (marks, msrc) in marks_of.items():
                            eps[seed] = episode_edges(d["episodes"][seed], marks, cs,
                                                      a.n_front, a.n_back, msrc)
                            if a.verify:
                                ok, bad = verify_edges(eps[seed], a.spectral_dir / task / level / status
                                                       / tag / cf.name, seed)
                                stats[f"verify_ok_{tag}"] += ok
                                stats[f"verify_bad_{tag}"] += bad
                        out = a.out_dir / task / level / status / tag / cf.name
                        out.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"k": k, "scale": scale, "sign_mode": sm,
                                    "n_front": a.n_front, "n_back": a.n_back,
                                    "src": str(cf), "model": d.get("model", MODEL),
                                    "episodes": eps}, out)
                        stats[f"episodes_{tag}"] += len(eps)
                        print(f"{task}/{level}/{status}/{tag}/{cf.name}: {len(eps)} eps "
                              f"({time.time() - t0:.0f}s)")
                    del d

    print(f"\nstats: {dict(stats)}")
    print(f"marker 토큰 수 분포 (텍스트로 센 step 구간): {dict(sorted(marker_hist.items()))}")
    print(f"marker 토큰 수 분포 (텍스트로 센 터미널 구간): {dict(sorted(term_hist.items()))}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -- stream: 토큰 단위 스트리밍 확인 / SVD 대조 ---------------------------
    s = sub.add_parser("stream", help="토큰 단위 누적 e 를 한 (task, level, status) 에 흘려 보기 / --verify",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("--task", required=True, choices=TASKS)
    s.add_argument("--level", required=True)
    s.add_argument("--status", default="success", choices=["success", "failure"])
    s.add_argument("-k", type=int, default=K_EIG)
    s.add_argument("--scale", default="true" if SCALE else "false", choices=["true", "false"])
    s.add_argument("--sign-mode", default=SIGN_MODE, choices=list(SIGN_MODES))
    s.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    s.add_argument("--limit", type=int, default=None, help="에피소드 개수 제한")
    s.add_argument("--verify", action="store_true",
                   help="spectral_embedding(SVD) 과 토큰마다 비교 (느림)")
    s.add_argument("--verify-last-only", action="store_true",
                   help="--verify 를 구간 마지막 토큰(e_t)에서만")
    s.set_defaults(func=main_stream)

    # -- edges: 가장자리 e 저장 ------------------------------------------------
    e = sub.add_parser("edges", help="구간마다 앞/뒤 가장자리 토큰의 누적 e 를 저장 (probing 입력)",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    e.add_argument("--task", nargs="+", default=list(TASKS), choices=TASKS)
    e.add_argument("--level", nargs="+", default=None, help="기본: hidden_states 에 있는 전부")
    e.add_argument("--status", nargs="+", default=["success", "failure"])
    e.add_argument("-k", type=int, nargs="+", default=[K_EIG], help="고유벡터 개수 k (여러 개 가능)")
    e.add_argument("--sign-mode", nargs="+", default=[SIGN_MODE], choices=list(SIGN_MODES),
                   help="부호 보정 규칙 (여러 개 가능)")
    e.add_argument("--scale", default="true" if SCALE else "false", choices=["true", "false"],
                   help="E/√n 스케일 (spectral_states 는 true)")
    e.add_argument("--n-front", type=int, default=5)
    e.add_argument("--n-back", type=int, default=5)
    e.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    e.add_argument("--spectral-dir", type=Path, default=ROOT / "latent" / "spectral_states")
    e.add_argument("--traj-dir", type=Path, default=ROOT / "generation" / "trajectory")
    e.add_argument("--out-dir", type=Path, default=ROOT / "latent" / "spectral_edges")
    e.add_argument("--device", default=DEVICE)
    e.add_argument("--verify", action="store_true",
                   help="back 마지막 e 를 spectral_states 의 구간 e_t 와 비교")
    e.add_argument("--fallback-marker", nargs="+", default=[], metavar="TASK=S:T",
                   help="원본 텍스트가 없을 때 쓸 태스크별 형식 토큰 수. "
                        "S=step 헤더, T=터미널 문구 (예: predict=5:7)")
    e.add_argument("--overwrite", action="store_true")
    e.set_defaults(func=main_edges)
    return ap


def main():
    a = build_parser().parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
