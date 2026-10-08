#!/usr/bin/env python
"""hidden state → spectral embedding e 를 만들어 저장한다.

    E (n x d): 구간(step) 하나의 토큰 hidden state 들.   e = [σ₁v₁; …; σ_k v_k]  (kd,)
    v_k 는 E 의 우특이벡터(토큰들이 퍼진 주축), σ_k² = λ_k 는 그람 G = E Eᵀ 의 고유값.

토큰 단위 누적. 구간 안에서 토큰이 하나 쌓일 때마다 "구간 첫 토큰부터 지금까지" 의 행렬로
e_i 를 다시 만든다 (구간 경계에서 리셋). 구간 마지막 토큰의 e_i 가 곧 구간 전체의 e_t 다.
계산은 CumulativeSpectral 이 한다: (n x d) SVD 를 토큰마다 다시 도는 대신 (n x n) 그람을
eigh 하고 σ_k v_k = Eᵀu_k 로 바꾼다. d=5120 ≫ n 이라 훨씬 싸고, 결과는 SVD 와 같다.

무엇을 저장하나 (Select). e 하나가 k·d 실수(k=8: 160KB)라 토큰 전부를 저장하면 에피소드당 ~32MB,
청크(256 에피소드)당 ~8GB 가 된다. 그래서 구간마다 어느 위치의 e 를 남길지 고른다.

    --n-front 0 --n-back 1    구간마다 마지막 e_t 하나 (구간 전체 SVD 와 같은 값)    tag …_f0_b1  ← 기본
    --n-front 5 --n-back 5    형식 문구 뒤 5개 + 마지막 5개 (probing 의 edges 입력)  tag …_f5_b5
    --all                     구간의 모든 토큰                                     tag …_all
    --pct 10 20 40 … 100      구간 길이의 q% 까지 누적한 지점 (t = ⌈q·n/100⌉, 위치 t-1)  tag …_p10-20-40-…
                              짧은 구간에서 겹치는 지점은 하나로 합쳐지고, 100 은 구간 마지막 e_t 와 같다.
                              t < k 인 지점은 그람 rank 가 t 라 뒤쪽 고유성분이 0 이다 (그대로 저장, pos 로 구분).
    --with-hidden             위 선택과 **같은 위치**의 hidden state 행 E[p] 도 레코드의 "h" 에 저장한다 (bf16 그대로).
                              누적 gram e 는 토큰 0..p 전부가 필요해서 띄엄띄엄 저장한 E 로는 다시 못 만든다 —
                              그래서 전체 E 가 손에 있는 이 단계에서 e 와 h 를 한 번에 뽑아 둔다. 전체 E 청크는
                              건드리지 않는다 (이 스크립트는 입력을 절대 지우지 않는다). 읽는 쪽은
                              predict/probing.py --source pct --input hidden.

    구간 t 의 토큰 0..n-1, marker = 구간 앞 형식 문구가 차지하는 토큰 수
        step 구간: 0 (헤더 줄은 입력에서 빠짐) / 터미널 구간: 정답 앞 문구 (extract.TERMINAL_PAT)
    front = 위치 marker .. marker+n_front-1   (형식 토큰의 e 는 버리되 누적에는 포함)
    back  = 위치 n-n_back .. n-1
    저장 위치 = front ∪ back (겹치면 한 번). 구간이 짧아 모자라면 있는 만큼만 넣는다 (짧은 구간 제외는
    읽는 쪽이 marker+n_front+n_back 로 한다). --dtype bfloat16 으로 e 의 저장 크기를 절반으로 줄일 수 있다.

marker 는 n_front > 0 (그리고 --all 이 아님) 일 때만 필요하다. 원본 jsonl (traj_dir) 을 추출 때와 똑같이
(extract.prepare_output 으로 "Step N" 마커를 벗기고) 토크나이즈해 TERMINAL_PAT 매치가 끝나는 문자
위치까지 걸친 토큰 수로 센다. 스텝 구간은 헤더가 입력에서 빠졌으므로 marker 가 항상 0 이다.
토큰 수/경계가 hidden_states 와 다르면 그 에피소드는 버리고 사유를 센다. 원본이 없으면 fallback
(TASK → (S, T)) 으로 태스크별 고정 길이를 쓴다 (헤더를 벗긴 뒤의 Qwen3 샘플값: decompose 0:3,
plan 0:9, predict 0:7). 어느 쪽인지는 에피소드마다 "marker_src" ("text" | "fixed" | "none") 에 남는다.

출력: <out_dir>/<task>/<level>/<status>/<tag>/chunk_XXXX.pt,  tag = make_tag(...) 예: k8_scaled_sign-data_f0_b1
    {"k", "scale", "sign_mode", "n_front", "n_back", "all", "pct", "dtype", "hidden", "src", "model",
     "episodes": {seed: {"seg": [...], "labels": [...], "marker": [m_0..m_N], "marker_src": str,
                         "pos": {t: [p_1 < … < p_m]}, "e": {t: (m, kd)}, "lam": {t: (m, k)},
                         "h": {t: (m, d)}   # --with-hidden 일 때만, E 의 저장 dtype 그대로}}}
    t 는 extract.gen_view 구간 번호 (0..N-1 = Step 1..N, N = 터미널, 프롬프트 제외). pos 는 구간 안
    0-based 위치이고 e[t][i] 가 pos[t][i] 까지 누적한 e 다. 구간 마지막 e_t 는 e[t][-1] (segment_last 가
    꺼내 준다). 단위 고유벡터가 필요하면 CumulativeSpectral.eigvecs(e, lam) 으로 복원한다.

Usage:
    python inference/main.py ...                                   # extract + 저장 (기본 f0_b1)
    python inference/spectral.py                                   # 전체 task/level/status, k8 data f0_b1
    python inference/spectral.py --n-front 5 --n-back 5 -k 4 8 16 --sign-mode data max
    python inference/spectral.py --task decompose --all --dtype bfloat16
    python inference/spectral.py --n-front 5 --fallback-marker decompose=0:3 plan=0:9 predict=0:7
    python inference/spectral.py --pct 1 5 10 20 50 80 90 95 99 100 --with-hidden --dtype bfloat16 \
        --hidden-dir latent/hidden_states_marker --device cpu      # e + h, GPU 없이
"""
from __future__ import annotations

import re
import sys
import time
import hashlib
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import torch
from tqdm import tqdm

import extract
from extract import (MODEL, load_episodes, load_chunk, gen_view, seg_labels,
                     prepare_output, tokenize_text)

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("decompose", "plan", "predict")

K_EIG = 8
SCALE = True        # E를 sqrt(n)로 나눠 토큰 수에 따른 고유값 증가를 방지
SIGN_MODE = "data"  # "none" | "first" | "max" | "data"
SIGN_MODES = ("none", "first", "max", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def pct_position(q: int, n: int) -> int:
    """구간 길이 n 의 q% 지점 (0-based 위치). 누적 토큰 수 t = ⌈q·n/100⌉ 이고 위치는 t-1.
    q=100 → n-1 (구간 마지막), 작은 q 는 최소 0. probing.py 가 같은 함수로 위치를 되찾는다."""
    t = -(-q * n // 100)
    return min(max(t, 1), n) - 1


@dataclass(frozen=True)
class Select:
    """구간 안에서 어느 위치의 e 를 저장할지.

    all=True 면 모든 토큰. pct 가 있으면 구간 길이의 q% 지점들 (pct_position). 아니면
    front (marker 뒤 n_front 개) ∪ back (뒤 n_back 개).
    문자열 형식은 tag 와 같다: "all" | "p<q>-<q>-…" | "f<n_front>_b<n_back>" (Select.parse / str())."""
    n_front: int = 0
    n_back: int = 1
    all: bool = False
    pct: tuple[int, ...] = ()

    def __post_init__(self):
        if self.all:
            if (self.n_front, self.n_back, self.pct) != (0, 0, ()):
                raise ValueError("all=True 면 n_front / n_back 은 0, pct 는 비어 있어야 한다")
        elif self.pct:
            if (self.n_front, self.n_back) != (0, 0):
                raise ValueError("pct 를 주면 n_front / n_back 은 0 이어야 한다")
            if any(not (0 < q <= 100) for q in self.pct) or list(self.pct) != sorted(set(self.pct)):
                raise ValueError(f"pct 는 0 < q ≤ 100 의 오름차순 중복 없는 정수 (got {self.pct})")
        elif self.n_front < 0 or self.n_back < 0 or self.n_front + self.n_back == 0:
            raise ValueError(f"need n_front, n_back ≥ 0 and not both 0 (got {self.n_front}, {self.n_back})")

    @classmethod
    def from_pct(cls, pct) -> "Select":
        return cls(0, 0, False, tuple(sorted(set(int(q) for q in pct))))

    @classmethod
    def parse(cls, text: str) -> "Select":
        if text == "all":
            return cls(0, 0, True)
        m = re.fullmatch(r"p(\d+(?:-\d+)*)", text)
        if m:
            return cls.from_pct(m.group(1).split("-"))
        m = re.fullmatch(r"f(\d+)_b(\d+)", text)
        if not m:
            raise ValueError(f"select 는 'all' | 'p<q>-<q>-…' | 'f<n>_b<n>' (got {text!r})")
        return cls(int(m.group(1)), int(m.group(2)))

    def __str__(self) -> str:
        if self.all:
            return "all"
        if self.pct:
            return "p" + "-".join(str(q) for q in self.pct)
        return f"f{self.n_front}_b{self.n_back}"

    @property
    def needs_marker(self) -> bool:
        return not self.all and not self.pct and self.n_front > 0

    def positions(self, n: int, marker: int) -> list[int]:
        """구간 길이 n 에서 저장할 위치 (오름차순, 중복 없음). 짧으면 있는 만큼만."""
        if self.all:
            return list(range(n))
        if self.pct:
            return sorted({pct_position(q, n) for q in self.pct}) if n > 0 else []
        front = range(min(marker, n), min(marker + self.n_front, n))
        back = range(max(n - self.n_back, 0), n)
        return sorted(set(front) | set(back))


ALL = Select(0, 0, True)


def make_tag(k: int, scale: bool, sign_mode: str, select: Select = Select(),
             with_hidden: bool = False) -> str:
    """저장 디렉토리 이름. 예: k8_scaled_sign-data_f0_b1, k8_scaled_sign-data_all, k4_f5_b5,
    k8_scaled_sign-data_p1-5-…-100_h (with_hidden — e 와 같은 위치의 hidden 행 h 가 들어 있다).

    읽는 쪽(visual/heatmap.py, visual/step_similarity.py)도 이 함수를 써야 한다 —
    문자열을 손으로 베끼면 규칙이 바뀔 때 조용히 어긋난다."""
    if sign_mode not in SIGN_MODES:
        raise ValueError(f"unknown sign_mode: {sign_mode!r} (choose from {SIGN_MODES})")
    tag = f"k{k}" + ("_scaled" if scale else "")
    if sign_mode != "none":
        tag += f"_sign-{sign_mode}"
    return f"{tag}_{select}" + ("_h" if with_hidden else "")


# ------------------------------------------------------------------ 부호 보정

def fix_sign(R: torch.Tensor, proj: torch.Tensor | None, mode: str):
    """고유벡터 부호 보정.

    R    : (r, d)  부호를 정할 벡터들 σ_k v_k = Eᵀu_k. 부호 규칙은 양수 배율에 무관하므로
           단위벡터 v_k 를 넘겨도 같은 부호가 나온다.
    proj : (n, r)  데이터 투영 v_k·x_i = E v_k = σ_k u_k. "data" 에서만 쓴다.

    - "none":  보정 안 함
    - "first": 각 벡터의 첫 성분이 양수가 되도록
    - "max":   각 벡터의 최대 절댓값 성분이 양수가 되도록
    - "data":  Bro, Acar & Kolda (2007)의 부호-가중 내적 점수
               s_k = Σ_i sign(v_k·x_i)(v_k·x_i)²  가 양수가 되도록. 대칭(Gram) 케이스 단순화 버전.

    반환: (부호 보정된 R, score 또는 None). "data" 의 score 는 보정 전 s_k (r,) 로, 부호는
    분해기가 뽑은 임의 초기 부호에 따르니 |score|/λ ∈ [0,1] 만 부호 신뢰도로 쓸 것.
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


# ---------------------------------------------------------- 토큰 단위 누적 그람

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


@torch.no_grad()
def tokens_cumulative(E: torch.Tensor, seg: list[int], k: int = K_EIG, scale: bool = SCALE,
                      sign_mode: str = SIGN_MODE, device: str | torch.device = DEVICE,
                      cs: CumulativeSpectral | None = None,
                      out_dtype: torch.dtype = torch.float32):
    """토큰열 E (n x d) 를 seg 경계마다 리셋하며 토큰 전부의 누적 e 를 (n x kd) 로 모은다.
    반환 (e_all, lam_all (n x k)).

    seg = [0, b1, ..., n] (extract.gen_view 규약). 어떤 경계든 받는다 — step 경계 대신
    GSBS 경계를 넣어 리셋 지점을 바꿔 볼 수도 있다. heatmap.py / gsbs_batch.py 처럼 토큰
    전체 행렬이 필요한 곳용이고 파일은 만들지 않는다. 메모리: n·k·d·(dtype 바이트),
    k=8 / 600 토큰이면 float32 ≈ 98MB — 에피소드 하나 단위로만 들고 있을 것.
    cs 를 넘기면 그 누적기를 재사용한다 (버퍼 재할당 없이 여러 에피소드를 돌릴 때).
    """
    n, d = E.shape
    if seg[0] != 0 or seg[-1] != n or any(a >= b for a, b in zip(seg, seg[1:])):
        raise ValueError(f"bad seg {seg} for E of {n} tokens")
    if cs is None:
        cs = CumulativeSpectral(k, scale, sign_mode, device)
    elif (cs.k, cs.scale, cs.sign_mode) != (k, scale, sign_mode):
        raise ValueError(f"cs config {(cs.k, cs.scale, cs.sign_mode)} != {(k, scale, sign_mode)}")
    e_all = torch.empty(n, k * d, dtype=out_dtype)
    lam_all = torch.empty(n, k, dtype=torch.float32)
    for s, en in zip(seg, seg[1:]):
        cs.reset()
        for i, (e, lam, _) in enumerate(cs.feed(E[s:en])):
            e_all[s + i] = e.to("cpu", out_dtype)
            lam_all[s + i] = lam.cpu()
    cs.reset()
    return e_all, lam_all


# ------------------------------------------------------------------ 마커 길이

def load_sources(traj_dir: Path, task: str, mode: str = "no_thinking") -> dict:
    """(env_name, env_seed, output_sha1) → jsonl 에피소드. 파일이 없으면 빈 dict."""
    path = traj_dir / f"{task}_{mode}.jsonl"
    if not path.is_file():
        print(f"warning: {path} 없음 — marker 를 텍스트로 못 세니 fallback 만 쓴다", file=sys.stderr)
        return {}
    out = {}
    for ep in load_episodes(path):
        text = ep.get("all_llm_output") or ""
        sha = hashlib.sha1(text.encode()).hexdigest()
        out[(ep["env_name"], int(ep["env_seed"]), sha)] = ep
    return out


def marker_lengths(src_ep: dict, task: str, boundaries: list[int]):
    """구간별 형식 토큰 수 [m_step1, ..., m_stepN, m_terminal]. 실패 시 (None, 사유).

    extract.prepare_output 이 "Step N" 마커를 입력에서 빼므로 스텝 구간의 형식 토큰은
    0 이다 (헤더 줄의 제목은 본문으로 친다). 터미널 구간만 정답 앞 문구("The LLM's action sequence is:" 등)가 남아 그 토큰 수를
    센다. 원본을 추출 때와 똑같이 (마커 제거 → 토크나이즈) 처리해서 토큰 수·경계가
    hidden_states 와 일치하는지 확인하고, 다르면 버린다.

    extract.tok 이 세팅되어 있어야 한다 (ensure_tokenizer)."""
    cleaned, reason = prepare_output(src_ep["all_llm_output"], task)
    if reason is not None:
        return None, f"bounds:{reason}"
    prompt = extract.render_prompt(src_ep)
    enc, tb, reason = tokenize_text(prompt, cleaned.text, cleaned.bounds)
    if reason is not None:
        return None, f"tokenize:{reason}"
    if len(enc.input_ids) != boundaries[-1]:
        return None, "token_count_mismatch"
    if tb != list(boundaries):
        return None, "boundary_mismatch"

    offs = enc.offset_mapping
    s_tok, e_tok, char_end = boundaries[-2], boundaries[-1], len(prompt) + cleaned.term_end
    m = 0                                                        # char_end 까지 걸친 토큰 수
    while s_tok + m < e_tok and offs[s_tok + m][0] < char_end:
        m += 1
    n_steps = len(boundaries) - 3
    return [0] * n_steps + [m], None


def ensure_tokenizer():
    if extract.tok is None:
        from transformers import AutoTokenizer
        extract.tok = AutoTokenizer.from_pretrained(MODEL)


def parse_fallback(items: list[str]) -> dict[str, tuple[int, int]]:
    """CLI 의 --fallback-marker TASK=S:T 목록 → {task: (S, T)}."""
    out = {}
    for kv in items:
        task_, v = kv.split("=")
        st, te = v.split(":")
        out[task_] = (int(st), int(te))
    return out


# ------------------------------------------------------------ 에피소드 레코드

@torch.no_grad()
def episode_record(episode: dict, marks: list[int], cs: CumulativeSpectral, select: Select,
                   marker_src: str, dtype: torch.dtype = torch.float32,
                   with_hidden: bool = False) -> dict:
    """에피소드 하나 → 저장 레코드 (모듈 docstring 의 episodes[seed] 형식). e 만 dtype 으로, lam 은 float32.

    with_hidden=True 면 같은 위치의 hidden state 행 E[s+p] 를 rec["h"][t] 에 (m, d) 로 넣는다 — E 의
    dtype 그대로 (bf16 저장본이면 bf16). e[t][i] 와 h[t][i] 는 같은 pos[t][i] 의 것이다."""
    E, seg = gen_view(episode)
    k, kd, d = cs.k, cs.k * E.shape[1], E.shape[1]
    rec = {"seg": seg, "labels": seg_labels(seg), "marker": marks, "marker_src": marker_src,
           "pos": {}, "e": {}, "lam": {}}
    if with_hidden:
        rec["h"] = {}
    for t, (s, e) in enumerate(zip(seg, seg[1:])):
        ps = select.positions(e - s, marks[t])
        cs.reset()
        out = cs.feed_at(E[s:e], ps) if ps else []
        cs.reset()
        rec["pos"][t] = ps
        rec["e"][t] = torch.stack([r[0] for r in out]).to("cpu", dtype) if ps else torch.empty(0, kd, dtype=dtype)
        rec["lam"][t] = torch.stack([r[1] for r in out]).cpu() if ps else torch.empty(0, k)
        if with_hidden:
            idx = torch.as_tensor([s + p for p in ps], dtype=torch.long)
            rec["h"][t] = E[idx] if ps else torch.empty(0, d, dtype=E.dtype)   # 인덱싱은 복사 — E 전체가 view 로 안 남는다
    return rec


def segment_last(rec: dict) -> dict[int, torch.Tensor]:
    """레코드에서 구간 마지막 e_t 사전 {t: (kd,)}. 마지막 위치가 저장된 구간만 (n_back ≥ 1 또는 all)."""
    seg = rec["seg"]
    return {t: rec["e"][t][-1] for t, ps in rec["pos"].items()
            if ps and ps[-1] == seg[t + 1] - seg[t] - 1}


# ------------------------------------------------------------------- 저장 실행

def load_hidden_states(data_dir: Path, task: str, level: str, status: str):
    """extract.py 출력 경로: <data_dir>/<task>/<level>/<status>/chunk_*.pt"""
    target = data_dir / task / level / status
    if not target.is_dir():
        raise FileNotFoundError(f"no such directory: {target}")

    files = sorted(target.glob("chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"no chunk_*.pt in {target}")
    return files


Config = tuple[int, bool, str]      # (k, scale, sign_mode)


@torch.no_grad()
def process_chunk(cf: Path, task: str, level: str, status: str, out_root: Path,
                  configs: list[Config], select: Select = Select(),
                  sources: dict | None = None, fallback: dict | None = None,
                  device: str | torch.device = DEVICE, dtype: torch.dtype = torch.float32,
                  overwrite: bool = False, stats: Counter | None = None,
                  with_hidden: bool = False) -> Counter:
    """hidden_states 청크 하나를 읽어 configs 마다 저장 파일 하나씩 쓴다.

    청크를 한 번만 메모리에 올리고 marker 도 한 번만 세서 여러 (k, scale, sign_mode) 에 재사용한다.
    overwrite=False 면 이미 있는 출력은 건너뛴다. 입력 청크(cf)는 읽기만 한다.
    with_hidden=True 면 e 와 같은 위치의 hidden 행 h 도 넣고 태그에 _h 가 붙는다."""
    stats = Counter() if stats is None else stats
    out_of = {c: out_root / task / level / status / make_tag(*c, select, with_hidden) / cf.name
              for c in configs}
    todo = [c for c in configs if overwrite or not out_of[c].exists()]
    if not todo:
        stats["chunk_skipped"] += 1
        return stats
    d = torch.load(cf, map_location="cpu", weights_only=False)

    marks_of: dict = {}                                          # seed → (marks, marker_src)
    for seed, ep in d["episodes"].items():
        n_seg = len(ep["boundaries"]) - 2                        # step N 개 + 터미널
        if not select.needs_marker:
            marks_of[seed] = ([0] * n_seg, "none")
            continue
        src = (sources or {}).get((level, int(seed), ep["output_sha1"]))
        if src is None:
            if not fallback or task not in fallback:
                stats["no_source"] += 1
                continue
            st, te = fallback[task]
            marks_of[seed] = ([st] * (n_seg - 1) + [te], "fixed")
            stats["marker_fixed"] += 1
            continue
        ensure_tokenizer()
        marks, reason = marker_lengths(src, task, ep["boundaries"])
        if reason:
            stats[reason] += 1
            continue
        marks_of[seed] = (marks, "text")
        stats["marker_text"] += 1
        for m in marks[:-1]:
            stats[f"hist_step_marker_{m}"] += 1
        stats[f"hist_term_marker_{marks[-1]}"] += 1

    for c in todo:
        k, scale, sm = c
        cs = CumulativeSpectral(k, scale, sm, device)
        eps = {}
        for seed, (marks, msrc) in marks_of.items():
            eps[seed] = episode_record(d["episodes"][seed], marks, cs, select, msrc, dtype, with_hidden)
        out = out_of[c]
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"k": k, "scale": scale, "sign_mode": sm,
                    "n_front": select.n_front, "n_back": select.n_back, "all": select.all,
                    "pct": list(select.pct), "hidden": with_hidden,
                    "dtype": str(dtype).removeprefix("torch."),
                    "src": str(cf), "model": d.get("model", MODEL), "episodes": eps}, out)
        stats["files"] += 1
        stats["episodes"] += len(eps)
    return stats


@torch.no_grad()
def spectral_run(hidden_dir: Path, out_root: Path, task: str, level: str, status: str,
                 configs: list[Config], select: Select = Select(),
                 traj_dir: Path | None = None, mode: str = "no_thinking",
                 fallback: dict | None = None, device: str | torch.device = DEVICE,
                 dtype: torch.dtype = torch.float32, overwrite: bool = False,
                 with_hidden: bool = False) -> Counter:
    """한 (task, level, status) 의 모든 청크를 configs 마다 저장한다. 반환: 집계 Counter.

    select.needs_marker 면 traj_dir 의 원본 jsonl 로 marker 를 센다 (없으면 fallback).
    overwrite=True 면 **이 스크립트가 만든 출력 디렉토리**의 낡은 chunk 파일만 지운다
    (hidden_states 청크 수가 줄었을 때). 입력 hidden_states 는 어떤 경우에도 건드리지 않는다."""
    hidden_dir = Path(hidden_dir).resolve()
    chunk_files = load_hidden_states(hidden_dir, task, level, status)
    sources = None
    if select.needs_marker and traj_dir is not None:
        sources = load_sources(Path(traj_dir), task, mode)

    if overwrite:
        for c in configs:
            out_dir = out_root / task / level / status / make_tag(*c, select, with_hidden)
            for old in out_dir.glob("chunk_*.pt"):
                old.unlink()

    stats = Counter()
    desc = f"{task}/{level}/{status} {select}{'+h' if with_hidden else ''} x{len(configs)}"
    for cf in tqdm(chunk_files, desc=desc, unit="chunk"):
        process_chunk(cf, task, level, status, out_root, configs, select,
                      sources, fallback, device, dtype, overwrite, stats, with_hidden)
    return stats


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--level", nargs="+", default=None, help="기본: hidden_states 에 있는 전부")
    ap.add_argument("--status", nargs="+", default=["success", "failure"], choices=["success", "failure"])
    ap.add_argument("-k", type=int, nargs="+", default=[K_EIG], help="고유벡터 개수 k (여러 개 가능)")
    ap.add_argument("--scale", nargs="+", default=["true" if SCALE else "false"], choices=["true", "false"],
                    help="E/√n 스케일 (여러 개 가능)")
    ap.add_argument("--sign-mode", nargs="+", default=[SIGN_MODE], choices=list(SIGN_MODES),
                    help="부호 보정 규칙 (여러 개 가능)")
    ap.add_argument("--n-front", type=int, default=0, help="구간 앞(형식 문구 뒤)에서 저장할 토큰 수")
    ap.add_argument("--n-back", type=int, default=1, help="구간 뒤에서 저장할 토큰 수 (1 = 마지막 e_t 만)")
    ap.add_argument("--all", action="store_true", help="구간의 모든 토큰을 저장 (--n-front/--n-back 무시)")
    ap.add_argument("--pct", type=int, nargs="+", default=None, metavar="Q",
                    help="구간 길이의 Q%% 지점들만 저장 (예: 10 20 40 60 80 90 100; --n-front/--n-back 무시)")
    ap.add_argument("--with-hidden", action="store_true",
                    help="선택한 위치의 hidden state 행 E[p] 도 같이 저장 (\"h\"; 태그에 _h). "
                         "probing.py --source pct/edges --input hidden 이 읽는다")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES), help="e 저장 dtype (lam 은 float32)")
    ap.add_argument("--hidden-dir", type=Path, default=ROOT / "latent" / "hidden_states")
    ap.add_argument("--traj-dir", type=Path, default=ROOT / "generation" / "trajectory",
                    help="marker 를 셀 원본 jsonl 위치 (--n-front > 0 이고 --all 이 아닐 때만 쓴다)")
    ap.add_argument("--mode", default="no_thinking", choices=["no_thinking", "thinking"],
                    help="원본 jsonl 이름: <traj-dir>/<task>_<mode>.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "latent" / "spectral")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--fallback-marker", nargs="+", default=[], metavar="TASK=S:T",
                    help="원본 텍스트가 없을 때 쓸 태스크별 형식 토큰 수. "
                         "S=step 헤더, T=터미널 문구 (예: predict=5:7)")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 출력도 다시 만든다")
    a = ap.parse_args()

    configs = [(k, s == "true", sm) for k in a.k for s in a.scale for sm in a.sign_mode]
    select = ALL if a.all else Select.from_pct(a.pct) if a.pct else Select(a.n_front, a.n_back)
    fallback = parse_fallback(a.fallback_marker)
    total = Counter()
    t0 = time.time()
    for task in a.task:
        levels = a.level or sorted(p.name for p in (a.hidden_dir / task).iterdir() if p.is_dir())
        for level in levels:
            for status in a.status:
                if not (a.hidden_dir / task / level / status).is_dir():
                    continue
                stats = spectral_run(a.hidden_dir, a.out_dir, task, level, status, configs, select,
                                     a.traj_dir, a.mode, fallback, a.device, DTYPES[a.dtype], a.overwrite,
                                     a.with_hidden)
                print(f"{task}/{level}/{status}: {dict(stats)}  ({time.time() - t0:.0f}s)")
                total.update(stats)
    print(f"\ntotal: {dict(total)}")


if __name__ == "__main__":
    main()