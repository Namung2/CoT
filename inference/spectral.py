from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from extract import MODEL, load_chunk, load_step_views

K_EIG = 8
SCALE = True        # E_t를 sqrt(n_t)로 나눠 토큰 수에 따른 고유값 증가를 방지
SIGN_MODE = "data"  # "none" | "first" | "max" | "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIGN_MODES = ("none", "first", "max", "data")


def make_tag(k: int, scale: bool, sign_mode: str) -> str:
    """spectral_states 저장 디렉토리 이름. 예: k8_scaled_sign-data, k8_scaled (none).

    읽는 쪽(visual/heatmap.py, visual/step_similarity.py)도 이 함수를 써야 한다 —
    문자열을 손으로 베끼면 규칙이 바뀔 때 조용히 어긋난다."""
    if sign_mode not in SIGN_MODES:
        raise ValueError(f"unknown sign_mode: {sign_mode!r} (choose from {SIGN_MODES})")
    tag = f"k{k}" + ("_scaled" if scale else "")
    if sign_mode != "none":
        tag += f"_sign-{sign_mode}"
    return tag


def _fix_sign(V_k: torch.Tensor, Et: torch.Tensor, mode: str):
    """고유벡터 부호 보정. V_k: (r, d), Et: (n, d).

    - "none":  보정 안 함
    - "first": 각 벡터의 첫 성분이 양수가 되도록 (기존 방식)
    - "max":   각 벡터의 최대 절댓값 성분이 양수가 되도록
    - "data":  Bro, Acar & Kolda (2007)의 부호-가중 내적 점수
               s_k = Σ_i sign(v_k·x_i)(v_k·x_i)²  (x_i = Et의 i번째 행)
               가 양수가 되도록. 대칭(Gram) 케이스 단순화 버전.

    반환: (부호 보정된 V_k, score 또는 None)
    "data" 모드의 score는 (r,) — |score|/λ ∈ [0,1]이 부호 신뢰도 지표.
    """
    if mode == "none":
        return V_k, None

    if mode == "first":
        sign = torch.sign(V_k[:, :1])                     # (r, 1)
        sign[sign == 0] = 1.0
        return V_k * sign, None

    if mode == "max":
        idx = V_k.abs().argmax(dim=1)                     # (r,)
        vals = V_k.gather(1, idx[:, None])                # (r, 1)
        sign = torch.sign(vals)
        sign[sign == 0] = 1.0
        return V_k * sign, None

    if mode == "data":
        proj = Et @ V_k.T                                 # (n, r): v_k·x_i
        score = (torch.sign(proj) * proj**2).sum(dim=0)   # (r,)
        sign = torch.sign(score)
        sign[sign == 0] = 1.0                             # 완전 대칭이면 그대로 둠
        return V_k * sign[:, None], score

    raise ValueError(f"unknown sign_mode: {mode!r}")


@torch.no_grad()
def spectral_embedding(Et: torch.Tensor, k: int, scale: bool, sign_mode: str):
    Et = Et.float()  # torch.linalg.svd는 bf16 미지원

    if scale:
        Et = Et / (Et.shape[0] ** 0.5)  # Et.shape[0] = n_t (부호에는 영향 없음)

    _, S, Vh = torch.linalg.svd(Et, full_matrices=False)   # S:(r,), Vh:(r,d)
    r = min(k, S.shape[0])                                 # rank(G_t) ≤ n_t
    S_k, V_k = S[:r], Vh[:r]                               # 내림차순 보장됨

    V_k, score = _fix_sign(V_k, Et, sign_mode)

    if r < k:  # 부족분은 0 (λ=0이면 √λ·q = 0이므로 정확한 값)
        d = Et.shape[1]
        S_k = torch.cat([S_k, S_k.new_zeros(k - r)])
        V_k = torch.cat([V_k, V_k.new_zeros(k - r, d)])
        if score is not None:
            score = torch.cat([score, score.new_zeros(k - r)])

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


def load_hidden_states(data_dir: Path, task: str, level: str,
                       status: str, ctx_tag: str = "with_prompt"):
    target = data_dir / task / level / ctx_tag / status
    if not target.is_dir():
        raise FileNotFoundError(f"no such directory: {target}")

    files = sorted(target.glob("chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"no chunk_*.pt in {target}")
    return files


@torch.no_grad()
def spectral_run(data_root: Path, out_root: Path, task: str, level: str,
                 status: str, k: int = K_EIG, scale: bool = SCALE,
                 sign_mode: str = SIGN_MODE, ctx_tag: str = "with_prompt"):

    data_root = data_root.resolve()
    chunk_files = load_hidden_states(data_root, task, level, status, ctx_tag)

    tag = make_tag(k, scale, sign_mode)
    rel = Path(task) / level
    out_dir = out_root / rel / ctx_tag / status / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.pt"):   # hidden_states 청크 개수가 줄었을 때 낡은 파일 안 남게
        old.unlink()

    n_episodes = 0
    for cf in tqdm(chunk_files, desc=f"{rel}/{tag}", unit="chunk"):
        chunk = load_chunk(cf)
        out = {}
        for seed, episode in chunk.items():
            _, views = load_step_views(episode)
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
