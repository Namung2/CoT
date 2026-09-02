from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from extract import MODEL, load_step_views

K_EIG = 8
SCALE = True      # E_t를 sqrt(n_t)로 나눠 토큰 수에 따른 고유값 증가를 방지
FIX_SIGN = True   # 고유벡터 부호 고정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def spectral_embedding(Et: torch.Tensor, k: int, scale: bool, fix_sign: bool):
    Et = Et.float()  # torch.linalg.svd는 bf16 미지원

    if scale:
        Et = Et / (Et.shape[0] ** 0.5)  # Et.shape[0] = n_t

    _, S, Vh = torch.linalg.svd(Et, full_matrices=False)   # S:(r,), Vh:(r,d)
    r = min(k, S.shape[0])                                 # rank(G_t) ≤ n_t
    S_k, V_k = S[:r], Vh[:r]                               # 내림차순 보장됨

    if fix_sign:  # q와 -q 모호성 제거: 첫 번째 성분을 양수로 통일
        sign = torch.sign(V_k[:, :1])
        sign[sign == 0] = 1.0
        V_k = V_k * sign

    if r < k:  # 부족분은 0 (λ=0이면 √λ·q = 0이므로 정확한 값)
        d = Et.shape[1]
        S_k = torch.cat([S_k, S_k.new_zeros(k - r)])
        V_k = torch.cat([V_k, V_k.new_zeros(k - r, d)])

    e_t = (S_k[:, None] * V_k).reshape(-1)   # [√λ₁q₁; ...; √λ_k q_k], (kd,)
    lam = S_k ** 2                           # λ_i = σ_i²
    return e_t.cpu(), lam.cpu(), V_k.cpu()


@torch.no_grad()
def episode_embeddings(views: list[torch.Tensor], k: int, scale: bool, fix_sign: bool):
    e, lam, V = {}, {}, {}
    for t, Et in enumerate(views):
        e[t], lam[t], V[t] = spectral_embedding(Et.to(DEVICE), k, scale, fix_sign)
    return e, lam, V

def load_hidden_states(data_dir: Path, task: str, level: str, method: str,
                       status: str, ctx_tag: str = "with_prompt"):
    target = data_dir / task / level / method / ctx_tag / status
    if not target.is_dir():
        raise FileNotFoundError(f"no such directory: {target}")

    files = sorted(target.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt in {target}")
    return files

@torch.no_grad()
def spectral_run(data_root: Path, out_root: Path, task: str, level: str, method: str,
                 status: str, k: int = K_EIG, scale: bool = SCALE, fix_sign: bool = FIX_SIGN,
                 ctx_tag: str = "with_prompt"):

    data_root = data_root.resolve()
    files = load_hidden_states(data_root, task, level, method, status, ctx_tag)

    tag = f"k{k}" + ("_scaled" if scale else "") + ("_signfix" if fix_sign else "")
    rel = Path(task) / level
    out_dir = out_root / rel / method / ctx_tag / status / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(files, desc=f"{rel}/{method}/{tag}", unit="ep"):
        _, views = load_step_views(f)
        e, lam, V = episode_embeddings(views, k, scale, fix_sign)

        out_path = out_dir / f.name
        torch.save({"k": k, "scale": scale, "fix_sign": fix_sign,
                    "src": str(f), "model": MODEL,
                    "e": e, "eigvals": lam, "V": V}, out_path)

    print(f"saved {len(files)} files under {out_dir}")
