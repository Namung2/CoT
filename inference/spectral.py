from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

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

    if fix_sign:  # q와 -q 모호성 제거: 최대 절댓값 성분을 양수로
        idx = V_k.abs().argmax(dim=1)
        sign = torch.sign(V_k.gather(1, idx[:, None]))
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
def episode_embeddings(E: dict[int, torch.Tensor], k: int, scale: bool, fix_sign: bool):
    e, lam, V = {}, {}, {}
    for t in sorted(E):
        e[t], lam[t], V[t] = spectral_embedding(E[t].to(DEVICE), k, scale, fix_sign)
    return e, lam, V

@torch.no_grad()
def spectral_run(src: Path, k: int = K_EIG, scale: bool = SCALE, fix_sign: bool = FIX_SIGN):
    data = torch.load(src, map_location="cpu", weights_only=False)
    e, lam, V = episode_embeddings(data["E"], k, scale, fix_sign)
    return {"k": k, "scale": scale, "fix_sign": fix_sign,
            "src": str(src), "model": data["model"],
            "e": e, "eigvals": lam, "V": V}


def spectral_dir_run(src_dir: Path, dst: Path, method: str,
                     k: int = K_EIG, scale: bool = SCALE, fix_sign: bool = FIX_SIGN):
    """디렉터리 안의 에피소드 .pt를 전부 돌려 하나로 합쳐 저장."""
    files = sorted(src_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt in {src_dir}")

    merged = {"k": k, "scale": scale, "fix_sign": fix_sign, "method": method,
              "src": str(src_dir),
              "model": None, "e": {}, "eigvals": {}, "V": {}}

    for f in tqdm(files, desc=dst.stem, unit="ep"):  # 설정별로 구분되도록 dst 이름 사용
        out = spectral_run(f, k, scale, fix_sign)
        if merged["model"] is None:
            merged["model"] = out["model"]
        merged["e"][f.stem] = out["e"]
        merged["eigvals"][f.stem] = out["eigvals"]
        merged["V"][f.stem] = out["V"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, dst)
    print(f"saved {dst}")
    return merged