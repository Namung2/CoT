from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import tqdm

HID_DIR = Path("hidden_states")
OUT_DIR = Path("e")

K_EIG = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = True # sqrt 

@torch.no_grad()
def step_e(Et: torch.Tensor, k: int, scale: bool):
    Et = Et.float()
    if scale:
        Et = Et / (Et.shape[0] ** 0.5) #Et.shape[0] = 토큰 수(n_t), 토큰 수가 길어져서 생기는 고유값 증가를 방지

    _, S, Vh = torch.linalg.svd(Et, full_matrices=False)   # S:(r,), Vh:(r,d)
    r = min(k, S.shape[0])
    S_k, V_k = S[:r], Vh[:r]                               # 내림차순 보장됨

    # 고유벡터 부호 고정 (q와 -q 모호성 제거): 최대 절댓값 성분을 양수로
    idx = V_k.abs().argmax(dim=1)
    sign = torch.sign(V_k.gather(1, idx[:, None]))
    sign[sign == 0] = 1.0
    V_k = V_k * sign

    d = Et.shape[1]
    if r < k:                                              # rank(G_t) ≤ n_t 보정
        S_k = torch.cat([S_k, S_k.new_zeros(k - r)])
        V_k = torch.cat([V_k, V_k.new_zeros(k - r, d)])

    e_t = (S_k[:, None] * V_k).reshape(-1)                 # [σ₁v₁; ...; σ_k v_k] = [√λ₁q₁; ...]
    lam = S_k ** 2
    return e_t.cpu(), lam.cpu()


@torch.no_grad()
def sample_embeddings(E: dict[int, torch.Tensor], k: int, scale: bool):
    e_out, lam_out = {}, {}
    for t in sorted(E):
        e_t, lam = step_e(E[t].to(DEVICE), k, scale)
        e_out[t], lam_out[t] = e_t, lam
    return e_out, lam_out


def run(level: str, case: str, k: int = K_EIG, scale: bool = SCALE):
    src = HID_DIR / level / f"{case}.pt"
    print(f"loading {src} ...")
    data = torch.load(src, map_location="cpu")
    print(f"  {len(data['E'])} samples, device={DEVICE}, k={k}, scale={scale}")

    result = {"k": k, "scale": scale, "model": data["model"],
              "e": {}, "eigvals": {}}

    t0 = time.time()
    for sample_id, E in tqdm(data["E"].items(), desc=f"{level}/{case}",unit="sample"):
        e_out, lam_out = sample_embeddings(E, k, scale)
        result["e"][sample_id] = e_out
        result["eigvals"][sample_id] = lam_out

    print(f"done in {time.time() - t0:.1f}s")
    dst_dir = OUT_DIR / level
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{case}.pt"
    torch.save(result, dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    for c in ("c3",):
        run(level="gotoseq", case=c)