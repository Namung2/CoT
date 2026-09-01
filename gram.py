from __future__ import annotations

import torch


@torch.no_grad()
def gram(E: torch.Tensor, scale: bool = False):

    if E.ndim != 2:
        raise ValueError(f"expected 2-D (n, d), got {tuple(E.shape)}")

    E = E.float()          # bf16 은 linalg 미지원
    if scale:
        E = E / (E.shape[0] ** 0.5)     # G -> G/n (trace 가 토큰 수를 인코딩하지 않도록)

    _, S, Vh = torch.linalg.svd(E, full_matrices=False)
    return S ** 2, Vh
