from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

HID_DIR = Path("hidden_states")
OUT_DIR = Path("e")

K_EIG = 8
SCALE = True # sqrt
PREFETCH = 4  # in-flight file loads, overlapped with GPU compute


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    idx = max(range(len(free)), key=free.__getitem__)  # least-loaded GPU, not always cuda:0
    return f"cuda:{idx}"


DEVICE = pick_device()


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
    return e_t, lam                                        # stays on DEVICE; caller syncs once per sample


@torch.no_grad()
def sample_embeddings(E: dict[int, torch.Tensor], k: int, scale: bool):
    ts = sorted(E)
    e_gpu, lam_gpu = {}, {}
    for t in ts:
        e_gpu[t], lam_gpu[t] = step_e(E[t].to(DEVICE, non_blocking=True), k, scale)
    # one host sync for the whole sample instead of one per step
    e_out = {t: e_gpu[t].cpu() for t in ts}
    lam_out = {t: lam_gpu[t].cpu() for t in ts}
    return e_out, lam_out


def _load(path: Path):
    # no mmap: force the real disk read to happen here, in the background
    # thread, so it actually overlaps with the main thread's GPU work
    # (mmap defers the read to first access, which would happen on the
    # main thread inside step_e and defeat the point of prefetching)
    return torch.load(path, map_location="cpu")


def _iter_prefetched(files: list[Path], prefetch: int = PREFETCH):
    """Yield (file, data) pairs while loading up to `prefetch` files ahead in background threads."""
    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        window = deque()
        for f in files[:prefetch]:
            window.append((f, pool.submit(_load, f)))
        for f in files[prefetch:]:
            nf, nfut = window.popleft()
            yield nf, nfut.result()
            window.append((f, pool.submit(_load, f)))
        while window:
            nf, nfut = window.popleft()
            yield nf, nfut.result()


def run(level: str, case: str, method: str, k: int = K_EIG, scale: bool = SCALE):
    src_dir = HID_DIR / level / "cases" / case / method
    files = sorted(src_dir.glob("*.pt"))
    print(f"loading {len(files)} samples from {src_dir} ... device={DEVICE}")

    result = {"k": k, "scale": scale, "method": method, "model": None,
              "e": {}, "eigvals": {}}

    t0 = time.time()
    for f, data in tqdm(_iter_prefetched(files), total=len(files),
                         desc=f"{level}/{case}/{method}", unit="sample"):
        if result["model"] is None:
            result["model"] = data["model"]
        sample_id = f.stem
        e_out, lam_out = sample_embeddings(data["E"], k, scale)
        result["e"][sample_id] = e_out
        result["eigvals"][sample_id] = lam_out

    print(f"done in {time.time() - t0:.1f}s")
    dst_dir = OUT_DIR / level
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{case}_{method}.pt"
    torch.save(result, dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    for c in ("c3",):
        for m in ("A", "B"):
            run(level="gotoseq_10to50", case=c, method=m)