"""
STEP 1 진단 — 파일럿 1 에피소드의 실시퀀스로 전부 측정.

setting.md §6 STEP 1.  batch=1 이므로 패딩 6조합과 배치 수치 실험은 삭제됐고,
남은 노이즈원은 T1(KV캐시 vs 전체 attention)과 T6(dtype) 둘뿐이다.

  1-a  인과성      뒤쪽 토큰 j 교체 -> H[:, :j] 전 layer 불변인지
  1-b  prefix 불변  H_full[:M] vs H_pref[:M]        (1-a 의 따름정리)
  1-c  span 정합    generate.py 가 이미 기록 -> 여기선 재확인만
  1-d  T1          생성 시 hidden state vs teacher-forcing 재투입  ★ 저장 전략 결정
  1-e  T6          bf16 vs fp32
  1-f  A1          off-diag cosine, effective rank, λ1/Σλ, eigengap
  1-g  A3          T1/T6 전후 q_i cosine · 부호 뒤집힘  -> k 상한
  1-h  trace       trA 단조성, trA/N_t 상수성, trB/n_t

1-a 가 통과해야 나머지가 의미를 갖는다 (method A/B 구분의 전제).

사용
  python -m inference.probe_step1 --steps data/P1/steps.jsonl \
      --hidden data/P1/hidden_0.pt --out data/P1/step1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------- 스펙트럼


def spectrum(E: torch.Tensor, k: int = 16, scale: bool = False):
    """E: (n, d) -> (λ[:k], q[:k] (k,d)).

    G = E^T E 와 K = E E^T 는 0 이 아닌 고유값이 완전히 동일하므로 SVD 한 번이면
    둘 다 나온다 (E = U S V^T -> G = V S^2 V^T, K = U S^2 U^T).
    setting.md M4: 부호 모호성 때문에 q 는 관례를 고정해야 비교가 가능하다.
    """
    X = E.float()
    if scale:
        X = X / (X.shape[0] ** 0.5)          # setting.md M2 대응
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    lam = (S ** 2)[:k]
    q = Vh[:k]
    # 부호 관례: 최대 절댓값 성분을 양수로 (M4 의 일부만 해결, 축퇴는 미해결)
    idx = q.abs().argmax(dim=1)
    sign = torch.sign(q[torch.arange(q.shape[0]), idx])
    sign[sign == 0] = 1.0
    q = q * sign[:, None]
    return lam, q, U[:, :k]


def et_from(lam, q):
    """CTRLS Eq.(5): e_t = [sqrt(λ1)q1; ...; sqrt(λk)qk]"""
    return (lam.sqrt()[:, None] * q).reshape(-1)


def a1_stats(E: torch.Tensor, k: int = 16) -> dict:
    """A1: step 내 토큰끼리의 유사도와 스펙트럼 집중도."""
    X = E.float()
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
    C = Xn @ Xn.T
    n = C.shape[0]
    off = C[~torch.eye(n, dtype=bool)] if n > 1 else torch.tensor([float("nan")])

    lam, q, _ = spectrum(X, k=min(k, min(X.shape)))
    lam_all = torch.linalg.svdvals(X) ** 2
    p = lam_all / lam_all.sum()
    eff_rank_pr = float((lam_all.sum() ** 2) / (lam_all ** 2).sum())
    eff_rank_ent = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    gaps = (lam[:-1] / lam[1:].clamp_min(1e-12)).tolist()
    return {
        "n_tokens": int(n),
        "mean_offdiag_cos": float(off.mean()),
        "p05_offdiag_cos": float(off.quantile(0.05)) if n > 1 else None,
        "p95_offdiag_cos": float(off.quantile(0.95)) if n > 1 else None,
        "eff_rank_participation": eff_rank_pr,
        "eff_rank_entropy": eff_rank_ent,
        "lambda1_ratio": float(p[0]),
        "eigengap_ratio": gaps,
        "trace": float(lam_all.sum()),
        "trace_per_token": float(lam_all.sum() / n),
    }


def compare(A: torch.Tensor, B: torch.Tensor, k: int = 16) -> dict:
    """두 E 행렬을 사다리 각 층에서 비교. no-op 변환 검증용."""
    A, B = A.float(), B.float()
    hc = torch.nn.functional.cosine_similarity(A, B, dim=1)
    rel = (A - B).norm(dim=1) / A.norm(dim=1).clamp_min(1e-9)

    KA, KB = A @ A.T, B @ B.T
    lamA, qA, _ = spectrum(A, k)
    lamB, qB, _ = spectrum(B, k)
    qc = torch.nn.functional.cosine_similarity(qA, qB, dim=1)
    eA, eB = et_from(lamA, qA), et_from(lamB, qB)
    return {
        "h_cos": {"p50": float(hc.median()), "p05": float(hc.quantile(.05)),
                  "min": float(hc.min())},
        "h_rel_l2": {"p50": float(rel.median()), "p95": float(rel.quantile(.95)),
                     "max": float(rel.max())},
        "K_frob_rel": float((KA - KB).norm() / KA.norm().clamp_min(1e-9)),
        "lambda_rel_err": ((lamA - lamB).abs() / lamA.clamp_min(1e-12)).tolist(),
        "q_cos": qc.tolist(),
        "q_sign_flip": [bool(c < 0) for c in qc.tolist()],
        "e_cos": float(torch.nn.functional.cosine_similarity(eA[None], eB[None])),
    }


# ---------------------------------------------------------------- 테스트

@torch.no_grad()
def _causality_once(model, ids, device, j) -> dict:
    x = torch.tensor([ids], device=device)
    x2 = x.clone()
    x2[0, j] = (x[0, j].item() + 1000) % model.config.vocab_size
    h1 = model(x, output_hidden_states=True).hidden_states
    h2 = model(x2, output_hidden_states=True).hidden_states
    per_layer = [(a[0, :j].float() - b[0, :j].float()).abs().max().item()
                 for a, b in zip(h1, h2)]
    return {"j": j,
            "max_abs_diff_before_j_per_layer": per_layer,
            "max_over_layers": max(per_layer),
            "causal": max(per_layer) == 0.0}
    
def test_causality(model_name, ids, device, prompt_len,
                   dtypes=("bfloat16", "float32")) -> dict:
    """1-a: 위치 j 의 토큰 하나만 바꾸고 H[:, :j] 가 전 layer 불변인지.

    마스킹된 위치는 softmax 후 정확히 0 이므로 결과는 0.0 이거나 명백히 큰
    값이지, 애매한 중간값이 나오지 않는 결정적 테스트다.

    두 축을 스윕한다:
      * dtype  — bf16 에서 0.0 이 나와도 정밀도가 미세 누출을 가렸을 수 있다.
                 fp32 에서도 0.0 이어야 구현이 확정된다.
      * 위치 j — 프롬프트 구간과 **생성 구간** 양쪽. step_spans 는 전부 생성
                 구간에 있으므로 거기서 재야 검증 대상과 일치한다.
    """
    n = len(ids)
    sites = {
        "prompt_mid": prompt_len // 2,
        "prompt_end": max(prompt_len - 2, 0),
        "gen_mid": prompt_len + (n - prompt_len) // 2,
        "gen_late": max(n - 5, prompt_len),
    }
    sites = {k: v for k, v in sites.items() if 0 < v < n}

    out = {}
    for dt in dtypes:
        m = AutoModel.from_pretrained(
            model_name, dtype=getattr(torch, dt), device_map=device
        ).eval()
        out[dt] = {k: _causality_once(m, ids, device, j) for k, j in sites.items()}
        del m
        torch.cuda.empty_cache()

    out["causal"] = all(r["causal"] for d in dtypes for r in out[d].values())
    out["max_over_all"] = max(r["max_over_layers"]
                              for d in dtypes for r in out[d].values())
    return out


@torch.no_grad()
def test_prefix_invariance(model, ids, device, m_frac=0.7) -> dict:
    """1-b: H_full[:M] vs H_pref[:M]. method A/B 구분의 전제."""
    M = int(len(ids) * m_frac)
    full = model(torch.tensor([ids], device=device)).last_hidden_state[0]
    pref = model(torch.tensor([ids[:M]], device=device)).last_hidden_state[0]
    return compare(full[:M], pref[:M])


@torch.no_grad()
def test_t1(model, ids, prompt_len, gen_hidden, device) -> dict:
    """1-d: 생성 시 hidden state vs teacher-forcing 재투입.

    ★ 이 결과가 저장 전략을 결정한다.
      통과 -> token_ids 만 저장하고 나중에 재투입 (layer/k 스윕 공짜)
      실패 -> 생성 시점 hidden state 를 전부 저장해야 함
    """
    # generate(output_hidden_states=True) 의 구조:
    #   hidden_states[step][layer] -> (batch, seq, d)
    #   step 0 은 prompt 전체, 이후는 각 1 토큰
    last = -1
    gen_rows = [gen_hidden[0][last][0]]                 # (prompt_len, d)
    for st in gen_hidden[1:]:
        gen_rows.append(st[last][0])                    # (1, d)
    G = torch.cat(gen_rows, dim=0)                      # (N, d)

    tf = model(torch.tensor([ids], device=device)).last_hidden_state[0]
    n = min(G.shape[0], tf.shape[0])
    out = {"n_compared": n, "prompt_len": prompt_len}
    out["all"] = compare(G[:n].to(tf.device), tf[:n])
    out["generated_only"] = compare(G[prompt_len:n].to(tf.device), tf[prompt_len:n])
    return out


@torch.no_grad()
def test_dtype(model_name, ids, device) -> dict:
    """1-e: bf16 vs fp32."""
    outs = {}
    for dt in ("bfloat16", "float32"):
        m = AutoModel.from_pretrained(model_name, dtype=getattr(torch, dt),
                                      device_map=device).eval()
        outs[dt] = m(torch.tensor([ids], device=device)).last_hidden_state[0].float().cpu()
        del m
        torch.cuda.empty_cache()
    return compare(outs["float32"], outs["bfloat16"])


def test_trace(H: torch.Tensor, prompt_len: int, step_spans) -> dict:
    """1-h: method A(누적) trace 단조성 / 토큰당 상수성, method B 비교.

    setting.md M2, M3 의 실증.
      tr(G_t) = Σ_i ||h_i||^2  -> 토큰 수에 대한 합
      method A 는 누적이므로 t 에 단조증가가 보장된다.
    """
    rows = []
    for t, (s, e) in enumerate(step_spans):
        EA = H[prompt_len:e]                 # 누적 prefix (Yu et al. 방식)
        EB = H[s:e]                          # 현재 step 만 (CTRLS 문자 그대로)
        NA, nB = EA.shape[0], EB.shape[0]
        trA = float((EA.float() ** 2).sum())
        trB = float((EB.float() ** 2).sum())
        rows.append({"t": t, "N_t": NA, "n_t": nB,
                     "trA": trA, "trA_per_tok": trA / max(NA, 1),
                     "trB": trB, "trB_per_tok": trB / max(nB, 1)})
    trA = [r["trA"] for r in rows]
    pa = np.array([r["trA_per_tok"] for r in rows])
    pb = np.array([r["trB_per_tok"] for r in rows])
    return {
        "rows": rows,
        "trA_monotone": all(x < y for x, y in zip(trA, trA[1:])),
        "trA_per_tok_cv": float(pa.std() / pa.mean()) if len(pa) else None,
        "trB_per_tok_cv": float(pb.std() / pb.mean()) if len(pb) else None,
        "corr_nt_t": float(np.corrcoef(np.arange(len(rows)),
                                       [r["n_t"] for r in rows])[0, 1])
        if len(rows) > 2 else None,
    }


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--hidden", default=None, help="generate --save-hidden 산출물")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--out", default="step1")
    args = ap.parse_args()

    rec = json.loads(open(args.steps).readline())
    ids = rec["token_ids"]
    plen = rec["prompt_len"]
    spans = rec["step_spans"]

    tok = AutoTokenizer.from_pretrained(args.model)

    res = {"id": rec["id"], "n_tokens": len(ids), "prompt_len": plen,
           "n_steps": len(spans), "layer": args.layer, "dtype": args.dtype}

    # 1-a 는 자체적으로 모델을 로드/해제한다. 공유 모델을 잡기 전에 돌려야
    # bf16 + fp32 + 공유 bf16 이 동시에 올라가는 것을 막는다.
    # 그리고 이게 게이트이므로 실패 시 여기서 멈추는 게 맞다.
    print("[1-a] causality ...", flush=True)
    res["causality"] = test_causality(args.model, ids, args.device, plen)
    print("      causal =", res["causality"]["causal"],
          " max_diff =", res["causality"]["max_over_all"])

    # 이제 공유 bf16 모델 로드
    model = AutoModel.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map=args.device
    ).eval()
    dev = next(model.parameters()).device

    print("[1-b] prefix invariance ...", flush=True)
    res["prefix_invariance"] = test_prefix_invariance(model, ids, dev)

    print("[1-c] span integrity (from generate) ...", flush=True)
    res["span_report"] = rec["span_report"]

    with torch.no_grad():
        H = model(torch.tensor([ids], device=dev),
                  output_hidden_states=True).hidden_states[args.layer][0].cpu()

    if args.hidden:
        print("[1-d] T1 generate vs teacher-forcing ...", flush=True)
        blob = torch.load(args.hidden, map_location=dev)
        res["T1"] = test_t1(model, ids, plen, blob["hidden"], dev)
    else:
        res["T1"] = None
        print("[1-d] skipped (no --hidden)")

    del model
    torch.cuda.empty_cache()

    print("[1-e] T6 bf16 vs fp32 ...", flush=True)
    res["T6"] = test_dtype(args.model, ids, args.device)

    print("[1-f/g] A1 spectrum per step ...", flush=True)
    res["A1"] = [a1_stats(H[s:e], args.k) for s, e in spans]
    res["A1_summary"] = {
        "mean_offdiag_cos": float(np.mean([a["mean_offdiag_cos"] for a in res["A1"]])),
        "eff_rank_participation": float(
            np.mean([a["eff_rank_participation"] for a in res["A1"]])),
        "lambda1_ratio": float(np.mean([a["lambda1_ratio"] for a in res["A1"]])),
    }

    print("[1-h] trace ...", flush=True)
    res["trace"] = test_trace(H.cpu(), plen, spans)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "step1_report.json").write_text(json.dumps(res, indent=2))

    # ---- 요약
    print("\n" + "=" * 60)
    print(f"causal                : {res['causality']['causal']}  "
          f"(max |Δ| = {res['causality']['max_over_all']:.2e})")
    print(f"prefix invariant h_cos: p50 {res['prefix_invariance']['h_cos']['p50']:.6f}  "
          f"min {res['prefix_invariance']['h_cos']['min']:.6f}")
    print(f"span ok               : {res['span_report']['ok']} "
          f"({res['span_report']['n_bad']}/{res['span_report']['n']} bad)")
    if res["T1"]:
        g = res["T1"]["generated_only"]
        print(f"T1 gen vs TF   h_cos  : p50 {g['h_cos']['p50']:.6f}  min {g['h_cos']['min']:.6f}")
        print(f"T1             e_cos  : {g['e_cos']:.6f}")
        print(f"T1             q_cos  : {[round(c,4) for c in g['q_cos'][:8]]}")
    t6 = res["T6"]
    print(f"T6 fp32 vs bf16 e_cos : {t6['e_cos']:.6f}")
    print(f"T6              q_cos : {[round(c,4) for c in t6['q_cos'][:8]]}")
    s = res["A1_summary"]
    print(f"A1 mean off-diag cos  : {s['mean_offdiag_cos']:.4f}")
    print(f"A1 effective rank     : {s['eff_rank_participation']:.2f}")
    print(f"A1 lambda1 / sum      : {s['lambda1_ratio']:.4f}")
    tr = res["trace"]
    print(f"trA monotone          : {tr['trA_monotone']}")
    print(f"trA/N_t  CV           : {tr['trA_per_tok_cv']:.4f}"
          if tr['trA_per_tok_cv'] is not None else "")
    print(f"corr(n_t, t)          : {tr['corr_nt_t']}")
    print("=" * 60)
    print(f"-> {out/'step1_report.json'}")


if __name__ == "__main__":
    main()