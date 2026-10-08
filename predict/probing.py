#!/usr/bin/env python
"""chunk_*.pt ? ?? ?? step-wise one-vs-rest ?? probe ?? (??? ???).

.pt ?? (extract.py ??):
    {"model": ..., "boundary_layout": "prompt|steps|terminal", "label_priority": ...,
     "episodes": {env_seed: {"E": bf16[n_tok, D],
                             "boundaries": [0, b_prompt, b_step1, ..., b_stepN, n_tok],
                             "output_sha1": str}}}

boundaries ?? (?? N+3):
    ?? = [????][step 1]...[step N][???]
    b[0]=0, b[1]=???? ?, b[1+k]=step k ?, b[-2]=step N ?(=??? ??), b[-1]=?? ?

??: <hidden_root>/<task>/<level>/<status>/chunk_*.pt
    ? group id = "<task>/<level>/<status>/<seed>"  (?? level ? ?? task ? ????
      task ?? ??? glob ? ??? ? ???? ???)

?? ??: ??? ??? ?? E[end-1-offset].
    step k ? ??? ?? = "Step k+1" ?? ?? ??. ??(Sun et al.)? ?? ???
    "Step k+1 activation" ?? ??? ? ???? ?? ???? step_k ? ???.
    ??? ??? ??? ??? "answer" ???.

    --with-prompt ? ?? ???? ??? ??? ??? "prompt" ???? ???.
    (??? "Step 1" ? ? ??. ?? ???? ??? ?? ?? ????? ??? ??)

?? ?? (--source). ??? ?? ??: ?? ???? ?, ?? ???? ??.
    hidden   [1] hidden_states ? ?? ??? ?? (? ??)
                 --pct 10 20 ? 100 ? ?? ???? q% ?? ?? E[s + pct_position(q, n)] ? q ?? ???
                 ??? ??? (?? E ??? ?? ???? hidden_states ????? ??? ??; 100 = ?? ??? ??).
                 probe ? ?? q ? ?? ?? ??? ????, ??? test ? q ?? ?? ?? ??
                 (summary.json ? by_pct, summary.png ? q ? AUC ??). --offset ? --pct ? ?? ? ??.
                 ?? E ? ?? spectral.py --with-hidden ???? ??? --source pct --input hidden ? ??.
    spectral [2] latent/spectral ???? ?? ??? e_t (e[t][-1], ???? kd ?? ??).
                 n_back ? 1 ??? --all ? ??? ???? ?? ??? ??. --with-prompt/--offset ? ????.
                 ? latent/spectral_states ?? (e[t] ? (kd,) ??, seg/pos ??) ? ??? ???.
    edges    [3-5] ?? ???? ?? ? ??? ?? e_i (inference/spectral.py --n-front 5 --n-back 5).
                 ??? ?? ???? ???? e_i ????? ????. --all ? ??? ??? --part both ? ??.
                 --part both  [3] "Step N" ?? ??? ? ? 5? + ??? 5?
                 --part front [4] ??? ? ? 5?
                 --part back  [5] ??? 5? (? ? ?? = [2] ? e_t)
                 ??? ??? ?? ? ?? ??("<START>" ?)? ???? ??.
                 ?? ?? + ? 5 + ? 5 ?? ?? ??? 3/4/5 ???? ?? (--keep-short ? ?).
                 --k / --sign-mode ? spectral ??? ???.
    pct      [6] ?? ??? ? --pct ? ??? ?? (inference/spectral.py --pct 10 20 ? 100) ??
                 ?? ??? Q% ???? ??? e ?? (--pct Q ? ??? ???, ???? ?? ??).
                 ??? spectral.pct_position ?? ????. 100 ? [2] ? e_t ? ??.
                 t < k ? ??(?? ??? ?? Q) ? ??? ?? ???; ??? stats ? t_lt_k ? ??.
                 --min-len L ? ?? ?? L ?? ??? ?? (?? ???? ?? Q ? ?? ??? ???? ?? ?? ?).
    --input hidden|spectral  (edges / pct) ???? ?? ??? ??. spectral(??) = ?? gram e (kd ??),
                 hidden = ?? ??? hidden state ? h (d ??; spectral.py --with-hidden ?? ??? _h ?? ???).
                 ?? pos ?? e ? h ? ??? ??? "??? vs gram" ??? ?? ?? ?? ??? ??.
    ??: acc / f1 / auc ? ?? margin = ? ??? ??? ?? (acc ? 1.0 ? ??? ??? ???? ??? ??).
    --max-step M ? step M ??? probe ??. --drop-above-max ? ?? ?? ? ? step ?
    ?? ????? ???.
    ??? ?? ???? ???? ??? ?? ????? ??? train/test ? ??? ???.
    --n-episodes N ? ?? ?? ? ???? N ?? ??? (success ??, ???? failure ? ??,
    --sample-seed ? ??). ?? ???? ??? <output>/episodes.json ? ??.

Usage:
    python probing.py --pt 'latent/hidden_states/plan/*/*/chunk_*.pt' --output out/plan_all
    python probing.py --source spectral \
        --pt 'latent/spectral/*/*/*/k8_scaled_sign-data_f0_b1/chunk_*.pt' --output out/spec_all
    python probing.py --source edges --part front \
        --pt 'latent/spectral/*/*/*/k8_scaled_sign-data_f5_b5/chunk_*.pt' --output out/edge_front
    python probing.py --pt 'latent/hidden_states/plan/BabyAI-GoTo-v0/*/chunk_*.pt' \
        --output out/plan_goto --offset 1 --cv
    python probing.py --source pct --pct 40 \
        --pt 'latent/spectral/*/*/*/k8_scaled_sign-data_p10-20-40-60-80-90-100/chunk_*.pt' --output out/pct_40
    python probing.py --pt 'latent/hidden_states_marker/decompose/BabyAI-GoToObj-v0/*/chunk_*.pt' \
        --pct 10 20 40 60 80 90 100 --output out/hidden_pct_gotoobj   # ?? E ?? ??, q ? ??
    python probing.py --source pct --pct 5 --input hidden --min-len 20 \
        --pt 'latent/spectral/decompose/BabyAI-GoToObj-v0/*/k8_scaled_sign-data_p1-5-10-20-50-80-90-95-99-100_h/chunk_*.pt' \
        --output out/pct_h_05                       # ?? ????? --input spectral ? ??? gram ?
    (?? % ? ? ?? ?? ????: script/pct_curve.py)
"""
import sys
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
from spectral import pct_position  # noqa: E402  (?? ?? ?? q% ? ?? ??)

PROMPT_LABEL = 0          # step_num 0 = ???? ?
ANSWER_LABEL = -1         # step_num -1 = ???(?? ??)


# ---------------------------------------------------------------- .pt ??

SOURCES = ("hidden", "spectral", "edges", "pct")


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


def load_pt(patterns, offset, with_prompt=False, pct=None):
    """[1] hidden_states: ?? ??? ?? E[end-1-offset] ??.
    pct ? ?? ???? q% ?? ??? q ?? ??? (offset ? 0 ??? ??). ??? q ? Ps ? ???."""
    Xs, Ns, Gs, Ps = [], [], [], []
    stats = Counter()
    if pct and offset:
        raise SystemExit("--pct ? --offset ? ?? ? ??")

    for p in _paths(patterns):
        d = torch.load(p, map_location="cpu", weights_only=False)

        for seed, ep in d["episodes"].items():
            gid = _gid(p, "hidden", seed)
            E = ep["E"]                                   # bf16 torch ??? (GPU ??? ??? ???)
            b = [int(x) for x in ep["boundaries"]]
            n_tok = E.shape[0]

            # ??: [0, prompt, step1..stepN, terminal ?] ? ?? ?? 4 (step 1?)
            if len(b) < 4 or b[0] != 0 or b[-1] != n_tok:
                stats["bad_boundaries"] += 1
                continue
            if any(x >= y for x, y in zip(b, b[1:])):
                stats["nonmonotonic_boundaries"] += 1
                continue
            stats["episodes"] += 1

            def take(start, end, label):
                if pct:                                       # q% ?? ???, q ?? ? ?
                    for q in pct:
                        i = start + pct_position(q, end - start)
                        Xs.append(E[i].clone()); Ns.append(label); Gs.append(gid); Ps.append(q)
                    return
                i = end - 1 - offset
                if i < start:
                    stats["segment_too_short"] += 1
                    return
                Xs.append(E[i].clone()); Ns.append(label); Gs.append(gid); Ps.append(100)  # view ? E ??? ???? 300GB ?? ?

            if with_prompt:                                   # ????? ?? ??? ?? ??
                i = b[1] - 1 - offset
                Xs.append(E[i].clone()); Ns.append(PROMPT_LABEL); Gs.append(gid); Ps.append(100)
            for k, (s, e) in enumerate(zip(b[1:-2], b[2:-1]), start=1):   # step 1..N
                take(s, e, k)
            take(b[-2], b[-1], ANSWER_LABEL)                              # ???

    return _pack(Xs, Ns, Gs, stats, Ps)


def _seg_label(t: int, n_seg: int) -> int:
    """gen_view ?? ?? t (0..N-1 = step 1..N, N = ???) ? step_num."""
    return ANSWER_LABEL if t == n_seg - 1 else t + 1


def _config_ok(d, p, k, sign_mode, seen, stats):
    """??? k / sign_mode ? ??? ???. ?? ?? ??? ??? ??? ?? ???."""
    cfg = (d.get("k"), d.get("sign_mode"), d.get("scale"))
    if (k is not None and cfg[0] != k) or (sign_mode is not None and cfg[1] != sign_mode):
        stats["file_config_skipped"] += 1
        return False
    seen.add(cfg)
    if len(seen) > 1:
        raise SystemExit(f"?? ?? spectral ??? ??? {sorted(seen, key=str)} ? "
                         f"--k / --sign-mode ? ??? ???? glob ? ?? ? ({p})")
    return True


def load_spectral(patterns, k=None, sign_mode=None):
    """[2] spectral: ?? ??? ??? e_t = e[t][-1] (???? ??, ??? ??? ??? ???)."""
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
                if row.dim() == 1:                                    # ? spectral_states: ???? e_t ??
                    Xs.append(row.clone()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
                    continue
                ps = ep["pos"][t]
                if not ps or ps[-1] != seg[t + 1] - seg[t] - 1:      # ??? ??? ??? ??? e_t
                    stats["no_last_token"] += 1
                    continue
                Xs.append(row[-1].clone()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
    return _pack(Xs, Ns, Gs, stats)


def _vec_key(d, p, inp):
    """????? ?? ?. hidden ?? --with-hidden ?? ??? ????? ??."""
    if inp == "hidden":
        if not d.get("hidden"):
            raise SystemExit(f"{p}: hidden ?? ??? ?? ?? (spectral.py --with-hidden ?? ?? _h ?? ?? ??)")
        return "h"
    return "e"


def load_edges(patterns, part, k=None, sign_mode=None, keep_short=False, inp="spectral"):
    """[3-5] edges: ?? ? ??? ?? e_i ? ?(?? ?? ??)/? ????.

    part: both (3) | front (4) | back (5). e_i ????? ???? ??? ? ??.
    ???? ?? pos ? e ? ?? ???? front/back ? ??? n_front/n_back ? marker ? ???:
    front = marker ? p < marker+n_front, back = p ? n-n_back. --all ???? part both ? ??.

    ?? ?? ??: ?? ?? < marker + n_front + n_back ?? (?/?? ???? ???)
    ? ??? ??? ?? ? ????? ????? ? ??. part ? ???? ?? ????
    3/4/5 ? ?? ?? ?? ??? ????. keep_short=True ? ?? (all ???? ?? ??).
    """
    Xs, Ns, Gs = [], [], []
    stats, seen = Counter(), set()
    for p in _paths(patterns):
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not _config_ok(d, p, k, sign_mode, seen, stats):
            continue
        n_front, n_back, is_all = d["n_front"], d["n_back"], d.get("all", False)
        if is_all and part != "both":
            raise SystemExit(f"--all ? ??? ??? --part both ? ?? ({p})")
        key = _vec_key(d, p, inp)
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
                for pos, row in zip(ep["pos"][t], ep[key][t]):
                    if part == "front" and not (m <= pos < m + n_front):
                        continue
                    if part == "back" and pos < n - n_back:
                        continue
                    Xs.append(row.clone()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
    return _pack(Xs, Ns, Gs, stats)


def load_pct(patterns, pct, k=None, sign_mode=None, inp="spectral", min_len=0):
    """[6] pct: ?? ??? pct% ???? ??? e (?? ?? ??? hidden ? h; inp) ? ???? ??.
    --pct ? ??? ??? ???.

    ???? pos ? e/h ? ?? ???? pct_position(pct, n) ?? ??? ??? ? ?? ???.
    ?? ???? ?? % ? ?? ??? ???? ??? ??? ??. ??? ??? ??? ?
    t < k ? ??(?? rank ??) ??? t_lt_k ? ??. min_len > 0 ?? ??? ?? ??? ??
    (short_segment ? ??) ? ??·?? ?? ???? ? ??."""
    Xs, Ns, Gs = [], [], []
    stats, seen = Counter(), set()
    for p in _paths(patterns):
        d = torch.load(p, map_location="cpu", weights_only=False)
        if not _config_ok(d, p, k, sign_mode, seen, stats):
            continue
        if pct not in d.get("pct", []):
            raise SystemExit(f"{p}: {pct}% ??? ??? ?? ?? (??? pct={d.get('pct')})")
        key = _vec_key(d, p, inp)
        k_eig = d["k"]
        for seed, ep in d["episodes"].items():
            gid = _gid(p, "edges", seed)                      # spectral/edges ? ?? ?? ??
            seg, n_seg = ep["seg"], len(ep["e"])
            stats["episodes"] += 1
            for t in range(n_seg):
                n = seg[t + 1] - seg[t]
                if n < min_len:
                    stats["short_segment"] += 1
                    continue
                want = pct_position(pct, n)
                ps = ep["pos"][t]
                if want not in ps:
                    stats["pos_missing"] += 1
                    continue
                if want + 1 < k_eig:
                    stats["t_lt_k"] += 1
                Xs.append(ep[key][t][ps.index(want)].clone()); Ns.append(_seg_label(t, n_seg)); Gs.append(gid)
    return _pack(Xs, Ns, Gs, stats, pct_value=pct)


def _pack(Xs, Ns, Gs, stats, Ps=None, pct_value=100):
    """? ?? ? data. X ? torch ???, ?? dtype ??? ?? (bf16 ????? bf16 ? fp32 ? ???
    edges ? ?? GB ? ? ?? ??). CPU(sklearn) ??? main ?? ? ? float32 numpy ? ???.
    pct: ??? "??? ? % ????" (Ps ? ??? ?? pct_value). ?? % ? ?? ?? ? q ? ??? ??."""
    if not Xs:
        raise SystemExit("?? ?? ? ?? ??/?? ??")
    X = torch.stack(Xs)
    Xs.clear()
    P = np.array(Ps if Ps is not None else [pct_value] * len(Ns), np.int32)
    data = dict(X=X, step_num=np.array(Ns, np.int32), group=np.array(Gs, dtype=object), pct=P)
    print(f"loaded: {dict(stats)}")
    print(f"  X={tuple(X.shape)} {str(X.dtype).removeprefix('torch.')}  "
          f"??? ??={dict(sorted(Counter(Ns).items()))}  groups={len(set(Gs))}"
          + (f"  pct? ??={dict(sorted(Counter(P.tolist()).items()))}" if len(set(P.tolist())) > 1 else ""))
    return data


def _take(data, idx):
    """? ???? data ? ??? (X ? torch, ???? numpy)."""
    idx = np.asarray(idx)
    return {k: (v[torch.as_tensor(idx)] if torch.is_tensor(v) else v[idx]) for k, v in data.items()}


# ---------------------------------------------------------------- ???? / ??

def target_name(label: int) -> str:
    if label == PROMPT_LABEL:
        return "prompt"
    if label == ANSWER_LABEL:
        return "answer"
    return f"step_{label}"


def subsample_episodes(data, n, seed):
    """????(group) ??? n ?? ???. success ? ?? ??? ???? failure ?.

    group id ? "<task>/<level>/<status>/<seed>" ? status ? ??? ???. success ? n ??
    ??? ? ??? n ?? ???? ???. ??: (?? data, ?? group ?? ??)."""
    groups = np.unique(data["group"])
    status = np.array([g.split("/")[2] for g in groups])
    rng = np.random.default_rng(seed)
    succ = rng.permutation(groups[status == "success"])
    fail = rng.permutation(groups[status != "success"])
    keep = np.concatenate([succ[:n], fail[:max(0, n - len(succ))]])
    mask = np.isin(data["group"], keep)
    print(f"--n-episodes {n}: success {min(n, len(succ))}/{len(succ)} + failure "
          f"{max(0, n - len(succ))}/{len(fail)} ? {len(keep)} episodes, X={int(mask.sum())} rows")
    if not mask.all():
        data = _take(data, np.flatnonzero(mask))
    return data, sorted(keep.tolist())


def make_binary(data, label):
    """one-vs-rest ??: ?? ??? 1, ??? ?? ?? 0. X ? ???? ??? (edges ? ?? GB)."""
    y = (data["step_num"] == label).astype(int)
    return data["X"], y, data["group"]


def split(n, y, g, test_size, seed, group):
    """? ??? (train, test). group=True ? ???? ??."""
    dummy = np.empty((n, 0))
    if group:
        return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(dummy, y, g))
    return train_test_split(np.arange(n), test_size=test_size, random_state=seed, stratify=y)


class GpuData:
    """X ??? GPU ? ? ?? ?? ?? (?? dtype ??? ? bf16 ?? fp32 ? ??), fit / predict ?
    ? ???? ??? ?? fp32 ? ?? ??. 70? x 40960 bf16 ? 57GB ? GPU ? ????."""

    def __init__(self, X: torch.Tensor, device, chunk: int = 32768):
        self.device = torch.device(device)
        self.X = X.to(self.device)
        self.chunk = chunk

    def batches(self, idx):
        """idx ???? (m x D) fp32 ??? ??."""
        idx_t = torch.as_tensor(np.asarray(idx), device=self.device)
        for i in range(0, len(idx_t), self.chunk):
            yield self.X[idx_t[i:i + self.chunk]].float()


class TorchLogReg:
    """GPU ???? ??. sklearn LogisticRegression(C, L2, class_weight="balanced") ? ?? ????

        C · ?_i w_i · logloss_i(?)  +  ½??_w?²      (w_i = n / (2·n_class(i)), ??? ?? ??)

    ? torch L-BFGS(strong Wolfe) ? ??. ??? ???? GpuData ??? ?? ????? ?????
    autograd ????, X ? fp32 ???? ??? ???. scale=True ? StandardScaler ? ?? ???
    (?? ? ?? ??/????, 2-pass) ? ????. ?? ? ????? CPU ? ?? pickle ??."""

    def __init__(self, C=1.0, scale=True, max_iter=500, tol=1e-5):
        self.C, self.scale, self.max_iter, self.tol = C, scale, max_iter, tol

    def fit(self, gd: GpuData, idx, y):
        dev, d = gd.device, gd.X.shape[1]
        n, n_pos = len(idx), float(np.sum(y))
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=dev)
        w_t = torch.where(y_t > 0.5, torch.tensor(n / (2 * n_pos), device=dev),
                          torch.tensor(n / (2 * (n - n_pos)), device=dev))        # balanced

        if self.scale:                                                           # 2-pass ??/????
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
    """Xnp: CPU ??? float32 numpy (GPU ??? None). gd: GpuData (CPU ??? None)."""
    if gd is not None:
        if cv:
            raise SystemExit("--cv ? --device cpu ??? ??")
        return TorchLogReg(C=1.0, scale=scale).fit(gd, idx, y[idx])
    if cv:
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=5, scoring="roc_auc",
                                  class_weight="balanced", max_iter=2000,
                                  random_state=seed, n_jobs=-1)
    else:
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    return (make_pipeline(StandardScaler(), lr) if scale else make_pipeline(lr)).fit(Xnp[idx], y[idx])


def evaluate(clf, Xnp, gd, idx, y):
    """acc / f1 / auc + margin. margin = ? ???? ? ??? ?? (y=1 ?? p, y=0 ?? 1-p):
    0.5 = ??, 1.0 = ?? ??. acc ? ???? ??? ?? ???? ??? ???."""
    p = clf.predict_proba(gd, idx)[:, 1] if gd is not None else clf.predict_proba(Xnp[idx])[:, 1]
    yhat = (p > 0.5).astype(int)
    return dict(acc=accuracy_score(y, yhat), f1=f1_score(y, yhat, zero_division=0),
                auc=roc_auc_score(y, p), margin=float(np.mean(np.where(y == 1, p, 1 - p))))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", nargs="+", required=True, help="chunk_*.pt glob (?? ? ??)")
    ap.add_argument("--source", choices=SOURCES, default="hidden",
                    help="hidden: ?? ??? ?? hidden (1) | spectral: ?? ?? ?? e_t (2) | "
                         "edges: ?? ?/? ??? ?? e_i (3-5, --part) | pct: ?? Q%% ?? ?? e (6, --pct)")
    ap.add_argument("--pct", type=int, nargs="+", default=None, metavar="Q",
                    help="hidden: ?? ??? Q%% ?? ??? (?? ?; ??? ???? ??? Q ?) | "
                         "pct ??: ???? --pct ?? ? ?? (100 = ?? ?)")
    ap.add_argument("--part", choices=("both", "front", "back"), default="both",
                    help="--source edges ? ?: both=?5+?5 (3), front=?? ? ?5 (4), back=?5 (5)")
    ap.add_argument("--k", type=int, default=None,
                    help="spectral/edges: ? k ? ??? ?? (??: glob ? ?? ???)")
    ap.add_argument("--sign-mode", default=None, choices=("none", "first", "max", "data"),
                    help="spectral/edges: ? ?? ??? ??? ??")
    ap.add_argument("--keep-short", action="store_true",
                    help="edges: marker+?+? ?? ?? ??? ??? (??? ??)")
    ap.add_argument("--input", choices=("spectral", "hidden"), default="spectral",
                    help="edges/pct ????? ?? gram e(spectral) ? ??, ?? ??? hidden ? h ? ?? "
                         "(h ? spectral.py --with-hidden ?? ??? _h ???? ??)")
    ap.add_argument("--min-len", type=int, default=0,
                    help="pct: ? ??(??) ?? ??? ???? ?? (?? %% ? ? ??? ???? ?? ?? ??)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--offset", type=int, default=0,
                    help="?? ??? ???? ? ? ?? ??? ?? (ablation)")
    ap.add_argument("--with-prompt", action="store_true",
                    help="???? ?? ??? ??? ???? ??")
    ap.add_argument("--max-step", type=int, default=None,
                    help="? ??? ?? step ? probe ???? ?? (??: negative ?? ??)")
    ap.add_argument("--drop-above-max", action="store_true",
                    help="--max-step ? ?? step ??? ????? ?? ?? (negative ?? ? ?)")
    ap.add_argument("--n-episodes", type=int, default=None,
                    help="?? ? ???? ? ??? ?? (success ??, ???? failure ? ??)")
    ap.add_argument("--sample-seed", type=int, default=0, help="--n-episodes ?? ??")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-group-split", action="store_true",
                    help="???? ?? ??? ?? (?? ????? train/test ? ???)")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--device", default="cpu",
                    help="cpu: sklearn LogisticRegression | cuda[:i]: ?? ????? torch L-BFGS ? (TorchLogReg)")
    a = ap.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "classifiers").mkdir(exist_ok=True)

    if a.source == "hidden":
        pct = sorted(set(a.pct)) if a.pct else None
        data = load_pt(a.pt, a.offset, a.with_prompt, pct)
    elif a.source == "spectral":
        data = load_spectral(a.pt, a.k, a.sign_mode)
    elif a.source == "pct":
        if a.pct is None or len(a.pct) != 1:
            raise SystemExit("--source pct ? --pct Q ??? ????")
        data = load_pct(a.pt, a.pct[0], a.k, a.sign_mode, a.input, a.min_len)
    else:
        data = load_edges(a.pt, a.part, a.k, a.sign_mode, a.keep_short, a.input)
    if a.n_episodes is not None:
        data, kept = subsample_episodes(data, a.n_episodes, a.sample_seed)
        with open(a.output / "episodes.json", "w") as f:
            json.dump({"n_episodes": a.n_episodes, "sample_seed": a.sample_seed, "groups": kept}, f)
    if a.max_step is not None and a.drop_above_max:
        keep = data["step_num"] <= a.max_step          # answer(-1)·prompt(0) ? ???
        print(f"--drop-above-max: step>{a.max_step} ?? {int((~keep).sum())}? ??")
        data = _take(data, np.flatnonzero(keep))
    group, scale = not a.no_group_split, not a.no_scale
    if group and len(np.unique(data["group"])) < 2:
        print("[warn] ??? 1?? ? ? ?? split ?? ??")
        group = False

    labels = sorted(np.unique(data["step_num"]).tolist())
    if a.max_step is not None:
        labels = [l for l in labels if l <= a.max_step]
    # ?? ?? ??: prompt, step 1..N, answer
    labels = ([PROMPT_LABEL] if PROMPT_LABEL in labels else []) \
             + [l for l in labels if l > 0] \
             + ([ANSWER_LABEL] if ANSWER_LABEL in labels else [])

    pcts = sorted(set(data["pct"].tolist()))                 # ???? ?? % ?? (????? q ? ?? ??)
    if len(pcts) == 1:
        pcts = []
    print(f"targets={[target_name(l) for l in labels]} "
          f"source={a.source} input={a.input} part={a.part} pct={a.pct} eval_pcts={pcts or '-'} "
          f"min_len={a.min_len} group={group} scale={scale} cv={a.cv} "
          f"offset={a.offset} device={a.device}")

    if a.device == "cpu":
        Xnp, gd = data["X"].float().numpy(), None            # float32 ????? ?? ??
    else:
        Xnp, gd = None, GpuData(data["X"], a.device)
        print(f"X on {a.device}: {tuple(gd.X.shape)} {str(gd.X.dtype).removeprefix('torch.')} "
              f"({gd.X.numel() * gd.X.element_size() / 2**30:.1f} GiB)")

    rows = []
    for label in labels:
        t = target_name(label)
        X, y, g = make_binary(data, label)
        if y.sum() < 2 or (1 - y).sum() < 2:
            print(f"[skip] {t}: ?? ??")
            continue
        for s in a.seeds:
            tr, te = split(len(y), y, g, a.test_size, s, group)
            if len(np.unique(y[te])) < 2:
                print(f"[skip] {t} seed={s}: test ? ? ????")
                continue
            clf = fit(Xnp, gd, y, tr, s, a.cv, scale)
            m_tr, m_te = evaluate(clf, Xnp, gd, tr, y[tr]), evaluate(clf, Xnp, gd, te, y[te])
            lr = clf if isinstance(clf, TorchLogReg) else clf[-1]
            C = float(np.atleast_1d(getattr(lr, "C_", lr.C))[0])
            rows.append(dict(target=t, seed=s, eval_pct="all", n_train=int(len(tr)), n_test=int(len(te)),
                             n_pos_test=int(y[te].sum()), C=C,
                             **{f"train_{k}": float(v) for k, v in m_tr.items()},
                             **{f"test_{k}": float(v) for k, v in m_te.items()}))
            per_pct = ""
            for q in pcts:                                   # ?? probe ? test ? q% ???? ??
                te_q = te[data["pct"][te] == q]
                if len(te_q) == 0 or len(np.unique(y[te_q])) < 2:
                    continue
                m_q = evaluate(clf, Xnp, gd, te_q, y[te_q])
                rows.append(dict(target=t, seed=s, eval_pct=int(q), n_train=int(len(tr)), n_test=int(len(te_q)),
                                 n_pos_test=int(y[te_q].sum()), C=C,
                                 **{f"test_{k}": float(v) for k, v in m_q.items()}))
                per_pct += f" {q}%={m_q['auc']:.3f}"
            with open(a.output / "classifiers" / f"{t}_seed{s}.pkl", "wb") as f:
                pickle.dump(clf, f)
            print(f"{t:8s} seed={s:5d}  AUC={m_te['auc']:.3f} "
                  f"Acc={m_te['acc']:.3f} F1={m_te['f1']:.3f} margin={m_te['margin']:.3f}"
                  + (f"  | AUC by pct:{per_pct}" if per_pct else ""))

    if not rows:
        raise SystemExit("??? probe ? ??")

    with open(a.output / "results_all.json", "w") as f:
        json.dump(rows, f, indent=2)

    def agg(r):
        return {k: dict(mean=float(np.mean([x[k] for x in r])), std=float(np.std([x[k] for x in r])))
                for k in ("test_auc", "test_acc", "test_f1", "test_margin")}

    summary = {}
    for label in labels:
        t = target_name(label)
        r = [x for x in rows if x["target"] == t and x["eval_pct"] == "all"]
        if not r:
            continue
        summary[t] = agg(r)
        summary[t]["n_seeds"] = len(r)
        if pcts:
            summary[t]["by_pct"] = {str(q): agg(rq) for q in pcts
                                    if (rq := [x for x in rows if x["target"] == t and x["eval_pct"] == q])}
    with open(a.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 62)
    print(f"{'target':8s} {'AUC':>15s} {'Acc':>15s} {'F1':>15s} {'margin':>15s}")
    for t, s in summary.items():
        fmt = lambda k: f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}"
        print(f"{t:8s} {fmt('test_auc'):>15s} {fmt('test_acc'):>15s} {fmt('test_f1'):>15s} "
              f"{fmt('test_margin'):>15s}")
    if pcts:
        for met, lab in (("test_auc", "AUC"), ("test_margin", "margin")):
            print(f"\n{lab} by eval pct (??? ?? %, ??? ? % ??)")
            print(f"{'target':8s}" + "".join(f"{q:>9d}%" for q in pcts))
            for t, s in summary.items():
                print(f"{t:8s}" + "".join(f"{s['by_pct'][str(q)][met]['mean']:10.3f}"
                                           if str(q) in s.get("by_pct", {}) else f"{'-':>10s}" for q in pcts))

    ts = list(summary)
    fig, ax = plt.subplots(figsize=(9, 5))
    if pcts:                                                  # q ? AUC ??, ???? ? ??
        for t in ts:
            bp = summary[t].get("by_pct", {})
            qs = [q for q in pcts if str(q) in bp]
            ax.errorbar(qs, [bp[str(q)]["test_auc"]["mean"] for q in qs],
                        yerr=[bp[str(q)]["test_auc"]["std"] for q in qs], marker="o", capsize=3, label=t)
        ax.set_xlabel("eval position in segment (%)")
        ax.set_ylabel("test AUC")
        ax.set_xticks(pcts)
    else:
        x, w = np.arange(len(ts)), 0.25
        for i, (k, lab) in enumerate([("test_auc", "AUC"), ("test_acc", "Acc"), ("test_f1", "F1")]):
            ax.bar(x + (i - 1) * w, [summary[t][k]["mean"] for t in ts], w,
                   yerr=[summary[t][k]["std"] for t in ts], capsize=3, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(ts)
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.legend()
    vec = "hidden h" if a.input == "hidden" else "spectral e"
    src = {"hidden": f"hidden offset={a.offset}" + (f" pct={a.pct}" if a.pct else ""), "spectral": "spectral e_t",
           "edges": f"{vec} edges ({a.part})", "pct": f"{vec} @ {a.pct}%"}[a.source]
    ax.set_title(f"Last-layer probes: {src} (seeds={len(a.seeds)}, group={group})")
    fig.tight_layout()
    fig.savefig(a.output / "summary.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved ? {a.output}")


if __name__ == "__main__":
    main()