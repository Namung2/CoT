"""
P1~P4 one-shot 데이터 생성.

setting.md §2, §4, §5 준수.

핵심 규약
  * batch = 1 고정. one-shot 이므로 배치가 불필요하고, 그러면 left-padding
    RoPE 시프트(M7)와 샘플링 RNG 의 배치 의존(M8)이 발생 자체를 안 한다.
  * 샘플링 사용 시 torch.manual_seed(env_seed) 를 매 generate() 직전.
    -> (level, seed) 하나가 환경 배치·미션·CoT 전부를 결정.
    이는 레퍼런스 없는 **우리 규약**이다 (BabyAI utils.seed() 는 실행 단위 1회,
    CTRLS/LLM-BabyBench 는 시드 규약 없음).
  * 성공 판정은 env reward > 0 이진. seed 필터 없음
    (BabyAI make_agent_demos.py: 상한만 있고 하한 없음, 기본 무필터).
  * step_spans 와 action_spans 사이 개수 등식은 걸지 않는다.

사용
  python -m datagen.generate --cell P1 --n 1  --save-hidden      # STEP 0
  python -m datagen.generate --cell P1 --n 30                    # STEP 2
  python -m datagen.generate --cell P1 --n 150                   # STEP 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .babyai_env import (ACTIONS, bot_reference_len, make_episode,
                         replay_and_label)
from .prompt import (ASSISTANT_PREFIX, TERMINAL_TEXT, build_messages)
from .spans import char_offsets, char_to_token_span, segment, verify_spans

# ---------------------------------------------------------------- 셀 정의

CELLS = {
    # cell : (obs_mode, n_exemplars)
    "P1": ("partial", 0),   # 본 실험
    "P2": ("full", 0),      # C1 : 좌표 대조. P1 과 동일 seed 집합
    "P4": ("partial", 3),   # C2 : few-shot exemplar
}

DEFAULT_LEVEL = "BabyAI-PutNextLocalS6N4-v0"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

_ACTIONS_RE = re.compile(r"<actions>(.*?)</actions>", re.S | re.I)


# ---------------------------------------------------------------- 파싱


def parse_output(text: str) -> dict:
    """생성 텍스트를 (추론부, 액션 리스트) 로 나눈다.

    반환의 reasoning_end 는 **문자 오프셋**이며, 분절은 이 앞부분에만 적용한다.
    """
    m = _ACTIONS_RE.search(text)
    if not m:
        return {"reasoning_end": len(text), "actions": [],
                "action_char_span": None, "parse_ok": False, "n_raw": 0}

    raw = [w.strip().lower() for w in re.split(r"[,\n]+", m.group(1)) if w.strip()]
    acts = [w for w in raw if w in ACTIONS]
    return {
        "reasoning_end": m.start(),
        "actions": acts,
        "action_char_span": [m.start(1), m.end(1)],
        "parse_ok": len(acts) > 0,
        "n_raw": len(raw),
        "n_dropped": len(raw) - len(acts),
    }


def leakage(reasoning: str) -> dict:
    """추론부 안 액션명 누출 + 좌표 언급. 프롬프트 지시만으로는 막히지 않는다."""
    low = reasoning.lower()
    hits = [a for a in ACTIONS if re.search(rf"\b{a}\b", low)]
    syn = [w for w in ("turn", "move", "step", "grab", "take", "pick up",
                       "put down", "place", "toggle") if w in low]
    coords = re.findall(r"\(\s*\d+\s*,\s*\d+\s*\)", reasoning)
    return {
        "action_words": hits,
        "action_synonyms": syn,
        "has_action_leak": bool(hits),
        "n_coord_mentions": len(coords),
    }


# ---------------------------------------------------------------- 생성


class Generator:
    def __init__(self, model_name: str, dtype: str = "bfloat16",
                 device: str = "cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map=device
        ).eval()
        self.model_name = model_name
        self.dtype = dtype
        self.device = next(self.model.parameters()).device

    def build_prompt_ids(self, messages) -> list[int]:
        """chat template + assistant prefix. 추출 시 이 문자열을 그대로 재현해야 한다."""
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + ASSISTANT_PREFIX + "\n\n"
        return self.tok(text, add_special_tokens=False).input_ids, text

    @torch.no_grad()
    def generate(self, prompt_ids, *, seed, decoding, temperature, top_k,
                 max_new_tokens, save_hidden=False):
        if decoding == "sample":
            torch.manual_seed(seed)          # batch=1 이므로 이것으로 충분
        ids = torch.tensor([prompt_ids], device=self.device)
        kw = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        if decoding == "greedy":
            kw.update(do_sample=False)
        else:
            kw.update(do_sample=True, temperature=temperature, top_k=top_k)
        if save_hidden:
            kw.update(output_hidden_states=True, return_dict_in_generate=True)
            out = self.model.generate(ids, **kw)
            return out.sequences[0].tolist(), out.hidden_states
        out = self.model.generate(ids, **kw)
        return out[0].tolist(), None


# ---------------------------------------------------------------- 저장


class Sink:
    def __init__(self, root: Path, cell: str):
        self.dir = root / cell
        self.dir.mkdir(parents=True, exist_ok=True)
        self.steps = self.dir / "steps.jsonl"
        self.labels = self.dir / "labels.jsonl"
        self.rejects = self.dir / "rejects.jsonl"

    def seen(self) -> set[str]:
        if not self.steps.exists():
            return set()
        with open(self.steps) as f:
            return {json.loads(l)["id"] for l in f if l.strip()}

    def write(self, rec, lab):
        with open(self.steps, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(self.labels, "a") as f:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")

    def reject(self, rec):
        """성공하지 않은 에피소드. 파일은 남기되 cases 에는 넣지 않는다.
        성공 필터가 다양성을 줄이는지 감사하려면 이 통계가 필요하다."""
        with open(self.rejects, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def manifest(self, m):
        (self.dir / "manifest.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2)
        )


# ---------------------------------------------------------------- 메인


def run_one(gen: Generator, level, seed, obs_mode, args, exemplars=None,
            save_hidden=False):
    ep = make_episode(level, seed, obs_mode)
    msgs = build_messages(ep.mission, ep.obs_text, obs_mode, exemplars)
    prompt_ids, prompt_text = gen.build_prompt_ids(msgs)

    token_ids, hidden = gen.generate(
        prompt_ids, seed=seed, decoding=args.decoding,
        temperature=args.temperature, top_k=args.top_k,
        max_new_tokens=args.max_new_tokens, save_hidden=save_hidden,
    )
    gen_ids = token_ids[len(prompt_ids):]
    gen_text = gen.tok.decode(gen_ids, skip_special_tokens=False)

    p = parse_output(gen_text)
    reasoning = gen_text[: p["reasoning_end"]]

    # --- 분절 (추론부에만) + char -> token
    char_spans, seg_mode = segment(reasoning, args.segment)
    step_texts = [reasoning[a:b] for a, b in char_spans]

    cum = char_offsets(gen.tok, gen_ids)
    off = len(prompt_ids)
    step_spans = [[off + s, off + e]
                  for s, e in (char_to_token_span(cum, a, b) for a, b in char_spans)]
    action_spans = []
    if p["action_char_span"] is not None:
        s, e = char_to_token_span(cum, *p["action_char_span"])
        action_spans = [[off + s, off + e]]

    span_report = verify_spans(
        gen.tok, token_ids,
        [(a, b) for a, b in step_spans], step_texts
    )

    # --- 실행 & 라벨
    rep = replay_and_label(level, seed, p["actions"]) if p["actions"] else {
        "labels": {k: [] for k in ("action", "pos", "dir", "carrying", "front_obj")},
        "n_exec": 0, "reward": 0.0, "success": False, "outcome": "unparsed",
        "final_pos": None, "final_dir": None,
    }

    lk = leakage(reasoning)

    rec = {
        "id": f"{level}|{seed}|{obs_mode}",
        "level": level, "seed": seed,
        "mode": "oneshot", "obs_mode": obs_mode,
        "n_exemplars": len(exemplars or []),
        "mission": ep.mission,
        "prompt": {"text": prompt_text, "n_tokens": len(prompt_ids)},
        "raw_output": gen_text,
        "token_ids": token_ids,
        "prompt_len": len(prompt_ids),
        "step_spans": step_spans,
        "action_spans": action_spans,
        "steps": step_texts,
        "segment_mode": seg_mode,
        "terminal": TERMINAL_TEXT if rep["success"] else None,
        "answer": {
            "action_seq": p["actions"],
            "success": rep["success"],
            "reward": rep["reward"],
            "final_pos": rep["final_pos"],
            "final_dir": rep["final_dir"],
        },
        "n_planned": len(p["actions"]),
        "n_dropped": p.get("n_dropped", 0),
        "n_exec": rep["n_exec"],
        "parse_ok": p["parse_ok"],
        "truncated": len(gen_ids) >= args.max_new_tokens,
        "outcome": rep["outcome"],
        "span_report": span_report,
        "leakage": lk,
    }
    lab = {
        "id": rec["id"],
        "n_exec": rep["n_exec"],
        "outcome": rep["outcome"],
        **rep["labels"],
        **ep.covariates,
        "bot_reference_len": bot_reference_len(level, seed),
        "terminal": {"pos": rep["final_pos"], "dir": rep["final_dir"]},
    }
    return rec, lab, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="P1", choices=list(CELLS))
    ap.add_argument("--level", default=DEFAULT_LEVEL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=1, help="목표 성공 에피소드 수")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--max-seeds", type=int, default=0,
                    help="0 이면 n*20 까지 시도")
    ap.add_argument("--decoding", default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=0.7)  # CTRLS §6.2
    ap.add_argument("--top-k", type=int, default=50)           # CTRLS 미명시 -> 우리 결정
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--segment", default="nl", choices=["nl", "marker", "equal"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data")
    ap.add_argument("--save-hidden", action="store_true",
                    help="STEP 0 전용. 생성 시 hidden state 를 .pt 로 저장")
    ap.add_argument("--keep-all", action="store_true",
                    help="실패도 steps.jsonl 에 저장 (진단용)")
    args = ap.parse_args()

    obs_mode, n_ex = CELLS[args.cell]
    root = Path(args.out)
    sink = Sink(root, args.cell)
    done = sink.seen()

    gen = Generator(args.model, args.dtype, args.device)
    sink.manifest({
        "cell": args.cell, "level": args.level, "obs_mode": obs_mode,
        "model": args.model, "dtype": args.dtype,
        "chat_template_applied": True,
        "assistant_prefix": ASSISTANT_PREFIX,
        "decoding": {
            "mode": args.decoding, "temperature": args.temperature,
            "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
            "seed_policy": "torch.manual_seed(env_seed) per generate, batch=1",
        },
        "segment": args.segment,
        "pad_token_id": gen.tok.pad_token_id,
        "tokenizer": args.model,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "minigrid_version": __import__("minigrid").__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    max_seeds = args.max_seeds or args.n * 20
    ok = 0
    t0 = time.time()
    for k in range(max_seeds):
        if ok >= args.n:
            break
        seed = args.start_seed + k
        rid = f"{args.level}|{seed}|{obs_mode}"
        if rid in done:
            continue

        rec, lab, hidden = run_one(gen, args.level, seed, obs_mode, args,
                                   exemplars=None if n_ex == 0 else [],
                                   save_hidden=args.save_hidden)

        if rec["outcome"] == "success" or args.keep_all:
            sink.write(rec, lab)
            if rec["outcome"] == "success":
                ok += 1
        else:
            sink.reject({k: rec[k] for k in
                         ("id", "seed", "outcome", "n_planned", "n_exec",
                          "parse_ok", "truncated", "span_report", "leakage",
                          "segment_mode")}
                        | {"n_steps": len(rec["steps"])})

        if hidden is not None:
            torch.save({"token_ids": rec["token_ids"],
                        "prompt_len": rec["prompt_len"],
                        "hidden": hidden},
                       sink.dir / f"hidden_{seed}.pt")

        print(f"[{k:4d}] seed={seed} {rec['outcome']:10s} "
              f"steps={len(rec['steps']):2d} acts={rec['n_planned']:3d} "
              f"span_ok={rec['span_report']['ok']} "
              f"leak={rec['leakage']['has_action_leak']} "
              f"vis={lab['n_targets_visible']} ok={ok}/{args.n}",
              flush=True)

    print(f"\n{ok}/{args.n} success in {time.time()-t0:.0f}s "
          f"({k+1} seeds tried) -> {sink.dir}")


if __name__ == "__main__":
    main()