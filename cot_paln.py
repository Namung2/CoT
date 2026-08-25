"""BabyAI-Plan CoT 궤적 수집 (Qwen3 + vLLM, 로컬). cot_predict.py와 동일 구조.

Predict와의 결정적 차이:
  - 액션 시퀀스는 프롬프트에 넣지 않는다 (Plan에서는 그게 곧 정답 = 누출).
    OmniBot 시퀀스는 optimal_action_seq로 채점기에만 전달 (efficiency 기준선).
  - 채점: 생성된 시퀀스를 환경에서 실제 실행 → CR(도달 여부), efficiency, ball_distance.

이 실험은 논문 재현이 아님 — LLM-BabyBench(arXiv:2505.12135)는 Plan을 ToT로만
평가했고 Plan-CoT 수치는 미보고. 저자 코드에 존재하는 Plan CoT 템플릿을 써서
그 빈칸을 자체 측정한다. 환경/템플릿/채점기는 원 레포 그대로
(커스텀 GoToRedBall 환경 — 템플릿이 red ball 하드코딩).

청크 단위로 jsonl에 append하므로 중간에 죽어도 다시 실행하면 이어서 간다.
프롬프트 빌드/컨텍스트 초과로 실패하는 시드는 스킵하고 계속 진행한다
(OmniBot이 특정 시드에서 assert로 죽는 경우가 있음 — 그 한 시드 때문에
전체 프로세스가 죽지 않도록).

    python cot_plan.py                # thinking off
    python cot_plan.py --thinking     # thinking on
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_VENDOR = ROOT / "third_party" / "llm-babybench"
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from runner.env_loader import make_env
from formatters import get_formatter
from processors import get_processor
from prompters import get_prompter
from evaluators import get_evaluator
from llms.utils import parser

TASK = "plan"
FORMATTER, PROMPTER, PROCESSOR = "structured", "cot", "omniscient_babyai_bot"


def build_prompt(fmt, proc, prompter, env_name: str, seed: int):
    """(prompt, optimal_action_seq) 생성. CPU 작업이라 thinking 모드와 무관.
    optimal_action_seq는 채점 기준선일 뿐 프롬프트엔 안 들어감 (누출 방지).

    실패하면 None 반환 — OmniBot이 특정 시드(잠긴 문 등 edge case)에서
    assert로 죽는 경우가 있는데, 그 한 시드 때문에 전체 배치를 죽이지 않고
    skip하기 위함."""
    try:
        env = make_env(env_name, seed)
        desc = fmt.format(env)
        env.close()
        optimal_seq = proc.process(env_name, seed, TASK)
        return prompter.prompt(desc, TASK), optimal_seq
    except Exception as e:
        print(f"  skip {env_name} seed={seed}: {type(e).__name__}: {e}")
        return None


def done_keys(path: Path) -> set:
    """이미 저장된 (env_name, seed) — 재개용."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["env_name"], r["env_seed"]) for r in map(json.loads, f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    # 논문 Plan 조건: Small 4,5,6,7 | Medium 20,40,50,60 | Large 60,80,100,120 | Ultra 120,140,160,180
    ap.add_argument("--levels", nargs="+", default=[
        "CustomBabyAI-GoToRedBall-Small-4Dists-v0",
        "CustomBabyAI-GoToRedBall-Medium-40Dists-v0",
        "CustomBabyAI-GoToRedBall-Large-100Dists-v0",
        "CustomBabyAI-GoToRedBall-Ultra-180Dists-v0",
    ])
    ap.add_argument("--seeds", type=int, default=10000, help="레벨당 seed 개수 (0..N-1)")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--chunk", type=int, default=256, help="이 단위로 생성 후 append")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel size")
    ap.add_argument("--sampling-seed", type=int, default=0, help="재현용")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = "thinking" if args.thinking else "no_thinking"
    out = Path(args.out) if args.out else ROOT / "data" / f"plan_{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = get_formatter(FORMATTER)
    proc = get_processor(PROCESSOR)
    prompter = get_prompter(PROMPTER)
    evaluator = get_evaluator(TASK)

    skip = done_keys(out)
    todo = [(lv, sd) for lv in args.levels for sd in range(args.seeds)
            if (lv, sd) not in skip]
    print(f"{len(todo)} to generate ({len(skip)} already in {out})")
    if not todo:
        return

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len)
    tok = AutoTokenizer.from_pretrained(args.model)
    sp = SamplingParams(temperature=args.temp, max_tokens=args.max_tokens,
                        seed=args.sampling_seed)

    n_skipped_build = 0
    n_skipped_ctx = 0

    for i in range(0, len(todo), args.chunk):
        raw_batch = todo[i:i + args.chunk]

        built = [build_prompt(fmt, proc, prompter, lv, sd) for lv, sd in raw_batch]
        batch, prompts, opt_seqs = [], [], []
        for (lv, sd), r in zip(raw_batch, built):
            if r is None:
                n_skipped_build += 1
                continue
            batch.append((lv, sd))
            prompts.append(r[0])
            opt_seqs.append(r[1])
        if not batch:
            continue

        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=args.thinking)
                 for p in prompts]

        # 컨텍스트 예산 검사. 넘치면 vLLM이 생성분을 조용히 깎아 잘린 CoT가 섞이므로
        # 매 청크마다 확인한다 (레벨마다 프롬프트 길이가 크게 다르다, 특히 Ultra-180Dists).
        # 예산 초과 항목은 전체를 죽이지 않고 그 시드만 skip한다.
        n_toks = [len(tok(t).input_ids) for t in texts]
        kept = [j for j, n in enumerate(n_toks) if n + args.max_tokens <= args.max_model_len]
        if len(kept) < len(texts):
            for j in range(len(texts)):
                if j not in kept:
                    lv, sd = batch[j]
                    print(f"  skip {lv} seed={sd}: prompt {n_toks[j]} + max_tokens "
                          f"{args.max_tokens} > max_model_len {args.max_model_len}")
                    n_skipped_ctx += 1
        if not kept:
            continue
        batch = [batch[j] for j in kept]
        prompts = [prompts[j] for j in kept]
        opt_seqs = [opt_seqs[j] for j in kept]
        texts = [texts[j] for j in kept]

        if i == 0:   # thinking 토글이 실제로 템플릿에 먹었는지 1회 확인
            print(f"longest prompt = {max(n_toks)} tok")
            print(f"template tail  = {texts[0][-160:]!r}")

        outputs = llm.generate(texts, sp)

        with out.open("a", encoding="utf-8") as f:
            for (env_name, seed), opt_seq, prompt, o in zip(batch, opt_seqs, prompts, outputs):
                text = o.outputs[0].text.strip()
                pred = parser(text, TASK)          # "" 이면 마커 미출력 = 파싱 실패
                try:
                    ev = evaluator.evaluate(env_name, seed,
                                            optimal_action_seq=opt_seq,
                                            llm_action_seq=pred)
                    err = None
                except Exception as e:             # 파싱 실패/미등록 액션명 등을 뭉개지 않는다
                    ev, err = None, f"{type(e).__name__}: {e}"
                f.write(json.dumps({
                    "env_name": env_name,
                    "env_seed": seed,
                    "task": TASK,
                    "formatter": FORMATTER,
                    "prompter": PROMPTER,
                    "model": args.model,
                    "thinking": args.thinking,
                    "prompt": prompt,
                    "optimal_action_seq": opt_seq,     # 채점 기준선 (efficiency 계산용)
                    "all_llm_output": text,
                    "parsed_llm_output": pred,
                    "eval_result": ev,
                    "eval_error": err,
                    "truncated": o.outputs[0].finish_reason == "length",
                }, ensure_ascii=False) + "\n")
        print(f"[{min(i + args.chunk, len(todo))}/{len(todo)}] appended -> {out}")

    if n_skipped_build or n_skipped_ctx:
        print(f"skipped: {n_skipped_build} build errors, {n_skipped_ctx} over context budget")

    with out.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for lv in args.levels:
        rs = [r for r in rows if r["env_name"] == lv]
        if not rs:
            continue
        ok = sum((r["eval_result"] or {}).get("CR", 0) == 1 for r in rs)
        effs = [(r["eval_result"] or {})["llm_efficiency"] for r in rs
                if (r["eval_result"] or {}).get("CR", 0) == 1
                and "llm_efficiency" in (r["eval_result"] or {})]
        eff = sum(effs) / len(effs) if effs else 0.0
        print(f"{lv}: {len(rs)} eps | CR {ok}/{len(rs)} ({ok/len(rs):.0%})  "
              f"eff(succ) {eff:.2f}  eval_error {sum(r['eval_error'] is not None for r in rs)}  "
              f"truncated {sum(r['truncated'] for r in rs)}")


if __name__ == "__main__":
    main()
