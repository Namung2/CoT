"""BabyAI-Predict CoT 궤적 수집 (Qwen3 + vLLM, 로컬).

llm-babybench 의 formatter/processor/prompter/evaluator/parser 를 그대로 쓰고
LLM 호출과 배치 오케스트레이션만 교체한다. 업스트림 runner/pipeline.py 는
config 하나당 에피소드 하나 + API 백엔드 전용이라 대량 로컬 생성에 못 쓴다.

출력 필드명은 업스트림 result_entry 와 맞춘다 (truncated/eval_error 만 추가).
청크 단위로 jsonl 에 append 하므로 중간에 죽어도 다시 실행하면 이어서 간다.

    python cot_predict.py                # thinking off
    python cot_predict.py --thinking     # thinking on
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "llm-babybench"))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from runner.env_loader import make_env
from formatters import get_formatter
from processors import get_processor
from prompters import get_prompter
from evaluators import get_evaluator
from llms.utils import parser

TASK = "predict"
FORMATTER, PROMPTER, PROCESSOR = "structured", "cot", "omniscient_babyai_bot"


def build_prompt(fmt, proc, prompter, env_name: str, seed: int):
    """(prompt, action_seq) 생성. CPU 작업이라 thinking 모드와 무관."""
    env = make_env(env_name, seed)
    desc = fmt.format(env)
    env.close()
    action_seq = proc.process(env_name, seed, TASK)   # OmniBot 궤적 -> "left, forward, ..."
    return prompter.prompt(desc, TASK, action_seq), action_seq


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
    ap.add_argument("--levels", nargs="+",
                    default=["BabyAI-GoToObj-v0", "BabyAI-BossLevel-v0"])
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
    out = Path(args.out) if args.out else ROOT / "data" / f"predict_{mode}.jsonl"
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

    for i in range(0, len(todo), args.chunk):
        batch = todo[i:i + args.chunk]
        prompts, action_seqs = zip(*(build_prompt(fmt, proc, prompter, lv, sd)
                                     for lv, sd in batch))

        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=args.thinking)
                 for p in prompts]

        # 컨텍스트 예산 검사. 넘치면 vLLM 이 생성분을 조용히 깎아 잘린 CoT 가 섞이므로
        # 매 청크마다 확인한다 (레벨마다 프롬프트 길이가 크게 다르다).
        longest = max(len(tok(t).input_ids) for t in texts)
        assert longest + args.max_tokens <= args.max_model_len, (
            f"prompt {longest} + max_tokens {args.max_tokens} "
            f"> max_model_len {args.max_model_len}")

        if i == 0:   # thinking 토글이 실제로 템플릿에 먹었는지 1회 확인
            print(f"longest prompt = {longest} tok")
            print(f"template tail  = {texts[0][-160:]!r}")

        outputs = llm.generate(texts, sp)

        with out.open("a", encoding="utf-8") as f:
            for (env_name, seed), aseq, prompt, o in zip(batch, action_seqs, prompts, outputs):
                text = o.outputs[0].text.strip()
                pred = parser(text, TASK)          # "" 이면 마커 미출력 = 파싱 실패
                try:
                    ev = evaluator.evaluate(env_name, seed, str_action_seq=aseq,
                                            predicted_output=pred)
                    err = None
                except Exception as e:             # 파싱 실패와 환경 오류를 뭉개지 않는다
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
                    "all_llm_output": text,
                    "parsed_llm_output": pred,
                    "eval_result": ev,
                    "eval_error": err,
                    "truncated": o.outputs[0].finish_reason == "length",
                }, ensure_ascii=False) + "\n")
        print(f"[{min(i + args.chunk, len(todo))}/{len(todo)}] appended -> {out}")

    with out.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for lv in args.levels:
        rs = [r for r in rows if r["env_name"] == lv]
        if not rs:
            continue
        ok = sum((r["eval_result"] or {}).get("success", False) for r in rs)
        print(f"{lv}: success {ok}/{len(rs)} ({ok/len(rs):.0%})  "
              f"unparsed {sum(not r['parsed_llm_output'] for r in rs)}  "
              f"eval_error {sum(r['eval_error'] is not None for r in rs)}  "
              f"truncated {sum(r['truncated'] for r in rs)}")


if __name__ == "__main__":
    main()
