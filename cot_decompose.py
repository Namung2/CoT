"""BabyAI-Decompose CoT 궤적 수집 (Qwen3 + vLLM, 로컬). cot_predict.py/cot_plan.py와 동일 구조.

주의 — 이 실험은 논문 재현이 아님:
  LLM-BabyBench(arXiv:2505.12135)는 Decompose를 ToT로만 평가했고 Decompose-CoT 수치는 미보고.
  저자 코드에 존재하는 Decompose CoT 템플릿을 사용해 그 빈칸을 자체 측정한다.
  환경(표준 16레벨) / 템플릿 / 채점기는 원 레포 그대로.

태스크: 미션 문장 → 서브골 리스트 (<START>...(GoNextToSubgoal,(x,y))/(OpenSubgoal)/...<END>)
채점: LLM 서브골로 OmniBot 스택을 초기화해 실행 —
  CR  = 봇이 서브골을 추가해서라도 미션 완주 (관대)
  PR  = 봇이 서브골 추가 0회로 완주 (엄격)
  ACI = 필요 추가량 대비 절약 비율 [0,1]
(evaluator가 live env 객체를 받으므로 predict/plan과 달리 채점 시점에 env를 새로 만든다.)

청크 단위로 jsonl에 append하므로 중간에 죽어도 다시 실행하면 이어서 간다.
프롬프트 빌드/컨텍스트 초과로 실패하는 시드는 스킵하고 계속 진행한다
(env 생성이 특정 시드에서 예외로 죽는 경우가 있음 — 그 한 시드 때문에
전체 프로세스가 죽지 않도록).

    python cot_decompose.py                # thinking off
    python cot_decompose.py --thinking     # thinking on
"""
from __future__ import annotations

import argparse
import json
import signal
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
from prompters import get_prompter
from evaluators import get_evaluator
from llms.utils import parser

TASK = "decompose"
FORMATTER, PROMPTER = "structured", "cot"


class _EvalTimeout(Exception):
    """evaluator.evaluate()가 시간 내 안 끝날 때. BabyAIBot.replan()의 서브골
    push/pop 루프가 env.step() 없이 순환에 빠지면 예외 없이 영원히 안 끝나서
    (실제로 BabyAI-Synth-v0 seed=3000에서 관측됨) 일반 try/except로는 못 잡는다."""


def _eval_timeout_handler(signum, frame):
    raise _EvalTimeout("evaluator.evaluate() timed out")


def build_prompt(fmt, prompter, env_name: str, seed: int):
    """(prompt, mission) 생성. CPU 작업이라 thinking 모드와 무관.

    실패하면 None 반환 — env 생성이 특정 시드(edge case)에서 예외를 던지는
    경우가 있는데, 그 한 시드 때문에 전체 배치를 죽이지 않고 skip하기 위함."""
    try:
        env = make_env(env_name, seed)
        desc = fmt.format(env)
        mission = env.unwrapped.mission
        env.close()
        return prompter.prompt(desc, TASK), mission
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
    # 논문 Decompose는 16레벨 전부 사용. 여기서는 난이도 스펙트럼 4개를 기본값으로.
    ap.add_argument("--levels", nargs="+", default=[
        "BabyAI-GoToObj-v0",     # Easy
        "BabyAI-GoTo-v0",        # Medium (문/미로 등장 → PR 갈리기 시작)
        "BabyAI-Synth-v0",       # Hard
        "BabyAI-BossLevel-v0",   # Very Hard
    ])
    ap.add_argument("--seeds", type=int, default=10000, help="레벨당 seed 개수 (0..N-1)")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--chunk", type=int, default=256, help="이 단위로 생성 후 append")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel size")
    ap.add_argument("--sampling-seed", type=int, default=0, help="재현용")
    ap.add_argument("--eval-timeout", type=int, default=10,
                    help="evaluator.evaluate() 1건당 제한 시간(초). "
                         "봇 리플랜 루프가 안 끝나는 시드를 건너뛰기 위함. "
                         "정상 케이스는 BossLevel(가장 무거움)도 실측 max 0.5s 이내라 "
                         "10s면 오탐 없이 충분한 마진")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = "thinking" if args.thinking else "no_thinking"
    out = Path(args.out) if args.out else ROOT / "data" / f"decompose_{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = get_formatter(FORMATTER)
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

        built = [build_prompt(fmt, prompter, lv, sd) for lv, sd in raw_batch]
        batch, prompts, missions = [], [], []
        for (lv, sd), r in zip(raw_batch, built):
            if r is None:
                n_skipped_build += 1
                continue
            batch.append((lv, sd))
            prompts.append(r[0])
            missions.append(r[1])
        if not batch:
            continue

        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=args.thinking)
                 for p in prompts]

        # 컨텍스트 예산 검사. 넘치면 vLLM이 생성분을 조용히 깎아 잘린 CoT가 섞이므로
        # 매 청크마다 확인한다 (레벨마다 프롬프트 길이가 크게 다르다, 특히 BossLevel).
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
        missions = [missions[j] for j in kept]
        texts = [texts[j] for j in kept]

        if i == 0:   # thinking 토글이 실제로 템플릿에 먹었는지 1회 확인
            print(f"longest prompt = {max(n_toks)} tok")
            print(f"template tail  = {texts[0][-160:]!r}")

        outputs = llm.generate(texts, sp)

        with out.open("a", encoding="utf-8") as f:
            for (env_name, seed), mission, prompt, o in zip(batch, missions, prompts, outputs):
                text = o.outputs[0].text.strip()
                pred = parser(text, TASK)          # <START>..<END> 사이
                try:
                    # Decompose evaluator는 (env_name, seed)가 아니라 살아있는 env 객체를 받음
                    env = make_env(env_name, seed)
                    signal.signal(signal.SIGALRM, _eval_timeout_handler)
                    signal.alarm(args.eval_timeout)
                    try:
                        ev = evaluator.evaluate(env, pred)
                    finally:
                        signal.alarm(0)
                    env.close()
                    err = None
                except Exception as e:              # 파싱 실패(빈 서브골/포맷 위반), 봇 실행 예외, eval 타임아웃
                    ev, err = None, f"{type(e).__name__}: {e}"
                    if isinstance(e, _EvalTimeout):
                        print(f"  eval timeout {env_name} seed={seed} (>{args.eval_timeout}s), skip")
                f.write(json.dumps({
                    "env_name": env_name,
                    "env_seed": seed,
                    "task": TASK,
                    "formatter": FORMATTER,
                    "prompter": PROMPTER,
                    "model": args.model,
                    "thinking": args.thinking,
                    "prompt": prompt,
                    "mission": mission,
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
        cr  = sum((r["eval_result"] or {}).get("CR", 0) == 1 for r in rs)
        pr_ = sum((r["eval_result"] or {}).get("PR", 0) == 1 for r in rs)
        aci = sum((r["eval_result"] or {}).get("ACI", 0) for r in rs) / len(rs)
        print(f"{lv}: {len(rs)} eps | CR {cr/len(rs):.0%} | PR {pr_/len(rs):.0%} | ACI {aci:.2f}  "
              f"eval_error {sum(r['eval_error'] is not None for r in rs)}  "
              f"truncated {sum(r['truncated'] for r in rs)}")


if __name__ == "__main__":
    main()
