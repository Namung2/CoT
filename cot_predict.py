"""BabyAI-Predict CoT 궤적 수집 (Qwen3 + vLLM, 로컬) — 타임아웃 + 목표치 채우기.

cot_predict.py 와 동일한 생성 로직이지만 두 가지가 다르다.

1. 시드 하나당 build_prompt 에 타임아웃을 건다. OmniBot 이 특정 시드에서 BFS 가
   끝나지 않는 경우가 있어서 (예: BossLevel seed=9322 —
   omniscient_fixed_plan_baby_ai_bot.py 의 _find_drop_pos -> _shortest_path ->
   _breadth_first_search 가 종료 안 함), 그 한 시드 때문에 전체 프로세스가
   멈추지 않도록 스킵한다.

2. --seeds 는 "시도할 시드 범위"가 아니라 "레벨당 최종적으로 채워야 할 개수"
   로 취급한다. 0..seeds-1 범위에서 스킵(빌드 실패/타임아웃/컨텍스트 초과)된
   만큼, seeds, seeds+1, ... 범위에서 추가로 시드를 끌어와 목표 개수를 채운다
   (레벨당 정확히 --seeds 개, 즉 기본 설정이면 20000개 전체를 다 채움).

출력 파일은 cot_predict.py 와 동일한 경로 규칙을 쓰므로, cot_predict.py 로 만들다
중단된 데이터를 그대로 이어서 생성할 수 있다 (done_keys 기반 resume 공유).

    python cot_predict2.py                # thinking off
    python cot_predict2.py --thinking     # thinking on
"""
from __future__ import annotations

import argparse
import json
import signal
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


class _BuildTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _BuildTimeout()


def build_prompt(fmt, proc, prompter, env_name: str, seed: int, timeout: int):
    """(prompt, action_seq) 생성. CPU 작업이라 thinking 모드와 무관.

    실패하거나 timeout 안에 안 끝나면 None 반환 — OmniBot 이 특정 시드에서
    assert 로 죽거나 BFS 가 끝나지 않는 경우가 있는데, 그 한 시드 때문에
    전체 배치를 죽이지 않고 skip 하기 위함."""
    env = None
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        env = make_env(env_name, seed)
        desc = fmt.format(env)
        env.close()
        env = None
        action_seq = proc.process(env_name, seed, TASK)   # OmniBot 궤적 -> "left, forward, ..."
        return prompter.prompt(desc, TASK, action_seq), action_seq
    except _BuildTimeout:
        print(f"  skip {env_name} seed={seed}: timed out after {timeout}s (OmniBot 무한루프 의심)")
        return None
    except Exception as e:
        print(f"  skip {env_name} seed={seed}: {type(e).__name__}: {e}")
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def done_keys(path: Path) -> set:
    """이미 저장된 (env_name, seed) — 재개/목표치 계산용."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["env_name"], r["env_seed"]) for r in map(json.loads, f)}


def seed_candidates(level: str, done: set):
    """0, 1, 2, ... 순서로 계속 시드를 끌어온다 (범위 제한 없음).
    이미 done 에 있는 시드는 건너뛴다. 무한 제너레이터 — 호출 측이 필요한 만큼만 소비
    (target 개수만큼 성공할 때까지, 또는 seed_cap 안전장치에 걸릴 때까지)."""
    cursor = 0
    while True:
        if (level, cursor) not in done:
            yield cursor
        cursor += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--levels", nargs="+",
                    default=["BabyAI-GoToObj-v0", "BabyAI-BossLevel-v0"])
    ap.add_argument("--seeds", type=int, default=10000,
                    help="레벨당 최종적으로 채울 개수 (스킵된 만큼 이 범위 밖에서 추가로 끌어옴)")
    ap.add_argument("--seed-cap-multiplier", type=int, default=5,
                    help="안전장치: seeds * 이 배수까지 시도해도 목표를 못 채우면 포기하고 다음 레벨로")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--chunk", type=int, default=256, help="이 단위로 생성 후 append")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel size")
    ap.add_argument("--sampling-seed", type=int, default=0, help="재현용")
    ap.add_argument("--build-timeout", type=int, default=20,
                    help="시드 하나당 build_prompt 타임아웃(초) — OmniBot이 안 끝나는 시드 스킵용")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = "thinking" if args.thinking else "no_thinking"
    out = Path(args.out) if args.out else ROOT / "data" / f"predict_{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = get_formatter(FORMATTER)
    proc = get_processor(PROCESSOR)
    prompter = get_prompter(PROMPTER)
    evaluator = get_evaluator(TASK)

    done = done_keys(out)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len)
    tok = AutoTokenizer.from_pretrained(args.model)
    sp = SamplingParams(temperature=args.temp, max_tokens=args.max_tokens,
                        seed=args.sampling_seed)

    n_skipped_build = 0
    n_skipped_ctx = 0
    printed_debug = False
    seed_cap = args.seeds * args.seed_cap_multiplier

    for level in args.levels:
        target = args.seeds
        have = sum(1 for (lv, _) in done if lv == level)
        print(f"[{level}] {have}/{target} already done")
        if have >= target:
            continue

        cand = seed_candidates(level, done)

        while have < target:
            # 남은 개수보다 많이 뽑으면 target 을 초과한다. 스킵이 나면
            # 다음 라운드에서 부족분만큼 다시 끌어오므로 정확히 target 에 수렴한다.
            need = min(args.chunk, target - have)
            raw_batch = []
            for sd in cand:
                raw_batch.append((level, sd))
                if len(raw_batch) >= need:
                    break
                if sd >= seed_cap:
                    break
            if not raw_batch:
                break
            if raw_batch[-1][1] >= seed_cap:
                print(f"[{level}] seed {raw_batch[-1][1]} >= cap {seed_cap}, "
                      f"목표({target}) 못 채우고 포기 (지금까지 {have})")
                break

            built = [build_prompt(fmt, proc, prompter, lv, sd, args.build_timeout)
                     for lv, sd in raw_batch]
            batch, prompts, action_seqs = [], [], []
            for (lv, sd), r in zip(raw_batch, built):
                if r is None:
                    n_skipped_build += 1
                    continue
                batch.append((lv, sd))
                prompts.append(r[0])
                action_seqs.append(r[1])
            if not batch:
                continue

            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=args.thinking)
                     for p in prompts]

            # 컨텍스트 예산 검사. 넘치면 vLLM 이 생성분을 조용히 깎아 잘린 CoT 가 섞이므로
            # 매 청크마다 확인한다. 예산 초과 항목은 전체를 죽이지 않고 그 시드만 skip 한다.
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
            action_seqs = [action_seqs[j] for j in kept]
            texts = [texts[j] for j in kept]

            if not printed_debug:
                print(f"longest prompt = {max(n_toks)} tok")
                print(f"template tail  = {texts[0][-160:]!r}")
                printed_debug = True

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
                    done.add((env_name, seed))
                    have += 1
            print(f"[{level}] {have}/{target} appended -> {out}")

    if n_skipped_build or n_skipped_ctx:
        print(f"skipped: {n_skipped_build} build errors/timeouts, {n_skipped_ctx} over context budget")

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
