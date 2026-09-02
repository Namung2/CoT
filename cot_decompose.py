"""BabyAI-Decompose CoT 궤적 수집 (Qwen3 + vLLM, 로컬). cot_predict.py/cot_paln.py 와 동일 구조.

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

predict/plan 과 동일하게 두 가지를 갖춘다.

1. --seeds 는 "시도할 시드 범위"가 아니라 "레벨당 최종적으로 채워야 할 개수"로
   취급한다. 0..seeds-1 범위에서 스킵(빌드 실패/컨텍스트 초과)된 만큼,
   seeds, seeds+1, ... 범위에서 추가로 시드를 끌어와 목표 개수를 채운다.

2. 스킵된 시드도 skipped 필드를 달아 jsonl 에 남긴다 — 남기지 않으면 재시작할
   때마다 같은 시드를 다시 시도한다.

청크 단위로 jsonl에 append하므로 중간에 죽어도 다시 실행하면 이어서 간다.

    python cot_decompose.py                # thinking off
    python cot_decompose.py --thinking     # thinking on
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import Counter
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
    """(prompt, mission, error) 생성. CPU 작업이라 thinking 모드와 무관.

    error 가 not None 이면 이 시드는 skip — env 생성이 특정 시드(edge case)에서
    예외를 던지는 경우가 있는데, 그 한 시드 때문에 전체 배치를 죽이지 않는다."""
    env = None
    try:
        env = make_env(env_name, seed)
        desc = fmt.format(env)
        mission = env.unwrapped.mission
        env.close()
        env = None
        return prompter.prompt(desc, TASK), mission, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def base_row(env_name: str, seed: int, args) -> dict:
    return {
        "env_name": env_name,
        "env_seed": seed,
        "task": TASK,
        "formatter": FORMATTER,
        "prompter": PROMPTER,
        "model": args.model,
        "thinking": args.thinking,
    }


def skip_row(env_name: str, seed: int, args, reason: str, detail: str) -> dict:
    """skip 된 시드도 기록 (재시작 시 재시도 방지 + 레벨별 편향 감사용)."""
    return {**base_row(env_name, seed, args),
            "skipped": reason, "skip_detail": detail,
            "prompt": None, "mission": None,
            "all_llm_output": None, "parsed_llm_output": None,
            "eval_result": None, "eval_error": None, "truncated": False}


def read_rows(path: Path) -> list:
    """저장된 줄들. 하드 킬로 마지막 줄이 잘려 있을 수 있어 깨진 줄은 건너뛴다."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def seed_candidates(level: str, attempted: set):
    """0, 1, 2, ... 순서로 시드를 끌어온다. 이미 시도한(성공/스킵 모두) 시드는 건너뛴다.
    무한 제너레이터 — 호출 측이 목표 개수만큼만 소비한다."""
    cursor = 0
    while True:
        if (level, cursor) not in attempted:
            yield cursor
        cursor += 1


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

    prev = read_rows(out)
    attempted = {(r["env_name"], r["env_seed"]) for r in prev}       # 스킵 포함
    have_by_level = Counter(r["env_name"] for r in prev if not r.get("skipped"))

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len)
    tok = AutoTokenizer.from_pretrained(args.model)
    sp = SamplingParams(temperature=args.temp, max_tokens=args.max_tokens,
                        seed=args.sampling_seed)

    diag_shown = False
    seed_cap = args.seeds * args.seed_cap_multiplier
    target = args.seeds

    for level in args.levels:
        have = have_by_level[level]
        print(f"[{level}] {have}/{target} already done")
        if have >= target:
            continue

        cand = seed_candidates(level, attempted)

        while have < target:
            # 남은 개수보다 많이 뽑으면 target 을 초과한다. 스킵이 나면 다음
            # 라운드에서 부족분만큼 다시 끌어오므로 정확히 target 에 수렴한다.
            need = min(args.chunk, target - have)
            raw_batch, hit_cap = [], False
            for sd in cand:
                if sd >= seed_cap:
                    hit_cap = True
                    break
                raw_batch.append(sd)
                if len(raw_batch) >= need:
                    break

            rows = []                   # 이 라운드에서 jsonl 에 쓸 줄 (skip 포함)
            batch, prompts, missions = [], [], []

            for sd in raw_batch:
                p, mission, err = build_prompt(fmt, prompter, level, sd)
                if err is not None:
                    print(f"  skip {level} seed={sd}: {err}")
                    rows.append(skip_row(level, sd, args, "build", err))
                    attempted.add((level, sd))
                    continue
                batch.append(sd)
                prompts.append(p)
                missions.append(mission)

            if batch:
                texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=args.thinking)
                         for p in prompts]

                # 컨텍스트 예산 검사. 넘치면 vLLM 이 생성분을 조용히 깎아 잘린 CoT 가
                # 섞이므로 매 라운드 확인한다 (레벨마다 프롬프트 길이가 크게 다르다,
                # 특히 BossLevel). 예산 초과 항목은 그 시드만 skip 한다.
                n_toks = [len(tok(t).input_ids) for t in texts]
                kept = [j for j, n in enumerate(n_toks)
                        if n + args.max_tokens <= args.max_model_len]
                for j in set(range(len(texts))) - set(kept):
                    detail = (f"prompt {n_toks[j]} + max_tokens {args.max_tokens} "
                              f"> max_model_len {args.max_model_len}")
                    print(f"  skip {level} seed={batch[j]}: {detail}")
                    rows.append(skip_row(level, batch[j], args, "context", detail))
                    attempted.add((level, batch[j]))

                batch = [batch[j] for j in kept]
                prompts = [prompts[j] for j in kept]
                missions = [missions[j] for j in kept]
                texts = [texts[j] for j in kept]

                if batch and not diag_shown:   # thinking 토글이 템플릿에 먹었는지 1회 확인
                    diag_shown = True
                    print(f"longest prompt = {max(n_toks)} tok")
                    print(f"template tail  = {texts[0][-160:]!r}")

                if batch:
                    outputs = llm.generate(texts, sp)
                    for seed, mission, prompt, o in zip(batch, missions, prompts, outputs):
                        text = o.outputs[0].text.strip()
                        pred = parser(text, TASK)   # <START>..<END> 사이
                        # Decompose evaluator 는 (env_name, seed) 가 아니라 살아있는
                        # env 객체를 받음 — eval 시점에 새로 만든다.
                        env = None
                        try:
                            env = make_env(level, seed)
                            old_handler = signal.signal(signal.SIGALRM, _eval_timeout_handler)
                            signal.alarm(args.eval_timeout)
                            try:
                                ev = evaluator.evaluate(env, pred)
                            finally:
                                signal.alarm(0)
                                signal.signal(signal.SIGALRM, old_handler)
                            err = None
                        except Exception as e:   # 파싱 실패(빈 서브골/포맷 위반), 봇 실행 예외, eval 타임아웃
                            ev, err = None, f"{type(e).__name__}: {e}"
                            if isinstance(e, _EvalTimeout):
                                print(f"  eval timeout {level} seed={seed} "
                                      f"(>{args.eval_timeout}s), skip")
                        finally:
                            if env is not None:
                                try:
                                    env.close()
                                except Exception:
                                    pass
                        rows.append({**base_row(level, seed, args),
                                     "skipped": None, "skip_detail": None,
                                     "prompt": prompt,
                                     "mission": mission,
                                     "all_llm_output": text,
                                     "parsed_llm_output": pred,
                                     "eval_result": ev,
                                     "eval_error": err,
                                     "truncated": o.outputs[0].finish_reason == "length"})
                        attempted.add((level, seed))
                        have += 1

            if rows:
                with out.open("a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if raw_batch:
                print(f"[{level}] {have}/{target} appended -> {out}")

            if hit_cap:
                print(f"[{level}] seed {seed_cap} 이상은 시도 안 함, "
                      f"목표({target}) 못 채우고 포기 (지금까지 {have})")
                break

    # ── 요약 (파일 전체 기준, 이전 run 포함) ──────────────────────────
    rows = read_rows(out)
    for lv in args.levels:
        rs = [r for r in rows if r["env_name"] == lv]
        if not rs:
            continue
        sk = Counter(r["skipped"] for r in rs if r.get("skipped"))
        ev_rows = [r for r in rs if not r.get("skipped")]
        if not ev_rows:
            print(f"{lv}: 0 evaluated, skipped {sum(sk.values())} {dict(sk)}")
            continue
        cr = sum((r["eval_result"] or {}).get("CR", 0) == 1 for r in ev_rows)
        pr_ = sum((r["eval_result"] or {}).get("PR", 0) == 1 for r in ev_rows)
        aci = sum((r["eval_result"] or {}).get("ACI", 0) for r in ev_rows) / len(ev_rows)
        print(f"{lv}: {len(ev_rows)} eps | CR {cr/len(ev_rows):.0%} | PR {pr_/len(ev_rows):.0%} | "
              f"ACI {aci:.2f}  "
              f"eval_error {sum(r['eval_error'] is not None for r in ev_rows)}  "
              f"truncated {sum(r['truncated'] for r in ev_rows)}  "
              f"skipped {sum(sk.values())} {dict(sk)}")


if __name__ == "__main__":
    main()
