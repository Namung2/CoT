"""BabyAI-Plan CoT 궤적 수집 (Qwen3 + vLLM, 로컬). cot_predict.py 와 동일 구조.

Predict 와의 결정적 차이:
  - 액션 시퀀스는 프롬프트에 넣지 않는다 (Plan 에서는 그게 곧 정답 = 누출).
    OmniBot 시퀀스는 optimal_action_seq 로 채점기에만 전달 (efficiency 기준선).
  - 채점: 생성된 시퀀스를 환경에서 실제 실행 -> CR(도달 여부), efficiency, ball_distance.
  - 파싱 실패가 예외를 던지지 않는다. str_action_seq_to_int("") 는 조용히 [] 를
    반환하므로 eval_error 가 아니라 CR=0 으로 집계된다. 따라서 요약에서
    parsed_llm_output == "" 개수(unparsed)를 반드시 따로 세야 원인이 구분된다.

이 실험은 논문 재현이 아님 — LLM-BabyBench(arXiv:2505.12135)는 Plan 을 ToT 로만
평가했고 Plan-CoT 수치는 미보고. 저자 코드에 존재하는 Plan CoT 템플릿을 써서
그 빈칸을 자체 측정한다. 환경/템플릿/채점기는 원 레포 그대로
(커스텀 GoToRedBall 환경 — 템플릿이 red ball 하드코딩).

청크 단위로 jsonl 에 append 하므로 중간에 죽어도 다시 실행하면 이어서 간다.
빌드 실패/타임아웃/컨텍스트 초과 시드는 skip 하되 skipped 필드를 달아 jsonl 에
남긴다 — 남기지 않으면 재시작할 때마다 같은 시드에 OmniBot 을 다시 돌린다.

--seeds 는 "시도할 시드 범위"가 아니라 "레벨당 최종적으로 채울 개수"다.
스킵된 만큼 seeds, seeds+1, ... 에서 추가로 끌어와 정확히 목표 개수를 채운다.

    python cot_paln.py                # thinking off
    python cot_paln.py --thinking     # thinking on
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
from processors import get_processor
from prompters import get_prompter
from evaluators import get_evaluator
from llms.utils import parser

TASK = "plan"
FORMATTER, PROMPTER, PROCESSOR = "structured", "cot", "omniscient_babyai_bot"


class _BuildTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _BuildTimeout()


def build_prompt(fmt, proc, prompter, env_name: str, seed: int, timeout: int):
    """(prompt, optimal_action_seq, error) 생성. CPU 작업이라 thinking 모드와 무관.
    optimal_action_seq 는 채점 기준선일 뿐 프롬프트엔 안 들어감 (누출 방지).

    error 가 not None 이면 이 시드는 skip — OmniBot 이 특정 시드에서 assert 로
    죽거나 _find_drop_pos -> _shortest_path -> _breadth_first_search 가 끝나지
    않는 경우가 있는데, 그 한 시드 때문에 전체 배치를 죽이지 않는다."""
    env = None
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        env = make_env(env_name, seed)
        desc = fmt.format(env)
        env.close()
        env = None
        optimal_seq = proc.process(env_name, seed, TASK)
        return prompter.prompt(desc, TASK), optimal_seq, None
    except _BuildTimeout:
        return None, None, f"BuildTimeout: {timeout}s 안에 안 끝남 (OmniBot 무한루프 의심)"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
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
            "prompt": None, "optimal_action_seq": None,
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
    # 논문 Plan 조건: Small 4,5,6,7 | Medium 20,40,50,60 | Large 60,80,100,120 | Ultra 120,140,160,180
    ap.add_argument("--levels", nargs="+", default=[
        "CustomBabyAI-GoToRedBall-Small-4Dists-v0",
        "CustomBabyAI-GoToRedBall-Medium-40Dists-v0",
        "CustomBabyAI-GoToRedBall-Large-100Dists-v0",
        "CustomBabyAI-GoToRedBall-Ultra-180Dists-v0",
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
    ap.add_argument("--build-timeout", type=int, default=20,
                    help="시드 하나당 build_prompt 타임아웃(초) — OmniBot이 안 끝나는 시드 스킵용")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = "thinking" if args.thinking else "no_thinking"
    out = Path(args.out) if args.out else ROOT / "data" / f"plan_{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = get_formatter(FORMATTER)
    proc = get_processor(PROCESSOR)
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
            raw_batch = []
            for sd in cand:
                raw_batch.append(sd)
                if len(raw_batch) >= need or sd >= seed_cap:
                    break
            if not raw_batch:
                break
            if raw_batch[-1] >= seed_cap:
                print(f"[{level}] seed {raw_batch[-1]} >= cap {seed_cap}, "
                      f"목표({target}) 못 채우고 포기 (지금까지 {have})")
                break

            rows = []                   # 이 라운드에서 jsonl 에 쓸 줄 (skip 포함)
            batch, prompts, opt_seqs = [], [], []

            for sd in raw_batch:
                p, oseq, err = build_prompt(fmt, proc, prompter, level, sd,
                                            args.build_timeout)
                if err is not None:
                    print(f"  skip {level} seed={sd}: {err}")
                    rows.append(skip_row(level, sd, args, "build", err))
                    attempted.add((level, sd))
                    continue
                batch.append(sd)
                prompts.append(p)
                opt_seqs.append(oseq)

            if batch:
                texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=args.thinking)
                         for p in prompts]

                # 컨텍스트 예산 검사. 넘치면 vLLM 이 생성분을 조용히 깎아 잘린 CoT 가
                # 섞이므로 매 라운드 확인한다 (레벨마다 프롬프트 길이가 크게 다르다,
                # 특히 Ultra-180Dists). 초과 항목은 그 시드만 skip 한다.
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
                opt_seqs = [opt_seqs[j] for j in kept]
                texts = [texts[j] for j in kept]

                if batch and not diag_shown:   # thinking 토글이 템플릿에 먹었는지 1회 확인
                    diag_shown = True
                    print(f"longest prompt = {max(n_toks)} tok")
                    print(f"template tail  = {texts[0][-160:]!r}")

                if batch:
                    outputs = llm.generate(texts, sp)
                    for seed, opt_seq, prompt, o in zip(batch, opt_seqs, prompts, outputs):
                        text = o.outputs[0].text.strip()
                        pred = parser(text, TASK)   # "" 이면 마커 미출력 = 파싱 실패
                        try:
                            ev = evaluator.evaluate(level, seed,
                                                    optimal_action_seq=opt_seq,
                                                    llm_action_seq=pred)
                            err = None
                        except Exception as e:
                            ev, err = None, f"{type(e).__name__}: {e}"
                        rows.append({**base_row(level, seed, args),
                                     "skipped": None, "skip_detail": None,
                                     "prompt": prompt,
                                     "optimal_action_seq": opt_seq,   # efficiency 기준선
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
            print(f"[{level}] {have}/{target} appended -> {out}")

    # ── 요약 (파일 전체 기준, 이전 run 포함) ──────────────────────────
    rows = read_rows(out)
    for lv in args.levels:
        rs = [r for r in rows if r["env_name"] == lv]
        if not rs:
            continue
        sk = Counter(r["skipped"] for r in rs if r.get("skipped"))
        ev = [r for r in rs if not r.get("skipped")]
        if not ev:
            print(f"{lv}: 0 evaluated, skipped {sum(sk.values())} {dict(sk)}")
            continue
        ok = sum((r["eval_result"] or {}).get("CR", 0) == 1 for r in ev)
        effs = [(r["eval_result"] or {})["llm_efficiency"] for r in ev
                if (r["eval_result"] or {}).get("CR", 0) == 1
                and "llm_efficiency" in (r["eval_result"] or {})]
        eff = sum(effs) / len(effs) if effs else 0.0
        # unparsed 를 반드시 따로 센다 — plan 은 파싱 실패가 예외를 안 던져서
        # eval_error 에 안 잡히고 CR=0 에 섞인다.
        print(f"{lv}: {len(ev)} eps | CR {ok}/{len(ev)} ({ok/len(ev):.0%})  "
              f"eff(succ) {eff:.2f}  "
              f"unparsed {sum(not r['parsed_llm_output'] for r in ev)}  "
              f"eval_error {sum(r['eval_error'] is not None for r in ev)}  "
              f"truncated {sum(r['truncated'] for r in ev)}  "
              f"skipped {sum(sk.values())} {dict(sk)}")


if __name__ == "__main__":
    main()
