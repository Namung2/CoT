"""
raw 데이터 생성. 텍스트만 만든다. 파싱도 분절도 라벨링도 하지 않는다.

생성기 축
    qwen    자기생성.  생성 모델 = 인코딩 모델. CTRLS Algorithm 1 이 전제하는 구성
            (E_t 는 "token embeddings from P_omega", 즉 그 CoT 를 만든 모델의 것).
    claude  교차생성.  Claude 가 쓴 CoT 를 Qwen 이 읽고 인코딩한다.
            CTRLS Assumption 5.2 가 문자 그대로는 성립하지 않는다. 대신
            **인코더를 고정한 채 텍스트 저자만 바꾸는 대조**가 생긴다.

왜 축으로 두는가
    B1 이 해석 불가였던 이유는 P_e(Qwen)와 P_surf(TF-IDF)가 서로 다른 인코더라
    "정보 내용 차이"와 "인코더 품질 차이"가 분리되지 않는 것이었다.
    qwen vs claude 는 **인코더가 같다.** 다른 것은 텍스트를 누가 썼느냐뿐이다.
      두 조건에서 결과가 같으면   -> e_t 는 읽은 것이지 추적한 것이 아니다
      qwen 이 유의하게 높으면     -> 생성 과정에서만 생기는 정보가 있다

프롬프트 동일성
    두 생성기에 **같은 단일 user 문자열**을 준다. 다르면 텍스트 차이가 모델 차이인지
    프롬프트 차이인지 갈리지 않는다. 렌더링만 다르다 — Qwen 은 chat template 을
    적용하고, Claude 는 messages 에 그대로 넣는다.
    system 파라미터도 assistant prefill 도 쓰지 않는다. 레퍼런스 어느 쪽도
    system/user 를 나누지 않고(LLM-BabyBench 는 문자열 하나, BALROG 는 전부
    role="user"), Claude 4.6+ 는 prefill 을 지원하지 않는다.

저장 원칙
    생성 단계에는 재계산 불가능한 것만 남긴다.
    token_ids 는 여기서 만들지 않는다 — 인코딩 모델의 tokenizer 소관이고
    텍스트로부터 결정론적으로 복원되므로 encode.py 가 담당한다.

사용
    python -m datagen.generate --cell P2 --generator claude --prompt bb     --seeds 0:20
    python -m datagen.generate --cell P2 --generator claude --prompt balrog --seeds 0:20
    python -m datagen.generate --cell P1 --generator claude --prompt bb     --seeds 0:20
    python -m datagen.generate --cell P2 --generator qwen   --prompt bb     --seeds 0:20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .babyai_env import ACTIONS, make_episode
from .prompt import PROBE_SEED, PROMPTS, build_messages, prompt_fingerprint
from .qwen import ENABLE_THINKING, MODEL as QWEN_MODEL, render_prefix

DEFAULT_LEVEL = "BabyAI-PutNextLocalS6N4-v0"

CLAUDE_MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------- 생성기


class QwenGenerator:
    """자기생성. chat template + assistant prefix 로 이어 쓴다."""

    name = "qwen"

    def __init__(self, model_name: str, dtype: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map=device
        ).eval()
        self.device = next(self.model.parameters()).device

    def meta(self, args) -> dict:
        return {
            "generator": "qwen", "gen_model": self.model_name, "dtype": self.dtype,
            # prefix 문자열을 바꾸는 값이다. encode.py 가 같은 값으로 렌더해야
            # prompt_len 이 맞으므로 반드시 남긴다.
            "enable_thinking": ENABLE_THINKING,

            "decoding": {"mode": args.decoding, "temperature": args.temperature,
                         "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
                         "seed_policy": "torch.manual_seed(seed) per generate, batch=1"},
        }

    def __call__(self, msgs: list[dict], seed: int, args) -> tuple[str, dict]:
        """(assistant 턴 전체 내용, 종료 정보).

        assistant prefill 은 쓰지 않는다. Claude 4.6 이후 모델이 지원하지 않으므로
        (400), Qwen 만 prefill 을 쓰면 두 셀의 prefix 가 달라져 generator 축 대조가
        교란된다. CoT 유발은 프롬프트 본문의 COT_TRIGGER/COT_SCAFFOLD 가 맡는다.

        prefix 렌더는 qwen.render_prefix 하나뿐이다. encode.py 가 같은 함수를 쓰므로
        생성 시와 인코딩 시의 prefix 가 문자 단위로 같다.
        """
        torch = self.torch
        text = render_prefix(self.tok, msgs)

        ids = self.tok(text, add_special_tokens=False).input_ids

        kw = dict(max_new_tokens=args.max_new_tokens,
                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        if args.decoding == "greedy":
            kw.update(do_sample=False)
        else:
            torch.manual_seed(seed)          # batch=1 이므로 이것으로 충분
            kw.update(do_sample=True, temperature=args.temperature, top_k=args.top_k)

        with torch.no_grad():
            out = self.model.generate(
                torch.tensor([ids], device=self.device), **kw)[0].tolist()
        gen = out[len(ids):]
        return self.tok.decode(gen, skip_special_tokens=True), {
            "stop_reason": "max_tokens" if len(gen) >= args.max_new_tokens else "stop",
            "truncated": len(gen) >= args.max_new_tokens,
            "out_tokens": len(gen),
        }


class ClaudeGenerator:
    """교차생성.

    API 제약 세 가지를 지킨다 (Claude 4.6 이후 / Sonnet 5, Opus 5).
      * sampling parameter (temperature / top_p / top_k) 를 비기본값으로 주면 400.
        -> 아예 보내지 않는다. 결정론적 재현은 불가능하고, 재현의 단위는
           **저장된 텍스트**다.
      * assistant prefill 미지원 (400). 요청은 user 메시지로 끝나야 한다.
        -> CoT 유발은 프롬프트 본문이 맡는다 (COT_TRIGGER / COT_SCAFFOLD).
      * thinking 이 **기본 on** 이다. 그대로 두면 추론이 thinking block 으로 빠지고
        text block 에는 추론이 끝난 뒤의 답만 남는다. 우리는 CoT 텍스트 자체를
        인코딩하므로 반드시 꺼야 한다.
        -> thinking={"type": "disabled"}
    """

    name = "claude"

    def __init__(self, model_name: str):
        import anthropic
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def meta(self, args) -> dict:
        return {"generator": "claude", "gen_model": self.model_name,
                "decoding": {"sampling_params": "미전송 (모델이 비기본값을 거부)",
                             "thinking": "disabled",
                             "max_tokens": args.max_new_tokens,
                             "seed_policy": "없음. 재현 단위는 저장된 텍스트"}}

    def __call__(self, msgs: list[dict], seed: int, args) -> tuple[str, dict]:
        # 프롬프트는 단일 user 메시지다 (레퍼런스에 system/user 분리가 없다).
        # system 파라미터를 쓰지 않으므로 Qwen 경로와 내용이 문자 단위로 같다.
        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=args.max_new_tokens,
            thinking={"type": "disabled"},
            messages=msgs,
        )
        # thinking 을 껐으므로 text block 이 곧 CoT 다. 혹시 thinking block 이
        # 섞여 오면 조용히 버리지 않고 드러낸다.
        kinds = {b.type for b in resp.content}
        if kinds - {"text"}:
            raise RuntimeError(f"예상 밖 content block: {sorted(kinds)}")
        # stop_reason 을 반드시 남긴다. "max_tokens" 면 답 블록이 통째로 잘려
        # score.py 에서 no_answer 로 보이는데, 포맷 불이행과 절단은 전혀 다른 사건이다.
        return "".join(b.text for b in resp.content), {
            "stop_reason": resp.stop_reason,
            "truncated": resp.stop_reason == "max_tokens",
            "out_tokens": resp.usage.output_tokens,
        }


# ---------------------------------------------------------------- 메인


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="P2", choices=["P1", "P2"],
                    help="P1=partial(GLAM) / P2=full(LLM-BabyBench Structured)")
    ap.add_argument("--generator", default="claude", choices=["qwen", "claude"])
    ap.add_argument("--prompt", default="bb", choices=list(PROMPTS),
                    help="bb=LLM-BabyBench Plan x cot / balrog=BALROG BabyAI x CoT")
    ap.add_argument("--seeds", default="0:20", help="시작:끝 (끝 미포함)")
    ap.add_argument("--level", default=DEFAULT_LEVEL)
    ap.add_argument("--gen-model", default=None, help="기본은 생성기별 기본값")
    ap.add_argument("--decoding", default="greedy", choices=["greedy", "sample"],
                    help="qwen 전용")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="qwen 전용. claude 는 sampling parameter 를 보내지 않는다")
    ap.add_argument("--top-k", type=int, default=50, help="qwen 전용")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="1024 는 bb 변형에서 실측 절단됨. Sonnet 5 는 새 tokenizer 로 "
                         "같은 텍스트에 약 30%% 더 많은 토큰을 쓴다")
    ap.add_argument("--dtype", default="bfloat16", help="qwen 전용")
    ap.add_argument("--device", default="cuda", help="qwen 전용")
    ap.add_argument("--out", default=None,
                    help="기본 data/{cell}-{generator}-{prompt}")

    args = ap.parse_args()

    obs_mode = {"P1": "partial", "P2": "full"}[args.cell]
    s0, s1 = (int(x) for x in args.seeds.split(":"))
    out = Path(args.out or f"data/{args.cell}-{args.generator}-{args.prompt}")
    out.mkdir(parents=True, exist_ok=True)
    raw_path, man_path = out / "raw.jsonl", out / "manifest.json"

    # 프롬프트 지문은 생성기와 무관하다 — 두 생성기가 같은 문자열을 받으므로
    # 셀 간·생성기 간 프롬프트 동일성을 이 값으로 대조할 수 있다.
    phash = prompt_fingerprint(args.level, obs_mode, args.prompt)
    # 모델 이름을 가드보다 먼저 확정한다. 지문은 프롬프트만 덮으므로(템플릿·미션
    # 배치·관측 렌더러) 모델을 갈아끼워도 지문은 그대로다. 따로 비교하지 않으면
    # 기존 seed 는 done 으로 스킵되고 새 seed 만 새 모델이 채워서, 한 raw.jsonl 에
    # 두 모델의 CoT 가 조용히 섞인다.
    model_name = args.gen_model or (
        QWEN_MODEL if args.generator == "qwen" else CLAUDE_MODEL)


    done = set()
    if raw_path.exists():
        done = {json.loads(l)["id"] for l in open(raw_path) if l.strip()}
    if man_path.exists() and done:
        old = json.loads(man_path.read_text())
        for key, now in (("prompt_fingerprint", phash), ("gen_model", model_name)):
            was = old.get(key)
            if was and was != now:
                raise SystemExit(
                    f"\n[중단] {out} 에 구 데이터가 있는데 {key} 가 바뀌었습니다.\n"
                    f"       manifest={was}\n"
                    f"       현재    ={now}\n"
                    f"       (지문은 seed={PROBE_SEED} 를 실제로 렌더한 결과라\n"
                    f"        템플릿뿐 아니라 관측 렌더러 변경도 잡습니다. 다만\n"
                    f"        모델까지는 못 덮으므로 gen_model 을 따로 봅니다.)\n"
                    f"       이어붙이려는 것이었다면:  mv {out} {out}_v0\n")

    gen = (QwenGenerator(model_name, args.dtype, args.device)
           if args.generator == "qwen" else ClaudeGenerator(model_name))


    man_path.write_text(json.dumps({
        "cell": args.cell, "level": args.level, "obs_mode": obs_mode,
        "prompt_variant": args.prompt,
        "prompt_fingerprint": phash, "fingerprint_probe_seed": PROBE_SEED,
        "actions": ACTIONS, "assistant_prefill": None,
        "seed_range": [s0, s1],
        **gen.meta(args),
        "minigrid_version": __import__("minigrid").__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=2))

    t0, n = time.time(), 0
    with open(raw_path, "a") as f:
        for seed in range(s0, s1):
            rid = f"{args.level}|{seed}"
            if rid in done:
                continue
            ep = make_episode(args.level, seed, obs_mode)
            msgs = build_messages(ep["mission"], ep["obs_text"], obs_mode, args.prompt)
            assistant_text, stop = gen(msgs, seed, args)

            f.write(json.dumps({
                # id 는 level|seed. 셀·생성기는 디렉토리로 갈리므로 넣지 않는다 —
                # 그래야 P1/P2, qwen/claude 를 같은 seed 로 조인할 수 있다.
                "id": rid, "level": args.level, "seed": seed,
                "obs_mode": obs_mode, "generator": gen.name,
                "prompt_variant": args.prompt,
                "mission": ep["mission"],
                "messages": msgs,                  # 렌더 전 원문. 생성기 무관
                "assistant_text": assistant_text,  # assistant 턴 전체 = CoT + 답
                "n_chars": len(assistant_text),
                **stop,
            }, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print(f"[{seed:5d}] {len(assistant_text):5d} chars  "
                  f"{stop['out_tokens']:5d} tok  {stop['stop_reason']:10s} "
                  f"({n} written)", flush=True)

    print(f"\n{n} episodes in {time.time()-t0:.0f}s -> {raw_path}")


if __name__ == "__main__":
    main()