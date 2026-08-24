"""
인코딩 단계. raw.jsonl 의 텍스트를 **Qwen 토큰열**로 바꾼다. hidden state 는 뽑지 않는다.

왜 생성과 분리하는가
    e_t 는 인코딩 모델의 표현이다. 생성기가 qwen 이든 claude 든 인코딩은 항상
    같은 Qwen 이어야 두 조건이 비교 가능하다. 그러려면 prefix 가 **문자 단위로
    동일**해야 하는데, 그 조립을 한 곳에서 하는 것이 이 파일이다.

    Qwen chat template
      user      : messages[0]      (단일 user 메시지. prompt.build_messages 참조)


    -> prompt_len 은 assistant 턴 시작 직전까지의 토큰 수.
       token_ids[prompt_len:] 을 디코드하면 assistant_text 와 같아야 한다.

    렌더는 qwen.render_prefix 하나로 한다. generate.py 가 생성 시 쓴 것과 **같은
    함수**라야 prefix 가 문자 단위로 같다. 특히 Qwen3 hybrid 의 enable_thinking 은
    prefix 자체를 바꾸므로(빈 <think></think> 주입) 두 곳이 갈리면 prompt_len 이
    어긋난다. claude 셀도 같은 값으로 렌더한다 — 두 셀의 prefix 가 같아야 generator
    축이 텍스트 저자 차이만 보게 된다.

왜 hidden state 를 저장하지 않는가
    layer 는 스윕 대상인데 전 layer 를 다 저장하면 에피소드당 수백 MB 다.
    token_ids 가 있으면 원하는 layer 만 그때 forward 하면 된다.
    단 이는 **teacher-forcing 재투입이 생성 시 hidden state 와 같다**는 전제 위에
    있고(1-d/T1), 그 전제는 qwen 자기생성 셀에서만 검증 가능하다.
    claude 셀은 애초에 teacher-forcing 밖에 없으므로 T1 이 정의되지 않는다.

정합 검증
    토큰 경계가 한 칸만 밀려도 E_t 가 옆 블록 토큰을 문다.
    exact  : token_ids[prompt_len:] 디코드 == assistant_text
    prefix : token_ids[:prompt_len] 디코드 == 렌더된 prefix
    둘 다 참이어야 이후 분절이 의미를 갖는다. 깨지면 여기서 드러난다.

사용
    python -m datagen.encode --dir data/P2-claude
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .qwen import ENABLE_THINKING, MODEL as ENCODER_MODEL, render_prefix



def encode_one(tok, rec: dict) -> dict:
    """한 레코드를 Qwen 토큰열로. (token_ids, prompt_len, 정합 리포트)"""
    prefix_text = render_prefix(tok, rec["messages"])

    full_text = prefix_text + rec["assistant_text"]

    prompt_ids = tok(prefix_text, add_special_tokens=False).input_ids
    token_ids = tok(full_text, add_special_tokens=False).input_ids
    n_prompt = len(prompt_ids)

    # 경계에서 BPE 병합이 일어나면 prompt 를 따로 토큰화한 길이가 full 의 앞부분과
    # 어긋난다. 그 경우를 조용히 넘기지 않고 리포트한다.
    prefix_ok = token_ids[:n_prompt] == prompt_ids
    got_assistant = tok.decode(token_ids[n_prompt:], skip_special_tokens=False)
    exact = got_assistant == rec["assistant_text"]

    return {
        "id": rec["id"], "seed": rec["seed"],
        "generator": rec["generator"], "obs_mode": rec["obs_mode"],
        "token_ids": token_ids,
        "prompt_len": n_prompt,
        "n_assistant_tokens": len(token_ids) - n_prompt,
        "span_ok": bool(prefix_ok and exact),
        "prefix_ok": bool(prefix_ok),
        "exact": bool(exact),
        # thinking 이 꺼지지 않았으면 여기에 <think> 가 남는다. 그 경우 assistant_text
        # 는 CoT 가 아니라 "사고 블록 + 결론"이 되어 분절 전제가 깨진다.
        "think_leak": "<think>" in rec["assistant_text"],

        # 불일치일 때만 원문을 남긴다. 전부 남기면 파일이 두 배가 된다.
        "mismatch": None if exact else {
            "expected": rec["assistant_text"][:400],
            "got": got_assistant[:400],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--encoder", default=ENCODER_MODEL)
    args = ap.parse_args()

    d = Path(args.dir)
    tok = AutoTokenizer.from_pretrained(args.encoder)

    rows = []
    with open(d / "raw.jsonl") as f, open(d / "encoded.jsonl", "w") as g:
        for line in f:
            if not line.strip():
                continue
            r = encode_one(tok, json.loads(line))
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows.append(r)

    if not rows:
        raise SystemExit(f"[중단] {d / 'raw.jsonl'} 에 레코드가 없습니다.")

    # manifest 에 인코더를 기록한다. 생성기와 다를 수 있으므로 별도 키다.
    man = d / "manifest.json"
    m = json.loads(man.read_text()) if man.exists() else {}
    # qwen 셀은 생성 모델 == 인코딩 모델이라야 "자기생성"이다 (CTRLS Algorithm 1).
    # 어긋나면 조용히 교차생성이 되어 generator 축이 무의미해지므로 드러낸다.
    if m.get("generator") == "qwen" and m.get("gen_model") != args.encoder:
        print(f"[경고] 자기생성 전제 위반: gen_model={m.get('gen_model')!r} "
              f"!= encoder={args.encoder!r}")
    m["encoder"] = {"model": args.encoder,
                    "enable_thinking": ENABLE_THINKING,

                    "transformers_version": __import__("transformers").__version__}
    man.write_text(json.dumps(m, ensure_ascii=False, indent=2))

    n = len(rows)
    bad = [r for r in rows if not r["span_ok"]]
    leak = sum(r["think_leak"] for r in rows)
    nt = sorted(r["n_assistant_tokens"] for r in rows)
    print(f"=== {d}  n={n}  encoder={args.encoder} ===")
    print(f"  span_ok        {n - len(bad)}/{n}")
    if leak:
        print(f"  think leak     {leak}/{n}   (<think> 가 assistant_text 에 남음 "
              f"-> qwen.ENABLE_THINKING 이 먹지 않았다)")

    print(f"  prompt_len     median {sorted(r['prompt_len'] for r in rows)[n//2]}")
    print(f"  assistant tok  median {nt[n//2]}  [{nt[0]}, {nt[-1]}]")
    for r in bad[:3]:
        print(f"  -- 불일치 {r['id']}  prefix_ok={r['prefix_ok']} exact={r['exact']}")
        print(f"     expected {r['mismatch']['expected'][:120]!r}")
        print(f"     got      {r['mismatch']['got'][:120]!r}")
    print(f"-> {d/'encoded.jsonl'}")


if __name__ == "__main__":
    main()