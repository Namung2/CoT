r"""
Qwen 설정 한 곳. generate.py 와 encode.py 가 공유한다.

왜 별도 모듈인가
    (1) 모델 이름이 두 곳에 있으면 안 된다.
        qwen 셀이 "자기생성"인 근거는 **생성 모델 == 인코딩 모델**이라는 것뿐이다
        (CTRLS Algorithm 1: E_t 는 그 CoT 를 만든 모델의 표현). 한쪽만 갈아끼우면
        qwen 셀이 조용히 교차생성이 되고 generator 축 자체가 무의미해진다.
    (2) prefix 렌더가 두 곳에 있으면 안 된다.
        prefix 는 생성 시(generate.py)와 인코딩 시(encode.py) 두 번 렌더된다.
        둘이 문자 단위로 같아야 prompt_len 이 맞고, 그래야 E_t 가 옳은 토큰을 문다.
        렌더 인자가 한쪽에서만 바뀌는 사고를 구조적으로 막는다.

    무거운 import 를 하지 않는다 (tok 을 인자로 받는다). claude 생성 경로가
    transformers/torch 없이 도는 성질을 유지하기 위해서다.

enable_thinking
    Qwen3(2504) hybrid 모델의 chat template 스위치. 기본값이 True 다.
    **이것은 생성 옵션이 아니라 prefix 문자열을 바꾼다:**

        True   ...<|im_start|>assistant\n
        False  ...<|im_start|>assistant\n<think>\n\n</think>\n\n

    False 는 빈 thinking 블록을 prefix 에 주입해 모델이 사고를 건너뛰게 만든다.
    즉 두 설정은 prompt_len 이 다르다. 생성과 인코딩이 다른 값을 쓰면 E_t 가 옆
    토큰을 문다 (encode.py 의 span_ok 가 잡지만, 애초에 어긋나지 않는 편이 낫다).

    항상 False 다. Claude 에 thinking={"type": "disabled"} 를 준 것과 같은 이유 —
    우리는 CoT 텍스트 자체를 인코딩하는데, 추론이 thinking block 으로 빠지면 남는
    것은 추론이 끝난 뒤의 결론뿐이다.

    Qwen3-*-Instruct-2507 이나 Qwen2.5 처럼 thinking 이 없는 모델의 템플릿은 이
    변수를 참조하지 않는다. Jinja 는 미사용 변수를 무시하므로 무해한 no-op 다.
    모델을 바꿨다면 실제 렌더를 한 번 확인할 것:

        python -c "from transformers import AutoTokenizer; \
                   from datagen.qwen import MODEL, render_prefix; \
                   print(repr(render_prefix(AutoTokenizer.from_pretrained(MODEL), \
                         [{'role':'user','content':'hi'}])[-140:]))"

디코딩 주의
    Qwen3 는 greedy decoding 을 권장하지 않는다 (무한 반복). generate.py 의 기본은
    아직 greedy 이므로, 파일럿에서 score.py 의 4-gram Jaccard 게이트가 WARN 을
    띄우면 --decoding sample 로 전환한다. non-thinking 권장값은
    temperature=0.7 / top_p=0.8 / top_k=20 이다.
"""
from __future__ import annotations

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ENABLE_THINKING = False


def render_prefix(tok, messages: list[dict]) -> str:
    """assistant 턴 직전까지의 prefix. 생성/인코딩 공용 — 유일한 렌더 지점."""
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
