from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen3-32B"
MAX_TOKENS = 32768                       # 초과 시 skip (meta에 기록)
HEAVY_FIELDS = ("prompt", "all_llm_output", "parsed_llm_output")

tok = None
model = None


def ensure_model():
    global tok, model
    if model is None:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(  # lm_head 없음 → logits 미계산
            MODEL, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        print("device:", model.device)
    return tok, model


# ------------------------------------------------------------- step boundaries

STEP_PAT = re.compile(r"(?mi)^(?:#+\s*|\*+\s*)?Step\s*(\d+)\s*[.:]")
MAX_STEPS = 6

# 정답을 뱉는 문장(터미널). llms/utils.py 의 parser() 와 같은 마커지만,
# 마크다운 변형(### / ** 접두, **...is**: 처럼 볼드가 콜론 앞에서 닫힘)과
# 곡선 아포스트로피까지 흡수한다.
_APOS = r"['’ʼ´`]"
_SEP = r"[\s*_]+"                        # 단어 사이에 낀 볼드 마커/공백
TERMINAL_PAT = {
    "plan": re.compile(
        rf"(?i)the{_SEP}llm\s*{_APOS}?\s*s{_SEP}action{_SEP}sequence{_SEP}is\s*\**\s*:"),
    "predict": re.compile(
        rf"(?i)the{_SEP}agent\s*{_APOS}?\s*s{_SEP}final{_SEP}state{_SEP}is\s*\**\s*:"),
    "decompose": re.compile(r"(?i)<+\s*start\s*>+"),
}
END_PAT = re.compile(r"(?i)<+\s*end\s*>+")     # decompose 전용


def _line_start(text: str, i: int) -> int:
    return text.rfind("\n", 0, i) + 1


def _segment(text: str, task: str):
    """공통 구조 검사. 성공: (heads, term, term_end, starts, None). 실패: (None,)*4 + (사유,).

    heads  = STEP_PAT 매치 목록 (Step 1..N 순서)
    term   = 터미널 문장이 시작하는 줄의 첫 문자 인덱스
    term_end = 터미널 마커 매치가 끝나는 문자 인덱스 (형식 토큰 수를 셀 때 씀)
    starts = 각 헤더가 있는 줄의 시작 (starts[0] 은 0)
    """
    fail = (None, None, None, None)
    if not text or not text.strip():
        return *fail, "empty_output"

    try:
        term_pat = TERMINAL_PAT[task]
    except KeyError:
        raise ValueError(f"no terminal marker defined for task {task!r}") from None

    marks = list(term_pat.finditer(text))
    if not marks:
        return *fail, "no_terminal_marker"
    # 마커가 여러 번 나오면 마지막 것. 마커가 걸린 줄을 통째로 터미널에 준다
    # (### / ** 접두가 마지막 스텝 쪽에 남지 않게).
    term = _line_start(text, marks[-1].start())

    if task == "decompose":
        ends = list(END_PAT.finditer(text, marks[-1].end()))
        if not ends:
            return *fail, "no_end_marker"
        if text[ends[-1].end():].strip():
            return *fail, "text_after_end"

    heads = list(STEP_PAT.finditer(text))
    if not heads:
        return *fail, "no_step_header"
    if text[:heads[0].start()].strip():
        return *fail, "preamble"

    nums = [int(m.group(1)) for m in heads]
    if nums != list(range(1, len(nums) + 1)):   # Step 1,2,4 처럼 끊기면 버린다
        return *fail, "step_sequence_break"
    if len(nums) > MAX_STEPS:
        return *fail, "too_many_steps"

    starts = [_line_start(text, m.start()) for m in heads]
    starts[0] = 0                               # 첫 헤더 앞 공백은 Step 1 이 흡수
    if starts[-1] >= term:                      # 헤더와 정답 마커가 같은 줄 → 길이 0 스텝
        return *fail, "step_in_terminal"
    return heads, term, marks[-1].end(), starts, None


def step_char_bounds(text: str, task: str) -> tuple[list[int] | None, str | None]:
    """원문 기준 구간 경계(문자 인덱스). 헤더 줄을 **포함**한 채로 쪼갠다.

    성공: (bounds, None). bounds[0] == 0, bounds[-1] == len(text) 이고
          마지막 구간이 정답을 뱉는 터미널 문장, 그 앞이 Step 1..N.
    실패: (None, 사유) — 호출부에서 그 궤적을 통째로 skip.

    깨끗한 궤적만 남긴다. 헤더가 없거나, Step 1 앞에 서문이 있거나,
    스텝 번호가 1,2,3.. 으로 안 이어지거나, 스텝이 MAX_STEPS 를 넘으면 버린다.

    정규식은 출력 텍스트에만 건다. 프롬프트에도 "Step 1." 류 목차가 들어 있어서
    (cot_*.py 의 프롬프트가 6단계 지시문) 이어붙인 전체 텍스트에 걸면 프롬프트
    쪽 헤더가 먼저 잡힌다.

    추출(forward pass)에는 이 함수가 아니라 "Step N" 마커를 벗긴 prepare_output 을 쓴다.
    이 함수는 원문을 그대로 보는 통계·점검용이다 (script/header_stats.py).
    """
    heads, term, _, starts, reason = _segment(text, task)
    if reason is not None:
        return None, reason
    bounds = starts + [term, len(text)]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        return None, "empty_segment"
    return bounds, None


@dataclass
class Cleaned:
    """"Step N" 마커를 벗긴 출력과 그 좌표계의 경계.

    text     : "### Step N." 류 마커만 빠진 출력 (헤더 줄의 제목·본문은 남는다). 모델에는 이것이 들어간다.
    bounds   : text 기준 [0, step2 시작, ..., stepN 시작, 터미널 시작, len(text)]
    term_end : text 기준 터미널 마커 매치 끝 (spectral.py 가 터미널 형식 토큰 수를 셀 때)
    n_header_chars : 벗겨낸 문자 수 (meta 기록용)
    """
    text: str
    bounds: list[int]
    term_end: int
    n_header_chars: int


_AFTER_MARK = re.compile(r"[*_]*[ \t]*")    # 마커 뒤에 남는 볼드 닫힘("**Step 3:**" 의 **) 과 가로 공백


def prepare_output(text: str, task: str) -> tuple[Cleaned | None, str | None]:
    """출력에서 "Step N" **마커만** 지우고, 지운 좌표계로 경계를 만든다.

    마커는 구간을 어디서 자를지 정하는 구분자로만 쓰고 forward pass 에서는 뺀다.
    프로빙 때 "Step N" 토큰을 보고 스텝 번호를 맞히는 치팅을 막기 위해서다.

    지우는 범위는 STEP_PAT 매치("### Step 3." / "**Step 3:" 등) 와 그 뒤의 볼드 닫힘·가로 공백까지.
    같은 줄의 나머지(제목이든 본문이든) 는 남긴다. 예전엔 헤더 줄 전체를 지웠는데, predict 는
    "Step 1: The agent starts at (3, 4)..." 처럼 본문이 헤더 줄에 붙고 decompose 는 Step 6 이
    제목 한 줄이라 전체의 15~70% 가 empty_segment 로 빠졌다. 제목("Identify the mission goal")
    이 프롬프트가 지시한 고정 문구라 스텝 번호를 드러내는 문제는 남지만, 그건 본문 내용도
    마찬가지이므로 프로빙 쪽에서 대조 실험(초반 층·cross-level)으로 가린다.

    마커만 있던 줄("Step 2: \n")은 줄 끝 개행과 뒤따르는 빈 줄까지 지워 구간이 본문 첫 글자에서
    시작하게 한다. 남겨두면 본문이 "\n" 으로 시작해 앞 구간 끝의 "\n\n" 과 한 토큰으로 묶여
    boundary_straddle 로 버려진다. 마커 뒤에 아무것도 없는 스텝은 길이 0 → "empty_segment".
    """
    heads, term, term_end, starts, reason = _segment(text, task)
    if reason is not None:
        return None, reason

    spans = []                                   # 지울 [ls, le)
    for m in heads:
        ls = m.start()                           # STEP_PAT 은 줄 머리(^)에서만 맞으므로 줄 시작과 같다
        le = _AFTER_MARK.match(text, m.end()).end()
        nl = text.find("\n", le)
        line_end = nl + 1 if nl >= 0 else len(text)
        if not text[le:line_end].strip():        # 마커만 있던 줄: 개행 + 뒤따르는 빈 줄까지
            le = line_end
            while le < len(text):
                nl = text.find("\n", le)
                end = nl + 1 if nl >= 0 else len(text)
                if text[le:end].strip():
                    break
                le = end
        spans.append((ls, le))
    # 마커는 모두 term 앞에서 끝난다 (starts[-1] < term 이고 빈 줄 소거는 term 줄 앞에서 멈춘다).
    assert spans[-1][1] <= term, "header marker crosses terminal"

    def mapped(p: int) -> int:                   # 원문 위치 → 지운 좌표계
        return p - sum(le - ls for ls, le in spans if le <= p)

    pieces, prev = [], 0
    for ls, le in spans:
        pieces.append(text[prev:ls])
        prev = le
    pieces.append(text[prev:])
    clean = "".join(pieces)

    clean_starts = [mapped(ls) for ls, _ in spans]
    clean_starts[0] = 0                          # 원문처럼 첫 마커 앞 공백은 Step 1 이 흡수
    bounds = clean_starts + [mapped(term), len(clean)]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        return None, "empty_segment"
    return Cleaned(clean, bounds, mapped(term_end), len(text) - len(clean)), None


def char_to_token_bounds(char_bounds: list[int], offsets) -> list[int]:
    tok_bounds, k = [], 0
    for cb in char_bounds:
        while k < len(offsets) and offsets[k][0] < cb:
            k += 1
        tok_bounds.append(k)
    return tok_bounds


# ---------------------------------------------------------------- tokenization

def render_prompt(episode: dict) -> str:
    """생성 시점에 모델이 실제로 본 입력을 재현한다.

    generation/scripts/cot_*.py 는 apply_chat_template(add_generation_prompt=True)
    결과를 vLLM 에 넘기고 jsonl 에는 raw 프롬프트만 저장한다. 그대로 쓰면
    <|im_start|> 류 특수 토큰과 (no_thinking 의) 빈 <think> 블록이 통째로 빠진다.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": episode["prompt"]}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(episode.get("thinking", False)),
    )


def tokenize_text(prompt: str, output: str, char_bounds: list[int]):
    """prompt + output 을 통째로 토크나이즈하고 output 기준 문자 경계를 토큰 경계로 바꾼다.

    반환 (enc, tok_bounds, 사유). tok_bounds = [0, 프롬프트 끝, ...char_bounds 를 옮긴 것].
    extract 와 spectral.py 가 같은 함수를 써야 토큰 수·경계가 일치한다.
    """
    enc = tok(prompt + output, add_special_tokens=False, return_offsets_mapping=True)
    abs_bounds = [0] + [len(prompt) + b for b in char_bounds]   # char_bounds[0] == 0

    # 경계 양쪽 글자가 한 토큰으로 묶이면 경계를 토큰 단위로 못 그어서 한쪽 첫 토큰이
    # 조용히 앞 구간에 먹힌다. 프롬프트/출력 이음매는 템플릿이 \n\n 로 끝나 보통 안 생기지만,
    # 마커를 벗긴 뒤의 스텝 이음매("...\n\n" + 본문 첫 글자)도 같은 위험이 있어 전부 검사한다.
    cuts = set(abs_bounds[1:-1])
    if any(s < c < e for s, e in enc.offset_mapping for c in cuts if s < c):
        return None, None, "boundary_straddle"

    tok_bounds = char_to_token_bounds(abs_bounds, enc.offset_mapping)
    assert tok_bounds[-1] == len(enc.input_ids), "unconsumed tokens"
    return enc, tok_bounds, None


def tokenize_episode(episode: dict, cleaned: Cleaned):
    """프롬프트 + ("Step N" 마커를 벗긴) 출력을 토크나이즈하고 토큰 경계를 만든다.

    boundaries 규약 (길이 N+3):
        [0, 프롬프트 끝, step1 끝, ..., stepN 끝, 전체 끝]
    즉 첫 구간 = 프롬프트, 마지막 구간 = 정답 문장. 호출부가 이 규약을 안다고 가정한다.
    스텝 구간에는 "Step N" 마커 토큰이 없다 (prepare_output 참고). 헤더 줄의 제목·본문은 남는다.

    반환: (input_ids, tok_bounds, 사유). 사유가 있으면 그 궤적은 skip.
    """
    ensure_model()
    enc, tok_bounds, reason = tokenize_text(render_prompt(episode), cleaned.text, cleaned.bounds)
    if reason is not None:
        return None, None, reason
    return enc.input_ids, tok_bounds, None


# ------------------------------------------------------------------- extractor

@torch.no_grad()
def extract_hidden(ids):
    ensure_model()
    H = model(torch.tensor([ids], device=model.device)).last_hidden_state[0]
    return H.to(torch.bfloat16).cpu().clone()   # 프롬프트 구간 포함 전체


# ------------------------------------------------------------------------ meta

def build_meta(episode: dict) -> dict:
    return {k: v for k, v in episode.items() if k not in HEAVY_FIELDS}


# 앞에 있는 키를 먼저 쓴다 — decompose→PR, predict→success, plan→CR.
LABEL_PRIORITY = ("PR", "success", "CR")


def episode_status(episode: dict) -> tuple[str | None, str | None]:
    """(status, 판정에 쓴 키). 판정 불가면 (None, None).

    PR 을 먼저 보는 이유: decompose 의 CR 은 봇이 서브골을 추가해서라도 완주하면
    1 이라 "성공" 안에 LLM 분해가 불완전한 궤적이 섞인다 (GoTo 에서 CR 91% vs
    PR 15%). PR 은 봇 추가 0회만 성공으로 친다.

    eval_error 가 있는 궤적은 호출부에서 이미 걸렀다고 가정한다. CR·ACI 등
    나머지 지표는 meta 의 eval_result 에 그대로 남으므로 나중에 재분류할 수 있다.
    """
    r = episode.get("eval_result") or {}
    for key in LABEL_PRIORITY:
        if key in r:
            ok = bool(r[key]) if key == "success" else r[key] == 1
            return ("success" if ok else "failure"), key
    return None, None


# ------------------------------------------------------------------------- run

def load_episodes(path: Path) -> list[dict]:

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    episodes = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def process_episode(episode: dict, task: str, src_name: str) -> tuple[dict, dict | None]:
    """에피소드 하나: skip 판정 → 라벨 → 경계 → 토크나이즈 → hidden state.

    반환 (meta, payload). payload 가 None 이면 skip 이고 사유는 meta["extract_skipped"].
    payload 는 청크에 들어가는 dict — {"E", "boundaries", "output_sha1"}.
    출력의 "Step N" 헤더 줄은 모델 입력에서 빠지므로 E 와 boundaries 는 헤더가
    없는 시퀀스 기준이다. output_sha1 은 원문 기준 (원본 jsonl 과 대조하는 열쇠).
    extract_run 과 script/resume_extract.py 가 같이 쓴다 (한 곳만 고치면 되게)."""
    meta = build_meta(episode)
    meta["src"] = src_name

    def skip(reason):
        meta["extract_skipped"] = reason
        return meta, None

    if episode.get("skipped"):               # 출력 자체가 없는 에피소드
        return skip("no_output")
    if episode.get("truncated"):             # max_tokens 에서 잘림 → 마지막 구간 불완전
        return skip("truncated")
    if episode.get("eval_error") is not None:
        # 파싱 실패 / 봇 실행 예외 / eval 타임아웃이 한 필드에 섞여 있다.
        # 타임아웃은 봇 리플랜 루프가 안 끝나는 환경 문제라 LLM 실패가
        # 아니고, 파싱 실패는 어차피 step_char_bounds 가 거른다. 통째로 뺀다.
        return skip("eval_error")
    if not (episode.get("prompt") or "").strip():
        return skip("empty_prompt")          # 프롬프트 구간이 길이 0이 된다

    status, label_key = episode_status(episode)
    if status is None:
        return skip("no_label")              # eval_result 에 판정 키가 없다
    meta["status"] = status
    meta["label_key"] = label_key

    # 구조 검사는 토크나이즈 전에 — 안 맞는 궤적엔 모델을 안 태운다
    cleaned, reason = prepare_output(episode["all_llm_output"], task)
    meta["output_sha1"] = hashlib.sha1(            # 원문(헤더 포함) 기준 — 원본 jsonl 대조용
        (episode["all_llm_output"] or "").encode()).hexdigest()
    if reason is not None:
        return skip(reason)

    ids, boundaries, reason = tokenize_episode(episode, cleaned)
    if reason is not None:
        return skip(reason)
    meta.update(
        n_tokens_total=len(ids),
        n_tokens_prompt=boundaries[1],
        n_tokens_output=boundaries[-1] - boundaries[1],   # 헤더 제외
        n_tokens_terminal=boundaries[-1] - boundaries[-2],
        n_steps=len(boundaries) - 3,         # 프롬프트·터미널 제외
        n_header_chars=cleaned.n_header_chars,   # forward pass 에서 빠진 "Step N" 마커 문자 수
    )

    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        return skip("empty_token_segment")   # 문자로는 갈렸는데 토큰으로 뭉개진 구간
    if len(ids) > MAX_TOKENS:
        return skip("too_long")

    E = extract_hidden(ids)
    assert E.shape[0] == boundaries[-1]
    return meta, {
        "E": E,                          # (프롬프트+출력 토큰수) x d
        "boundaries": boundaries,        # [0, prompt, step1..N, terminal 끝]
        "output_sha1": meta["output_sha1"],
    }


def chunk_header() -> dict:
    return {"model": MODEL, "boundary_layout": "prompt|steps|terminal",
            "step_headers": "marker_stripped",   # "Step N" 마커만 입력에서 뺐다, 제목·본문은 남음 (prepare_output)
            "label_priority": LABEL_PRIORITY}


def extract_run(
    data_dir: Path,
    out_root: Path,
    task: str,
    level: str,
    mode: str = "no_thinking",
    chunk: int = 256,
    limit: int | None = None,
):

    src = data_dir / f"{task}_{mode}.jsonl"
    all_episodes = load_episodes(src)

    episodes = [e for e in all_episodes
                if e.get("task") == task and e.get("env_name") == level]
    if not episodes:
        raise ValueError(f"no episodes with task == {task!r} and env_name == {level!r} "
                         f"in {src} ({len(all_episodes)} loaded total)")

    if limit is not None:        # smoke test — skip 사유 분포와 청크 크기만 보고 끊는다
        episodes = episodes[:limit]
        print(f"limit={limit} (of {len(all_episodes)} loaded)")

    run_dir = out_root / task / level
    for status in ("success", "failure"):
        d = run_dir / status
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("chunk_*.pt"):   # 재실행 시 이전 청크 개수와 안 맞게 남는 것 방지
            old.unlink()
    meta_path = run_dir / "meta.jsonl"

    buffers = {"success": {}, "failure": {}}
    chunk_idx = {"success": 0, "failure": 0}
    seen_seeds = {"success": set(), "failure": set()}

    def flush(status):
        if not buffers[status]:
            return
        out_path = run_dir / status / f"chunk_{chunk_idx[status]:04d}.pt"
        torch.save({"episodes": buffers[status], **chunk_header()}, out_path)
        buffers[status] = {}
        chunk_idx[status] += 1

    n_saved = n_skipped = 0
    reasons = Counter()

    with meta_path.open("w", encoding="utf-8") as mf:

        def skip(meta, reason):
            nonlocal n_skipped
            meta["extract_skipped"] = reason
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            reasons[reason] += 1
            n_skipped += 1

        for episode in tqdm(episodes, desc="episodes", unit="episode"):
            meta, payload = process_episode(episode, task, src.name)
            if payload is None:
                skip(meta, meta["extract_skipped"])
                continue
            status = meta["status"]

            seed = episode["env_seed"]
            if seed in seen_seeds[status]:
                raise ValueError(f"id collision: {task}/{level}/{status} seed={seed}")
            seen_seeds[status].add(seed)
            buffers[status][seed] = payload
            meta["chunk"] = chunk_idx[status]        # 이 episode가 들어갈 청크 파일 인덱스
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_saved += 1

            if len(buffers[status]) >= chunk:
                flush(status)

    for status in buffers:
        flush(status)

    n_success = sum(1 for s in seen_seeds["success"] for _ in (0,))
    print(f"saved {n_saved} episodes under {run_dir} ({n_skipped} skipped)")
    print(f"  success {len(seen_seeds['success'])} / failure {len(seen_seeds['failure'])}"
          f"  [label_priority={LABEL_PRIORITY}]")
    for reason, n in reasons.most_common():
        print(f"  skipped {reason}: {n}")


# ----------------------------------------------------------------- load helper

def load_chunk(pt_path: Path) -> dict:

    return torch.load(pt_path, map_location="cpu", weights_only=False)["episodes"]


def load_all_views(episode: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """[프롬프트, step1..stepN, 터미널] 전 구간."""
    E, b = episode["E"], episode["boundaries"]
    return E, [E[s:e] for s, e in zip(b, b[1:])]


def load_step_views(episode: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """스텝 구간만 — 프롬프트(첫 구간)와 터미널(마지막 구간)은 뺀다."""
    E, b = episode["E"], episode["boundaries"]
    return E, [E[s:e] for s, e in zip(b[1:-2], b[2:-1])]


def gen_view(episode: dict) -> tuple[torch.Tensor, list[int]]:
    """프롬프트를 뺀 생성 구간. 반환 (E_gen, seg).

        seg = [0, step1 끝, ..., stepN 끝, 전체 끝]   — 0 기준으로 재정렬한 경계
        구간 = [step 1]...[step N][터미널]            — N+1 개

    프롬프트는 생성 시점 문맥으로만 필요했고 분석 대상이 아니다. 여기 있는 토큰이
    전체의 2/3 라 (플랜 프롬프트 ~1000토큰 vs 출력 ~500토큰) 히트맵·GSBS 에 그대로
    넣으면 실제 관심 구간이 구석으로 밀린다.

    heatmap.py / gsbs.py / spectral.py 가 전부 이 함수를 써야 세 결과의 좌표가
    맞물린다 — 각자 boundaries 를 손으로 자르면 조용히 어긋난다.
    """
    E, b = episode["E"], episode["boundaries"]
    gen0 = b[1]                                   # 프롬프트 끝 = 생성 구간 시작
    return E[gen0:], [x - gen0 for x in b[1:]]


def gen_views(episode: dict) -> tuple[torch.Tensor, list[int], list[torch.Tensor]]:
    """gen_view 를 구간 리스트까지 잘라서 돌려준다. (E_gen, seg, views)"""
    E, seg = gen_view(episode)
    return E, seg, [E[s:e] for s, e in zip(seg, seg[1:])]


def seg_labels(seg: list[int]) -> list[str]:
    """gen_view 의 seg 에 대응하는 구간 이름. ["Step 1", ..., "Step N", "answer"]"""
    return [f"Step {t}" for t in range(1, len(seg) - 1)] + ["answer"]