"""LLM 추론 궤적(jsonl) → 구간별 hidden state 청크(.pt).

파이프라인 (extract_run):
    load_episodes → EpisodeExtractor.process (skip 판정 → 라벨 → 구간 경계 → 토크나이즈 → forward)
    → ChunkWriter (status 별 chunk_XXXX.pt) + meta.jsonl

구성 요소:
    StepSegmenter    출력 텍스트에서 Step 1..N + 정답 문장의 경계를 찾고 "Step N" 마커를 벗긴다.
    Encoder          토크나이저/모델을 lazy 로드. 프롬프트 렌더링, 토크나이즈, hidden state 추출.
    EpisodeExtractor 에피소드 하나를 (meta, payload) 로.
    ChunkWriter      payload 를 status 별로 모아 청크 파일로 쓴다.

저장 규약:
    payload = {"E": (토큰수 x d) bf16, "boundaries": [0, 프롬프트 끝, step1 끝, ..., stepN 끝, 전체 끝],
               "output_sha1": 원문 기준 sha1}
    boundaries 길이 = N+3. 첫 구간 = 프롬프트, 마지막 구간 = 정답 문장(터미널).
    스텝 구간에는 "Step N" 마커 토큰이 없다 (StepSegmenter.prepare 참고). 헤더 줄의 제목·본문은 남는다.
"""
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
MAX_STEPS = 6
HEAVY_FIELDS = ("prompt", "all_llm_output", "parsed_llm_output")
STATUSES = ("success", "failure")

# 태스크별 기대 step 수. 프롬프트가 지시한 개수와 다르면 skip (step_count_mismatch).
# plan 5 / decompose 6 가 각각 99% 이상이고, 나머지는 step 을 건너뛰거나 머리말에 번호를 붙인 궤적이다.
# predict 는 아직 데이터가 없어 검사하지 않는다.
EXPECTED_STEPS = {"plan": 5, "decompose": 6}

# 마지막 step 을 터미널에 합치는 태스크. decompose 의 Step 6 "Sequence the subgoals in logical order" 는
# 본문이 중앙값 1줄, 15% 는 제목뿐이라 사실상 <START>…<END> 의 머리말이다. 합치면 터미널이
# Step 6 줄부터 시작하고 step 구간은 1..5 가 된다. plan 의 Step 5 "Verify" 는 실질 본문이 있어 분리한다.
MERGE_LAST_STEP = {"decompose"}

# "Step N" 마커를 뗀 뒤 이보다 짧은 step 구간이 하나라도 있으면 skip (short_step).
# 제목만 남은 구간은 plan 에서 ≤9 토큰, decompose 에서 5~12 토큰이다.
MIN_STEP_TOKENS = 10

# 앞에 있는 키를 먼저 쓴다 — decompose→PR, predict→success, plan→CR.
LABEL_PRIORITY = ("PR", "success", "CR")


# ------------------------------------------------------------------ segmenter

@dataclass
class Segmentation:
    #원문 좌표계의 구간 정보
    heads: list[re.Match]       # "Step N" 헤더 매치 (N = 1..len). 마커 제거용이라 합쳐진 마지막 step 것도 포함
    starts: list[int]           # step 으로 남는 구간의 시작 인덱스 (starts[0] == 0). 합쳐진 마지막 step 은 빠진다
    term: int                   # 터미널이 시작하는 줄의 인덱스 (합치는 태스크는 마지막 step 헤더 줄, 아니면 정답 마커 줄)
    term_end: int               # 정답 마커 매치가 끝나는 인덱스
 

@dataclass
class Cleaned:
    # Step N" 마커를 벗긴 출력과 그 좌표계의 경계.    
    text: str                   # "### Step N."만 빠진 출력 (헤더 줄의 제목·본문은 남는다). 모델에는 이것이 들어간다.
    bounds: list[int]           # text 기준 [0, step2 시작, ..., stepN 시작, 터미널 시작, len(text)]
                                # 합치는 태스크(decompose)는 터미널이 마지막 step 헤더 줄에서 시작하고 step 구간은 N-1 개
    term_end: int               #text 기준 터미널 마커 매치 끝 (터미널 형식 토큰 수를 셀 때)
    n_header_chars: int         #벗겨낸 문자 수 (meta 기록용)


class StepSegmenter:
    """출력 텍스트를 Step 1..N + 정답 문장으로 쪼갠다. task 마다 정답 마커가 다르다.

    깨끗한 궤적만 남긴다. 
    1. Step N이 없음
    2. Step 1 앞에 서문 존재
    3. 스텝 번호가 1,2,3.. 으로 이어지지 않음
    4. 스텝이 max_steps 를 넘음
    5. 스텝 수가 태스크의 기대 개수(EXPECTED_STEPS)와 다름
    이면 사유를 붙여 버린다.

    MERGE_LAST_STEP 태스크는 마지막 step 줄부터를 터미널로 본다 (step 구간은 N-1 개).

    프롬프트에도 "Step 1."와 같은 목차가 들어 있어서, 정규식은 출력 텍스트에만 건다.
    """

    STEP_PAT = re.compile(r"(?mi)^(?:#+\s*|\*+\s*)?Step\s*(\d+)\s*[.:]") # Step N 마커 
    AFTER_MARK = re.compile(r"[*_]*[ \t]*")   # 마커 뒤 볼드 닫힘("**Step 3:**" 의 **)과 가로 공백

    _APOS = r"['’ʼ´`]"
    _SEP = r"[\s*_]+"
    TERMINAL_PAT = {
        "plan": re.compile( # The LLM's action sequence is:
            rf"(?i)the{_SEP}llm\s*{_APOS}?\s*s{_SEP}action{_SEP}sequence{_SEP}is\s*\**\s*:"),

        "predict": re.compile( #The agent's final state is: 
            rf"(?i)the{_SEP}agent\s*{_APOS}?\s*s{_SEP}final{_SEP}state{_SEP}is\s*\**\s*:"), #
            
        "decompose": re.compile( # <start> tage
            r"(?i)<+\s*start\s*>+"
            ),
    }
    END_PAT = re.compile(r"(?i)<+\s*end\s*>+")

    def __init__(self, task: str, max_steps: int = MAX_STEPS,
                 expected_steps: int | None = None, merge_last: bool | None = None):
        if task not in self.TERMINAL_PAT:
            raise ValueError(f"no terminal marker defined for task {task!r}")
        self.task = task
        self.term_pat = self.TERMINAL_PAT[task]
        self.max_steps = max_steps
        # None 이면 태스크 기본값 (EXPECTED_STEPS / MERGE_LAST_STEP)
        self.expected_steps = EXPECTED_STEPS.get(task) if expected_steps is None else expected_steps
        self.merge_last = (task in MERGE_LAST_STEP) if merge_last is None else merge_last

    @staticmethod
    def _line_start(text: str, i: int) -> int:
        return text.rfind("\n", 0, i) + 1

    def _segment(self, text: str) -> tuple[Segmentation | None, str | None]:
        #궤적 전체가 공백이면 버리기
        if not text or not text.strip():
            return None, "empty_output"

    
        marks = list(self.term_pat.finditer(text))
        if not marks:
            return None, "no_terminal_marker"
        # 마커가 여러 번 나오면 마지막 것. 마커가 걸린 줄을 통째로 터미널에 준다
        # (### / ** 접두가 마지막 스텝 쪽에 남지 않게).
        term = self._line_start(text, marks[-1].start())

        if self.task == "decompose":
            ends = list(self.END_PAT.finditer(text, marks[-1].end()))
            if not ends:
                return None, "no_end_marker"
            if text[ends[-1].end():].strip():
                return None, "text_after_end"

        heads = list(self.STEP_PAT.finditer(text))
        if not heads:
            return None, "no_step_header"
        if text[:heads[0].start()].strip():
            return None, "preamble"

        nums = [int(m.group(1)) for m in heads]
        if nums != list(range(1, len(nums) + 1)):   # Step 1,2,4 처럼 끊기면 버린다
            return None, "step_sequence_break"
        if len(nums) > self.max_steps:
            return None, "too_many_steps"
        if self.expected_steps is not None and len(nums) != self.expected_steps:
            return None, "step_count_mismatch"

        starts = [self._line_start(text, m.start()) for m in heads]
        starts[0] = 0                               # 첫 헤더 앞 공백은 Step 1 이 흡수
        if starts[-1] >= term:                      # 헤더와 정답 마커가 같은 줄 → 길이 0 스텝
            return None, "step_in_terminal"
        if self.merge_last:                         # 마지막 step 줄부터 터미널 (heads 는 마커 제거용으로 그대로)
            term = starts[-1]
            starts = starts[:-1]
        return Segmentation(heads, starts, term, marks[-1].end()), None

    def _marker_spans(self, text: str, seg: Segmentation) -> list[tuple[int, int]]:
        """지울 "Step N" 마커 범위 [ls, le). 마커만 있던 줄은 개행 + 뒤따르는 빈 줄까지."""
        spans = []
        for m in seg.heads:
            ls = m.start()                           # STEP_PAT 은 줄 머리(^)에서만 맞으므로 줄 시작과 같다
            le = self.AFTER_MARK.match(text, m.end()).end()
            nl = text.find("\n", le)
            line_end = nl + 1 if nl >= 0 else len(text)
            if not text[le:line_end].strip():
                le = line_end
                while le < len(text):
                    nl = text.find("\n", le)
                    end = nl + 1 if nl >= 0 else len(text)
                    if text[le:end].strip():
                        break
                    le = end
            spans.append((ls, le))
        # term 앞에서 시작한 마커는 term 앞에서 끝난다 (빈 줄 소거는 다음 본문 줄 앞에서 멈춘다).
        # 합쳐진 마지막 step 의 마커는 term 에서 시작해 터미널 안에서 지워진다.
        assert all(le <= seg.term for ls, le in spans if ls < seg.term), "header marker crosses terminal"
        return spans

    def prepare(self, text: str) -> tuple[Cleaned | None, str | None]:
        """출력에서 "Step N" **마커만** 지우고, 지운 좌표계로 경계를 만든다.

        마커는 구간을 어디서 자를지 정하는 구분자로만 쓰고 forward pass 에서는 뺀다.
        프로빙 때 "Step N" 토큰을 보고 스텝 번호를 맞히는 치팅을 막기 위해서다.

        지우는 범위는 STEP_PAT 매치("### Step 3." / "**Step 3:" 등) 와 그 뒤의 볼드 닫힘·가로 공백까지.
        같은 줄의 나머지(제목이든 본문이든) 는 남긴다. 헤더 줄 전체를 지우면 predict 는
        "Step 1: The agent starts at (3, 4)..." 처럼 본문이 헤더 줄에 붙고 decompose 는 Step 6 이
        제목 한 줄이라 전체의 15~70% 가 empty_segment 로 빠진다. 제목("Identify the mission goal")
        이 프롬프트가 지시한 고정 문구라 스텝 번호를 드러내는 문제는 남지만, 그건 본문 내용도
        마찬가지이므로 프로빙 쪽에서 대조 실험(초반 층·cross-level)으로 가린다.

        마커만 있던 줄("Step 2: \\n")은 줄 끝 개행과 뒤따르는 빈 줄까지 지워 구간이 본문 첫 글자에서
        시작하게 한다. 남겨두면 본문이 "\\n" 으로 시작해 앞 구간 끝의 "\\n\\n" 과 한 토큰으로 묶여
        boundary_straddle 로 버려진다. 마커 뒤에 아무것도 없는 스텝은 길이 0 → "empty_segment".

        MERGE_LAST_STEP 태스크는 마지막 step 의 마커도 지우되 그 줄부터는 터미널 구간이다.
        """
        seg, reason = self._segment(text)
        if reason is not None:
            return None, reason

        spans = self._marker_spans(text, seg)

        def mapped(p: int) -> int:                   # 원문 위치 → 지운 좌표계
            return p - sum(le - ls for ls, le in spans if le <= p)

        pieces, prev = [], 0
        for ls, le in spans:
            pieces.append(text[prev:ls])
            prev = le
        pieces.append(text[prev:])
        clean = "".join(pieces)

        clean_starts = [mapped(s) for s in seg.starts]   # step 으로 남는 구간만 (합쳐진 마지막 step 은 없다)
        clean_starts[0] = 0                          # 원문처럼 첫 마커 앞 공백은 Step 1 이 흡수
        bounds = clean_starts + [mapped(seg.term), len(clean)]
        if any(a >= b for a, b in zip(bounds, bounds[1:])):
            return None, "empty_segment"
        return Cleaned(clean, bounds, mapped(seg.term_end), len(text) - len(clean)), None


# -------------------------------------------------------------------- encoder

class Encoder:
    """토크나이저와 모델을 처음 쓸 때 각각 따로 로드한다.

    토크나이즈만 필요한 쪽(경계 검증 등)은 tokenizer 만 건드리면 되고 모델은 올라가지 않는다.
    """

    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self._tok = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
        return self._tok

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(  # lm_head 없음 → logits 미계산
                self.model_name, dtype=torch.bfloat16, device_map="auto"
            )
            self._model.eval()
            print("device:", self._model.device)
        return self._model

    def render_prompt(self, episode: dict) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": episode["prompt"]}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=bool(episode.get("thinking", False)),
        )

    @staticmethod
    def _char_to_token_bounds(char_bounds: list[int], offsets) -> list[int]:
        tok_bounds, k = [], 0
        for cb in char_bounds:
            while k < len(offsets) and offsets[k][0] < cb:
                k += 1
            tok_bounds.append(k)
        return tok_bounds

    def tokenize(self, prompt: str, output: str, char_bounds: list[int]):
        """prompt + output 을 통째로 토크나이즈하고 output 기준 문자 경계를 토큰 경계로 바꾼다.

        반환 (enc, tok_bounds, 사유). tok_bounds = [0, 프롬프트 끝, ...char_bounds 를 옮긴 것].
        추출과 검증이 같은 함수를 써야 토큰 수·경계가 일치한다.
        """
        enc = self.tokenizer(prompt + output, add_special_tokens=False, return_offsets_mapping=True)
        abs_bounds = [0] + [len(prompt) + b for b in char_bounds]   # char_bounds[0] == 0

        # 경계 양쪽 글자가 한 토큰으로 묶이면 경계를 토큰 단위로 못 그어서 한쪽 첫 토큰이
        # 조용히 앞 구간에 먹힌다. 프롬프트/출력 이음매는 템플릿이 \n\n 로 끝나 보통 안 생기지만,
        # 마커를 벗긴 뒤의 스텝 이음매("...\n\n" + 본문 첫 글자)도 같은 위험이 있어 전부 검사한다.
        cuts = set(abs_bounds[1:-1])
        if any(s < c < e for s, e in enc.offset_mapping for c in cuts if s < c):
            return None, None, "boundary_straddle"

        tok_bounds = self._char_to_token_bounds(abs_bounds, enc.offset_mapping)
        assert tok_bounds[-1] == len(enc.input_ids), "unconsumed tokens"
        return enc, tok_bounds, None

    @torch.no_grad()
    def hidden(self, ids: list[int]) -> torch.Tensor:
        """마지막 층 hidden state (프롬프트 구간 포함 전체). (토큰수 x d) bf16, CPU."""
        H = self.model(torch.tensor([ids], device=self.model.device)).last_hidden_state[0]
        return H.to(torch.bfloat16).cpu().clone()


# ------------------------------------------------------------------- episode

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


class EpisodeExtractor:
    """에피소드 하나: skip 판정 → 라벨 → 경계 → 토크나이즈 → hidden state."""

    def __init__(self, task: str, encoder: Encoder | None = None,
                 max_tokens: int = MAX_TOKENS, min_step_tokens: int = MIN_STEP_TOKENS):
        self.task = task
        self.segmenter = StepSegmenter(task)
        self.encoder = encoder or Encoder()
        self.max_tokens = max_tokens
        self.min_step_tokens = min_step_tokens

    @staticmethod
    def _build_meta(episode: dict, src_name: str) -> dict:
        meta = {k: v for k, v in episode.items() if k not in HEAVY_FIELDS}
        meta["src"] = src_name
        return meta

    def process(self, episode: dict, src_name: str) -> tuple[dict, dict | None]:
        """반환 (meta, payload). payload 가 None 이면 skip 이고 사유는 meta["extract_skipped"].

        payload = {"E", "boundaries", "output_sha1"}. "Step N" 마커는 모델 입력에서 빠지므로
        E 와 boundaries 는 마커가 없는 시퀀스 기준이다. output_sha1 은 원문 기준
        (원본 jsonl 과 대조하는 열쇠).
        """
        meta = self._build_meta(episode, src_name)

        def skip(reason: str):
            meta["extract_skipped"] = reason
            return meta, None

        if episode.get("skipped"):               # 출력 자체가 없는 에피소드
            return skip("no_output")
        if episode.get("truncated"):             # max_tokens 에서 잘림 → 마지막 구간 불완전
            return skip("truncated")
        if episode.get("eval_error") is not None:
            # 파싱 실패 / 봇 실행 예외 / eval 타임아웃이 한 필드에 섞여 있다.
            # 타임아웃은 봇 리플랜 루프가 안 끝나는 환경 문제라 LLM 실패가
            # 아니고, 파싱 실패는 어차피 segmenter 가 거른다. 통째로 뺀다.
            return skip("eval_error")
        if not (episode.get("prompt") or "").strip():
            return skip("empty_prompt")          # 프롬프트 구간이 길이 0이 된다

        status, label_key = episode_status(episode)
        if status is None:
            return skip("no_label")              # eval_result 에 판정 키가 없다
        meta["status"] = status
        meta["label_key"] = label_key

        # 구조 검사는 토크나이즈 전에 — 안 맞는 궤적엔 모델을 안 태운다
        output = episode["all_llm_output"] or ""
        meta["output_sha1"] = hashlib.sha1(output.encode()).hexdigest()   # 원문(마커 포함) 기준
        cleaned, reason = self.segmenter.prepare(output)
        if reason is not None:
            return skip(reason)

        prompt = self.encoder.render_prompt(episode)
        enc, boundaries, reason = self.encoder.tokenize(prompt, cleaned.text, cleaned.bounds)
        if reason is not None:
            return skip(reason)
        ids = enc.input_ids
        meta.update(
            n_tokens_total=len(ids),
            n_tokens_prompt=boundaries[1],
            n_tokens_output=boundaries[-1] - boundaries[1],   # 마커 제외
            n_tokens_terminal=boundaries[-1] - boundaries[-2],
            n_steps=len(boundaries) - 3,         # 프롬프트·터미널 제외
            n_header_chars=cleaned.n_header_chars,   # forward pass 에서 빠진 "Step N" 마커 문자 수
        )

        if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
            return skip("empty_token_segment")   # 문자로는 갈렸는데 토큰으로 뭉개진 구간
        step_lens = [e - s for s, e in zip(boundaries[1:-2], boundaries[2:-1])]   # step 1..N (프롬프트·터미널 제외)
        if step_lens and min(step_lens) < self.min_step_tokens:
            return skip("short_step")            # 마커를 뗀 뒤 제목 몇 토큰만 남은 step
        if len(ids) > self.max_tokens:
            return skip("too_long")

        E = self.encoder.hidden(ids)
        assert E.shape[0] == boundaries[-1]
        return meta, {
            "E": E,                          # (프롬프트+출력 토큰수) x d
            "boundaries": boundaries,        # [0, prompt, step1..N, terminal 끝]
            "output_sha1": meta["output_sha1"],
        }


# --------------------------------------------------------------------- writer

class ChunkWriter:
    """payload 를 status 별로 모아 <run_dir>/<status>/chunk_XXXX.pt 로 쓴다.

    같은 status 안에서 seed 가 겹치면 즉시 에러. 시작할 때 이전 청크 파일을 지워
    재실행 시 청크 개수가 안 맞게 남는 것을 막는다.
    """

    def __init__(self, run_dir: Path, chunk_size: int, header: dict):
        self.run_dir = run_dir
        self.chunk_size = chunk_size
        self.header = header
        self.buffers = {s: {} for s in STATUSES}
        self.chunk_idx = {s: 0 for s in STATUSES}
        self.seen = {s: set() for s in STATUSES}
        for s in STATUSES:
            d = run_dir / s
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("chunk_*.pt"):
                old.unlink()

    def add(self, status: str, seed, payload: dict) -> int:
        """payload 를 버퍼에 넣고 이것이 들어갈 청크 인덱스를 돌려준다."""
        if seed in self.seen[status]:
            raise ValueError(f"id collision: {self.run_dir}/{status} seed={seed}")
        self.seen[status].add(seed)
        idx = self.chunk_idx[status]
        self.buffers[status][seed] = payload
        if len(self.buffers[status]) >= self.chunk_size:
            self.flush(status)
        return idx

    def flush(self, status: str) -> None:
        if not self.buffers[status]:
            return
        out_path = self.run_dir / status / f"chunk_{self.chunk_idx[status]:04d}.pt"
        torch.save({"episodes": self.buffers[status], **self.header}, out_path)
        self.buffers[status] = {}
        self.chunk_idx[status] += 1

    def close(self) -> None:
        for s in STATUSES:
            self.flush(s)

    def counts(self) -> dict[str, int]:
        return {s: len(self.seen[s]) for s in STATUSES}


# ------------------------------------------------------------------------- run

def load_episodes(path: Path) -> list[dict]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_header() -> dict:
    return {"model": MODEL, "boundary_layout": "prompt|steps|terminal",
            "step_headers": "marker_stripped",   # "Step N" 마커만 입력에서 뺐다, 제목·본문은 남음
            "label_priority": LABEL_PRIORITY,
            "expected_steps": EXPECTED_STEPS,    # 이 개수가 아닌 궤적은 skip
            "terminal_includes_last_step": sorted(MERGE_LAST_STEP),   # 마지막 step 을 터미널에 합친 태스크
            "min_step_tokens": MIN_STEP_TOKENS}


def extract_run(
    data_dir: Path,
    out_root: Path,
    task: str,
    level: str,
    mode: str = "no_thinking",
    chunk: int = 256,
    limit: int | None = None,
) -> None:
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
    extractor = EpisodeExtractor(task)
    writer = ChunkWriter(run_dir, chunk, chunk_header())
    reasons = Counter()
    n_saved = 0

    with (run_dir / "meta.jsonl").open("w", encoding="utf-8") as mf:
        for episode in tqdm(episodes, desc="episodes", unit="episode"):
            meta, payload = extractor.process(episode, src.name)
            if payload is None:
                reasons[meta["extract_skipped"]] += 1
            else:
                meta["chunk"] = writer.add(meta["status"], episode["env_seed"], payload)
                n_saved += 1
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
    writer.close()

    counts = writer.counts()
    print(f"saved {n_saved} episodes under {run_dir} ({sum(reasons.values())} skipped)")
    print(f"  success {counts['success']} / failure {counts['failure']}"
          f"  [label_priority={LABEL_PRIORITY}]")
    for reason, n in reasons.most_common():
        print(f"  skipped {reason}: {n}")


# ----------------------------------------------------------------- load helper

def load_chunk(pt_path: Path) -> dict:
    return torch.load(pt_path, map_location="cpu", weights_only=False)["episodes"]


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


def seg_labels(seg: list[int]) -> list[str]:
    """gen_view 의 seg 에 대응하는 구간 이름. ["Step 1", ..., "Step N", "answer"]"""
    return [f"Step {t}" for t in range(1, len(seg) - 1)] + ["answer"]
