from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
tok = None
model = None


def ensure_model():
    """import 시점이 아니라 실제로 필요할 때만 모델을 올린다."""
    global tok, model
    if model is None:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(  # lm_head 없음 → logits 미계산
            MODEL, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        print("device:", model.device)
    return tok, model


## full
def verify_spans(steps: list[str]):
    ensure_model()
    ids, spans = [], []
    for j, step in enumerate(steps):
        text = step if j == 0 else "\n" + step
        s = len(ids)
        ids.extend(tok(text, add_special_tokens=(j == 0)).input_ids)
        spans.append((s, len(ids)))

    assert spans[0][0] == 0 and spans[-1][1] == len(ids)
    for (_, e), (s, _) in zip(spans, spans[1:]):
        assert e == s, f"gap/overlap at {e} != {s}"

    for t, (s, e) in enumerate(spans):
        got = tok.decode(ids[s:e])
        want = steps[t] if t == 0 else "\n" + steps[t]
        print(f"t={t:2d} n_tok={e-s:3d} {'OK ' if got == want else 'MISMATCH'} {got!r}")

def load_episodes(data_dir: Path, level: str | None = None, case: str | None = None):
    pattern = f"{level or '*'}/**/{case or '*'}/*.jsonl"
    episodes = []
    for path in sorted(data_dir.glob(pattern)):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append((path, json.loads(line)))
    return episodes

def build_babyai(episode: dict):
    mission = f"Mission: {episode['mission'].strip()}" # "mission: 실제 mission text"로 저장
    steps = [s.strip() for s in episode["steps"] if s.strip()] # "일련의 step들로 저장: [step1, step2, step3, ...]"
    terminal = f"Terminal: {episode['terminal'].strip()}" # "terminal: 실제 terminal text"로 저장

    steps = [mission] + steps + [terminal] #mission, sptes, terminal을 list로 합쳐서 저장. 

    meta = {
        "id": episode["id"],
        "level": episode["level"],
        "success": episode["answer"]["success"],
        "action_seq": episode["answer"]["action_seq"],
        "n_steps": len(steps),
    }
    return steps, meta

@torch.no_grad()
def extract_full_sequence_pass(steps: list[str]):
    ensure_model()
    ids = [] # 토큰 id
    spans = [] # 각 step의 범위를 기록
    
    for j, step in enumerate(steps): # full sequence 생성 및 step 위치 계산
        text = step if j == 0 else "\n" + step # mission(input quesition)을 제외하고 나머지는 모두 앞에 \n을 삽입
        s = len(ids)
        ids.extend(tok(text, add_special_tokens=(j == 0)).input_ids) # step별로 따로 토큰화
        spans.append((s, len(ids))) # 토큰화된 step의 좌표를 기록, 다시 step별로 잘라내기 위함
    
    out = model(torch.tensor([ids], device=model.device)) # forward
    H = out.last_hidden_state[0].to(torch.bfloat16).cpu()
    # H는 아직 step 별로 나눠지지 않은 하나의 시퀀스, H = N x d, (N = 전체 토큰수, d = model의 hidden 차원)

    E = {}
    for t, (s, e) in enumerate(spans): # mission을 포함(mission = step0)
        E[t] = H[s:e].clone()  # H를 step 단위로 잘라서 저장, E = n_t x d (n_t는 step t에서의 토큰 수)
    return E # 위와 같이 stpe 단위로 잘린 행렬을 T개 저장(T = 전체 step 수(missionm, terminal포함))

@torch.no_grad()
def extract_cumulative_prefix_passes(steps: list[str]):
    ensure_model()
    ids = []      # 누적 토큰 id
    E_A = {}      # 방법 A: 입력 시퀀스 전체 N_t x d
    E_B = {}      # 방법 B: x_t 구간만      n_t x d

    for t, step in enumerate(steps):
        text = step if t == 0 else "\n" + step
        s = len(ids)                                        # x_t 시작 위치
        ids.extend(tok(text, add_special_tokens=(t == 0)).input_ids)
        e = len(ids)                                        # == N_t

        H = model(torch.tensor([ids], device=model.device)).last_hidden_state[0]   # N_t x d

        E_A[t] = H.to(torch.bfloat16).cpu()        # 전체 step만 뽑기
        E_B[t] = H[s:e].to(torch.bfloat16).cpu()   # 마지막 step만

    return E_A, E_B

def extract_run(episodes: list, data_dir: Path, out_root: Path):
    opened: set[Path] = set()

    for path,episode in tqdm(episodes, desc="episodes", unit="episode"):
        steps, meta = build_babyai(episode)
        E = extract_full_sequence_pass(steps)
        #E_A, E_B = extract_cumulative_prefix_passes(steps)

        rel = path.relative_to(data_dir).with_suffix("")
        out_dir = out_root / rel
        for sub in ("full", "A", "B"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

        sid = meta["id"]
        torch.save({"E": E,   "method": "full", "model": MODEL}, out_dir / "full" / f"{sid}.pt")
        #torch.save({"E": E_A, "method": "A",    "model": MODEL}, out_dir / "A"    / f"{sid}.pt")
        #torch.save({"E": E_B, "method": "B",    "model": MODEL}, out_dir / "B"    / f"{sid}.pt")

        meta_path = out_dir / "meta.jsonl"
        mode = "a" if meta_path in opened else "w"
        opened.add(meta_path)
        with meta_path.open(mode, encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"saved under {out_root}")