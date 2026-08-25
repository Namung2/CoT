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
    global tok, model
    if model is None:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(  # lm_head 없음 → logits 미계산
            MODEL, dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        print("device:", model.device)
    return tok, model


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

def load_episodes(data_dir: Path, case: str, level: str, step: str):
    target = data_dir / case / level / step
    if not target.is_dir():
        raise FileNotFoundError(f"no such directory: {target}")
    
    episodes = []
    for path in sorted(target.glob("*.jsonl")):
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

    steps = [mission] + steps + [terminal] # mission, sptes, terminal을 list로 합쳐서 저장. 

    meta = {
        "id": episode["id"],
        "level": episode["level"],
        "success": episode["answer"]["success"],
        "action_seq": episode["answer"]["action_seq"],
        "n_steps": len(steps),
    }
    return steps, meta

@torch.no_grad() 
def extract_full_sequence_pass(steps: list[str]): # full-sequence로 1번 forward pass
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
def extract_cumulative_prefix_passes(steps: list[str]): # prefix 누적해서 T번 forward pass
    ensure_model()
    ids = []      # 누적 토큰 id
    E = {}

    for t, step in enumerate(steps):
        text = step if t == 0 else "\n" + step
        s = len(ids)                                        # x_t 시작 위치
        ids.extend(tok(text, add_special_tokens=(t == 0)).input_ids)
        e = len(ids)                                        # N_t

        H = model(torch.tensor([ids], device=model.device)).last_hidden_state[0]   # N_t x d

        E[t] = H[s:e].to(torch.bfloat16).cpu()   # 마지막 step만

    return E

EXTRACTORS = {
    "full_sequence": extract_full_sequence_pass,
    "cumulative_prefix": extract_cumulative_prefix_passes,
}

def extract_run(data_root: Path, out_root: Path, case: str, level: str, step: str, method: str):

    if method not in EXTRACTORS:
        raise ValueError(f"unknown method: {method!r} (choose from {list(EXTRACTORS)})")
    extractor = EXTRACTORS[method]

    data_root = data_root.resolve()
    episodes = load_episodes(data_dir = data_root, case=case, level=level, step=step)

    rel = Path(case) / level / step
    out_dir = out_root / rel / method
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / rel / "meta.jsonl"

    with meta_path.open("w", encoding="utf-8") as mf:
        for path, episode in tqdm(episodes, desc="episodes", unit="episode"):
            steps, meta = build_babyai(episode)
            E = extractor(steps)

            out_path = out_dir / f"{meta['id']}.pt"
            if out_path.exists():
                raise FileExistsError(f"id collision: {out_path}")
            torch.save({"E": E, "method": method, "model": MODEL}, out_path)

            meta["src"] = path.name
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"saved {len(episodes)} episodes under {out_dir}")