from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path("data")
OUT_DIR = Path("hidden_states")

MODEL = "Qwen/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(  # lm_head 없음 → logits 미계산
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print("device:", model.device)


def load_samples(data_dir: Path = DATA_DIR, level: str | None = None, case: str | None = None): # data_dir, level, case
    pattern = f"{level or '*'}/{case or '*'}.jsonl" 
    for path in sorted(data_dir.glob(pattern)):
        with path.open(encoding="utf-8") as f:
            for line in f: 
                line = line.strip() 
                if line:
                    yield path, json.loads(line) # Path(파일경로), line 1개(id, level, seed, mission, steps, terminal ...)

def build_babyai(sample: dict):
    x0 = f"Mission: {sample['mission'].strip()}"
    steps = [s.strip() for s in sample["steps"] if s.strip()]
    if sample.get("terminal"):
        steps.append(sample["terminal"].strip())
    meta = {
        "id": sample["id"],
        "level": sample["level"],
        "success": sample["answer"]["success"],
        "action_seq": sample["answer"]["action_seq"],
        "n_steps": len(steps),
    }
    return x0, steps, meta

@torch.no_grad()
def extract_step_hidden_states(x0: str, steps: list[str]):
    pieces = [x0] + steps # piece를 prompt or 어느 한 step라고 하고~
    ids = [] # 토큰 id
    spans = [] # 각 Piece의 범위를 기록
    for j, piece in enumerate(pieces): 
        text = piece if j == 0 else "\n" + piece # mission(input quesition)을 제외하고 나머지는 모두 앞에 \n을 삽입
        piece_ids = tok(text, add_special_tokens=(j == 0)).input_ids # piece별로 따로 토큰화
        spans.append((len(ids), len(ids) + len(piece_ids))) # 토큰화된 step의 좌표를 기록, 다시 step별로 잘라내기 위함
        ids.extend(piece_ids) # 다시 하나로 합치기

    input_ids = torch.tensor([ids], device=model.device) # input tensor

    out = model(input_ids, output_hidden_states=True) # forward
    H = out.hidden_states[-1][0] 
    # [-1]: 마지막 레이어, [0]: 마지막 레이어에서 첫 번째 배치, H = N x d, (N = 전체 토큰수, d = model의 hidden 차원)
    # H는 아직 step 별로 나눠지지 않은 하나의 시퀀스

    E = {}
    for t in range(len(pieces)): # mission을 포함(mission을 step0이라고 생각해도 될 듯?), 토큰 단위가 아니라 piece 단위로 
        s, e = spans[t] # piece단위로 기록한 위치(start, end)
        E[t] = H[s:e].float().cpu()  # H를 piece 단위로 잘라서 저장, E = n_t x d (n_t는 step t에서의 토큰 수)
    return E # 위와 같이 piece단위로 잘린 행렬을 T개 저장(T = 전체 step 수(mission포함))


def run(level: str | None = None, case: str | None = None):
    buckets: dict[tuple[str, str], dict] = {} # key = tuple[level_dir, case_name], value = E or meta

    for path, sample in load_samples(level=level, case=case): # 해당 "level의 case"에서 line 1개씩 반복 
        x0, steps, meta = build_babyai(sample) 
        E = extract_step_hidden_states(x0, steps) # E = n_t x d

        key = (path.parent.name, path.stem) # key = (level, case)
        bucket = buckets.setdefault(key, {"E": {}, "meta": []}) # 이미 (level,case) 쌍 bucket이 없으면 만들고, 있으면 그대로
        bucket["E"][meta["id"]] = E
        bucket["meta"].append(meta)

        # log 출력(이해안할래)
        n = [E[t].shape[0] for t in sorted(E)]
        print(f"[{key[0]}/{key[1]}] {meta['id']}: T+1={len(E)} n_t={min(n)}~{max(n)} d={E[0].shape[1]}")


    
    for (level_dir, case_name), bucket in buckets.items():
        out_dir = OUT_DIR / level_dir
        out_dir.mkdir(parents=True, exist_ok=True)


        torch.save({"E": bucket["E"], "model": MODEL}, out_dir / f"{case_name}.pt")
        with (out_dir / f"{case_name}.meta.jsonl").open("w", encoding="utf-8") as f:
            for m in bucket["meta"]:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"saved {out_dir / f'{case_name}.pt'} ({len(bucket['E'])} samples)")


if __name__ == "__main__":
    run(level="gotoseq", case="c1")
        