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
    MODEL, dtype=torch.bfloat16, device_map="auto"
)
model.eval()
print("device:", model.device)


def load_samples(data_dir: Path = DATA_DIR, level: str | None = None, case: str | None = None):
    pattern = f"{level or '*'}/**/{case or '*'}.jsonl"  # 중간 디렉토리 깊이 무관
    for path in sorted(data_dir.glob(pattern)):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield path, json.loads(line)


def build_babyai(sample: dict):
    mission = f"Mission: {sample['mission'].strip()}" # "mission: 실제 mission text"로 저장
    steps = [s.strip() for s in sample["steps"] if s.strip()] # "일련의 step들로 저장: [step1, step2, step3, ...]"
    terminal = f"Terminal: {sample['terminal'].strip()}" # "terminal: 실제 terminal text"로 저장

    steps = [mission] + steps + [terminal] #mission, sptes, terminal을 합쳐서 저장. 

    meta = {
        "id": sample["id"],
        "level": sample["level"],
        "success": sample["answer"]["success"],
        "action_seq": sample["answer"]["action_seq"],
        "n_steps": len(steps),
    }
    return steps, meta

@torch.no_grad()
def extract_step_hidden_states(steps: list[str]):
    ids = []      # 누적 토큰 id
    E_A = {}      # 방법 A: 입력 시퀀스 전체 토큰,  N_t x d
    E_B = {}      # 방법 B: x_t 구간 토큰만,       n_t x d

    for t, step in enumerate(steps):
        input = step if t == 0 else "\n" + step
        s = len(ids)                                        # x_t 시작 위치
        ids.extend(tok(input, add_special_tokens=(t == 0)).input_ids)
        e = len(ids)                                        # == N_t

        H = model(torch.tensor([ids], device=model.device),
                  output_hidden_states=True).hidden_states[-1][0]   # N_t x d

        E_A[t] = H.to(torch.bfloat16).cpu()        # 전체 step만 뽑기
        E_B[t] = H[s:e].to(torch.bfloat16).cpu()   # 마지막 step만

    return E_A, E_B

def run(level: str | None = None, case: str | None = None):
    opened: set[Path] = set()  
    
    for path, sample in load_samples(level=level, case=case):
        steps, meta = build_babyai(sample)
        E_A, E_B = extract_step_hidden_states(steps)

        rel = path.relative_to(DATA_DIR).with_suffix("")  # 예: gotoseq/cased/c1
        out_dir = OUT_DIR / rel
        (out_dir / "A").mkdir(parents=True, exist_ok=True)
        (out_dir / "B").mkdir(parents=True, exist_ok=True)

        sid = meta["id"]
        torch.save({"E": E_A, "method": "A", "model": MODEL}, out_dir / "A" / f"{sid}.pt")
        torch.save({"E": E_B, "method": "B", "model": MODEL}, out_dir / "B" / f"{sid}.pt")

        meta_path = out_dir / "meta.jsonl"
        mode = "a" if meta_path in opened else "w"
        opened.add(meta_path)
        with meta_path.open(mode, encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        n = [E_B[t].shape[0] for t in sorted(E_B)]
        N_T = E_A[max(E_A)].shape[0]
        print(f"[{rel}] {sid}: T+1={len(E_B)} n_t={min(n)}~{max(n)} N_T={N_T} d={E_B[0].shape[1]}")

    print(f"saved under {OUT_DIR}")

if __name__ == "__main__":
    run(level="gotoseq", case="c1")