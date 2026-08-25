"""에피소드 하나의 토큰 간 유사도 행렬 S = H Hᵀ 를 뽑아 본다.

목적
    "한 스텝 안의 토큰은 서로 가깝고, 먼 스텝의 토큰과는 먼가"
    -> 대각 블록이 밝고 비대각이 어두우면 그렇다.

extract.py 와 다른 점
    extract.py 는 스텝별로 따로 토큰화해서 E_t 를 스텝 단위로 잘라 저장한다.
    그러면 스텝 **간** 관계가 애초에 존재하지 않는다. 여기서는 에피소드 전체를
    한 번 토큰화하고 스텝 경계를 토큰 인덱스로만 표시한다 — S 는 한 장이고
    세그먼트는 그 위에 긋는 선일 뿐이다.

    char span -> token span 변환은 offset_mapping 으로 한다. segment.py 독스트링이
    "extract.py 가 담당한다"고 적어둔 그 부분인데 실제로는 없어서 여기서 처음 만든다.

이 파일은 파이프라인이 아니라 탐색용이다. 그림을 보고 나서 어떤 형태로 정식화할지
정하기 위한 것이므로 extract.py / spectral.py 는 건드리지 않는다.

    python probe_gram.py data/segmented/BabyAI-GoToObj-v0/success.jsonl
    python probe_gram.py data/segmented/BabyAI-BossLevel-v0/fail.jsonl --index 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# 읽는 모델은 쓴 모델과 같아야 한다 (자기생성). 다르면 "3B 가 32B 의 글을 읽으며
# 만드는 표현"이 되어 32B 의 추론 상태와는 별개의 것을 보게 된다.
# extract.py 는 아직 Qwen2.5-3B 로 하드코딩돼 있어 여기와 갈린다.
DEFAULT_MODEL = "Qwen/Qwen3-32B"


def load_episode(path: Path, index: int, ep_id: str | None):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if ep_id:
        for r in rows:
            if r["id"] == ep_id:
                return r
        raise SystemExit(f"id={ep_id!r} 없음 (총 {len(rows)}개)")
    if not 0 <= index < len(rows):
        raise SystemExit(f"index {index} 범위 밖 (0..{len(rows)-1})")
    return rows[index]


def token_spans(offsets, segments):
    """char span 세그먼트 -> [(name, tok_start, tok_end)].

    토큰은 "시작 offset 이 어느 세그먼트에 들어가는가" 로 배정한다. 토큰 하나가
    경계를 걸치는 경우("...\\n\\n**Action" 처럼 BPE 가 개행을 붙여 묶는 경우가 있다)
    양쪽에 중복 배정하거나 어디에도 안 넣으면 블록이 어긋나므로, 시작점 기준으로
    딱 한 곳에 넣어 분할이 되게 한다.
    """
    out = []
    for seg in segments:
        idx = [i for i, (a, b) in enumerate(offsets)
               if b > a and seg["start"] <= a < seg["end"]]
        if idx:
            out.append((seg["name"], idx[0], idx[-1] + 1))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="segment.py 가 만든 jsonl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--id", dest="ep_id", default=None, help="index 대신 id 로 지정")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"인코더. 기본 {DEFAULT_MODEL} (= CoT 를 쓴 모델)")
    ap.add_argument("--out", type=Path, default=None,
                    help="확장자 없는 저장 경로. 기본 visual/gram_<id>")
    args = ap.parse_args()

    ep = load_episode(args.input, args.index, args.ep_id)
    text = ep["all_llm_output"]

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    N = len(ids)

    blocks = {"step": token_spans(offs, ep["segments_step"]),
              "action": token_spans(offs, ep["segments_action"])}

    print(f"id={ep['id']}  task={ep['task']}  success={ep['success']}")
    print(f"answer={ep['parsed_llm_output']!r}")
    print(f"chars={len(text)}  tokens={N}")
    for kind, bs in blocks.items():
        print(f"  {kind:6}: {len(bs)}개  " +
              " ".join(f"{n}[{s}:{e}]" for n, s, e in bs[:6]) +
              (" ..." if len(bs) > 6 else ""))

    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    H = model(torch.tensor([ids], device=model.device)).last_hidden_state[0].float()

    def cosine(X):
        n = X.norm(dim=1)
        return ((X @ X.T) / torch.outer(n, n)).clamp(-1, 1).cpu(), n.cpu()

    S = (H @ H.T).cpu()                      # 내적. 대각 = ‖h_i‖²
    C, norm = cosine(H)

    # residual stream 은 모든 토큰이 공유하는 큰 평균 방향을 갖는다. 그대로 코사인을
    # 재면 무엇이든 +0.4~0.6 이 나와 실제 구조가 공통 성분에 묻힌다 (3B 로 잰 첫 시도에서
    # 대각-비대각 격차가 0.05 에 그쳤다). 토큰 평균을 빼고 다시 잰다.
    Hc = H - H.mean(dim=0, keepdim=True)
    Cc, norm_c = cosine(Hc)

    # id 에 success/fail 구분이 없다. 같은 seed 의 성공/실패를 둘 다 뽑으면
    # 파일명이 겹치므로 입력 파일 이름(success/fail)을 접두사로 붙인다.
    out = args.out or Path("visual") / f"{args.input.stem}_{ep['id'].replace('|', '_')}"
    out.parent.mkdir(parents=True, exist_ok=True)
    # extract.py / spectral.py 와 같은 .pt 로 통일한다. 블록 정보가 (str, int, int)
    # 리스트라 npz 에 넣으려면 dtype=object 로 감싸고 allow_pickle 을 켜야 하는데,
    # 그러면 결국 pickle 이라 torch.save 와 안전성이 같으면서 코드만 번거로워진다.
    torch.save({
        "S": S, "C": C, "norm": norm,
        # 중심화 코사인. 공통 성분을 뺀 것이라 보통 이쪽이 구조를 보여준다.
        "C_centered": Cc, "norm_centered": norm_c,
        "tokens": tok.convert_ids_to_tokens(ids),
        # 토큰 인덱스 (name, tok_start, tok_end) — C[s:e, s:e] 로 바로 잘린다
        "blocks": blocks,
        # segment.py 가 준 원래 char span 과 원문. 이게 있어야 "step2.a3 블록이
        # 무슨 내용이냐" 를 jsonl 을 다시 열지 않고 text[s:e] 로 볼 수 있다.
        "segments": {"step": ep["segments_step"],
                     "action": ep["segments_action"]},
        "text": text,
        # 토큰 <-> char 변환표. 임의의 토큰 i 가 원문 어디인지 offsets[i] 로 안다.
        "offsets": offs,
        # 그림 제목에 쓸 정보. 이 파일 하나만 넘기면 heatmap.py 가 다 그릴 수 있게.
        "meta": {"id": ep["id"], "task": ep["task"], "success": ep["success"],
                 "model": args.model, "src": str(args.input)},
    }, f"{out}.pt")
    print(f"saved {out}.pt")

    # 블록 평균 코사인 — 그림 없이도 대각/비대각 대비를 숫자로 본다.
    # gap(대각-비대각)이 클수록 세그먼트가 실제 의미 단위라는 뜻이다.
    import statistics as st
    for label, M in (("raw", C), ("centered", Cc)):
        for kind, bs in blocks.items():
            diag = st.mean(M[s:e, s:e].mean().item() for _, s, e in bs)
            off = [M[s1:e1, s2:e2].mean().item()
                   for i, (_, s1, e1) in enumerate(bs)
                   for _, s2, e2 in bs[i + 1:]]
            o = st.mean(off) if off else float("nan")
            print(f"  {label:8} {kind:6} diag {diag:+.4f}  off {o:+.4f}  gap {diag - o:+.4f}")

    print(f"\n그림: python visual/heatmap.py {out}.pt")


if __name__ == "__main__":
    main()
