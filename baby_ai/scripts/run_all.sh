#!/bin/bash
# PutNextLocal, 원샷 LLM CoT 2 조건
#   c1  --obs partial   부분관측
#   c2  --obs full      완전관측
#
# References: CTRLS arXiv 2507.08182 / LLM-BabyBench arXiv 2505.12135
#             GLAM arXiv 2302.02662 / BabyAI arXiv 1810.08272
#             설계 근거는 AGENT_CONTEXT.md
set -e

N=${1:-20}
OBJS=${OBJS:-8}
BACKEND=${BACKEND:-local}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
WORKERS=${WORKERS:-4}
OUT=data/putnext

if [ "$BACKEND" = "openai" ] && [ -z "$OPENAI_API_KEY" ]; then
  echo "BACKEND=openai 인데 OPENAI_API_KEY 가 없습니다." >&2
  exit 1
fi

for OBS in partial full; do
  echo "=== ${OBS} ==="
  python baby_ai/datagen/gen_oneshot.py --obs "$OBS" \
    --out "$OUT" --n "$N" --num-objs "$OBJS" \
    --backend "$BACKEND" --model "$MODEL" --workers "$WORKERS"
  echo
done

echo "=== 완료 ==="
for c in c1 c2; do
  for d in cases fail/cases; do
    f="$OUT/$d/$c.jsonl"
    [ -f "$f" ] && echo "  $(wc -l < "$f") eps  $f"
  done
done