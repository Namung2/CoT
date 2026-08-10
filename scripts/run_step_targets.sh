#!/bin/bash
# 레벨별 x 정확한 step 수(10/20/30/40/50)별로 생성 + c3 후처리.
# min-steps == max-steps 로 걸어서 "정확히 그 스텝 수로 풀리는 에피소드"만 채택.
# 결과: data/{level}_step{N}/train.jsonl, train.labels.jsonl, cases/c3.jsonl
#       총 2 레벨 x 5 step 타겟 = 10 개 디렉터리
set -e
N=${1:-1000}

for level_cfg in configs/gotoseq.yaml configs/bosslevel.yaml; do
  level=$(basename "$level_cfg" .yaml)
  for steps in 10 20 30 40 50; do
    out="data/${level}_step${steps}"
    echo "=== $level step=$steps -> $out ==="
    python datagen/gen.py "$level_cfg" --n "$N" --max-steps "$steps" --min-steps "$steps" --out "$out"
    python datagen/case.py "$out" --cases c3
  done
done
