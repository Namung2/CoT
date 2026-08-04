#!/bin/bash
# 레벨별 생성 + 조건 c1~c6 후처리. 
# 결과: data/{level}/train.jsonl, train.labels.jsonl, cases/c1..c6.jsonl
#       총 2 레벨 x 6 조건 = 12 케이스
set -e
N=${1:-5000}
for cfg in configs/gotoseq.yaml configs/bosslevel.yaml; do
  echo "=== $cfg ==="
  python datagen/gen.py "$cfg" --n "$N"
done
for d in data/gotoseq data/bosslevel; do
  echo "=== cases: $d ==="
  python datagen/case.py "$d"
done