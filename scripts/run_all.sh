#!/bin/bash
# 레벨별로 따로 생성. 섞지 않는다.
set -e
N=${1:-5000}
for cfg in configs/gotoseq.yaml configs/bosslevel.yaml; do
  echo "=== $cfg ==="
  python datagen/datagen.py "$cfg" --n "$N"
done