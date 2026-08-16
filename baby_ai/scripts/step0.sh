#!/usr/bin/env bash
# STEP 0 + STEP 1 : 파일럿 1 에피소드 -> 실시퀀스 진단
set -e
cd "$(dirname "$0")/.."
python -m datagen.generate --cell P1 --n 1 --keep-all --save-hidden --out data
python -m inference.probe_step1 \
    --steps data/P1/steps.jsonl \
    --hidden "$(ls data/P1/hidden_*.pt | head -1)" \
    --out data/P1/step1