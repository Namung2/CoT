#!/usr/bin/env bash
# STEP 2 : 파일럿 30 -> 게이트
set -e
cd "$(dirname "$0")/.."
python -m datagen.generate --cell P1 --n 30 --keep-all --out data
python -m datagen.check --cell-dir data/P1