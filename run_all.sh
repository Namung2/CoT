#!/usr/bin/env bash
# 세 태스크(plan/predict/decompose)를 순서대로 이어서 돌린다. GPU를 하나만 쓰므로
# 병렬이 아니라 순차 실행이고, 한 스크립트가 실패해도(비정상 종료) 다음 스크립트는
# 계속 진행한다 - 무인 실행이라 하나 걸렸다고 나머지까지 안 도는 게 더 손해다.
#
# 사용:
#   tmux new -s cot
#   ./run_all.sh
set -uo pipefail
cd "$(dirname "$0")"

PY=/home/hail/anaconda3/envs/cot_llm/bin/python
export CUDA_VISIBLE_DEVICES=0,2
SEEDS=30000

LOG_DIR="logs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

declare -A STATUS
declare -A ELAPSED_MIN

run_stage() {
    local name="$1"; shift
    echo ""
    echo "===== [$name] start $(date '+%F %T') ====="
    local t0=$(date +%s)
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local rc=${PIPESTATUS[0]}
    local t1=$(date +%s)
    STATUS[$name]=$rc
    ELAPSED_MIN[$name]=$(( (t1 - t0) / 60 ))
    echo "===== [$name] end $(date '+%F %T')  elapsed=${ELAPSED_MIN[$name]}min  exit=$rc ====="
}

run_stage plan      "$PY" cot_paln.py      --seeds "$SEEDS"
run_stage predict   "$PY" cot_predict.py   --seeds "$SEEDS"
run_stage decompose "$PY" cot_decompose.py --seeds "$SEEDS"

echo ""
echo "================ SUMMARY ================"
for name in plan predict decompose; do
    printf "%-10s exit=%-4s elapsed=%smin\n" "$name" "${STATUS[$name]:-?}" "${ELAPSED_MIN[$name]:-?}"
done
echo "logs: $LOG_DIR"
