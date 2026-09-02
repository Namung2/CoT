#!/usr/bin/env bash
# task x level 조합을 순서대로 inference/main.py(extract+spectral)로 돌린다.
# GPU 하나 쓰는 순차 실행, 하나 실패해도 나머지는 계속 진행한다(무인 실행이라
# 하나 걸렸다고 나머지까지 안 도는 게 더 손해다). run_all.sh와 동일한 구조.
#
# extract는 success/failure를 한 번에 같이 저장하므로, status=failure는
# --no-extract로 이미 뽑힌 hidden state를 재사용해 spectral만 다시 돈다
# (모델 forward를 두 번 안 하기 위함).
#
# 사용:
#   tmux new -s inference
#   ./inference.sh
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0                      # Qwen3-32B라 GPU 여러 개 필요하면 수정

# task:level — 실제 data/*_no_thinking.jsonl의 env_name 그대로
TASK_LEVELS=(
    "decompose:BabyAI-GoToObj-v0"
    "decompose:BabyAI-GoTo-v0"
    "decompose:BabyAI-Synth-v0"
    "decompose:BabyAI-BossLevel-v0"
    "plan:CustomBabyAI-GoToRedBall-Small-4Dists-v0"
    "plan:CustomBabyAI-GoToRedBall-Medium-40Dists-v0"
    "plan:CustomBabyAI-GoToRedBall-Large-100Dists-v0"
    "plan:CustomBabyAI-GoToRedBall-Ultra-180Dists-v0"
    "predict:BabyAI-GoToObj-v0"
    "predict:BabyAI-BossLevel-v0"
)

LOG_DIR="logs/inference_$(date +%Y%m%d_%H%M%S)"
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

for tl in "${TASK_LEVELS[@]}"; do
    task="${tl%%:*}"
    level="${tl#*:}"

    run_stage "${task}_${level}_success" \
        python inference/main.py --task "$task" --level "$level" --status success

    run_stage "${task}_${level}_failure" \
        python inference/main.py --task "$task" --level "$level" --status failure --no-extract
done

echo ""
echo "================ SUMMARY ================"
for name in "${!STATUS[@]}"; do
    printf "%-45s exit=%-4s elapsed=%smin\n" "$name" "${STATUS[$name]}" "${ELAPSED_MIN[$name]}"
done
echo "logs: $LOG_DIR"
