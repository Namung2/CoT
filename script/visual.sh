#!/usr/bin/env bash
# task x level x status 조합마다 visual/step_similarity.py를 돌린다. GPU/모델 필요
# 없고 저장된 hidden_states/spectral_states만 읽으므로 inference.sh보다 훨씬 가볍다.
#
# 사용:
#   ./visual.sh              # 백그라운드로 떨어지고 바로 셸로 돌아옴
#   tail -f nohup_*.out      # 진행 상황 보기
set -uo pipefail
cd "$(dirname "$0")"

if [[ "${VISUAL_SH_BG:-}" != "1" ]]; then
    nohup_log="nohup_$(date +%Y%m%d_%H%M%S).out"
    VISUAL_SH_BG=1 nohup setsid "$0" "$@" > "$nohup_log" 2>&1 &
    echo "started in background: pid=$! log=$nohup_log"
    exit 0
fi

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

LOG_DIR="logs/visual_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

declare -A STATUS

for tl in "${TASK_LEVELS[@]}"; do
    task="${tl%%:*}"
    level="${tl#*:}"

    for status in success failure; do
        name="${task}_${level}_${status}"
        echo ""
        echo "===== [$name] start $(date '+%F %T') ====="
        python visual/step_similarity.py --task "$task" --level "$level" --status "$status" \
            2>&1 | tee "$LOG_DIR/${name}.log"
        rc=${PIPESTATUS[0]}
        STATUS[$name]=$rc
        echo "===== [$name] end $(date '+%F %T')  exit=$rc ====="
    done
done

echo ""
echo "================ SUMMARY ================"
for name in "${!STATUS[@]}"; do
    printf "%-45s exit=%-4s\n" "$name" "${STATUS[$name]}"
done
echo "logs: $LOG_DIR"
echo "results: visual/step_similarity/"
