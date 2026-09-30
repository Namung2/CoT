#!/usr/bin/env bash
# 모든 task x level x status 에 metric/gsbs_batch.py(step 안 GSBS) 를 돌리고
# 끝나면 metric/gsbs_summary.py 로 집계한다. GPU 필요 없음 (CPU 만 씀).
#
# GSBS 가 순수 CPU 라 (task, level, status) 조합을 PARALLEL 개씩 동시에 돌린다.
# 조합 하나가 config 7개(tokens + k4/8/16 x data/max) x 에피소드 전부를 순서대로 처리한다.
# 중간에 끊겨도 같은 명령으로 다시 돌리면 끝난 (에피소드, config) 는 건너뛴다.
#
# 사용:
#   ./script/gsbs.sh                 # 백그라운드로 떨어지고 바로 셸로 돌아옴
#   PARALLEL=8 ./script/gsbs.sh      # 동시 실행 개수 (기본 4)
#   EXTRA="--kmax 30" ./script/gsbs.sh   # gsbs_batch.py 에 추가 인자 (긴 스텝이 느리면 kmax 를 제한)
#   tail -f nohup_gsbs_*.out
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ "${GSBS_SH_BG:-}" != "1" ]]; then
    nohup_log="nohup_gsbs_$(date +%Y%m%d_%H%M%S).out"
    # setsid 는 리눅스 전용 (macOS 에는 없음) → 있으면 쓰고 없으면 nohup 만.
    if command -v setsid >/dev/null 2>&1; then
        GSBS_SH_BG=1 nohup setsid "$0" "$@" > "$nohup_log" 2>&1 &
    else
        GSBS_SH_BG=1 nohup "$0" "$@" > "$nohup_log" 2>&1 &
    fi
    echo "started in background: pid=$! log=$nohup_log"
    exit 0
fi

PARALLEL="${PARALLEL:-4}"
EXTRA="${EXTRA:-}"

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

LOG_DIR="logs/gsbs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "logs: $LOG_DIR   parallel=$PARALLEL   extra='$EXTRA'"

# 조합 목록을 만들어 xargs 로 PARALLEL 개씩 실행. 하나 실패해도 나머지는 계속.
jobs=()
for tl in "${TASK_LEVELS[@]}"; do
    for status in success failure; do
        jobs+=("${tl%%:*} ${tl#*:} $status")
    done
done

printf '%s\n' "${jobs[@]}" | xargs -P "$PARALLEL" -L 1 bash -c '
    task=$0; level=$1; status=$2
    name="${task}_${level}_${status}"
    echo "===== [$name] start $(date "+%F %T") ====="
    python metric/gsbs_batch.py --task "$task" --level "$level" --status "$status" '"$EXTRA"' \
        > "'"$LOG_DIR"'/${name}.log" 2>&1
    echo "===== [$name] end $(date "+%F %T") exit=$? ====="
'

echo ""
echo "===== summary $(date '+%F %T') ====="
python metric/gsbs_summary.py 2>&1 | tee "$LOG_DIR/summary.log"
echo "results: latent/gsbs/batch/ (jsonl), latent/gsbs/summary/ (csv + fig)"
