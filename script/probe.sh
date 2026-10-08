#!/usr/bin/env bash
# task x level x sign_mode 조합을 predict/probing.py 로 돌린다 (k=8).
# SOURCE=spectral (기본): 구간 마지막 e_t. 입력은 옛 포맷
#     latent/spectral_states/<task>/<level>/{success,failure}/k8_scaled_sign-<mode>/chunk_*.pt
# SOURCE=edges PART=both|front|back TAG_SUFFIX=_f5_b5 SPECTRAL_DIR=latent/spectral: 구간 앞5/뒤5 토큰별 e_i.
#     latent/spectral/<task>/<level>/{success,failure}/k8_scaled_sign-<mode>_f5_b5/chunk_*.pt
# SOURCE=pct PCT=40 TAG_SUFFIX=_p10-20-40-60-80-90-100 SPECTRAL_DIR=latent/spectral: 구간 40% 지점 누적 e.
#     → out/probe_pct_40/
# SOURCE=hidden HIDDEN_DIR=latent/hidden_states_marker PCTS="10 20 40 60 80 90 100": 토큰 hidden state 자체.
#     구간의 q% 지점 토큰을 q 마다 샘플로 넣어 학습하고 평가는 q 별로 나눈다 (probing.py --pct). sign_mode 는 무관해서
#     런 이름은 <task>_<level>_hidden. PCTS 를 비우면 구간 마지막 토큰만. → out/probe_hidden_pct/
# 레벨마다 에피소드 N_EPISODES 개만 쓴다 (success 우선, 모자라면 failure). 같은 SAMPLE_SEED 라
# max / data 가 같은 에피소드 집합 위에서 비교된다.
# 런들은 DEVICES 의 GPU 에 라운드로빈으로 배정되어 GPU 마다 하나씩 동시에 돈다 (probing.py --device).
# 하나 실패해도 나머지는 계속 돈다. 기본은 nohup 백그라운드, NO_BG=1 이면 지금 셸(tmux)에서 그대로 돈다.
#
# 사용:
#   ./script/probe.sh                            # 전체 (10 레벨 x 2 부호 = 20 런), GPU 3장
#   SEEDS="42" NO_BG=1 ./script/probe.sh         # 시드 1개, tmux 안에서 포그라운드로
#   DEVICES="cpu" ./script/probe.sh              # sklearn CPU 경로 (느림)
#   SOURCE=edges PART=front TAG_SUFFIX=_f5_b5 SPECTRAL_DIR=latent/spectral SIGN_MODES=data ./script/probe.sh
#   SOURCE=pct PCT=40 TAG_SUFFIX=_p10-20-40-60-80-90-100 SPECTRAL_DIR=latent/spectral SIGN_MODES=data ./script/probe.sh
#   SOURCE=hidden HIDDEN_DIR=latent/hidden_states_marker TASK_LEVELS="decompose:BabyAI-GoToObj-v0" ./script/probe.sh
#   SIGN_MODES="max" TASK_LEVELS="plan:CustomBabyAI-GoToRedBall-Small-4Dists-v0" ./script/probe.sh
#   tail -f nohup_probe_*.out
set -uo pipefail
cd "$(dirname "$0")/.."
# 셸에 CUDA_VISIBLE_DEVICES=0 같은 값이 남아 있으면 cuda:1/2 가 "invalid device ordinal" 로 죽는다.
# 장치는 DEVICES(probe.sh) / DEVICE(spectral_edges.sh) 로 명시하므로 여기서 지운다.
unset CUDA_VISIBLE_DEVICES

if [[ "${PROBE_SH_BG:-}" != "1" && "${NO_BG:-}" != "1" ]]; then
    nohup_log="nohup_probe_$(date +%Y%m%d_%H%M%S).out"
    PROBE_SH_BG=1 nohup setsid "$0" "$@" > "$nohup_log" 2>&1 &
    echo "started in background: pid=$! log=$nohup_log"
    exit 0
fi

SOURCE="${SOURCE:-spectral}"                 # spectral | edges | pct | hidden
PART="${PART:-both}"                         # edges 일 때: both | front | back
PCT="${PCT:-100}"                            # pct 일 때: 저장본의 --pct 목록 중 하나
HIDDEN_DIR="${HIDDEN_DIR:-latent/hidden_states_marker}"   # hidden 일 때
PCTS="${PCTS:-10 20 40 60 80 90 100}"                      # hidden 일 때 학습/평가 % 지점 (빈 문자열 = 마지막 토큰만)
TAG_SUFFIX="${TAG_SUFFIX:-}"                 # 새 포맷이면 _f5_b5 처럼 (옛 spectral_states 는 빈 문자열)
SPECTRAL_DIR="${SPECTRAL_DIR:-latent/spectral_states}"
if [[ "$SOURCE" == "edges" ]]; then
    OUT_ROOT="${OUT_ROOT:-out/probe_edges_${PART}}"
elif [[ "$SOURCE" == "pct" ]]; then
    OUT_ROOT="${OUT_ROOT:-out/probe_pct_${PCT}}"
elif [[ "$SOURCE" == "hidden" ]]; then
    OUT_ROOT="${OUT_ROOT:-out/probe_hidden_pct}"
    SIGN_MODES="hidden"                      # sign_mode 는 쓰지 않는다; 런 이름 접미사로만
else
    OUT_ROOT="${OUT_ROOT:-out/probe_spectral}"
fi
K="${K:-8}"
SIGN_MODES="${SIGN_MODES:-max data}"
N_EPISODES="${N_EPISODES:-10000}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
SEEDS="${SEEDS:-42 123 456 789 1011}"
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2}"

# task:level — hidden_states 디렉토리 이름 그대로
TASK_LEVELS="${TASK_LEVELS:-
decompose:BabyAI-GoToObj-v0
decompose:BabyAI-GoTo-v0
decompose:BabyAI-Synth-v0
decompose:BabyAI-BossLevel-v0
plan:CustomBabyAI-GoToRedBall-Small-4Dists-v0
plan:CustomBabyAI-GoToRedBall-Medium-40Dists-v0
plan:CustomBabyAI-GoToRedBall-Large-100Dists-v0
plan:CustomBabyAI-GoToRedBall-Ultra-180Dists-v0
predict:BabyAI-GoToObj-v0
predict:BabyAI-BossLevel-v0
}"

# 태스크별 probe 할 최대 step. 그 위 step 은 양성 표본이 너무 적다 (plan step_6 수십 개 등).
declare -A MAX_STEP=( [decompose]=6 [plan]=5 [predict]=4 )

LOG_DIR="logs/probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 런 목록: "task level sign_mode"
JOBS=()
for tl in $TASK_LEVELS; do
    for sm in $SIGN_MODES; do
        JOBS+=("${tl%%:*} ${tl#*:} $sm")
    done
done
read -ra DEVS <<< "$DEVICES"
echo "source=$SOURCE part=$PART pct=$PCT tag_suffix=$TAG_SUFFIX spectral_dir=$SPECTRAL_DIR out=$OUT_ROOT"
echo "runs=${#JOBS[@]} devices=${DEVS[*]} seeds=[$SEEDS] n_episodes=$N_EPISODES logs=$LOG_DIR"
PART_ARGS=()
[[ "$SOURCE" == "edges" ]] && PART_ARGS=(--part "$PART")
[[ "$SOURCE" == "pct" ]] && PART_ARGS=(--pct "$PCT")
[[ "$SOURCE" == "hidden" && -n "$PCTS" ]] && PART_ARGS=(--pct $PCTS)

run_one() {                       # run_one <device> <task> <level> <sign_mode>
    local dev="$1" task="$2" level="$3" sm="$4"
    local tag="k${K}_scaled_sign-${sm}${TAG_SUFFIX}" name="${task}_${level}_${sm}"
    if [[ -f "$OUT_ROOT/$name/summary.json" ]]; then
        echo "[$dev] skip  $name (summary.json 있음)"
        echo "$name exit=skip" >> "$LOG_DIR/status.txt"
        return 0
    fi
    local t0=$(date +%s)
    echo "[$dev] start $name $(date '+%T')"
    local pt_args=(--pt "$SPECTRAL_DIR/$task/$level/*/$tag/chunk_*.pt" --k "$K" --sign-mode "$sm")
    [[ "$SOURCE" == "hidden" ]] && pt_args=(--pt "$HIDDEN_DIR/$task/$level/*/chunk_*.pt")
    python predict/probing.py --source "$SOURCE" "${PART_ARGS[@]}" "${pt_args[@]}" \
        --n-episodes "$N_EPISODES" --sample-seed "$SAMPLE_SEED" \
        --max-step "${MAX_STEP[$task]}" --seeds $SEEDS --device "$dev" \
        --output "$OUT_ROOT/$name" > "$LOG_DIR/${name}.log" 2>&1
    local rc=$?
    echo "[$dev] end   $name $(date '+%T')  elapsed=$(( ($(date +%s) - t0) / 60 ))min  exit=$rc"
    echo "$name exit=$rc elapsed=$(( ($(date +%s) - t0) / 60 ))min" >> "$LOG_DIR/status.txt"
}

# GPU 마다 자기 몫(라운드로빈)을 순서대로 도는 워커를 하나씩 띄운다
for ((i = 0; i < ${#DEVS[@]}; i++)); do
    (
        for ((j = i; j < ${#JOBS[@]}; j += ${#DEVS[@]})); do
            run_one "${DEVS[$i]}" ${JOBS[$j]}
        done
    ) &
done
wait

echo ""; echo "================ SUMMARY ================"
sort "$LOG_DIR/status.txt"
echo "logs: $LOG_DIR   results: $OUT_ROOT/<task>_<level>_<sign>/summary.json"
