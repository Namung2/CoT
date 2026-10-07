#!/usr/bin/env bash
# 프로빙(script/probe.sh)이 끝나길 기다린 뒤, 그 프로빙이 쓴 에피소드(각 레벨 1만 개)만으로
# 구간 앞5/뒤5 토큰별 spectral e_i 를 뽑고(k8, sign data, bf16, 헤더 "Step N:" 토큰 제외),
# 이어서 그 저장본으로 edges 프로빙(both/front/back)을 돌린다.
#
#   1) 대기: pgrep 으로 probe.sh 가 사라질 때까지 60초마다 확인
#   2) spectral: inference/spectral.py --n-front 5 --n-back 5 --episodes <probing episodes.json>
#        → latent/spectral/<task>/<level>/<status>/k8_scaled_sign-data_f5_b5/chunk_*.pt
#      레벨은 task 마다 하나 (아래 TASK_LEVELS). HDD 읽기가 병목이라 순차로 돈다.
#   3) probing: script/probe.sh SOURCE=edges PART={both,front,back} → out/probe_edges_<part>/
#
# 사용:  NO_BG=1 ./script/spectral_edges.sh      (tmux 안에서)   |   ./script/spectral_edges.sh (nohup)
#        SKIP_SPECTRAL=1 NO_BG=1 ./script/spectral_edges.sh   # 저장본이 이미 있을 때 3) 프로빙만
set -uo pipefail
cd "$(dirname "$0")/.."
# 셸에 CUDA_VISIBLE_DEVICES=0 같은 값이 남아 있으면 cuda:1/2 가 "invalid device ordinal" 로 죽는다.
# 장치는 DEVICES(probe.sh) / DEVICE(spectral_edges.sh) 로 명시하므로 여기서 지운다.
unset CUDA_VISIBLE_DEVICES

if [[ "${SPECTRAL_EDGES_BG:-}" != "1" && "${NO_BG:-}" != "1" ]]; then
    nohup_log="nohup_spectral_edges_$(date +%Y%m%d_%H%M%S).out"
    SPECTRAL_EDGES_BG=1 nohup setsid "$0" "$@" > "$nohup_log" 2>&1 &
    echo "started in background: pid=$! log=$nohup_log"
    exit 0
fi

TASK_LEVELS="${TASK_LEVELS:-
decompose:BabyAI-GoToObj-v0
plan:CustomBabyAI-GoToRedBall-Small-4Dists-v0
predict:BabyAI-GoToObj-v0
}"
EPISODES_FROM="${EPISODES_FROM:-out/probe_spectral}"     # <task>_<level>_data/episodes.json 이 있는 곳
K="${K:-8}"; SIGN_MODE="${SIGN_MODE:-data}"; N_FRONT="${N_FRONT:-5}"; N_BACK="${N_BACK:-5}"
DTYPE="${DTYPE:-bfloat16}"; DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-latent/spectral}"
SEEDS="${SEEDS:-42}"
PARTS="${PARTS:-both front back}"
LOG_DIR="logs/spectral_edges_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"

if [[ "${SKIP_SPECTRAL:-}" != "1" ]]; then
echo "[$(date '+%F %T')] waiting for script/probe.sh to finish ..."
while pgrep -f "script/probe.sh" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] probing done"

# ---- 2) spectral f5_b5, 프로빙 에피소드만
EP_JSON=()
for tl in $TASK_LEVELS; do
    task="${tl%%:*}"; level="${tl#*:}"
    f="$EPISODES_FROM/${task}_${level}_${SIGN_MODE}/episodes.json"
    if [[ ! -f "$f" ]]; then echo "missing $f — 이 레벨은 건너뜀"; continue; fi
    EP_JSON+=("$f")
done
if [[ ${#EP_JSON[@]} -eq 0 ]]; then echo "episodes.json 이 하나도 없다"; exit 1; fi

t0=$(date +%s)
echo "[$(date '+%F %T')] spectral start: ${EP_JSON[*]}"
python inference/spectral.py \
    -k "$K" --scale true --sign-mode "$SIGN_MODE" \
    --n-front "$N_FRONT" --n-back "$N_BACK" --dtype "$DTYPE" \
    --episodes "${EP_JSON[@]}" --out-dir "$OUT_DIR" --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/spectral.log"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] spectral end exit=$rc elapsed=$(( ($(date +%s) - t0) / 60 ))min"
[[ $rc -ne 0 ]] && { echo "spectral 실패 — 프로빙은 하지 않음"; exit $rc; }
fi   # SKIP_SPECTRAL

# ---- 3) edges 프로빙
TL_ONE_LINE=$(echo $TASK_LEVELS)
for part in $PARTS; do
    echo "[$(date '+%F %T')] probing edges part=$part"
    SOURCE=edges PART="$part" TAG_SUFFIX="_f${N_FRONT}_b${N_BACK}" SPECTRAL_DIR="$OUT_DIR" \
        SIGN_MODES="$SIGN_MODE" SEEDS="$SEEDS" TASK_LEVELS="$TL_ONE_LINE" NO_BG=1 \
        ./script/probe.sh 2>&1 | tee "$LOG_DIR/probe_edges_${part}.log"
done
echo "[$(date '+%F %T')] all done. results: out/probe_edges_{both,front,back}/  logs: $LOG_DIR"
