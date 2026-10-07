#!/usr/bin/env bash
# decompose 전 레벨을 "Step N" 마커만 벗기는 새 prepare_output 으로 다시 추출한다.
# 출력은 latent/hidden_states_marker (기존 latent/hidden_states 는 건드리지 않는다).
#
# GPU 하나(기본 1)에서 레벨을 순서대로 돈다. Qwen3-32B bf16(~65GB) 은 GPU 하나(98GB)에 올라간다.
# 레벨 디렉토리에 chunk_*.pt 가 이미 있으면 script/resume_extract.py 로 이어서 뽑고(기존 청크 유지),
# 없으면 inference/main.py 로 처음부터 뽑는다. extract 는 success/failure 를 한 번에 저장하므로
# --status success 하나로 충분하다 (inference.sh 와 같음).
#
#   tmux cot 창에서:  conda activate cot_llm313 && ./script/extract_marker.sh
#   GPU 바꾸기:       GPU=2 ./script/extract_marker.sh
#   진행:            tail -f logs/extract_marker_*/decompose_<level>.log
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-1}
HIDDEN_DIR=${HIDDEN_DIR:-latent/hidden_states_marker}
LEVELS=(BabyAI-GoToObj-v0 BabyAI-BossLevel-v0 BabyAI-Synth-v0 BabyAI-GoTo-v0)
LOG_DIR="logs/extract_marker_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "gpu=$GPU hidden-dir=$HIDDEN_DIR logs=$LOG_DIR"

for level in "${LEVELS[@]}"; do
    name="decompose_${level}"
    if compgen -G "$HIDDEN_DIR/decompose/$level/*/chunk_*.pt" > /dev/null; then
        mode=resume
        cmd=(python script/resume_extract.py --task decompose --level "$level" --hidden-dir "$HIDDEN_DIR")
    else
        mode=fresh
        cmd=(python inference/main.py --task decompose --level "$level" --status success --no-spectral --hidden-dir "$HIDDEN_DIR")
    fi
    echo "===== [$name] gpu=$GPU $mode start $(date '+%F %T') ====="
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU "${cmd[@]}" > "$LOG_DIR/$name.log" 2>&1
    rc=$?
    echo "===== [$name] gpu=$GPU $mode end $(date '+%F %T')  elapsed=$(( ($(date +%s) - t0) / 60 ))min  exit=$rc ====="
done
echo "all done: $LOG_DIR"
