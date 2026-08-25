#!/usr/bin/env bash
# raw jsonl -> 분절 -> 유사도 행렬 -> 히트맵 까지 한 번에.
#
# 왜 GoToObj 성공/실패 한 쌍인가
#   블록 수와 크기가 성공/실패에서 거의 같아서(액션 9개 / 93자 median) "실패는
#   블록이 잘아서 유사도가 높게 나온 것"이라는 교란이 없다. BossLevel 은 실패가
#   블록 43개 vs 성공 15개라 그대로 비교하면 안 된다.
#
#   ./run_probe.sh
#   ./run_probe.sh data/predict_no_thinking.jsonl BabyAI-BossLevel-v0 3
#   FORCE=1 ./run_probe.sh                          # 분절을 다시 만든다
#   MODEL=Qwen/Qwen2.5-3B-Instruct ./run_probe.sh   # 32B 받기 전에 싸게 확인
set -euo pipefail

RAW="${1:-data/predict_no_thinking.jsonl}"
LEVEL="${2:-BabyAI-GoToObj-v0}"
INDEX="${3:-0}"
SEG_DIR="${SEG_DIR:-data/segmented}"
PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
TAG="$(basename "$MODEL")"

cd "$(dirname "$0")"

# ── 1. 분절 ──────────────────────────────────────────────────────────
# segment.py 는 출력 디렉토리를 "w" 로 덮어쓴다. 이미 있으면 건너뛰는 이유는
# 재실행 비용(3~4초)보다 그 사이 손댔을지 모르는 결과를 말없이 날리는 쪽이
# 위험해서다. 다시 만들려면 FORCE=1.
echo "[1/3] 분절  $RAW -> $SEG_DIR"
if [ -d "$SEG_DIR" ] && [ -z "${FORCE:-}" ]; then
    echo "  $SEG_DIR 이미 있음 — 건너뜀 (다시 만들려면 FORCE=1)"
else
    $PY segment.py "$RAW" "$SEG_DIR"
fi

for g in success fail; do
    f="$SEG_DIR/$LEVEL/$g.jsonl"
    [ -f "$f" ] || { echo "없음: $f" >&2; exit 1; }
done

# ── 2. 유사도 행렬 ───────────────────────────────────────────────────
# 성공 1 + 실패 1. probe_gram.py 는 에피소드 하나짜리라 모델을 두 번 올린다.
# 두 개뿐이라 감수한다.
#
# --out 을 직접 준다. 기본 이름은 id 로 정해져서 스크립트가 미리 알 수 없고,
# 그러면 히트맵 단계에서 visual/*.pt 로 긁어야 하는데 그러면 이전 run 이나 다른
# 모델로 만든 .pt 까지 딸려 들어간다 (필드가 안 맞으면 거기서 죽는다).
echo "[2/3] 유사도 행렬  $LEVEL index=$INDEX  model=$MODEL"
OUTS=()
for g in success fail; do
    o="visual/${g}_${LEVEL}_${INDEX}_${TAG}"
    echo "  ── $g ──"
    $PY probe_gram.py "$SEG_DIR/$LEVEL/$g.jsonl" \
        --index "$INDEX" --model "$MODEL" --out "$o"
    OUTS+=("$o.pt")
done

# ── 3. 그림 ──────────────────────────────────────────────────────────
echo "[3/3] 히트맵"
if ! $PY -c "import matplotlib" 2>/dev/null; then
    echo "  matplotlib 설치"
    $PY -m pip install --quiet matplotlib
fi
$PY visual/heatmap.py "${OUTS[@]}"

echo
echo "완료. visual/ 아래 .pt 와 .centered.png 를 보세요."
echo "  대각 블록이 밝고 비대각이 어두우면 '스텝 안은 가깝고 스텝 간은 멀다' 가 맞는 것."
echo "  action 쪽 gap 이 step 보다 크면 액션 단위가 더 응집력 있는 단위."
echo "  raw 코사인으로 대조: python visual/heatmap.py ${OUTS[*]} --which cosine"
