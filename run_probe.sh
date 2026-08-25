#!/usr/bin/env bash
# raw jsonl -> 분절 -> 유사도 행렬 -> 히트맵 까지 한 번에.
#
# 왜 GoToObj 성공/실패 한 쌍인가
#   블록 수와 크기가 성공/실패에서 거의 같아서(액션 9개 / 93자 median) "실패는
#   블록이 잘아서 유사도가 높게 나온 것"이라는 교란이 없다. BossLevel 은 실패가
#   블록 43개 vs 성공 15개라 그대로 비교하면 안 된다.
#
# 인덱스를 안 주면 find_pair.py 가 구조가 맞는 성공/실패 쌍을 골라준다. 손으로
# 고르면 성공 N=199 / 실패 N=334 처럼 크기가 어긋나 비교가 오염된다.
#
#   ./run_probe.sh                                  # 쌍 자동 선택
#   ./run_probe.sh data/predict_no_thinking.jsonl BabyAI-GoToObj-v0 5 21
#   FORCE=1 ./run_probe.sh                          # 분절을 다시 만든다
#   MODEL=Qwen/Qwen2.5-3B-Instruct ./run_probe.sh   # 32B 받기 전에 싸게 확인
set -euo pipefail

RAW="${1:-data/predict_no_thinking.jsonl}"
LEVEL="${2:-BabyAI-GoToObj-v0}"
S_IDX="${3:-}"        # 비우면 자동 선택
F_IDX="${4:-}"
SEG_DIR="${SEG_DIR:-data/segmented}"
PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
TAG="$(basename "$MODEL")"

cd "$(dirname "$0")"

# ── 1. 분절 ──────────────────────────────────────────────────────────
# segment.py 는 출력 디렉토리를 "w" 로 덮어쓴다. 이미 있으면 건너뛰는 이유는
# 재실행 비용(3~4초)보다 그 사이 손댔을지 모르는 결과를 말없이 날리는 쪽이
# 위험해서다. 다시 만들려면 FORCE=1.
echo "[1/4] 분절  $RAW -> $SEG_DIR"
if [ -d "$SEG_DIR" ] && [ -z "${FORCE:-}" ]; then
    echo "  $SEG_DIR 이미 있음 — 건너뜀 (다시 만들려면 FORCE=1)"
else
    $PY segment.py "$RAW" "$SEG_DIR"
fi

for g in success fail; do
    f="$SEG_DIR/$LEVEL/$g.jsonl"
    [ -f "$f" ] || { echo "없음: $f" >&2; exit 1; }
done

# ── 2. 비교할 쌍 고르기 ──────────────────────────────────────────────
if [ -z "$S_IDX" ] || [ -z "$F_IDX" ]; then
    echo "[2/4] 쌍 선택  $SEG_DIR/$LEVEL"
    read -r S_IDX F_IDX < <($PY find_pair.py "$SEG_DIR/$LEVEL" --quiet)
    echo "  success idx=$S_IDX   fail idx=$F_IDX"
else
    echo "[2/4] 쌍 지정  success idx=$S_IDX  fail idx=$F_IDX"
fi

# ── 3. 유사도 행렬 ───────────────────────────────────────────────────
# probe_gram.py 는 에피소드 하나짜리라 모델을 두 번 올린다. 두 개뿐이라 감수한다.
#
# --out 을 직접 준다. 기본 이름은 id 로 정해져서 스크립트가 미리 알 수 없고,
# 그러면 히트맵 단계에서 visual/*.pt 로 긁어야 하는데 그러면 이전 run 이나 다른
# 모델로 만든 .pt 까지 딸려 들어간다 (필드가 안 맞으면 거기서 죽는다).
echo "[3/4] 유사도 행렬  $LEVEL  model=$MODEL"
OUTS=()
for pair in "success:$S_IDX" "fail:$F_IDX"; do
    g="${pair%%:*}"; idx="${pair##*:}"
    o="visual/${g}_${LEVEL}_${idx}_${TAG}"
    echo "  ── $g  idx=$idx ──"
    $PY probe_gram.py "$SEG_DIR/$LEVEL/$g.jsonl" \
        --index "$idx" --model "$MODEL" --out "$o"
    OUTS+=("$o.pt")
done

# ── 3. 그림 ──────────────────────────────────────────────────────────
echo "[4/4] 히트맵"
if ! $PY -c "import matplotlib" 2>/dev/null; then
    echo "  matplotlib 설치"
    $PY -m pip install --quiet matplotlib
fi
$PY visual/heatmap.py "${OUTS[@]}" --drift --states --growth

echo
echo "완료. visual/ 아래를 보세요."
echo "  .centered.png  토큰 x 토큰. 대각 블록이 밝고 비대각이 어두우면 스텝 안은"
echo "                 가깝고 스텝 간은 멀다는 뜻. 단 'distance-matched gap' 을 보라"
echo "                 — 그냥 gap 은 '같은 블록 토큰은 원래 붙어 있다' 에 속는다."
echo "  .states.png    블록 끝 지점의 h 끼리. MDP 로 치면 상태 궤적이다. 인접 상태가"
echo "                 닮고 먼 상태가 안 닮으면 궤적이 실제로 진행하는 것."
echo "  .drift.png     시점 x 시점(누적). 포함 관계 때문에 0.99 로 포화하니 단독으로"
echo "                 믿지 말고 .states.png 와 같이 볼 것."
echo "  .growth.png    prefix 를 늘려가며 gap 이 어떻게 변하는지."
echo
echo "  raw 코사인으로 대조: python visual/heatmap.py ${OUTS[*]} --which cosine"
