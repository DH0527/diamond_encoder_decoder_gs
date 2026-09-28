#!/usr/bin/env bash
# 저장된 args.json(configs/*.args.json) 하나로 학습을 띄운다. 처음부터 또는 재개.
#
#   # 처음부터 (배경 클리핑 수정 설정)
#   bash scripts/launch_from_config.sh configs/B1_bgcap.args.json
#
#   # 체크포인트에서 재개 (안정화 설정)
#   bash scripts/launch_from_config.sh configs/B1g_stable.args.json \
#        --resume runs/B1_bgcap_xxx/ckpt_step00030000.pt
#
#   # 값 덮어쓰기 (여러 번 가능) + 백그라운드
#   bash scripts/launch_from_config.sh configs/B1_bgcap.args.json \
#        --set max_steps=40000 --set w_splat_area=0.1 --detached
#
#   # GPU 를 쓰기 전에 확인만: 계약 검사 + 데이터 경로 존재 + 최종 명령 출력 후 종료
#   bash scripts/launch_from_config.sh configs/B1_bgcap.args.json --dry-run
#
# 환경변수: GPUS(0,1,2) NPROC(3) TAG MASTER_PORT(29531) PYTHON TORCHRUN
#
# 왜 계약 플래그를 검사하나: train.py 의 파서는 새로 추가된 플래그(예:
# normalize_pooler_xyz)의 기본값을 1 로 둔다. args.json 에 그 키가 없으면 재개 시
# 기본값이 조용히 들어가 체크포인트가 학습된 적 없는 코드 경로가 켜진다.
# 실제로 이 때문에 21.5 dB 모델이 500 스텝 만에 16 dB 로 무너진 적이 있다
# (SETUP_KR.md §8.1). 그래서 키가 하나라도 빠지면 실행하지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-29531}

if [[ $# -lt 1 || "$1" == -* ]]; then
  sed -n '2,15p' "$0"; exit 1
fi
CONFIG="$1"; shift
[[ -f "$CONFIG" ]] || { echo "config 없음: $CONFIG" >&2; exit 1; }

RESUME=""
DETACHED=0
DRYRUN=0
SETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME="$2"; shift 2 ;;
    --set)    SETS+=(--set "$2"); shift 2 ;;
    --detached) DETACHED=1; shift ;;
    --dry-run)  DRYRUN=1; shift ;;
    *) echo "알 수 없는 인자: $1" >&2; exit 1 ;;
  esac
done

NAME=$(basename "$CONFIG" .args.json)
TAG=${TAG:-${NAME}_$(date +%Y%m%d_%H%M%S)}
OUT="runs/$TAG"
LOG="runs/$TAG.log"

# 계약 플래그 검사 (§8.1)
"$PYTHON" - "$CONFIG" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
need = ["normalize_pooler_xyz", "count_aware_template", "attr_slot_mask",
        "shared_cell_owner", "holdout_own_photo", "keep_extra_fullres",
        "reuse_prev_vis", "eval_pred_mask"]
miss = [k for k in need if k not in d]
if miss:
    sys.exit(f"[중단] {sys.argv[1]} 에 계약 플래그가 없습니다: {miss}\n"
             f"       파서 기본값이 들어가 체크포인트와 다른 모델이 됩니다. 키를 명시하세요.")
print("계약 플래그 확인: " + " ".join(f"{k}={d[k]}" for k in need))
PY

if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "체크포인트 없음: $RESUME" >&2; exit 1
fi

if [[ $DRYRUN -eq 1 ]]; then
  # 새 서버에서 가장 흔한 실패는 데이터/자산 경로가 옛 서버를 가리키는 것 (SETUP_KR.md §5.2)
  "$PYTHON" - "$CONFIG" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))
bad = 0
for key in ("root", "stats_path", "scene_anchors", "split_path", "photo_map", "view_pool"):
    val = str(d.get(key) or "")
    for p in [x for x in val.split(",") if x]:
        ok = os.path.exists(p)
        bad += not ok
        print(f"  {'ok ' if ok else '없음'}  {key:14s} {p}")
if bad:
    print(f"\n경로 {bad}개가 없습니다. tools/relocate_assets.py --assets configs 로 옮기거나 데이터를 준비하세요.")
PY
  echo
  echo "[dry-run] 실행할 명령:"
  printf '  CUDA_VISIBLE_DEVICES=%s %s --standalone --nproc_per_node=%s --master_port=%s train.py \\\n' \
    "$GPUS" "$TORCHRUN" "$NPROC" "$MASTER_PORT"
  "$PYTHON" scripts/argv_from_argsjson.py --src "$CONFIG" \
    --set "out_dir=$OUT" --set "resume=$RESUME" --set "init_from=" "${SETS[@]}" \
    | tr '\0' '\n' | grep -E '^--(out_dir|resume|max_steps|stats_path|root|w_splat_area|lr)$' -A1 \
    | grep -v '^--$' | paste -d' ' - - | sed 's/^/    /'
  echo "  (전체 인자 $("$PYTHON" scripts/argv_from_argsjson.py --src "$CONFIG" | tr '\0' '\n' | grep -c '^--')개 중 주요 항목만 표시)"
  exit 0
fi

if [[ $DETACHED -eq 1 && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs "$OUT"
  ARGS=("$CONFIG")
  [[ -n "$RESUME" ]] && ARGS+=(--resume "$RESUME")
  ARGS+=("${SETS[@]}")
  CAN3TOK_DETACHED=1 TAG="$TAG" GPUS="$GPUS" NPROC="$NPROC" MASTER_PORT="$MASTER_PORT" \
    PYTHON="$PYTHON" TORCHRUN="$TORCHRUN" \
    nohup setsid bash "$0" "${ARGS[@]}" >>"$LOG" 2>&1 &
  echo "$!" >"$OUT/run.pid"
  echo "Detached PID=$!  log=$LOG  out=$OUT"
  exit 0
fi

mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# resume 과 out_dir 은 항상 덮어쓴다: 스냅샷 안의 옛 경로를 그대로 쓰면 안 된다.
mapfile -d '' ARGV < <(
  "$PYTHON" scripts/argv_from_argsjson.py --src "$CONFIG" \
    --set "out_dir=$OUT" \
    --set "resume=$RESUME" \
    --set "init_from=" \
    "${SETS[@]}"
)

echo "launch $NAME | GPUs=$GPUS nproc=$NPROC out=$OUT"
[[ -n "$RESUME" ]] && echo "  resume $RESUME" || echo "  from scratch"
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py "${ARGV[@]}"
