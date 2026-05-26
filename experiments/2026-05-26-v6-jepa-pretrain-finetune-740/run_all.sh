#!/usr/bin/env bash
# Three-phase run for v6-jepa-pretrain-finetune-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) smoke pretrain (1 epoch, 50k rows + 1 mid-probe on 5k/2k)
#   2) smoke finetune (1 epoch from smoke encoder, 50k rows)
#   3) full pretrain (20 epochs, ~6h)
#   4) full linear probe (5 epochs, ~0.5h)
#   5) full multi-task fine-tune (20 epochs, ~6h)
#
# Uses `python -u` for unbuffered stdout under nohup.
# Per-phase subprocess retry wrapper (defensive; Blackwell torch bug
# closed on JEDEC 4800 MT/s but cost is zero).
#
# Reuses extended player_features + rich_cols sidecar parquets verbatim.
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-26-v6-jepa-pretrain-finetune-740/run_all.sh \
#     > experiments/2026-05-26-v6-jepa-pretrain-finetune-740/full_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=".venv/bin/python -u"
EXP="experiments/2026-05-26-v6-jepa-pretrain-finetune-740"
MAX_RETRIES=3

echo "===== v6-jepa-pretrain-finetune-740 run_all START $(date -Iseconds) ====="

# Required inputs.
EXTENDED_PF_TRAIN="data/snapshots/7.40-2025-12-16/processed/player_features_extended/train.parquet"
EXTENDED_PF_VAL="data/snapshots/7.40-2025-12-16/processed/player_features_extended/val.parquet"
EXTENDED_RC_TRAIN="data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/train.parquet"
EXTENDED_RC_VAL="data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/val.parquet"
VOCAB="experiments/2026-05-20-rich-supervision-multitask-740/results/item_vocab.json"
for f in "$EXTENDED_PF_TRAIN" "$EXTENDED_PF_VAL" "$EXTENDED_RC_TRAIN" "$EXTENDED_RC_VAL" "$VOCAB"; do
  if [ ! -e "$f" ]; then
    echo "REFUSED: required input missing: $f"
    exit 2
  fi
done

run_phase() {
  local phase="$1"; shift
  local smoke_flag="$1"; shift
  local label="$1"; shift
  local attempt=0
  while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    echo "[$(date -Iseconds)] $label attempt $attempt/$MAX_RETRIES START"
    $PY $EXP/train.py --phase "$phase" $smoke_flag
    local rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "[$(date -Iseconds)] $label attempt $attempt SUCCESS"
      return 0
    fi
    echo "[$(date -Iseconds)] $label attempt $attempt FAILED rc=$rc"
    sleep 5
  done
  echo "[$(date -Iseconds)] $label EXHAUSTED retries"
  return 1
}

# ---- Step 1: smoke pretrain ------------------------------------------------
echo "[$(date -Iseconds)] STEP 1/5: smoke pretrain"
run_phase pretrain "--smoke" "smoke-pretrain"
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

# ---- Step 2: smoke finetune ------------------------------------------------
echo "[$(date -Iseconds)] STEP 2/5: smoke finetune"
run_phase finetune "--smoke" "smoke-finetune"
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

# ---- Step 3: full pretrain --------------------------------------------------
echo "[$(date -Iseconds)] STEP 3/5: full pretrain (~6h)"
run_phase pretrain "" "pretrain"
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

# ---- Step 4: full linear probe ----------------------------------------------
echo "[$(date -Iseconds)] STEP 4/5: linear probe (~0.5h)"
run_phase probe "" "linear-probe"
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

# ---- Step 5: full finetune --------------------------------------------------
echo "[$(date -Iseconds)] STEP 5/5: full multi-task fine-tune (~6h)"
run_phase finetune "" "finetune"
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

if [ -f "$EXP/metrics_finetune.json" ]; then
  AUC=$($PY -c "import json; print(json.load(open('$EXP/metrics_finetune.json'))['val_auc'])")
  echo "[$(date -Iseconds)] FINAL val_auc=$AUC"
fi

echo "===== v6-jepa-pretrain-finetune-740 run_all DONE rc=0 $(date -Iseconds) ====="
exit 0
