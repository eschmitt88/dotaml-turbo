#!/usr/bin/env bash
# Full-run script for v4-iso-teambias-extended-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) smoke (1 epoch, 50k rows) -- verifies plumbing on the single ablation
#   2) v4_iso_teambias_extended full training (max 30 epochs, ~6h GPU)
#
# Uses `python -u` for unbuffered stdout under nohup.
# Per-trial subprocess retry wrapper (defensive against the rare segfault
# pattern; Blackwell torch DataLoader bug no longer fires on JEDEC 4800
# MT/s but the wrapper costs nothing).
#
# No data build phase. Reuses extended player_features + rich_cols parquets
# from foundation-v3-740 verbatim. No account_id sidecar / player vocab
# needed because use_player_embedding=false.
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-25-v4-iso-teambias-extended-740/run_all.sh \
#     > experiments/2026-05-25-v4-iso-teambias-extended-740/full_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=".venv/bin/python -u"
EXP="experiments/2026-05-25-v4-iso-teambias-extended-740"
ABL="v4_iso_teambias_extended"
MAX_RETRIES=3

echo "===== v4-iso-teambias-extended-740 run_all START $(date -Iseconds) ====="

# Reused inputs.
EXTENDED_PF_TRAIN="data/snapshots/7.40-2025-12-16/processed/player_features_extended/train.parquet"
EXTENDED_PF_VAL="data/snapshots/7.40-2025-12-16/processed/player_features_extended/val.parquet"
EXTENDED_RC_TRAIN="data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/train.parquet"
EXTENDED_RC_VAL="data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/val.parquet"
VOCAB="experiments/2026-05-20-rich-supervision-multitask-740/results/item_vocab.json"
PRIOR_SIDECAR_TRAIN="experiments/2026-05-19-player-embedding-prelim-740/sidecar/account_ids_train.parquet"
PRIOR_SIDECAR_VAL="experiments/2026-05-19-player-embedding-prelim-740/sidecar/account_ids_val.parquet"
# The extended-train account_id sidecar is loaded but NOT consumed
# (use_player_embedding=false). If it's missing, data.py routes those
# rows to anonymous which has no effect on the model.
for f in "$EXTENDED_PF_TRAIN" "$EXTENDED_PF_VAL" "$EXTENDED_RC_TRAIN" "$EXTENDED_RC_VAL" \
         "$VOCAB" "$PRIOR_SIDECAR_TRAIN" "$PRIOR_SIDECAR_VAL"; do
  if [ ! -e "$f" ]; then
    echo "REFUSED: required input missing: $f"
    exit 2
  fi
done

# ---- Step 1: smoke (single ablation) --------------------------------------
echo "[$(date -Iseconds)] STEP 1/2: smoke $ABL"
$PY $EXP/train.py --config $EXP/config.yaml --ablation $ABL --smoke
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 1/2 (smoke $ABL) FAILED rc=$rc -- aborting."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 1/2 DONE"

# ---- Step 2: full training (per-trial subprocess retry) -------------------
run_ablation() {
  local ab="$1"
  local sfx="$2"
  local attempt=0
  while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    echo "[$(date -Iseconds)] train.py $ab attempt $attempt/$MAX_RETRIES START"
    $PY $EXP/train.py --config $EXP/config.yaml \
      --ablation "$ab" --metrics-suffix "$sfx"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "[$(date -Iseconds)] $ab attempt $attempt SUCCESS"
      return 0
    fi
    echo "[$(date -Iseconds)] $ab attempt $attempt FAILED rc=$rc"
    sleep 5
  done
  echo "[$(date -Iseconds)] $ab EXHAUSTED retries"
  return 1
}

echo "[$(date -Iseconds)] STEP 2/2: $ABL full training"
run_ablation "$ABL" "_$ABL"
rc_v4=$?
echo "[$(date -Iseconds)] STEP 2/2 DONE rc=$rc_v4"

# Copy the metrics to metrics.json for downstream tooling.
if [ "$rc_v4" -eq 0 ]; then
  cp "$EXP/metrics_${ABL}.json" "$EXP/metrics.json"
  AUC=$($PY -c "import json; print(json.load(open('$EXP/metrics.json'))['val_auc'])")
  echo "metrics.json <- $ABL ($AUC)"
fi

echo "===== v4-iso-teambias-extended-740 run_all DONE rc=$rc_v4 $(date -Iseconds) ====="
exit "$rc_v4"
