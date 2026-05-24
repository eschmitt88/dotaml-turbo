#!/usr/bin/env bash
# Full-run script for foundation-component-isolation-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) smoke (all 3 isolation ablations, 1 epoch, 50k rows) -- verifies plumbing.
#   2) iso_uwso        -- baseline + UW-SO (with init-loss normalization fix).
#   3) iso_pmae        -- baseline + PMAE (with EMA-teacher fix).
#   4) iso_teambias    -- baseline + (team_q, team_k) attention bias.
#
# Reuses foundation-mvp-740's rich_cols sidecar + item_vocab.json (same window).
#
# Uses `python -u` for unbuffered stdout under nohup.
# Per-trial subprocess isolation (MAX_RETRIES=3) carried from prior experiments
# as defensive insurance (Blackwell torch DataLoader bug no longer fires on
# JEDEC 4800 MT/s but the wrapper costs nothing).
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-23-foundation-component-isolation-740/run_all.sh \
#     > experiments/2026-05-23-foundation-component-isolation-740/full_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=".venv/bin/python -u"
EXP="experiments/2026-05-23-foundation-component-isolation-740"
MAX_RETRIES=3

echo "===== foundation-component-isolation-740 run_all START $(date -Iseconds) ====="

# Pre-check: confirm reused inputs exist.
VOCAB="experiments/2026-05-20-rich-supervision-multitask-740/results/item_vocab.json"
RICH_TRAIN="data/snapshots/7.40-2025-12-16/processed/rich_cols/train.parquet"
RICH_VAL="data/snapshots/7.40-2025-12-16/processed/rich_cols/val.parquet"
CLEAN_TRAIN="data/snapshots/7.40-2025-12-16/processed/player_features_prepatch_clean/train.parquet"
CLEAN_VAL="data/snapshots/7.40-2025-12-16/processed/player_features_prepatch_clean/val.parquet"
for f in "$VOCAB" "$RICH_TRAIN" "$RICH_VAL" "$CLEAN_TRAIN" "$CLEAN_VAL"; do
  if [ ! -e "$f" ]; then
    echo "REFUSED: required input missing: $f"
    exit 2
  fi
done
echo "[$(date -Iseconds)] inputs verified."

run_ablation() {
  local ab="$1"
  local sfx="$2"
  local extra_flags="${3:-}"
  local attempt=0
  while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    echo "[$(date -Iseconds)] train.py $ab attempt $attempt/$MAX_RETRIES START $extra_flags"
    $PY $EXP/train.py --config $EXP/config.yaml \
      --ablation "$ab" --metrics-suffix "$sfx" $extra_flags
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

# ---- Step 1: smoke all three isolation ablations ---------------------------
echo "[$(date -Iseconds)] STEP 1/4: smoke all isolation ablations"
for ab in iso_uwso iso_pmae iso_teambias; do
  $PY $EXP/train.py --config $EXP/config.yaml --ablation "$ab" --smoke
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -Iseconds)] STEP 1 FAILED on smoke $ab rc=$rc"
    exit "$rc"
  fi
done
echo "[$(date -Iseconds)] STEP 1/4: smoke DONE"

# ---- Step 2: iso_uwso ------------------------------------------------------
echo "[$(date -Iseconds)] STEP 2/4: iso_uwso"
run_ablation "iso_uwso" "_iso_uwso"
rc_uwso=$?
echo "[$(date -Iseconds)] STEP 2/4 DONE rc=$rc_uwso"

# ---- Step 3: iso_pmae ------------------------------------------------------
echo "[$(date -Iseconds)] STEP 3/4: iso_pmae"
run_ablation "iso_pmae" "_iso_pmae"
rc_pmae=$?
echo "[$(date -Iseconds)] STEP 3/4 DONE rc=$rc_pmae"

# ---- Step 4: iso_teambias --------------------------------------------------
echo "[$(date -Iseconds)] STEP 4/4: iso_teambias"
run_ablation "iso_teambias" "_iso_teambias"
rc_teambias=$?
echo "[$(date -Iseconds)] STEP 4/4 DONE rc=$rc_teambias"

# Promote whichever ablation has the best val_auc to canonical metrics.json,
# but we don't auto-pick here -- main agent will read each metrics_*.json and
# interpret. Default copy: iso_uwso (alphabetical first by name).
if [ "$rc_uwso" -eq 0 ]; then
  cp "$EXP/metrics_iso_uwso.json" "$EXP/metrics.json"
fi

echo "===== foundation-component-isolation-740 run_all DONE iso_uwso=$rc_uwso iso_pmae=$rc_pmae iso_teambias=$rc_teambias $(date -Iseconds) ====="
# Exit non-zero iff ALL three failed.
if [ "$rc_uwso" -ne 0 ] && [ "$rc_pmae" -ne 0 ] && [ "$rc_teambias" -ne 0 ]; then
  exit 1
fi
exit 0
