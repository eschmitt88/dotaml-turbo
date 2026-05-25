#!/usr/bin/env bash
# Full-run script for foundation-v3-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) extended player_features build  (~3-4h CPU)
#   2) extended rich_cols sidecar build (~3-4h CPU)
#   3) smoke (1 epoch, 50k rows) - verifies plumbing
#   4) foundation_v3 full training (max 30 epochs, ~6-7h GPU)
#
# Uses `python -u` for unbuffered stdout under nohup.
# Per-trial subprocess retry wrapper (defensive; Blackwell torch DataLoader
# bug no longer fires on JEDEC 4800 MT/s but the wrapper costs nothing).
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-24-foundation-v3-740/run_all.sh \
#     > experiments/2026-05-24-foundation-v3-740/full_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=".venv/bin/python -u"
EXP="experiments/2026-05-24-foundation-v3-740"
MAX_RETRIES=3

echo "===== foundation-v3-740 run_all START $(date -Iseconds) ====="

# Reused inputs (item_vocab.json + duration bucket edges).
VOCAB="experiments/2026-05-20-rich-supervision-multitask-740/results/item_vocab.json"
for f in "$VOCAB"; do
  if [ ! -e "$f" ]; then
    echo "REFUSED: required input missing: $f"
    exit 2
  fi
done

# ---- Step 1: build extended player_features parquet ------------------------
EXTENDED_PF_TRAIN="data/snapshots/7.40-2025-12-16/processed/player_features_extended/train.parquet"
EXTENDED_PF_VAL="data/snapshots/7.40-2025-12-16/processed/player_features_extended/val.parquet"
if [ -e "$EXTENDED_PF_TRAIN" ] && [ -e "$EXTENDED_PF_VAL" ]; then
  echo "[$(date -Iseconds)] STEP 1/4: extended player_features already built, skipping."
else
  echo "[$(date -Iseconds)] STEP 1/4: building extended player_features (~3-4h)..."
  $PY $EXP/build_features_extended.py --config $EXP/config.yaml
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -Iseconds)] STEP 1/4 FAILED rc=$rc -- aborting pipeline."
    exit "$rc"
  fi
  echo "[$(date -Iseconds)] STEP 1/4 DONE"
fi

# ---- Step 2: build extended rich_cols sidecar ------------------------------
EXTENDED_RC_TRAIN="data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/train.parquet"
EXTENDED_RC_VAL="data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/val.parquet"
if [ -e "$EXTENDED_RC_TRAIN" ] && [ -e "$EXTENDED_RC_VAL" ]; then
  echo "[$(date -Iseconds)] STEP 2/4: extended rich_cols already built, skipping."
else
  echo "[$(date -Iseconds)] STEP 2/4: building extended rich_cols sidecar (~3-4h)..."
  $PY $EXP/build_rich_cols_extended.py --config $EXP/config.yaml
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -Iseconds)] STEP 2/4 FAILED rc=$rc -- aborting pipeline."
    exit "$rc"
  fi
  echo "[$(date -Iseconds)] STEP 2/4 DONE"
fi

# ---- Step 3: smoke ---------------------------------------------------------
echo "[$(date -Iseconds)] STEP 3/4: smoke foundation_v3"
$PY $EXP/train.py --config $EXP/config.yaml --ablation foundation_v3 --smoke
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 3/4 FAILED on smoke rc=$rc -- aborting."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 3/4 DONE"

# ---- Step 4: foundation_v3 full training ------------------------------------
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

echo "[$(date -Iseconds)] STEP 4/4: foundation_v3"
run_ablation "foundation_v3" "_foundation_v3"
rc_v3=$?
echo "[$(date -Iseconds)] STEP 4/4 DONE rc=$rc_v3"

if [ "$rc_v3" -eq 0 ]; then
  cp "$EXP/metrics_foundation_v3.json" "$EXP/metrics.json"
fi

echo "===== foundation-v3-740 run_all DONE rc_v3=$rc_v3 $(date -Iseconds) ====="
exit $rc_v3
