#!/usr/bin/env bash
# Full-run convenience script for upstream-data-cleanup-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) build_features.py       — rebuild CLEAN prepatch parquet (~3 h, CPU-bound)
#                                 to data/snapshots/.../processed/player_features_prepatch_clean/
#   2) train_lgbm.py --ablation features_only  — LightGBM A/B (~30 min)
#   3) train_tfm.py  --ablation transformer_plus_features  — Transformer A/B (~25 min)
#                                 with per-trial subprocess isolation retry wrapper
#                                 (MAX_RETRIES=3) per Blackwell torch DataLoader bug memory
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-19-upstream-data-cleanup-740/run_all.sh \
#     > /tmp/dotaml_cleanup.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=.venv/bin/python
EXP=experiments/2026-05-19-upstream-data-cleanup-740
MAX_RETRIES=3

echo "===== upstream-data-cleanup-740 run_all START $(date -Iseconds) ====="

# ---- Step 1: rebuild clean parquet (CPU-bound; no retry — failures here are
# almost always data/config issues, not transient) -----------------------------
echo "[$(date -Iseconds)] STEP 1/3: build_features.py START"
$PY $EXP/build_features.py --config $EXP/config.yaml
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 1 FAILED rc=$rc — pipeline halted."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 1/3: build_features.py DONE"

# ---- Step 2: LightGBM features_only ablation (CPU-bound; no retry) ----------
echo "[$(date -Iseconds)] STEP 2/3: train_lgbm.py features_only START"
$PY $EXP/train_lgbm.py --config $EXP/config.yaml \
  --ablation features_only \
  --metrics-suffix _ablation_features_only
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 2 FAILED rc=$rc — pipeline halted."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 2/3: train_lgbm.py features_only DONE"

# ---- Step 3: Transformer + features (retry wrapper for Blackwell DataLoader) -
run_tfm() {
  local ab="$1"
  local sfx="$2"
  local attempt=0
  while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    echo "[$(date -Iseconds)] STEP 3/3: train_tfm $ab attempt $attempt/$MAX_RETRIES START"
    $PY $EXP/train_tfm.py --config $EXP/config.yaml \
      --ablation "$ab" --metrics-suffix "$sfx"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "[$(date -Iseconds)] STEP 3/3: $ab attempt $attempt SUCCESS"
      return 0
    fi
    echo "[$(date -Iseconds)] STEP 3/3: $ab attempt $attempt FAILED rc=$rc"
    sleep 5
  done
  echo "[$(date -Iseconds)] STEP 3/3: $ab EXHAUSTED retries — moving on"
  return 1
}

run_tfm "transformer_plus_features" "_transformer_plus_features"
tfm_rc=$?

echo "===== upstream-data-cleanup-740 run_all DONE tfm_rc=$tfm_rc $(date -Iseconds) ====="
exit "$tfm_rc"
