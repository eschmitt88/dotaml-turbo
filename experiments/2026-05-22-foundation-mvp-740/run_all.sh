#!/usr/bin/env bash
# Full-run convenience script for foundation-mvp-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) smoke (all 3 ablations, 1 epoch, 50k rows) -- verifies plumbing.
#   2) baseline_multitask_repro -- 5M-scale anchor, NO PMAE/patch/team-bias.
#   3) foundation_mvp           -- PRIMARY, full design.
#   4) foundation_no_patch_token -- ablation.
#
# Reuses multitask-740's rich_cols sidecar + item_vocab.json -- so no new
# data build is needed for the MVP. (See README "Important deviation from
# proposal" for why we are not extending the training window to Aug 2025.)
#
# Uses `python -u` (Python defaults to block-buffered stdout under nohup
# which hides progress; lesson from multitask-740 saga).
# Per-trial subprocess isolation (MAX_RETRIES=3) carried over from prior
# experiments as harmless defensive insurance (Blackwell torch DataLoader
# bug no longer fires on JEDEC 4800 MT/s but the wrapper costs nothing).
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-22-foundation-mvp-740/run_all.sh \
#     > experiments/2026-05-22-foundation-mvp-740/full_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=".venv/bin/python -u"
EXP="experiments/2026-05-22-foundation-mvp-740"
MAX_RETRIES=3

echo "===== foundation-mvp-740 run_all START $(date -Iseconds) ====="

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

# ---- Step 1: smoke all three ablations -------------------------------------
echo "[$(date -Iseconds)] STEP 1/4: smoke all ablations"
for ab in baseline_multitask_repro foundation_mvp foundation_no_patch_token; do
  $PY $EXP/train.py --config $EXP/config.yaml --ablation "$ab" --smoke
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -Iseconds)] STEP 1 FAILED on smoke $ab rc=$rc"
    exit "$rc"
  fi
done
echo "[$(date -Iseconds)] STEP 1/4: smoke DONE"

# ---- Step 2: baseline_multitask_repro --------------------------------------
echo "[$(date -Iseconds)] STEP 2/4: baseline_multitask_repro"
run_ablation "baseline_multitask_repro" "_baseline_multitask_repro"
rc_baseline=$?
echo "[$(date -Iseconds)] STEP 2/4 DONE rc=$rc_baseline"

# ---- Step 3: foundation_mvp (PRIMARY) --------------------------------------
echo "[$(date -Iseconds)] STEP 3/4: foundation_mvp"
run_ablation "foundation_mvp" "_foundation_mvp"
rc_primary=$?
echo "[$(date -Iseconds)] STEP 3/4 DONE rc=$rc_primary"

# ---- Step 4: foundation_no_patch_token -------------------------------------
echo "[$(date -Iseconds)] STEP 4/4: foundation_no_patch_token"
run_ablation "foundation_no_patch_token" "_foundation_no_patch_token"
rc_no_patch=$?
echo "[$(date -Iseconds)] STEP 4/4 DONE rc=$rc_no_patch"

# Promote PRIMARY to canonical metrics.json.
if [ "$rc_primary" -eq 0 ]; then
  cp "$EXP/metrics_foundation_mvp.json" "$EXP/metrics.json"
fi

echo "===== foundation-mvp-740 run_all DONE baseline=$rc_baseline primary=$rc_primary no_patch=$rc_no_patch $(date -Iseconds) ====="
exit "$rc_primary"
