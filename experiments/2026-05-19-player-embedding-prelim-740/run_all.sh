#!/usr/bin/env bash
# Full-run convenience script for player-embedding-prelim-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) build_account_sidecar.py — walk raw history JSON, emit train/val
#      account_id sidecar parquets keyed by match_id (~30-45 min on 80 days).
#      Single CPU walk; no retry (failures here are config/data issues).
#   2) build_vocab.py             — stream the train sidecar, build top-N
#      vocab.json (~1-3 min). No retry.
#   3) train.py --ablation baseline_extended_clean   — sanity replication
#      of cleanup-740. ~25 min. Per-trial subprocess isolation retry wrapper.
#   4) train.py --ablation with_player_embedding     — PRIMARY. ~30 min.
#      Per-trial subprocess isolation retry wrapper.
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-19-player-embedding-prelim-740/run_all.sh \
#     > experiments/2026-05-19-player-embedding-prelim-740/full_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=.venv/bin/python
EXP=experiments/2026-05-19-player-embedding-prelim-740
MAX_RETRIES=3

echo "===== player-embedding-prelim-740 run_all START $(date -Iseconds) ====="

# ---- Step 1: build account_id sidecar (CPU-bound; no retry) -----------------
echo "[$(date -Iseconds)] STEP 1/4: build_account_sidecar.py START"
$PY $EXP/build_account_sidecar.py --config $EXP/config.yaml
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 1 FAILED rc=$rc — pipeline halted."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 1/4: build_account_sidecar.py DONE"

# ---- Step 2: build vocab from sidecar (CPU-bound; no retry) -----------------
echo "[$(date -Iseconds)] STEP 2/4: build_vocab.py START"
$PY $EXP/build_vocab.py --config $EXP/config.yaml
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 2 FAILED rc=$rc — pipeline halted."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 2/4: build_vocab.py DONE"

# ---- Steps 3-4: Transformer ablations (Blackwell DataLoader retry wrapper) --
run_tfm() {
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
  echo "[$(date -Iseconds)] $ab EXHAUSTED retries — moving on"
  return 1
}

echo "[$(date -Iseconds)] STEP 3/4: baseline_extended_clean START"
run_tfm "baseline_extended_clean" "_baseline_extended_clean"
rc_baseline=$?
echo "[$(date -Iseconds)] STEP 3/4: baseline_extended_clean DONE rc=$rc_baseline"

echo "[$(date -Iseconds)] STEP 4/4: with_player_embedding START"
run_tfm "with_player_embedding" "_with_player_embedding"
rc_primary=$?
echo "[$(date -Iseconds)] STEP 4/4: with_player_embedding DONE rc=$rc_primary"

# Copy the primary metrics file to canonical metrics.json (per experiment rule).
if [ "$rc_primary" -eq 0 ]; then
  cp "$EXP/metrics_with_player_embedding.json" "$EXP/metrics.json"
fi

echo "===== player-embedding-prelim-740 run_all DONE baseline_rc=$rc_baseline primary_rc=$rc_primary $(date -Iseconds) ====="
exit "$rc_primary"
