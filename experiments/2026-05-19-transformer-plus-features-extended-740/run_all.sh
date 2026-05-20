#!/usr/bin/env bash
# Full-run convenience script for transformer-plus-features-extended-740.
#
# Single ablation in a fresh Python subprocess for Blackwell + torch 2.9
# stability (per-trial subprocess isolation; see
# docs/decisions/0001-per-trial-subprocess-isolation.md):
#   1) transformer_plus_features (~25-50 min — up to 30 epochs with
#      patience=5 early stopping on val_log_loss)
#
# With auto-retry on rc!=0 (intermittent Blackwell DataLoader crashes).
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-19-transformer-plus-features-extended-740/run_all.sh \
#     > /tmp/dotaml_tpfe.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=.venv/bin/python
EXP=experiments/2026-05-19-transformer-plus-features-extended-740
MAX_RETRIES=3

run_ablation() {
  local ab="$1"
  local sfx="$2"
  local attempt=0
  while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    echo "[$(date -Iseconds)] $ab attempt $attempt/$MAX_RETRIES START"
    "$PY" "$EXP/train.py" --ablation "$ab" --metrics-suffix "$sfx"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "[$(date -Iseconds)] $ab attempt $attempt SUCCESS"
      return 0
    fi
    echo "[$(date -Iseconds)] $ab attempt $attempt FAILED rc=$rc"
    sleep 2
  done
  echo "[$(date -Iseconds)] $ab EXHAUSTED retries — moving on"
  return 1
}

echo "===== transformer-plus-features-extended-740 run_all START $(date -Iseconds) ====="
run_ablation "transformer_plus_features" "_transformer_plus_features"
prim_rc=$?

echo "===== run_all DONE primary_rc=$prim_rc $(date -Iseconds) ====="
exit "$prim_rc"
