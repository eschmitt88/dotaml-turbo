#!/usr/bin/env bash
# Full-run convenience script for transformer-plus-features-740.
#
# Two ablations, each in a fresh Python subprocess for Blackwell + torch 2.9
# stability (per-trial subprocess isolation; see
# docs/decisions/0001-per-trial-subprocess-isolation.md):
#   1) architecture_only         (~12-15 min, sanity vs plateau-architectures-740)
#   2) transformer_plus_features (~12-15 min, PRIMARY)
#
# With auto-retry on rc!=0 (intermittent Blackwell DataLoader crashes).
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-18-transformer-plus-features-740/run_all.sh \
#     > /tmp/dotaml_tpf.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=.venv/bin/python
EXP=experiments/2026-05-18-transformer-plus-features-740
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

echo "===== transformer-plus-features-740 run_all START $(date -Iseconds) ====="
run_ablation "architecture_only" "_architecture_only"
arch_rc=$?
run_ablation "transformer_plus_features" "_transformer_plus_features"
prim_rc=$?

echo "===== run_all DONE arch_rc=$arch_rc primary_rc=$prim_rc $(date -Iseconds) ====="
# Exit nonzero only if the PRIMARY ablation failed — the sanity is informative
# but its failure shouldn't block analysis of the primary if that one passed.
if [ "$prim_rc" -ne 0 ]; then
  exit "$prim_rc"
fi
exit 0
