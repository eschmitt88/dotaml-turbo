#!/usr/bin/env bash
# Full-run convenience script for rich-supervision-multitask-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) build_rich_cols.py        -- walk raw turbo parquets filtered to clean
#                                   match_ids, parse raw_json, emit per-match
#                                   sidecar with duration + per-slot items/KDA/
#                                   GPM/XPM/hero_damage/net_worth. ~3-4 h CPU.
#                                   Multi-checkpoint defense + row-group-stats
#                                   verification (NO full re-read).
#   2) build_item_vocab.py       -- stream train sidecar, build item vocab
#                                   (freq >= cutoff) + 8-quantile duration
#                                   buckets. ~1-3 min. No retry.
#   3) train.py --ablation win_only_sanity   -- sanity replication of cleanup-740.
#                                                ~25 min. Per-trial subprocess
#                                                isolation retry wrapper (Blackwell
#                                                torch DataLoader bug workaround).
#   4) train.py --ablation multitask_all     -- PRIMARY. ~60-90 min (~2x per-epoch
#                                                from the per-slot item head).
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-20-rich-supervision-multitask-740/run_all.sh \
#     > experiments/2026-05-20-rich-supervision-multitask-740/full_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=.venv/bin/python
EXP=experiments/2026-05-20-rich-supervision-multitask-740
MAX_RETRIES=3

echo "===== rich-supervision-multitask-740 run_all START $(date -Iseconds) ====="

# ---- Step 1: build rich-cols sidecar (CPU-bound; no retry) -----------------
echo "[$(date -Iseconds)] STEP 1/4: build_rich_cols.py START"
$PY $EXP/build_rich_cols.py --config $EXP/config.yaml
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 1 FAILED rc=$rc -- pipeline halted."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 1/4: build_rich_cols.py DONE"

# ---- Step 2: build item vocab + duration buckets (CPU-bound; no retry) ----
echo "[$(date -Iseconds)] STEP 2/4: build_item_vocab.py START"
$PY $EXP/build_item_vocab.py --config $EXP/config.yaml
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 2 FAILED rc=$rc -- pipeline halted."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 2/4: build_item_vocab.py DONE"

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
  echo "[$(date -Iseconds)] $ab EXHAUSTED retries -- moving on"
  return 1
}

echo "[$(date -Iseconds)] STEP 3/4: win_only_sanity START"
run_tfm "win_only_sanity" "_win_only_sanity"
rc_sanity=$?
echo "[$(date -Iseconds)] STEP 3/4: win_only_sanity DONE rc=$rc_sanity"

echo "[$(date -Iseconds)] STEP 4/4: multitask_all START"
run_tfm "multitask_all" "_multitask_all"
rc_primary=$?
echo "[$(date -Iseconds)] STEP 4/4: multitask_all DONE rc=$rc_primary"

# Copy the primary metrics file to canonical metrics.json (per experiment rule).
if [ "$rc_primary" -eq 0 ]; then
  cp "$EXP/metrics_multitask_all.json" "$EXP/metrics.json"
fi

echo "===== rich-supervision-multitask-740 run_all DONE sanity_rc=$rc_sanity primary_rc=$rc_primary $(date -Iseconds) ====="
exit "$rc_primary"
