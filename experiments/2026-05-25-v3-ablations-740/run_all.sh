#!/usr/bin/env bash
# Full-run script for v3-ablations-740.
#
# Sequential pipeline (each step gates the next on rc=0):
#   1) build extended train account_id sidecar (~30-60 min raw walk on
#      pre-patch days only; reuses the prior 7.40-only sidecar for
#      post-patch + val coverage)
#   2) build player_id vocab from sidecars (~5-15 min streaming scan)
#   3) smoke (1 epoch, 50k rows) on both ablations - verifies plumbing
#   4) v3_dur_ce full training (max 30 epochs, ~6h GPU)
#   5) v3_player_emb full training (max 30 epochs, ~6-7h GPU)
#
# Uses `python -u` for unbuffered stdout under nohup.
# Per-trial subprocess retry wrapper (defensive against the rare segfault
# pattern; Blackwell torch DataLoader bug no longer fires on JEDEC 4800
# MT/s but the wrapper costs nothing).
#
# Main agent invokes via:
#   nohup bash experiments/2026-05-25-v3-ablations-740/run_all.sh \
#     > experiments/2026-05-25-v3-ablations-740/full_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../.."  # project root

PY=".venv/bin/python -u"
EXP="experiments/2026-05-25-v3-ablations-740"
MAX_RETRIES=3

echo "===== v3-ablations-740 run_all START $(date -Iseconds) ====="

# Reused inputs.
EXTENDED_PF_TRAIN="data/snapshots/7.40-2025-12-16/processed/player_features_extended/train.parquet"
EXTENDED_PF_VAL="data/snapshots/7.40-2025-12-16/processed/player_features_extended/val.parquet"
EXTENDED_RC_TRAIN="data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/train.parquet"
EXTENDED_RC_VAL="data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/val.parquet"
VOCAB="experiments/2026-05-20-rich-supervision-multitask-740/results/item_vocab.json"
PRIOR_SIDECAR_TRAIN="experiments/2026-05-19-player-embedding-prelim-740/sidecar/account_ids_train.parquet"
PRIOR_SIDECAR_VAL="experiments/2026-05-19-player-embedding-prelim-740/sidecar/account_ids_val.parquet"
for f in "$EXTENDED_PF_TRAIN" "$EXTENDED_PF_VAL" "$EXTENDED_RC_TRAIN" "$EXTENDED_RC_VAL" \
         "$VOCAB" "$PRIOR_SIDECAR_TRAIN" "$PRIOR_SIDECAR_VAL"; do
  if [ ! -e "$f" ]; then
    echo "REFUSED: required input missing: $f"
    exit 2
  fi
done

# ---- Step 1: build extended train account_id sidecar (A2-only need) -------
EXT_SIDECAR="$EXP/sidecar/account_ids_train_extended.parquet"
if [ -e "$EXT_SIDECAR" ]; then
  echo "[$(date -Iseconds)] STEP 1/5: extended account_id sidecar already built, skipping."
else
  echo "[$(date -Iseconds)] STEP 1/5: building extended train account_id sidecar (~30-60min)..."
  $PY $EXP/build_account_sidecar_extended.py --config $EXP/config.yaml
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -Iseconds)] STEP 1/5 FAILED rc=$rc -- aborting pipeline."
    exit "$rc"
  fi
  echo "[$(date -Iseconds)] STEP 1/5 DONE"
fi

# ---- Step 2: build player_id vocab from sidecars --------------------------
PLAYER_VOCAB="$EXP/vocab/player_id_vocab.json"
if [ -e "$PLAYER_VOCAB" ]; then
  echo "[$(date -Iseconds)] STEP 2/5: player_id vocab already built, skipping."
else
  echo "[$(date -Iseconds)] STEP 2/5: building player_id vocab (~5-15min)..."
  $PY $EXP/build_vocab.py --config $EXP/config.yaml
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -Iseconds)] STEP 2/5 FAILED rc=$rc -- aborting pipeline."
    exit "$rc"
  fi
  echo "[$(date -Iseconds)] STEP 2/5 DONE"
fi

# ---- Step 3: smoke both ablations -----------------------------------------
echo "[$(date -Iseconds)] STEP 3/5: smoke v3_dur_ce"
$PY $EXP/train.py --config $EXP/config.yaml --ablation v3_dur_ce --smoke
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 3/5 (smoke v3_dur_ce) FAILED rc=$rc -- aborting."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 3/5: smoke v3_player_emb"
$PY $EXP/train.py --config $EXP/config.yaml --ablation v3_player_emb --smoke
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[$(date -Iseconds)] STEP 3/5 (smoke v3_player_emb) FAILED rc=$rc -- aborting."
  exit "$rc"
fi
echo "[$(date -Iseconds)] STEP 3/5 DONE"

# ---- Step 4/5: full training (sequential, per-trial subprocess retry) -----
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

echo "[$(date -Iseconds)] STEP 4/5: v3_dur_ce full training"
run_ablation "v3_dur_ce" "_v3_dur_ce"
rc_a1=$?
echo "[$(date -Iseconds)] STEP 4/5 DONE rc=$rc_a1"

echo "[$(date -Iseconds)] STEP 5/5: v3_player_emb full training"
run_ablation "v3_player_emb" "_v3_player_emb"
rc_a2=$?
echo "[$(date -Iseconds)] STEP 5/5 DONE rc=$rc_a2"

# Copy the better ablation's metrics to metrics.json for downstream tooling.
if [ "$rc_a1" -eq 0 ] && [ "$rc_a2" -eq 0 ]; then
  A1_AUC=$($PY -c "import json; print(json.load(open('$EXP/metrics_v3_dur_ce.json'))['val_auc'])")
  A2_AUC=$($PY -c "import json; print(json.load(open('$EXP/metrics_v3_player_emb.json'))['val_auc'])")
  echo "A1 (v3_dur_ce) val_auc=$A1_AUC; A2 (v3_player_emb) val_auc=$A2_AUC"
  USE_A1=$($PY -c "print('1' if $A1_AUC >= $A2_AUC else '0')")
  if [ "$USE_A1" = "1" ]; then
    cp "$EXP/metrics_v3_dur_ce.json" "$EXP/metrics.json"
    echo "metrics.json <- v3_dur_ce ($A1_AUC)"
  else
    cp "$EXP/metrics_v3_player_emb.json" "$EXP/metrics.json"
    echo "metrics.json <- v3_player_emb ($A2_AUC)"
  fi
elif [ "$rc_a1" -eq 0 ]; then
  cp "$EXP/metrics_v3_dur_ce.json" "$EXP/metrics.json"
elif [ "$rc_a2" -eq 0 ]; then
  cp "$EXP/metrics_v3_player_emb.json" "$EXP/metrics.json"
fi

echo "===== v3-ablations-740 run_all DONE rc_a1=$rc_a1 rc_a2=$rc_a2 $(date -Iseconds) ====="
# Exit 0 if at least one ablation succeeded.
if [ "$rc_a1" -eq 0 ] || [ "$rc_a2" -eq 0 ]; then
  exit 0
fi
exit 1
