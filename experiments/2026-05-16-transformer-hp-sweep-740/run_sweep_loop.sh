#!/usr/bin/env bash
# Per-trial subprocess isolation wrapper. Each iteration spawns a FRESH
# Python process running ONE trial, then exits. CUDA crashes can't poison
# sibling trials. Optuna's SQLite study at results/optuna.db is the
# coordination point. Loops until the study has TARGET_TRIALS in
# {COMPLETE, PRUNED}, then runs the top-k retraining once.
set -u
cd "$(dirname "$0")/../.."
EXP=experiments/2026-05-16-transformer-hp-sweep-740
PY=.venv/bin/python
DB=$EXP/results/optuna.db
TARGET_TRIALS=60
MAX_ITER=200          # safety cap; ~3.3x target to allow for crashed retries
crash_count=0
sleep 1

count_done() {
  "$PY" - <<EOF
import optuna
study = optuna.load_study(study_name="transformer-hp-sweep-740",
                          storage="sqlite:///$DB")
done = sum(1 for t in study.trials
           if str(t.state).endswith(("COMPLETE", "PRUNED")))
print(done)
EOF
}

for iter in $(seq 1 "$MAX_ITER"); do
  done=$(count_done 2>/dev/null || echo 0)
  echo "===== iter $iter / $MAX_ITER  done=$done / $TARGET_TRIALS  crashes=$crash_count  $(date -Iseconds) ====="
  if [ "$done" -ge "$TARGET_TRIALS" ]; then
    echo "===== reached target $TARGET_TRIALS trials ====="
    break
  fi
  # Cleanup any leftover RUNNING/FAIL/WAITING from a prior crash.
  "$PY" "$EXP/cleanup_failed_trials.py" 2>&1 | tail -2
  # Run exactly ONE trial in a fresh process.
  "$PY" "$EXP/run_sweep.py" --n-trials 1 --skip-top-k
  rc=$?
  if [ "$rc" -ne 0 ]; then
    crash_count=$((crash_count + 1))
    echo "----- iter $iter rc=$rc (crash count: $crash_count) -----"
    # Clear torch pycache if we suspect a corrupted .pyc from SIGSEGV-mid-import.
    if [ "$rc" -eq 1 ] && grep -q "bad marshal data" /tmp/dotaml_sweep.log 2>/dev/null; then
      echo "----- bad marshal data detected, clearing torch __pycache__ -----"
      find .venv/lib/python3.12/site-packages/torch -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    fi
    sleep 2
  fi
done

# Top-k retraining (one final call, isolated subprocess).
echo "===== running top-k retraining ====="
"$PY" "$EXP/cleanup_failed_trials.py" 2>&1 | tail -2
"$PY" "$EXP/run_sweep.py" --retrain-only
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "===== top-k retraining failed rc=$rc — sweep results still in optuna.db ====="
  exit "$rc"
fi
echo "===== sweep + top-k retraining complete ====="
exit 0
