#!/usr/bin/env bash
# Full-run convenience script for player-features-prepatch-740.
#
# Sequential pipeline:
#   1) pull_history.py    — Azure pull, ~127 days [2025-08-01, 2025-12-15] (~30 min, ~100 GB)
#   2) build_features.py  — chronological walk across history/ then snapshot raw (~3 h)
#   3) train.py x3        — heroes_plus_features, heroes_only, features_only (~15 min total)
#
# Main agent should invoke via:
#   nohup bash experiments/2026-05-18-player-features-prepatch-740/run_all.sh \
#     > /tmp/dotaml_pfp.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."  # project root

PY=.venv/bin/python
EXP=experiments/2026-05-18-player-features-prepatch-740

echo "[$(date -Iseconds)] pull_history.py START"
$PY $EXP/pull_history.py 2>&1 | tee /tmp/dotaml_pfp_pull.log
echo "[$(date -Iseconds)] pull_history.py DONE"

echo "[$(date -Iseconds)] build_features.py START"
$PY $EXP/build_features.py 2>&1 | tee /tmp/dotaml_pfp_build.log
echo "[$(date -Iseconds)] build_features.py DONE"

echo "[$(date -Iseconds)] train.py heroes_plus_features START"
$PY $EXP/train.py --ablation heroes_plus_features 2>&1 | tee /tmp/dotaml_pfp_train.log
echo "[$(date -Iseconds)] train.py heroes_plus_features DONE"

echo "[$(date -Iseconds)] train.py heroes_only START"
$PY $EXP/train.py --ablation heroes_only \
  --metrics-suffix _ablation_heroes_only 2>&1 | tee -a /tmp/dotaml_pfp_train.log
echo "[$(date -Iseconds)] train.py heroes_only DONE"

echo "[$(date -Iseconds)] train.py features_only START"
$PY $EXP/train.py --ablation features_only \
  --metrics-suffix _ablation_features_only 2>&1 | tee -a /tmp/dotaml_pfp_train.log
echo "[$(date -Iseconds)] train.py features_only DONE"

echo "[$(date -Iseconds)] ALL DONE"
