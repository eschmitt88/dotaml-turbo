#!/usr/bin/env bash
# Full-run convenience script for player-features-740.
# Main agent should invoke via:
#   nohup bash experiments/2026-05-17-player-features-740/run_all.sh \
#     > /tmp/dotaml_pf.log 2>&1 &
#
# Estimated wall: ~2-3 h for build_features.py (sequential JSON parse
# over 81 GB raw); ~5-10 min for each of 3 LightGBM training runs.
# Total ~3-4 h.

set -euo pipefail
cd "$(dirname "$0")/../.."  # project root

PY=.venv/bin/python
EXP=experiments/2026-05-17-player-features-740

echo "[$(date -Iseconds)] build_features.py START"
$PY $EXP/build_features.py 2>&1 | tee /tmp/dotaml_pf_build.log
echo "[$(date -Iseconds)] build_features.py DONE"

echo "[$(date -Iseconds)] train.py heroes_plus_features START"
$PY $EXP/train.py --ablation heroes_plus_features 2>&1 | tee /tmp/dotaml_pf_train.log
echo "[$(date -Iseconds)] train.py heroes_plus_features DONE"

echo "[$(date -Iseconds)] train.py heroes_only START"
$PY $EXP/train.py --ablation heroes_only \
  --metrics-suffix _ablation_heroes_only 2>&1 | tee -a /tmp/dotaml_pf_train.log
echo "[$(date -Iseconds)] train.py heroes_only DONE"

echo "[$(date -Iseconds)] train.py features_only START"
$PY $EXP/train.py --ablation features_only \
  --metrics-suffix _ablation_features_only 2>&1 | tee -a /tmp/dotaml_pf_train.log
echo "[$(date -Iseconds)] train.py features_only DONE"

echo "[$(date -Iseconds)] ALL DONE"
