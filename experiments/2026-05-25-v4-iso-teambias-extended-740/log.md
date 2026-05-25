# v4-iso-teambias-extended-740 log

## 2026-05-25 — scaffold

- Forked code verbatim from `experiments/2026-05-25-v3-ablations-740/`
  (data.py, models.py, train.py, loss.py, mae.py).
- Single-line change in `train.py:828-829`: `--ablation` argparse
  choices updated to `["v4_iso_teambias_extended"]`.
- Wrote `config.yaml` with the single ablation entry
  `v4_iso_teambias_extended` (use_patch_token=false, use_pmae=false,
  use_uw_so=false, dur_loss_mode=ce, use_player_embedding=false,
  use_team_team_bias=true, use_features=true, multitask=true).
- Wrote `run_all.sh` — two steps: smoke (1 epoch, 50k rows) then full
  training (max 30 epochs, ~6h). No data build phase needed (reuses
  v3-built extended parquets verbatim; no account_id sidecar / player
  vocab needed because use_player_embedding=false).
- Smoke run: (filled in after smoke succeeds).
