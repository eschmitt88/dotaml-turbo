---
kind: experiment
slug: transformer-plus-features-extended-740
date: 2026-05-19
status: done
hypothesis: "Extending training on the transformer-plus-features-740 architecture and feature set from 14 epochs to up to 30 epochs with early stopping (patience=5 on val_log_loss) raises val_auc by >= 0.001 over the prior 0.6452 (target val_auc >= 0.6462)."
result: "CONFIRMED. val_auc=0.6477 @ best_epoch=22 (early-stopped at 27), +0.0025 over parent 0.6452 and +0.0015 over proposal target 0.6462. All three coverage buckets lifted; HIGH bucket reached 0.6588. 25.1-min wall, zero Blackwell retries."
related_concepts:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/draft-only-win-prediction]]"
related_literature:
  - "[[literature/papers/hodge2017win]]"
related_experiments:
  - "[[experiments/2026-05-18-transformer-plus-features-740]]"
tags:
  - transformer
  - player-features
  - extended-training
  - plateau-740
respects:
  - "~/.claude/rules/evaluation.md"
---

# transformer-plus-features-extended-740

## Hypothesis

The parent experiment `transformer-plus-features-740` reached val_auc=0.6452
with `best_epoch=14=max_epochs` — i.e., val_loss was still improving at the
epoch cap. A natural cheap follow-up is to train longer with early stopping
and see whether the schedule had been the binding constraint.

**Target**: val_auc >= 0.6462 (a +0.001 lift over 0.6452). Anything smaller
than that is within noise of the parent and would suggest the 14-epoch cap
was already at-or-near convergence.

## Setup

- **Config**: `config.yaml` — identical to parent except:
  - `optim.max_epochs`: 14 -> 30
  - `optim.early_stopping_metric`: `val_log_loss` (documenting the
    pre-existing patience semantics)
  - parent `optim.patience: 5` was already present but never bound
- **Code**: copied verbatim from
  `experiments/2026-05-18-transformer-plus-features-740/` —
  `models.py`, `data.py`, `train.py`. Only `train.py` was touched, to
  drop the `architecture_only` ablation choice (already validated in
  parent at 0.6319 vs anchor 0.6322) and add a
  `delta_vs_transformer_plus_features_740` field to `metrics.json`.
- **Data** (validation split only — test window [2026-03-10, 2026-03-23]
  is sealed, asserted at train time):
  - `data/snapshots/7.40-2025-12-16/processed/player_features_prepatch/train.parquet`
    (13,018,393 rows; same 5M stratified subsample with seed=42 as the
    parent)
  - `.../val.parquet` (2,419,185 rows; full val every run)
- **Training**: 30-epoch cap, Adam lr=1e-3, batch_size=8192, bf16
  autocast, num_workers=0, math SDP backend forced. Early stopping with
  patience=5 on `val_log_loss` (the existing
  `train.py:train_model`'s `patience` argument; selects best epoch
  by `val_loss` decrease and breaks when `epochs_since_improve >= patience`).

## Ablations

| name | use_features | role |
|---|---|---|
| `transformer_plus_features` | True | **PRIMARY**. Target val_auc >= 0.6462 |

`architecture_only` is NOT re-run — its parent run produced 0.6319 within
0.0003 of `plateau-architectures-740` (0.6322), so the pipeline is already
verified.

## Result

**CONFIRMED.** val_auc = **0.6477** at `best_epoch=22`; training ran 27 epochs
total before early-stopping (5 epochs of no val_loss improvement after
epoch 22 → `epochs_since_improve >= patience=5`). Total wall 25.1 min,
including data load (14.1 s); zero Blackwell retries.

Coverage-bucket val_auc (val rows partitioned into terciles by mean
`n_games_log1p` across the 10 players in the match):

| bucket | n     | val_auc | parent  | delta  |
|--------|------:|--------:|--------:|-------:|
| low    | 805 K | 0.6367  | 0.6347  | +0.0020 |
| medium | 808 K | 0.6467  | 0.6443  | +0.0024 |
| high   | 806 K | 0.6588  | 0.6560  | +0.0028 |

All three buckets lifted by similar magnitude (+0.0020 to +0.0028);
the LOW–HIGH gap is 0.0221, essentially unchanged from the parent's
0.0213. Extended training is a uniform lift, NOT a targeted fix for
the cold-start / anonymous tail.

Deltas vs anchors:

- vs proposal target 0.6462 → **+0.0015** (cleared)
- vs parent transformer-plus-features-740 (0.6452) → **+0.0025**
- vs Transformer-only plateau-architectures-740 (0.6322) → +0.0155
- vs LightGBM-only player-features-prepatch-740 (0.6256) → +0.0221
- vs LightGBM-baseline (0.6161) → +0.0316

## Diagnostics

- intended_effect_confirmed: yes — val_auc=0.6477 > target 0.6462 (`metrics.json:val_auc`).
- leakage_check: HCE date-window assertion live in `data.py`; train ends 2025-12-16..2026-02-23, val ends 2026-02-24..2026-03-09, both strictly < test_start 2026-03-10 (`metrics.json:train_date_max`, `val_date_max`). No tool calls touched `test/` or the test-window date range (verified out-of-band).
- overfitting_signal: train_loss=0.6495 val_loss=0.6547 gap=0.0052 at `best_epoch=22` (`metrics.json:train_loss_at_best`, `val_loss`). Parent at epoch 14 had train=0.6523 val=0.6558 gap=0.0035; the extended schedule widened the gap by ~0.0017 — modest, and early-stopping correctly fired at epoch 27 once val_loss plateaued.
- delta_from_prior: vs `2026-05-18-transformer-plus-features-740` (val_auc 0.6452 @ epoch 14), +0.0025 val_auc attributed to ~57% additional training (14 → 22 epochs) before early stopping. The marginal gain per epoch shrinks but never inverts in the 14–22 window (`metrics_transformer_plus_features.json:history`).
- unexpected_findings: none. Curve behaved as predicted (continued improvement past epoch 14, convergence in low-20s, plateau by mid-20s). One small observation: val_auc reached 0.6477 at both epoch 22 AND epoch 23 (0.6477163 vs 0.6477298), so the best-epoch selection is essentially insensitive within the noise floor — a multi-seed run would be informative if precision matters.
- seeds_run: 1 (single run; seed=42)
- metric_aggregation: single-run
- next_candidates:
  - **upstream-data-cleanup** — patch the divide-by-zero in `experiments/2026-05-18-player-features-prepatch-740/build_features.py` that produces fp32-max sentinels (2,497 cells sanitized at load this run; 6,482 in prior). Rebuild prepatch parquet (~3 h), then re-validate downstream. Likely tiny effect on the headline but matters for downstream cleanliness.
  - **player-embedding-prelim-740** — learned per-player embeddings (~1.3M accounts × 32-64 dim, ~167-333 MB, easy on 16 GB VRAM). Now anchored against a 0.6477 reference. Anonymous embedding (66% of player-slots) is naturally well-trained and serves as a shrinkage target via a learned gate.
  - (Optional) **multi-seed sanity at the new ceiling** — re-run 3 seeds at this config to put error bars on the 0.6477 number before downstream comparisons depend on it.

## Follow-up

User's stated queue (after this experiment):

1. `upstream-data-cleanup` — patch the divide-by-zero in
   `player-features-prepatch-740/build_features.py`. Rebuild prepatch
   parquet (~3 h re-run), then re-run player-features-prepatch and
   transformer-plus-features against clean data. Expected effect tiny
   (~2.5K sentinels in 130 M cells is 0.002%); the value is downstream
   correctness, not headline AUC.
2. `player-embedding-prelim-740` — learned per-player embeddings as
   a richer player representation than the 8 aggregated features.

User explicitly de-prioritized `anonymous-aware-modeling-740` despite
the persistent low-vs-high coverage gap.
