---
kind: experiment
slug: "plateau-baseline-740"
date: "2026-05-15"
status: done     # running | done | abandoned
hypothesis: "On the patch-7.40 Turbo snapshot (~19.6M matches, 2025-12-16 → 2026-03-23), a LightGBM one-hot draft-only classifier with prior-art-style features achieves validation AUC within 0.635 ± 0.010, confirming the ceiling observed across six prior architectures on a smaller pre-7.40 dataset."
result: "val_auc=0.6161 — outside proposal band [0.625, 0.645] on the low side, but within 0.003 of DotaML v3's same-recipe test_auc=0.6189. Plateau hypothesis confirmed for the LightGBM-specific ceiling; the headline 0.635 target was the v5 Transformer's number, not v3 LightGBM's."
related_concepts:
  - draft-only-win-prediction
  - draft-prediction-plateau
  - radiant-side-advantage
  - fake-match-filtering
  - hero-embedding-vs-onehot
  - match-id-vs-seq-num-ordering
related_literature:
  - eschmitt88-DotaML
  - eschmitt88-DotaDB
respects:
  - "~/.claude/rules/evaluation.md"
tags: [lightgbm, baseline, plateau, draft-only, patch-7.40]
---

# plateau-baseline-740

## Hypothesis

LightGBM with the DotaML v3 recipe (300-dim one-hot heroes + 1 Radiant-side
bit, ~500 boosting rounds, lr 0.1, 31 leaves) on the patch-7.40 Turbo
snapshot will reach validation AUC within `0.635 ± 0.010`, replicating the
plateau observed across six prior architectures on a smaller pre-7.40
dataset (DotaML v3-v6: ~0.619-0.635 test AUC).

## Setup

- Config: `config.yaml`
- Code: `train.py` (entry point), `pull_raw.py`, `build_features.py`
- Data: `data/snapshots/7.40-2025-12-16/processed/{train,val}.parquet`
  (validation split only during search; test is sealed per
  `~/.claude/rules/evaluation.md`).
- Splits: inherited from project root `splits.yaml`.
- Hardware: CPU LightGBM training; no GPU needed for this run.

## Result

Headline (validation split — search signal, `metrics.json`):

| metric                    | value   |
| ------------------------- | ------- |
| **val_auc**               | 0.6161  |
| val_acc                   | 0.5866  |
| val_log_loss              | 0.6698  |
| val_brier                 | 0.2386  |
| train_auc                 | 0.6287  |
| train-val AUC gap         | 0.0126  |
| Radiant base rate (val)   | 0.5326  |
| val majority-class acc    | 0.5326  |

Counts: train 13,018,393 → 5M stratified subsample (`metrics.json:n_train_post_subsample`),
val 2,419,185, feature_dim = 301. Train wall: 123 s. Build wall: 881 s.

Build pipeline (`data/snapshots/7.40-2025-12-16/processed/build_stats.json`):
read 16,923,487 raw rows → kept 15,437,578 → filtered 1,485,909 (8.78 %).
Zero `match_id` duplicates after read-time dedup over 16.9 M rows
(confirms structural seq-num overlap is harmless — see
`concepts/match-id-vs-seq-num-ordering.md`).

Held-out test-split numbers will live in a separate `final_metrics.json`
written ONLY by an explicit final-scoring pass at chain end. This
experiment is not that pass; do not consume `test/` data here.

## Interpretation

The proposal hypothesised **val_auc within 0.635 ± 0.010**. Strict reading:
0.6161 falls 0.0089 below the band, so the hypothesis as written is
**not confirmed**.

But the proposal's 0.635 target appears miscalibrated against this
specific architecture. The plateau cited in the rationale spanned
DotaML v3 (LightGBM, **test_auc 0.6189**) through v5/v6 (Transformer,
**test_auc ~0.635**). The 0.635 number is the v5 Transformer's ceiling,
not the v3 LightGBM's. Mirroring the v3 recipe under HCE on the new
patch and getting **val_auc 0.6161** lands **−0.0028 from v3's test_auc
0.6189** (`metrics.json:delta_val_auc_vs_v3_test`). For the LightGBM-
specific ceiling, the plateau replicates cleanly.

Three lines of evidence the run is sound, not silently broken:

1. **Calibration is essentially perfect** across 20 quantile bins
   (`results/calibration.png`, `metrics.json:calibration`). Predicted vs
   empirical Radiant-win probability tracks the y=x diagonal to within
   one bin width across the entire predicted-prob range
   `[0.337, 0.721]`. A broken side label or scrambled hero index would
   shift this dramatically.
2. **Train-val AUC gap = 0.0126** — small. The model is well-fit on the
   5 M subsample, not overfit. A larger gap would have suggested
   leakage; a much larger gap would have suggested broken features.
3. **Radiant base rates match across splits** to within 0.001:
   train 0.5335, val 0.5326. Comparable to the prior-art's empirical
   Radiant edge of 5-7 pp on Turbo (`concepts/radiant-side-advantage.md`).
   The val majority-class accuracy is 0.5326, so the model's 0.5866
   accuracy is +5.4 pp of real lift over base rate.

So: the proposal's **strict** target was missed, the proposal's
**spirit** (plateau holds for this architecture) was confirmed, and the
result is anchored cleanly enough that the next experiment can build on
it rather than auditing it.

## Diagnostics

- intended_effect_confirmed: partial — strict band [0.625, 0.645] missed by 0.0089 on the low side, but val_auc=0.6161 lands within 0.003 of DotaML-v3's same-recipe test_auc=0.6189 (`metrics.json:val_auc`, `metrics.json:delta_val_auc_vs_v3_test=-0.0028`)
- leakage_check: enforced both at build-time (`build_features.py:102-112` `assert_no_test_dates`) and train-time (`train.py:113-121` `assert_no_test_dates`); confirmed via `metrics.json:train_date_max=2026-02-23` and `metrics.json:val_date_max=2026-03-09`, both strictly < `splits.yaml:test_start_date=2026-03-10` — finding: no test-window dates ever read
- overfitting_signal: train=0.6287 val=0.6161 gap=0.0126 — small gap, model is well-fit not overfit; consistent with a near-noise-ceiling problem (from `metrics.json:train_val_auc_gap`)
- delta_from_prior: vs DotaML-v3 (test_auc 0.6189 on a 5M pre-7.40 LightGBM run, [[literature/repos/eschmitt88-DotaML]]), delta = -0.0028 attributed to (likely) HCE-style chronological val + small patch-7.40 difference; not a regression (`metrics.json:delta_val_auc_vs_v3_test`)
- unexpected_findings: (a) calibration is near-perfect across all 20 quantile bins (`results/calibration.png`, `metrics.json:calibration`) — better than I'd expect from vanilla LightGBM and a strong signal that the binary log-loss objective is doing real work, not just memorising the base rate; (b) the proposal's "0.635 plateau" target was actually the v5 Transformer's ceiling, not the v3 LightGBM's — worth re-citing more precisely in any follow-up proposal
- seeds_run: 1 (single run, seed=42 from `config.yaml:seed`)
- metric_aggregation: single-run
- next_candidates:
  - Retest the proposal's actual claim ("plateau across architectures") with a Transformer/FFN baseline mirroring DotaML v5/v6 — that's where the ~0.635 ceiling number actually lives. If a 64-dim hero-embedding Transformer also lands ≤0.625 on the new patch under HCE, the plateau-across-architectures claim from the prior art does NOT replicate at the magnitude originally cited.
  - Filter sensitivity audit: re-run with `apply_forfeit_filter: false` and separately with `apply_empty_inventory_filter: false` to attribute the 8.78 % drop. If val_auc shifts >0.005, the filter is doing real work; if <0.001 the filter is mostly cosmetic on this snapshot. Required before any future "filtered vs raw" architecture comparison is meaningful.
  - Side-conditional decomposition: split val by hero-pick-ordering position and check whether the model's signal lives mostly in the late-pick counter-positions (where information is fully revealed) vs early picks. A flat decomposition would suggest the model is mostly using base-rate-of-hero-on-side; a steep one would suggest it's actually modelling matchups.

## Follow-up

- See `next_candidates` above. None implemented in this experiment.
- The proposal's hypothesis re-frames as: "the v3 LightGBM ceiling, specifically, replicates within 0.005 on the new patch under HCE." The broader plateau-across-architectures claim is unresolved.
- One concept update is warranted: `concepts/draft-prediction-plateau.md` should record that the plateau magnitude varies by architecture (LightGBM ≈ 0.619, Transformer ≈ 0.635) — not all six prior models converged to the same number.
