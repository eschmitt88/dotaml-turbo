---
kind: proposal
slug: plateau-baseline-740
date: 2026-05-15
status: implemented
experiment: experiments/2026-05-15-plateau-baseline-740/
hypothesis: "On the patch-7.40 Turbo snapshot (~19.6M matches, 2025-12-16 → 2026-03-23), a LightGBM one-hot draft-only classifier with prior-art-style features achieves validation AUC within 0.635 ± 0.010, confirming the ceiling observed across six prior architectures on a smaller pre-7.40 dataset."
rationale: >
  In the DotaML prior art, six successive architectures spanning LightGBM,
  SimpleFFN, ResidualFFN, and a Transformer with learned hero embeddings —
  4x capacity range — all converged to ~59.9% test accuracy and ~0.635
  test AUC on 7-9M matches from a pre-7.40 window. Before proposing any
  technique to break that ceiling, we need to know whether it survives a
  2x larger dataset on a different patch under HCE evaluation, and we
  need a calibrated number of our own to anchor every subsequent
  experiment against. A LightGBM baseline with 300-dim one-hot features
  is the cheapest, most reproducible, and most directly comparable point
  in that grid.
reads:
  - "[[literature/repos/eschmitt88-DotaML]]"
  - "[[literature/repos/eschmitt88-DotaDB]]"
  - "[[concepts/draft-only-win-prediction]]"
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/radiant-side-advantage]]"
  - "[[concepts/fake-match-filtering]]"
  - "[[concepts/hero-embedding-vs-onehot]]"
  - "[[concepts/match-id-vs-seq-num-ordering]]"
expected_metric:
  name: val_auc
  target: 0.635
  direction: higher-is-better
design_sketch:
  - Read patch-7.40 Parquet from Azure Data Lake (DefaultAzureCredential, dota2datalake/matches) for start_time in [2025-12-16, 2026-03-23].
  - Deduplicate by match_id at read time (covers any residual seq-num overlap).
  - Apply fake-match filter (both T4s standing on losing side OR >2 empty inventories).
  - Split chronologically on start_time per splits.yaml; train = first ~70 days, validation = next ~14 days, test = final ~14 days (sealed, HCE rule).
  - Features; one-hot 300-dim (radiant_hero_id, dire_hero_id x 150) + 1-bit Radiant-side indicator. Hero IDs 1-150 per DotaML v3 fix.
  - Model; LightGBM, ~500 boosting rounds, lr 0.1, 31 leaves, default L1/L2 - mirrors DotaML v3 to keep the comparison clean.
  - Metrics; val_auc, val_acc, val_log_loss, val_brier, calibration curve, Radiant-side base-rate sanity.
risks:
  - LightGBM may not fit 19.6M x 301 in RAM (64 GB). Mitigation; stratified 5M-row training subset (matches v3's memory-constrained recipe) plus a sanity check that 5M ~ 19.6M.
  - Patch-7.40 window contains within-patch meta drift (new heroes / balance patches mid-window). Single chronological val/test may not be the right HCE split forever; sufficient for the first run.
  - A "plateau holds" result is consistent with both a genuine ceiling and silent label corruption - must inspect calibration + base-rate to rule the latter out.
  - The prior-art number is from a different patch; even a confirmed ~0.635 on 7.40 is not directly comparable, only suggestive.
related_prior: []
estimated_runtime: "4-6 h CPU once raw Parquet is local; ~50 GB SN850X for data download; well under budget.yaml max_wall_hours=24."
---

# Plateau replication — LightGBM baseline on patch-7.40

The single most informative number we can produce right now is whether
the ~0.635 test AUC reported in DotaML v3-v6 reproduces on the patch-7.40
snapshot, in this repo, under HCE rules. Six prior architectures across
4x of capacity converged to that ceiling on a pre-7.40 window of 7-9M
matches. The new snapshot is twice the size, on a different patch, and
will be evaluated with a sealed test split this project's prior work did
not have. Any improvement on top of the prior art must be measured
against a number we trust on our own data — this proposal produces it.

The architecture choice is deliberately the dullest available. LightGBM
with 300-dim one-hot features is the same recipe DotaML v3 used to reach
58.82% acc / 0.6189 AUC on a 5M subset of the pre-7.40 data. Anything
fancier reopens the v4-v6 question of "did we change the ceiling or did
we just hit it again," which is exactly the question we are trying to
ground first.

How the answer updates downstream work:

- **Val AUC inside 0.635 ± 0.010.** The plateau survives a patch boundary
  and a 2x data increase. Any future architectural innovation can be
  meaningfully judged against ≈0.635. The next proposals should attack
  the ceiling from a hypothesis-driven angle (information leakage,
  calibration vs accuracy, side-conditional models, draft-order signal)
  rather than try yet another architecture.
- **Val AUC > 0.645.** The ceiling moved. Likely culprits are more data,
  patch 7.40 being "easier," or our pipeline introducing a leak we
  should hunt before celebrating.
- **Val AUC < 0.625.** Either the patch is harder than the prior window
  or — more likely — something is wrong (fake-match filter not working,
  hero ID range off, side label scrambled). Either way the run becomes a
  data-pipeline audit, not a model question.

This is also the experiment that forces the project's first `splits.yaml`,
its first read pipeline, and its first sealed-test discipline. Every
follow-up will inherit those decisions.
