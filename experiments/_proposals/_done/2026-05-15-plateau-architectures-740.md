---
kind: proposal
slug: plateau-architectures-740
date: 2026-05-15
status: implemented
experiment: experiments/2026-05-15-plateau-architectures-740/
hypothesis: "On the patch-7.40 Turbo snapshot under HCE, three deep-learning architectures mirroring DotaML v4-v6 (SimpleFFN ~47k params, ResidualFFN ~230k params, Transformer with 64-dim hero embeddings ~150k params) reproduce the prior-art rank order Transformer ≥ ResidualFFN ≥ SimpleFFN > LightGBM-baseline (val_auc 0.6161 from plateau-baseline-740), with each pairwise gap matching prior art within ±0.005 AUC."
rationale: >
  The plateau-baseline-740 experiment confirmed the v3 LightGBM ceiling
  (val_auc 0.6161 vs prior test_auc 0.6189) but in doing so revealed
  that the proposal's "0.635 plateau" target was actually the v5/v6
  Transformer's number, not LightGBM's. Whether the architecture-spread
  observed in DotaML v4-v6 (≈0.0165 AUC across SimpleFFN, ResidualFFN,
  Transformer) reproduces on patch-7.40 under HCE is the next load-
  bearing question — both for the plateau concept and for any future
  ceiling-breaking work, which has to know what to compare against. A
  three-architecture mirror of the prior art's deep-learning ladder is
  the cleanest test: same data, same recipes, different ranks.
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/hero-embedding-vs-onehot]]"
  - "[[concepts/draft-only-win-prediction]]"
  - "[[literature/repos/eschmitt88-DotaML]]"
  - "[[experiments/2026-05-15-plateau-baseline-740]]"
expected_metric:
  name: val_auc
  target: 0.635
  direction: higher-is-better
design_sketch:
  - Reuse data/snapshots/7.40-2025-12-16/processed/{train,val}.parquet from plateau-baseline-740 (13M train, 2.4M val, same 8.78% filter rate, same chronological splits per splits.yaml). No new Azure pull.
  - Keep the 5M stratified train subset (same seed=42) for direct comparability with plateau-baseline-740.
  - Three architectures, all ingesting 10 hero IDs + 1 Radiant-side bit:
    - SimpleFFN (mirrors v4); 64-dim hero-embedding lookup → concat(640) → dense[256, 128, 64] → sigmoid. ~50k params.
    - ResidualFFN (mirrors v5); 64-dim hero-embedding lookup → concat(640) → 4 residual blocks (256-dim) → sigmoid. ~230k params.
    - Transformer (mirrors v6, masking OFF for this run); 64-dim hero embeddings + learned Radiant/Dire position embedding → 2 self-attention layers (4 heads, 64-dim) over 10 hero tokens + 1 side token → mean-pool → dense → sigmoid. ~150k params.
  - PyTorch on RTX 5080, Adam optimizer, lr 1e-3, batch size 8192, ≤30 epochs with early stopping on val_log_loss (patience 5). Same seed across architectures.
  - Per-architecture metrics; val_auc, val_acc, val_log_loss, val_brier, train_auc, calibration curve. Plus a comparison table vs LightGBM baseline + DotaML prior-art numbers.
risks:
  - DotaML's exact hyperparameters were tuned on a smaller pre-7.40 dataset; the same recipes may train sub-optimally on 5M / patch-7.40 data. Mitigation; early stopping prevents overshoot, and the comparison still holds as an "out-of-the-box DotaML recipes" test even if a follow-up tuning pass would do better.
  - A "spread doesn't reproduce" result is consistent with both (a) the prior art's spread being noise and (b) a patch-7.40-specific collapse. Distinguishing requires either a Transformer-only retest with masking ON, or a longer-trained run; both are out of scope here.
  - Reusing the processed parquet inherits the 8.78% fake-match filter rate. If a follow-up sensitivity audit (in NOTES.md Next) shows the filter materially shifts results, this comparison would need a re-run.
  - 64-dim embeddings + ~150k-param Transformer + 5M rows × 30 epochs at bs 8192 is well within RTX 5080 (16 GB) headroom, but the first run may need batch-size or precision tuning (mixed precision recommended).
related_prior:
  - 2026-05-15-plateau-baseline-740
estimated_runtime: "≈90 min on RTX 5080 (3 architectures × ~30 min each); processed data already local; <1 GB additional disk for model checkpoints. Well under budget.yaml max_wall_hours=24 and max_disk_gb=500."
---

# Architecture-spread test on the patch-7.40 plateau

The previous experiment (`plateau-baseline-740`) replicated the v3 LightGBM ceiling cleanly but, in doing so, exposed a sharper question than the original proposal had framed. The "≈0.635 plateau" cited in the rationale was specifically the v5 Transformer's ceiling — v3 LightGBM was 0.0165 AUC below it. So the prior art's architecture-spread within the plateau is real and well-defined; the question this proposal tests is whether that spread *reproduces* on the new snapshot.

If yes, the next ceiling-breaking work should target what makes Transformer attention add ~0.0044 AUC over a residual FFN, and what makes a residual FFN add ~0.0096 AUC over LightGBM. These deltas tell us where information lives that the plateau-baseline can't extract. If no — if the three architectures land within ±0.005 of one another and of the LightGBM baseline — the prior-art's architecture-spread was either noise or pre-7.40-specific, and the patch-7.40 ceiling is genuinely architecture-insensitive at ~0.616-0.620.

Either result is informative and shapes the next round of proposals:

- **Spread reproduces** (Transformer ≥ ResidualFFN ≥ SimpleFFN > LightGBM, each gap ≥ 0.005). The plateau is real but its magnitude depends on architecture. Next: target the gap between Transformer and ResidualFFN with attention-aware features (e.g. cross-team attention on lane assignments, draft-order conditioning).
- **Spread compresses** (all three within ±0.005 of LightGBM-0.6161). The architecture spread was noise on a smaller / older snapshot. Next: shift focus from architecture to data — look for label-noise (more aggressive fake-match filtering), feature additions (pick order, hero-pair history), or external signals (player MMR if obtainable).
- **Ceiling moves** (any architecture > 0.645). Treat as suspicious; first audit for HCE leakage in the new training pipeline before celebrating.

This experiment costs ~90 min and produces three numbers that constrain every subsequent proposal. It is the cheapest informative thing to do next.
