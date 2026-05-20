---
kind: proposal
slug: transformer-plus-features-extended-740
date: 2026-05-19
status: implemented
experiment: experiments/2026-05-19-transformer-plus-features-extended-740/
hypothesis: "Extending training on the transformer-plus-features-740 architecture and feature set from 14 epochs to up to 30 epochs with early stopping (patience=5 on val_log_loss) raises val_auc by ≥ 0.001 over the prior 0.6452 (target val_auc ≥ 0.6462). The prior run's `best_epoch=14=max_epochs` indicates the model was still improving at the cap, so additional training is essentially free signal."
rationale: >
  transformer-plus-features-740 reached val_auc=0.6452 in 14 epochs
  but hit best_epoch=14=max_epochs (val_loss still decreasing). The
  prior plateau-architectures-740 and transformer-hp-sweep-740 runs
  also converged around epoch 9-11 with cap=14 — for the
  ARCHITECTURE-ONLY models, 14 epochs was probably enough. But the
  combined model (with player features adding ~13% more degrees of
  freedom via the Linear(8, d_model) projection) plausibly needs
  longer to converge. This is the cheapest possible follow-up to a
  strong positive result; reuses all existing scripts with a single
  config change.
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/draft-only-win-prediction]]"
  - "[[literature/papers/hodge2017win]]"
  - "[[experiments/2026-05-18-transformer-plus-features-740]]"
  - "[[experiments/2026-05-18-player-features-prepatch-740]]"
  - "[[experiments/2026-05-15-plateau-architectures-740]]"
  - "[[experiments/2026-05-16-transformer-hp-sweep-740]]"
expected_metric:
  name: val_auc
  target: 0.6462
  direction: higher-is-better
design_sketch:
  - Reuse `experiments/2026-05-18-transformer-plus-features-740/{models.py, data.py, train.py}` verbatim (no code change to the model or training loop logic).
  - Single config change in this experiment's `config.yaml`; `max_epochs: 30` (was 14), `early_stopping_patience: 5` (was None or 0), `early_stopping_metric: val_log_loss`.
  - Single ablation; only the PRIMARY (`transformer_plus_features`). The architecture_only sanity already passed at 0.6319 in the prior experiment (matched plateau-architectures-740 within 0.0003); not re-run here.
  - Reuse processed parquet from player-features-prepatch-740 (same input as prior). Same 5M-row stratified subsample (seed=42).
  - Same data sanitization in data.py (fp32-max sentinel clip) — the upstream patch is a separate experiment (next in queue) and not gated on this.
  - Per-trial subprocess isolation via run_all.sh (same pattern as prior); single ablation = 1 invocation + retry.
  - HCE strict; never read [2026-03-10, 2026-03-23] dates. Asserted at train time.
  - Coverage-bucket val_auc diagnostic carried over from prior (low/med/high terciles).
  - Save the trained checkpoint to `results/transformer_plus_features_extended.pt` for downstream reuse.
risks:
  - Marginal gain may be smaller than the 0.001 target. The Transformer was already mostly converged by epoch 9-11 in prior architecture-only runs; the combined model may not need 16 more epochs. A +0.0005 result would still be informative but would suggest 14 was near-converged.
  - Overfitting risk grows with more epochs. The prior train-val AUC gap wasn't separately captured in metrics.json for the combined model; if extended training exposes a wide gap, the early-stopping patience=5 should pick a moderate epoch and that's the right outcome — but worth watching.
  - If `best_epoch=30=max_epochs` again (still not converged at 30), suggests we need either cosine LR schedule, warmup, or a different lr — a follow-up experiment, not this one.
  - Blackwell torch instability could trip the retry path (precedent: 5 of 6 prior Transformer experiments had at least one retry; transformer-plus-features-740 was the lucky exception). run_all.sh already handles MAX_RETRIES=3 per ablation.
related_prior:
  - 2026-05-18-transformer-plus-features-740
  - 2026-05-18-player-features-prepatch-740
estimated_runtime: "≈30-50 min on RTX 5080 (up to 30 epochs × ~55s each = 27 min; plus eval + checkpoint save + retry overhead). Disk; <50 MB for new checkpoint + plots. Well under budget.yaml max_wall_hours=24."
---

# Extended-training follow-up — cheapest possible win on the new ceiling

The `transformer-plus-features-740` run lifted whole-val val_auc from ~0.632 (either lever alone) to **0.6452** (both levers combined). But the run cut off at `epochs_run=14=max_epochs=best_epoch`, meaning val_loss was still decreasing when training ended. This is the single cheapest experiment in the queue: same code, same data, just bump the epoch cap and add early stopping.

If the model was genuinely converged at 14, the +0.001 target won't be met and we'll know the 14-epoch number was close to optimal. If it wasn't converged, we get free signal — and the next follow-ups (upstream data cleanup, player embeddings) start from a slightly stronger reference.

Three result forks:

- **val_auc ≥ 0.6462 (confirmed).** Extended training is essentially free signal. New whole-val reference for downstream experiments. May also lift the HIGH-coverage bucket above the current 0.6560 toward Hodge 2017's 75-76% in-game-telemetry ceiling.
- **val_auc in [0.6452, 0.6462) (essentially flat).** The 14-epoch number was near-converged; further epochs are diminishing returns. Move on; the 0.6452 reference stands.
- **val_auc < 0.6452 (regression).** Either overfitting kicked in past epoch 14 and early stopping wasn't aggressive enough, OR run-to-run noise without seed-variance control. The fix (tighter patience or multi-seed) goes into a separate follow-up.

There's also a useful operational benefit: this experiment captures the actual `best_epoch` under a longer schedule, which informs the right max_epochs for ALL future combined-model training. If the answer is "best_epoch=17" we know 20 is the right cap forever; if it's "best_epoch=30=max_epochs" again we have a different problem.
