---
kind: proposal
slug: transformer-hp-sweep-740
date: 2026-05-16
status: implemented
experiment: experiments/2026-05-16-transformer-hp-sweep-740/
hypothesis: "An Optuna TPE+ASHA hyperparameter sweep over a minimal-Transformer baseline (~40k params: shared 64-dim hero embeddings + binary team embedding + 1 attention layer + linear head) finds a configuration beating the current Transformer val_auc=0.6322 by ≥ 0.005 within a 60-trial budget on the patch-7.40 5M-row stratified subset."
rationale: >
  The plateau-architectures-740 Transformer (val_auc 0.6322) was sized
  by analogy with DotaML v6 rather than by any signal from this
  snapshot. Its architecture also carries decoration the task does not
  use — a side-token branch driven by a constant-1 side_bit, an
  11-position embedding mostly learning team membership, a two-layer
  output head with redundant LayerNorm. A clean minimal baseline
  + Bayesian HP search isolates the "best Transformer the existing
  architecture vocabulary can produce" before any structural mutation
  experiment (LLM-driven islands evolution) can claim it adds value
  beyond what TPE in a fixed search space can find. This is the proper
  anchor; any subsequent FunSearch/AlphaEvolve-style experiment must
  beat THIS number, not 0.6322.
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/hero-embedding-vs-onehot]]"
  - "[[concepts/draft-only-win-prediction]]"
  - "[[experiments/2026-05-15-plateau-architectures-740]]"
  - "[[experiments/2026-05-15-plateau-baseline-740]]"
expected_metric:
  name: val_auc
  target: 0.6372
  direction: higher-is-better
design_sketch:
  - Reuse data/snapshots/7.40-2025-12-16/processed/{train,val}.parquet from the prior two experiments. Same 5M stratified subset (seed=42) for every trial → fair comparison both within the sweep and against plateau-baseline-740 (LightGBM 0.6161) and plateau-architectures-740 (Transformer 0.6322).
  - Minimal MinimalTransformer architecture (new models.py:MinimalTransformer); 64-dim shared hero embedding + 2-vocab team embedding (Radiant=0/Dire=1) added to hero embeddings; 10-token sequence (no side-token, no 11-position embedding); search-space-determined attention layers; mean-pool over 10 tokens; single Linear(d_model → 1) head.
  - Optuna sweep using TPESampler (Bayesian) + ASHASuccessiveHalving pruner; min budget 3 epochs, max budget 14 epochs (current Transformer plateaus by epoch 9), reduction factor 3. SQLite study at experiments/.../optuna.db for resumability.
  - Search space; d_model ∈ {32, 64, 96, 128}, n_heads ∈ {1, 2, 4, 8} (sampled conditional on d_model % n_heads == 0), n_layers ∈ {1, 2, 3}, ff_mult ∈ {2, 4}, embed_dim ∈ {32, 64, 128}, lr log[1e-4, 5e-3], weight_decay log[1e-7, 1e-3], dropout [0, 0.3], batch_size ∈ {4096, 8192, 16384}.
  - First trial pinned as a "control"; a hand-selected MinimalTransformer config matching the current Transformer's HPs (d_model=64, n_heads=4, n_layers=2, ff_mult=2, lr=1e-3, dropout=0). This calibrates whether any val_auc gap vs 0.6322 comes from the architecture simplification or from HP search.
  - Workarounds carried forward from plateau-architectures-740; --num-workers 0, math SDP backend, deep-copied DraftDataset (torch 2.11 + Blackwell sm_120 stability).
  - Trial budget; 60 trials, expected ~3-5 min average each after ASHA pruning. Wall budget ≤ 6 hr.
  - Per-trial metrics; val_auc, val_log_loss, train_val_auc_gap, epochs_run, params_total, all stored in optuna.db. Top-3 trials get full 14-epoch retraining + checkpoints saved to results/.
risks:
  - Stripping the side-token + 11-position-embedding branches removes capacity the current Transformer used; if the minimal architecture's control trial undershoots 0.6322 by more than 0.005, attribute the gap before celebrating any sweep finding. The hypothesis is contingent on the simpler architecture being a sound starting point.
  - TPE on a low-dimensional mixed space converges fast but can exploit early; SOBOL warmup for the first 10 trials mitigates premature exploitation.
  - ASHA pruning at min_budget=3 epochs is aggressive; a slow-starting good HP combo (e.g. higher lr needing more warmup) might be killed early. Mitigation; track ASHA ratio in metrics.json and inspect pruned-trials' epoch-3 vs epoch-9 trajectories before drawing strong conclusions.
  - 60 trials is a small budget for 9-dim search; result is a "good enough" config, not a global optimum. Worth reporting "best 5 trials" rather than just "best 1" to convey the noise band.
  - torch 2.11 + Blackwell intermittent crashes (documented in plateau-architectures-740 log) may interrupt the sweep mid-trial. Mitigation; Optuna study is SQLite-backed and resumable; failed trials get retried up to 2x.
related_prior:
  - 2026-05-15-plateau-baseline-740
  - 2026-05-15-plateau-architectures-740
estimated_runtime: "≈4-6 hr on RTX 5080 (60 trials × ~3-5 min average with ASHA pruning); processed data already local; <1 GB additional disk for optuna.db + top-3 checkpoints. Well under budget.yaml max_wall_hours=24 and max_disk_gb=500."
---

# Transformer HP sweep on the patch-7.40 plateau

The current Transformer at `plateau-architectures-740` (val_auc 0.6322) was constructed by analogy with DotaML v6's recipe rather than by any tuning signal from the patch-7.40 snapshot. Its architecture carries decoration the task does not use: a side-token branch driven by a constant-1 input bit, an 11-position embedding that mostly recovers team membership, and a two-layer output head with a redundant LayerNorm. A clean minimal baseline — single attention layer, binary team embedding added to hero embeddings, single linear head — strips this ornamentation back to the structural choices that actually matter for a 10-hero classification task.

The Optuna TPE+ASHA setup is the standard Python recipe for ML hyperparameter optimization (Optuna 2024+, framework-agnostic, single-machine on RTX 5080 — Ray Tune's distributed machinery is overkill at this scale). 60 trials with ASHA pruning costs ~5 hr wall and produces a Bayesian-search-anchored "best Transformer this architecture vocabulary can produce" number.

This experiment is the necessary anchor before any LLM-driven evolutionary search (the FunSearch / AlphaEvolve family ingested into agentic-research this week). The reasoning: an islands-evolution experiment can only claim the LLM-mutated structural changes add value if they beat what TPE in a fixed parametric space already finds. Without this baseline, a +0.005 islands result is ambiguous between "structural mutation matters" and "the original Transformer was just under-tuned."

How the result moves downstream work:

- **Best trial val_auc ≥ 0.6372 (hypothesis confirmed).** The plateau is not architectural; it's at least partly a tuning artifact. Re-run all prior architectures with their own Optuna sweeps before claiming any new architecture is better.
- **Best trial 0.6322-0.6372** (control matches but no clear improvement). The current architecture was approximately optimal in the existing search space. Islands evolution becomes the higher-priority next move.
- **Best trial < 0.6317 (control undershoots current Transformer by > 0.005).** The architectural simplification removed something load-bearing — likely the position-embedding-as-team-signal. Audit what the minimal arch lost and either re-add it or run the sweep on the un-simplified architecture.
- **All trials cluster at val_majority_class_acc** (model can't beat base rate at small d_model). The search space's lower bound is too low; constrain it.

This experiment also tests, secondarily, whether Optuna's TPE on this problem converges quickly (< 30 trials) or slowly (> 50). That signal is itself useful for sizing future HP work in this project.
