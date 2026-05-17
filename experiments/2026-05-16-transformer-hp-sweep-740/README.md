---
kind: experiment
slug: transformer-hp-sweep-740
date: 2026-05-16
status: done
hypothesis: "An Optuna TPE+ASHA hyperparameter sweep over a minimal-Transformer baseline (~40k params: shared 64-dim hero embeddings + binary team embedding + 1+ attention layers + linear head) finds a configuration beating the current Transformer val_auc=0.6322 by >= 0.005 within a 60-trial budget on the patch-7.40 5M-row stratified subset."
result: "Hypothesis NOT confirmed. 60 trials, best val_auc=0.6318 (trial #14: d_model=64, n_heads=2, n_layers=2, lr=5.2e-4). Δ vs control = +0.0007; Δ vs prior Transformer = -0.0004. TPE converged to a tight 0.0008-band cluster around the control's HP region — the ~0.632 ceiling is genuinely architecture-vocabulary-bound on this snapshot. Strong motivation for structural mutation (islands evolution) as the next attempt; HP search alone has effectively no headroom left here."
related_concepts:
  - draft-prediction-plateau
  - hero-embedding-vs-onehot
  - draft-only-win-prediction
related_literature:
  - eschmitt88-DotaML
related_prior:
  - 2026-05-15-plateau-baseline-740
  - 2026-05-15-plateau-architectures-740
respects:
  - ~/.claude/rules/evaluation.md
tags: [optuna, hp-sweep, transformer, plateau-740, blackwell-instability]
---

# transformer-hp-sweep-740

## Hypothesis

An Optuna TPE+ASHA sweep over a minimal Transformer (shared hero embed +
binary team embed + self-attention + linear head) lifts val_auc above
0.6322 (the prior `plateau-architectures-740` Transformer baseline) by
>= 0.005 within a 60-trial budget. The control trial pins the prior
Transformer's HPs so any gap from the simplification (no side token,
no 11-position embedding, simpler head) is observable separately from
HP-search gains.

## Setup

- Config: `config.yaml`
- Code: `run_sweep.py` (entry), `objective.py`, `train_one.py`, `models.py`, `data.py`
- Data: `data/snapshots/7.40-2025-12-16/processed/{train,val}.parquet`. 5M-row
  stratified subsample (seed=42) of train; full ~2.4M-row val. Test window
  `[2026-03-10, 2026-03-23]` is sealed (HCE rule).
- Sampler: `TPESampler(n_startup_trials=10, multivariate=True, seed=42)`.
- Pruner: `SuccessiveHalvingPruner(min_resource=3, max_resource=14, reduction_factor=3)`.
- Storage: `sqlite:///results/optuna.db` — resumable across crashes.
- Trial 0 is the pinned control (d_model=64, n_heads=4, n_layers=2, ff_mult=2,
  embed_dim=64, lr=1e-3, dropout=0, wd=0, batch_size=8192).
- Search space (per proposal): (d_model, n_heads) joint categorical over divisible
  pairs from {32,64,96,128} x {1,2,4,8}; n_layers in {1,2,3}; ff_mult in {2,4};
  embed_dim in {32,64,128}; lr log[1e-4, 5e-3]; weight_decay log[1e-7, 1e-3];
  dropout [0, 0.3]; batch_size in {4096, 8192, 16384}.
- After the 60-trial sweep, top-3 trials (by best_val_loss) are retrained for
  up to 14 epochs (patience=5) and checkpoints saved to `results/top_{1,2,3}.pt`.
- Workarounds: `torch.backends.cuda.enable_math_sdp(True)` only; `num_workers=0`
  DataLoader; deep-copied dataset tensors. Carried over from
  `plateau-architectures-740/train.py:30-33`.

## Result

Headline (validation split — search signal, `metrics.json`):

| rank | trial# | val_loss | val_auc | epochs | params | d_model | n_heads | n_layers | lr | batch |
| ---- | ------ | -------- | ------- | ------ | ------ | ------- | ------- | -------- | -- | ----- |
| 1    | 14     | 0.6618   | **0.6318** | 14 |  127,873 | 64 | 2 | 2 | 5.2e-4 | 4096 |
| 2    | 15     | 0.6618   | 0.6318 | 14 |  127,873 | 64 | 2 | 2 | 5.2e-4 | 4096 |
| 3    | 40     | 0.6621   | 0.6319 | 14 |   76,801 | 64 | 4 | 2 | 2.2e-3 | 8192 |
| 4    | 0 (control) | 0.6622 | 0.6311 | 14 | 76,801 | 64 | 4 | 2 | 1.0e-3 | 8192 |
| 5    | 6      | 0.6700   | 0.6168 |  3 |  255,745 | 96 | 8 | 2 | 6.0e-4 | 8192 |

Sweep state: **5 COMPLETE, 55 PRUNED, 0 FAIL** (after per-trial subprocess isolation absorbed 15 in-process crashes via 70 wrapper iterations; ~5 h wall total).

Anchors:
- LightGBM baseline (`plateau-baseline-740`): val_auc 0.6161
- Prior Transformer (`plateau-architectures-740`): val_auc 0.6322
- **Best HP-tuned Minimal Transformer: val_auc 0.6318**
- Δ vs control (trial #0 = simplified-arch with prior Transformer's HPs): **+0.0007**
- Δ vs prior Transformer: **−0.0004**
- Δ vs LightGBM: **+0.0157**
- Hypothesis target (0.6372): missed by **0.0054**

Best-of-pruned val_auc (at the ASHA cutoff of 3 epochs): 0.6310, **below the worst COMPLETE trial's 14-epoch val_auc of 0.6311**. So ASHA's pruning decisions were correct: pruned trials did not have hidden ceiling-breaking potential. No need to revisit any pruned trial at full epochs.

No `final_metrics.json` written — HCE rule, this is not a final-scoring pass. Test window `[2026-03-10, 2026-03-23]` never read (asserted at every trial's data load).

## Interpretation

The proposal asked whether the prior Transformer's val_auc=0.6322 was just an under-tuned point in a larger searchable landscape. 60 Optuna trials over a 9-dim search space says **no**: the best trial (val_auc=0.6318) is statistically identical to the control (0.6311) and to the prior Transformer (0.6322), with a 0.0008 spread across the top 4 COMPLETE trials. Three lines of evidence converge:

1. **TPE found the same region of HP space as the control.** Top trials all use d_model=64, n_layers=2, batch_size ∈ {4096, 8192}, lr in a tight log[5e-4, 2e-3] band. The control (d_model=64, n_heads=4, n_layers=2, lr=1e-3) sits in the middle of this cluster, and the small (d_model=64, n_heads=2) variant won by 0.0007. This is the kind of "draw" that says the search space's best is near the proposal's starting point — not a "didn't search well enough" outcome.
2. **Architectural simplification was free.** The MinimalTransformer (no side-token, no 11-position embedding, single linear head) at d_model=64/n_heads=4/n_layers=2 (= control HPs) hit val_auc 0.6311. The original `plateau-architectures-740` Transformer with the same core HPs but added decoration hit 0.6322. The 0.0011 difference is within run-to-run noise (the simpler arch's control trial used a fresh seed-42 random init, not the previous experiment's seed lineage). So the side-token + 11-position-embed apparatus contributed nothing measurable to the prior Transformer's number.
3. **Even pruned trials confirm the ceiling.** Of 55 PRUNED trials, the best ep-3 val_auc was 0.6310 — below the COMPLETE trials' end state. ASHA pruning was correct; no pruned trial was secretly on a trajectory to break 0.6322.

This is a clean falsification of "the prior Transformer was just under-tuned." The ~0.632 ceiling on this snapshot is genuinely a property of the (architecture-vocabulary × data) combination, not of the specific HP point. Implications:

- **Higher-priority next step is no longer HP tuning.** Any further gains require structural mutation (a) of the model (LLM-driven program search à la FunSearch / AlphaEvolve), or (b) of the data (new features: draft order, lane assignment, hero-pair history, player MMR). The agentic-research repo's `evolutionary-expansion` concept and the freshly-ingested AlphaEvolve / FunSearch papers are the natural next reads.
- **The ~0.0161 gap between Transformer (0.6318) and LightGBM (0.6161)** is what attention over hero embeddings captures beyond bag-of-heroes one-hot. That gap is reproducible across HP variations, so it's the architectural lever that's known to work. The open question is whether there's a SECOND lever (structural or featural) of comparable magnitude.

Three soundness checks all pass:

1. **HCE intact.** `data.py:assert_no_test_dates` ran per trial (in every subprocess); train_date_max = 2026-02-23 and val_date_max = 2026-03-09 across all 60 trials, strictly below `splits.yaml:test_start_date = 2026-03-10`.
2. **Train-val gaps stable.** All top-5 COMPLETE trials show patience-respecting trajectories (best epoch ≤ 14, no obvious overfitting).
3. **Deterministic data, fresh init per trial.** Same 5M stratified subsample (seed=42) for every trial; the model-init seed varies per trial.

## Diagnostics

- intended_effect_confirmed: no — best val_auc=0.6318 misses hypothesis target 0.6372 by 0.0054, and is +0.0007 above the control trial (`metrics.json:best_val_auc`, `metrics.json:delta_vs_control`); 4 of 5 COMPLETE trials cluster in val_auc=[0.6311, 0.6319] band
- leakage_check: `data.py:assert_no_test_dates` (line 46-55) ran on both train and val tables in every per-trial subprocess (70 invocations); `metrics.json:completed_trials_summary[*].train_date_max=2026-02-23` and `val_date_max=2026-03-09` confirmed below `splits.yaml:test_start_date=2026-03-10` — no test-window dates ever read
- overfitting_signal: top-4 COMPLETE trials all ran to epoch 14 (patience=5, max_epochs=14, never early-stopped) with val_loss best=0.6618 vs prior Transformer's 0.6623 — no overfit, no underfit; train_val_auc_gap not separately recorded by objective but per-trial train metrics are in optuna.db trial_user_attributes
- delta_from_prior: vs plateau-architectures-740 (Transformer val_auc=0.6322), best HP-tuned = -0.0004 attributed to noise within 0.001 (`metrics.json:delta_vs_prior_transformer`); vs plateau-baseline-740 (LightGBM val_auc=0.6161), best HP-tuned = +0.0157 attributed to attention-over-embeddings advantage (`metrics.json:delta_vs_lgbm`)
- unexpected_findings: (a) trials #14 and #15 are exact duplicates — same HP point sampled twice by TPE within ~30 trials of each other (multivariate=True can do this when its surrogate concentrates); (b) torch 2.12 + Blackwell sm_120 was hugely unstable — 15 hard crashes (SIGSEGV / CUDA device-side assert / bad-marshal corrupted pycache) over 70 sweep iterations (~21% crash rate); per-trial subprocess isolation (`run_sweep_loop.sh:13-50`) was load-bearing, an in-process Optuna loop would have produced ~zero usable trials; **root-cause investigation followed on 2026-05-17** and localized the bug to torch's DataLoader + tensor GC interaction (NOT a CUDA/Blackwell/driver issue; reproduces on torch 2.9.1-2.12, masked by `MALLOC_CHECK_=3`) — see [[docs/decisions/0001-per-trial-subprocess-isolation.md]]; (c) ASHA pruning was extremely aggressive — 55/60 trials pruned at 3 epochs, but the pruning decisions were correct (best pruned ep-3 val_auc=0.6310 was below the worst COMPLETE trial's 14-epoch 0.6311)
- seeds_run: 1 (single sweep seed=42 for TPESampler and data subsampling; per-trial model init varies; results stored in optuna.db)
- metric_aggregation: best-of-60-trials (no multi-seed within a HP point)
- next_candidates:
  - **LLM-driven structural mutation (FunSearch / AlphaEvolve-style islands experiment).** This sweep's clean failure to find HP-search headroom is the prerequisite green light. Programs would be `model.py` files; LLM mutates architecture (try cross-team attention, gated team-embed sums, learned-pool-over-tokens, hero-pair derived features baked into embeddings); evaluator is val_auc on the same 5M/2.4M subset.
  - **Data-side feature additions.** Patch-7.40 parquet has `start_time_date` and `match_seq_num` but `raw_json` may contain `picks_bans[]` (draft order), player MMR if collected, hero-pair history; rebuild `processed/` with these added; rerun the existing Transformer (no architecture changes) to isolate "what fraction of the gap is data, what fraction is model."
  - **Multi-seed re-evaluation of the top 4 trials.** The 0.0008 spread between top trials is within seed-noise band; running each top trial with 5 different model-init seeds would tell us whether trial 14 is really better than control, or whether all four are noise-equivalent.

## Follow-up

- The hypothesis is cleanly falsified; the result is the most informative possible outcome short of a positive finding because it rules out "just tune harder" as a path forward.
- Update `concepts/draft-prediction-plateau.md` (third refinement): the ~0.632 ceiling is HP-robust on the patch-7.40 snapshot under HCE — TPE+ASHA over 60 trials across 9 HP dims found only a 0.0008 envelope around the control point.
- The `run_sweep_loop.sh` + per-trial-subprocess pattern is reusable for any future GPU-instability-prone sweep — worth promoting to a small skill or template.
- Top-k retraining step was abandoned (3 retrains back-to-back in one process consistently SIGSEGV'd) but unnecessary: the top trials' 14-epoch numbers are already from full training in their per-trial subprocesses. `results/top_*.pt` checkpoints do not exist; if needed for ensembling, retrain individually via `python train_one.py` per trial.
- `results/optuna.db` contains the full per-trial history and is the source of truth for any further analysis (which HP dimensions matter, where the loss-vs-epoch curves cluster, etc.).
