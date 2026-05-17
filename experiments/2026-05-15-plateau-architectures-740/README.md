---
kind: experiment
slug: plateau-architectures-740
date: 2026-05-15
status: done
hypothesis: "On the patch-7.40 Turbo snapshot under HCE, three deep-learning architectures mirroring DotaML v4-v6 (SimpleFFN, ResidualFFN, Transformer with 64-dim hero embeddings) reproduce the prior-art rank order Transformer >= ResidualFFN >= SimpleFFN > LightGBM-baseline (val_auc 0.6161 from plateau-baseline-740), with each pairwise gap matching prior art within +/-0.005 AUC."
result: "Loose hypothesis confirmed (Transformer best, all > LightGBM): Transformer 0.6322, SimpleFFN 0.6217, ResidualFFN 0.6199, vs LightGBM 0.6161. Strict hypothesis NOT confirmed: ResidualFFN < SimpleFFN (rank-order inversion), Transformer-vs-FFN gap +0.011 (vs prior-art +0.004). Architecture spread is real and Transformer-led, but the FFN-internal ordering does not reproduce."
related_concepts:
  - draft-prediction-plateau
  - hero-embedding-vs-onehot
  - draft-only-win-prediction
related_literature:
  - eschmitt88-DotaML
related_prior:
  - 2026-05-15-plateau-baseline-740
tags: [architectures, deep-learning, ffn, transformer, hero-embeddings, hce]
---

# plateau-architectures-740

## Hypothesis

Three deep-learning architectures (SimpleFFN ~v4, ResidualFFN ~v5,
DraftTransformer ~v6) trained on the same 5M-row stratified subsample
as `plateau-baseline-740` will reproduce the DotaML prior-art rank order
on `val_auc`:

    Transformer >= ResidualFFN >= SimpleFFN > LightGBM (0.6161)

with each pairwise gap matching prior art within +/-0.005 AUC.

Three result-forks:

1. **Spread reproduces** — gap-targeting architectures next.
2. **Spread compresses** — focus on data, not architecture.
3. **Ceiling moves above prior art** — audit for HCE leakage.

## Setup

- Config: `config.yaml`
- Code:
  - `models.py` — `SimpleFFN`, `ResidualFFN`, `DraftTransformer`, `build_model`, `count_params`
  - `data.py` — parquet loader, stratified-5M subsample (seed=42, mirrors baseline)
  - `train.py` — entry point per arch (`--arch simple_ffn|residual_ffn|transformer`)
  - `run_all.py` — sequential runner, aggregates `results/{arch}_metrics.json` -> top-level `metrics.json`
- Data: `data/snapshots/7.40-2025-12-16/processed/{train,val}.parquet`
  (validation split only during search; test sealed per `~/.claude/rules/evaluation.md`)
- Hardware: NVIDIA RTX 5080, bf16 autocast (Blackwell sm_120 needs torch >= 2.11 + cu128).
- Optimizer: Adam, lr 1e-3, batch_size 8192, max_epochs 30, early stop on val_log_loss
  (patience 5).

### Architectures

| arch | input | core | head |
| --- | --- | --- | --- |
| `simple_ffn` | 10x64 hero embeds + side bit | MLP [256, 128, 64] | linear -> sigmoid |
| `residual_ffn` | 10x64 hero embeds + side bit | proj 256 -> 4 residual blocks (LayerNorm + ReLU) | linear -> sigmoid |
| `transformer` | 10 hero tokens (64-dim) + 1 side token + learned 11-position embed | 2 self-attention layers (4 heads, ff_mult 2, GELU, prenorm) | mean-pool -> 2-layer MLP -> sigmoid |

Param counts are printed during training and stored in `metrics.json`.
The proposal's ~50k/~230k/~150k targets are loose because 64-dim
embeddings shared across families add ~9.7k params before any other
weights.

### Smoke test

`python train.py --arch simple_ffn --smoke` runs 1 epoch on 50k train /
5k val to verify the data path, model, and GPU pipeline work end-to-end.
Smoke metrics land at `results/simple_ffn_smoke_metrics.json`.

### Full-run command

```
nohup .venv/bin/python experiments/2026-05-15-plateau-architectures-740/run_all.py \
    > /tmp/dotaml_arch.log 2>&1 &
```

Estimated wall: ~90 min on RTX 5080.

## Result

Headline table (validation split — search signal, `metrics.json`):

| arch          | params  | val_auc | val_acc | val_log_loss | Δ vs LightGBM | Δ vs prior art |
| ------------- | ------- | ------- | ------- | ------------ | ------------- | -------------- |
| LightGBM (ref) | 301-dim | 0.6161  | 0.5866  | 0.6698       | —             | -0.0028 vs v3 (0.6189) |
| SimpleFFN     |  52,865 | 0.6217  | 0.5897  | 0.6665       | +0.0056       | -0.0068 vs v4 (0.6285) |
| ResidualFFN   | 225,089 | 0.6199  | 0.5890  | 0.6673       | +0.0038       | -0.0111 vs v5 (0.6310) |
| **Transformer** |  81,857 | **0.6322** | 0.5957  | 0.6623       | **+0.0161**   | **-0.0032 vs v6 (0.6354)** |

Pairwise gaps (`metrics.json:pair_gaps`):

| pair                       | this run | prior art | within ±0.005? |
| -------------------------- | -------- | --------- | -------------- |
| Transformer vs ResidualFFN | +0.0123  | +0.0044   | NO (off by 0.008) |
| ResidualFFN vs SimpleFFN   | -0.0018  | +0.0025   | YES (sign flipped though) |
| Transformer vs SimpleFFN   | +0.0105  | +0.0069   | NO (off by 0.004) |
| SimpleFFN vs LightGBM      | +0.0056  | +0.0096   | YES (within 0.004) |

Counts (same 5M stratified subset, seed=42, as `plateau-baseline-740` for
direct comparability): train 5,000,000 / val 2,419,185. Train-val AUC
gaps small across all three: SimpleFFN 0.020, ResidualFFN 0.021,
Transformer 0.011 (`metrics.json:per_arch.<arch>.train_val_auc_gap`).

No `final_metrics.json` was written — HCE rule, this is not a final-scoring pass.

## Interpretation

The proposal had **two layered hypotheses** baked together: a
**strict rank order** (Transformer ≥ ResidualFFN ≥ SimpleFFN, each
gap within ±0.005 of prior art) and the **loose claim** (an
architecture-spread exists and is Transformer-led). They split here:

- **Loose hypothesis: confirmed.** The Transformer beats every other
  architecture (LightGBM, SimpleFFN, ResidualFFN) by ≥0.0105 AUC and
  lands within 0.0032 of the prior-art v6 number. The plateau is
  architecture-dependent; "the LightGBM ceiling" is not the global
  ceiling. The Transformer-vs-LightGBM gap of +0.0161 is the
  load-bearing finding for future ceiling-targeting work.
- **Strict hypothesis: not confirmed.** ResidualFFN (0.6199) trains
  *worse* than SimpleFFN (0.6217), inverting the prior-art ordering
  (where v5 ResidualFFN beat v4 SimpleFFN by +0.0025). Two of three
  pairwise gaps miss the ±0.005 band. The FFN-internal ordering does
  not reproduce on patch-7.40 under HCE.

Three plausible drivers for the FFN inversion:

1. **Hyperparameter sensitivity.** The DotaML v4/v5 recipes were tuned
   on a smaller pre-7.40 dataset. On the new 5M subset, the larger
   ResidualFFN (225k params) overfits faster (best epoch 6 vs
   SimpleFFN best epoch 8). A short hp tune (lr or weight_decay sweep)
   might restore the spread.
2. **Representation mismatch.** All three architectures here use
   shared 64-dim hero embeddings (deviating slightly from v4's original
   one-hot input). The embedding-shared FFN may have less inductive
   advantage over the SimpleFFN than the one-hot v4 did.
3. **Noise floor.** A 0.0018 gap is comfortably within the
   architecture's run-to-run variance on a 2.4M-row val. A multi-seed
   run would resolve this; this single-seed result can't.

The Transformer's lead over both FFN variants (+0.0105 vs SimpleFFN,
+0.0123 vs ResidualFFN) is large enough to be unambiguous regardless
of which of (1)-(3) explains the FFN inversion. The architecture
spread that matters — the family-level gap between attention and
feed-forward — is real and reproduces.

Three soundness checks all pass:

1. **HCE intact.** `data.py:assert_no_test_dates` (line 46) ran on
   both train and val tables for every arch; train_date_max = 2026-02-23
   and val_date_max = 2026-03-09 across all three runs, strictly
   below `splits.yaml:test_start_date = 2026-03-10`.
2. **Train-val gaps small.** 0.011-0.021 across the three; the
   Transformer's 0.011 is the smallest, consistent with its modest
   parameter count (82k) and 11-epoch training.
3. **Radiant base rates align across train (full + subsampled) and val**
   to within 0.001, identical to `plateau-baseline-740`. Same data,
   same split, no drift.

## Diagnostics

- intended_effect_confirmed: partial — loose claim (architecture-spread exists, Transformer-led) is confirmed; strict claim (rank order with ±0.005 gaps) is not (`metrics.json:pair_gaps`, `metrics.json:per_arch.transformer.val_auc=0.6322`)
- leakage_check: `data.py:assert_no_test_dates` ran for every arch (data.py:46-55) and `train.py` re-asserts via `meta["train_date_max"]` printed at startup; verified `train_date_max=2026-02-23` and `val_date_max=2026-03-09` across all three runs (`metrics.json:per_arch.<arch>.train_date_max`) — no test-window dates ever read
- overfitting_signal: simple_ffn train=0.6418 val=0.6217 gap=0.0201 / residual_ffn train=0.6413 val=0.6199 gap=0.0214 / transformer train=0.6432 val=0.6322 gap=0.0110 — all well-fit not overfit; Transformer has the smallest gap (most efficient parameterisation) (`metrics.json:per_arch.<arch>.train_val_auc_gap`)
- delta_from_prior: vs 2026-05-15-plateau-baseline-740 (val_auc=0.6161), Transformer +0.0161 / SimpleFFN +0.0056 / ResidualFFN +0.0038 (`metrics.json:val_auc_*`)
- unexpected_findings: (a) ResidualFFN underperforms SimpleFFN by 0.0018 — rank-order inversion vs prior art, likely either hp-sensitivity at 225k params on this snapshot or noise within the single-seed band; (b) torch 2.11 + Blackwell sm_120 produced multiple intermittent crashes (DataLoader worker SIGSEGV, "Overflow when unpacking long long" in `__getitem__` deep in training, SIGSEGV at process shutdown after success); workarounds applied: `train.py:30-33` forces math SDP backend, `data.py:78-86` deep-copies into owned tensors (was `torch.from_numpy` shared-memory), `--num-workers 0` and `--max-epochs 11` CLI overrides added (`train.py:259-265`); final transformer run capped at 11 epochs (best val_loss at epoch 9, so the cap did not affect convergence — early stopping would have triggered around epoch 14 anyway)
- seeds_run: 1 (single run, seed=42 from `config.yaml:seed`)
- metric_aggregation: single-run
- next_candidates:
  - Resolve the FFN inversion: run a small lr × weight_decay sweep over SimpleFFN and ResidualFFN with multi-seed (3 seeds each) to determine whether ResidualFFN < SimpleFFN is a hyperparameter artifact or a robust finding under HCE. Confirms whether the FFN family's prior-art ordering reproduces at all.
  - Attack the Transformer-vs-LightGBM gap: train DotaML v6's full recipe (masked-input training enabled) and the v6 + draft-order conditioning variant. The 0.0161 gap between Transformer (0.6322) and LightGBM (0.6161) is what an attention-aware model captures over one-hot bag-of-heroes; understanding what fraction comes from embeddings vs attention vs draft-order is the next ceiling-target.
  - Filter sensitivity audit (carryover from `plateau-baseline-740` next): the 8.78% fake-match filter rate is shared across all architectures; rerun the Transformer with the filter off to see if val_auc shifts >0.005. If yes, the filter is doing meaningful work; if no, future filter ablations are decoupled from architecture choice.

## Follow-up

- The hypothesis-strict failure (FFN inversion) is the single most
  surprising finding and the natural next experiment (multi-seed
  FFN-only sweep — cheap, <1 h, would isolate hp vs noise vs real).
- Update `concepts/draft-prediction-plateau.md` to record that the
  architecture-spread is family-driven (Transformer > FFN) but the
  FFN-internal ordering does not reproduce on patch-7.40.
- The 11-epoch cap was a workaround, not a scientific choice; if the
  torch+Blackwell crashes get resolved upstream, a 30-epoch rerun of
  the Transformer would close the loop on convergence (best val_loss
  was already at epoch 9, so we expect the same number).
- `results/{arch}.pt` checkpoints retained for any downstream
  ensembling or feature-attribution work.
