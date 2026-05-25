---
kind: concept
name: "Embedding-vs-features gradient competition"
status: growing
added: "2026-05-25"
updated: "2026-05-25"
sources: []
related_concepts:
  - "[[hero-embedding-vs-onehot]]"
  - "[[tabular-foundation-model]]"
related_experiments:
  - "[[experiments/2026-05-19-embedding-prelim-740]]"
  - "[[experiments/2026-05-25-v3-ablations-740]]"
tags: [architecture, optimization, multi-task]
---

# Embedding-vs-features gradient competition

## Definition

When a model has both dense engineered features and a sparse
identity-embedding lookup competing to explain the same target,
the dense features dominate gradient flow and starve the embeddings
of learning signal — even when the embeddings could in principle
capture orthogonal information.

## Two failure modes — depending on data scale

The original framing was that embeddings fail by being **starved
of gradient signal**. The v3-ablations-740 A2 result proved that's
only one of two failure modes. Which one fires depends on the
data-per-player ratio:

| Data regime | Outcome | Mechanism |
|---|---|---|
| 7.40-only (~5M rows, ~3-5 obs/player) | NULL, val_auc=0.6476 (embedding-prelim-740) | Gradient starvation — embedding stays at init (clauses 1-3 below) |
| Extended cross-patch (~32M rows, more obs/player) | CATASTROPHIC OVERFIT, val_auc=0.6290 (v3-ablations-740 A2) | Train-val distribution shift — embedding memorizes train-time player behavior that doesn't transfer to a structurally different val window |

The overfit mode is the more dangerous one: a NULL result is easy
to recognize and easy to abandon. The overfit result looks like
working code (loss decreases, training is stable, no NaN) until
you compare val to train. Without the train-vs-val gap diagnostic,
it can be mistaken for "embeddings just need more epochs."

## Mechanism: gradient starvation (the 7.40-only failure mode)

`embedding-prelim-740` (2026-05-19) added 16M params of per-player
identity embeddings to the transformer-plus-features baseline on
7.40-only data and produced val_auc=0.6476 — exactly the cleanup
anchor, zero lift.

The mechanism — the original framing of this concept:

1. **Gradient redundancy.** The 8-feature projection already encodes
   "this player has 55% smoothed winrate on Pudge" — there is
   limited residual variance for the embedding to capture.

2. **Easy path wins.** Dense features provide a gradient signal on
   every training example to the feature_proj weights. Embeddings
   provide a sparse signal only when that specific player's row
   appears. Sparse signals lose to dense signals on the same loss
   surface — the optimizer "uses" the features first because the
   gradient flow is cleaner.

3. **Anonymous-slot dominance.** ~66% of slots are anonymous in
   Turbo and route to the shared anonymous-embedding row. That row
   gets the bulk of the gradient updates while individual
   non-anonymous embeddings get sparse, noisy updates.

## Mechanism: train-val distribution shift overfit (the extended-data failure mode)

`v3-ablations-740 A2` (2026-05-25) added a 4M-param player
embedding lookup (top-30k frequent + 1024 hash buckets + 1
anonymous) on top of the v3 foundation, trained on the extended
Aug 2025 → Feb 2026 corpus. Result: best val_auc=0.6290 at epoch 2,
then catastrophic overfit (val_auc declined to 0.6114 by epoch 7,
early-stopped). The trajectory diagnostic was unmistakable:

- train_win loss: 0.6812 → 0.6550 (going DOWN — train improving)
- vl_win loss:   0.6682 → 0.6840 (going UP — val WORSE)
- val_auc:       peaks at epoch 2, monotonic decline thereafter

Coverage-bucket val_auc hurt UNIFORMLY (low −0.015, medium −0.017,
high −0.019). Note the high-coverage bucket was hurt the MOST —
the opposite of what "embeddings help frequent players" would
predict.

The mechanism:

1. **Per-player signal is now learnable.** Top-K frequent players
   have enough observations in train (~32M rows) that gradient
   updates can shape their embedding rows meaningfully — the
   starvation mode is gone.

2. **Train-val distribution shift.** The train window covers
   Aug 2025 → Feb 2026 (multi-patch, ~6 months of player meta
   evolution). Val is purely 7.40 patch in late Feb / early Mar.
   A player's recent form, hero pool, and skill trajectory at val
   time differ from their behavior averaged across the training
   window. The embedding learns the latter; can't generalize to
   the former.

3. **No regularization on the embedding table.** AdamW with
   `weight_decay=0.0` on the embedding (standard practice for
   sparse embedding learning) means no shrinkage toward zero.
   Per-player rows free to grow unboundedly to fit train, then
   misfire at val.

4. **Composition amplifies.** PMAE + multi-task heads use the
   embedded representation as input to multiple auxiliary
   objectives. Overfit in the embedding propagates through every
   head, not just the win head.

## When to expect each failure mode

- **n_obs per top-K player < 10**: gradient starvation regime →
  use embedding-prelim-740 mitigations (asymmetric LR, drop
  features, auxiliary embedding loss).
- **n_obs per top-K player ≥ 50 AND val window ≠ train window
  (different patch / time / meta)**: overfit regime → use
  regularization (weight decay > 0 on embedding, dropout on
  embedding output, embedding L2 penalty as auxiliary loss),
  shorter training (early-stop aggressively), or align train-val
  distributions before training.
- **n_obs per top-K player ≥ 50 AND val window matches train
  distribution**: untested regime in this codebase — embeddings
  might actually work.

The v3-ablations-740 A2 vocab had a frequency cliff at 192
matches for the top-30k cutoff, so all individual-row players had
n_obs ≥ 192 — squarely in the overfit regime, and val was a
different time window.

## What embeddings could capture that engineered features can't

These are real signals in principle — the question is whether the
optimizer allocates enough gradient capacity to the embedding to
learn them despite the easier feature path:

- Player style beyond aggregate winrate (aggressive vs passive,
  mid-vs-late-game player, off-meta-pick tendencies).
- Skill in specific roles (carry vs support performance).
- Player synergies (player X wins more often paired with Y).
- Things the lookback window can't see (improvement trajectory,
  season-over-season changes, role flexibility).

## Diagnostics to run BEFORE concluding "embeddings don't work"

Required to disambiguate the two failure modes:

- **train_win vs vl_win trajectory per epoch.** If train improves
  while val degrades → overfit mode. If both stay flat → starvation
  mode.
- **Embedding L2-norm distribution at end of training.** Compare to
  init L2 (~0.22 for the standard init in this codebase). If
  ~unchanged → starvation. If grew significantly (especially for
  non-anonymous rows) → embedding learned, but possibly overfit.
- **Cosine similarity sample for top-K players.** Random pairs ~0
  → starvation. Structured (some +ve, some -ve) → learned.
- **Coverage-bucket val_auc breakdown.** Starvation: bucket val_auc
  matches the no-embedding baseline. Overfit: bucket val_auc
  HURT, especially the high-coverage bucket.

## Mitigations by failure mode

**Starvation mode** (7.40-only-style, embedding stays at init):

- Drop the engineered features in a parallel arm. Forces the
  embedding to carry the full per-player signal.
- Asymmetric optimizer: higher LR or longer warmup on embedding
  params. Lets sparse params catch up to dense ones.
- Auxiliary embedding loss: force the embedding to predict the
  engineered features from itself alone (reconstruction loss).
- Reset / freeze feature projection for a few epochs while
  embedding warms up.

**Overfit mode** (extended-data-style, train ↓ val ↑):

- **Weight decay > 0** on the embedding table (skip the
  "standard practice" of no decay; the standard practice was
  designed for embedding tasks without distribution shift).
- Dropout on the embedding lookup output before it enters the
  transformer.
- Auxiliary L2 penalty on the embedding L2 norm as a side loss.
- Aggressive early-stop with patience=2 on vl_win (rather than 5).
- Aligning train-val distributions before training: e.g., subsample
  train to match val's patch / time window, OR weight train
  examples by recency / patch similarity to val.
- For the foundation-model goal specifically: a player embedding
  may need its own out-of-sample evaluation (e.g., only count val
  rows where ≥ 1 player is OUT-OF-VOCAB in the embedding table,
  forcing the rest of the network to generalize).

## Connections

- `[[hero-embedding-vs-onehot]]` — same competition pattern but
  for hero identity (resolved positively: hero embeddings beat
  one-hot at ~10× param budget). Different shape: hero vocab is
  ~130, player vocab is ~5M.
- `[[tabular-foundation-model]]` — the general design space these
  trade-offs sit inside.
- The "feature-dominated regime" is a known issue in tabular ML
  literature; SAINT (Somepalli 2021) addresses it by making the
  feature interactions richer, not by adding identity embeddings.
