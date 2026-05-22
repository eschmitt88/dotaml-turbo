---
kind: paper
title: "Revisiting Deep Learning Models for Tabular Data"
authors:
  - Yury Gorishniy
  - Ivan Rubachev
  - Valentin Khrulkov
  - Artem Babenko
year: 2021
venue: "NeurIPS 2021 (arXiv:2106.11959)"
url: "https://arxiv.org/abs/2106.11959"
source: "raw/papers/gorishniy2021revisiting.pdf"
added: "2026-05-22"
relevance: 5
status: skimmed
related_experiments: []
related_concepts:
  - tabular-foundation-model
  - hero-embedding-vs-onehot
  - draft-prediction-plateau
tags: [tabular, transformer, ft-transformer, resnet, baseline, neurips2021, feature-tokenizer, cls-token]
---

# Revisiting Deep Learning Models for Tabular Data (FT-Transformer)

## TL;DR

Yandex team benchmarks DL architectures for tabular data under a uniform
HP-tuning protocol across 11 datasets, finding that (a) a properly tuned
ResNet is a stronger baseline than the field had acknowledged and (b) a
simple Transformer adaptation — FT-Transformer (Feature Tokenizer +
Transformer) — outperforms all other DL models on most tasks while
holding its own against tuned XGBoost/CatBoost. The "Feature Tokenizer"
projects every feature (categorical via lookup, numerical via
`b + x·W`) to a `d`-dim embedding, prepends a learned `[CLS]` token,
applies `L` pre-norm Transformer layers, and predicts from the final
`[CLS]` representation.

## Claims

- **FT-Transformer beats other tabular DL on most of 11 datasets** at
  average rank 1.8 (vs ResNet 3.3, NODE 3.9, MLP 4.8, AutoInt 5.7),
  with results averaged over 15 seeds per (dataset, model) pair
  (Section 4.4, Table 2).
- **FT-Transformer is "more universal"** than ResNet — it performs well
  across more task types, where ResNet sometimes degrades (Section 4.4).
- **Ensembling helps DL more than GBDT.** FT-Transformer ensembles
  beat ensembled tuned GBDT on the average dataset, though GBDT still
  wins specific datasets (Section 4.5, Table 4) — "no universally
  superior solution" between best DL and best GBDT.
- **Pre-Norm beats Post-Norm** for FT-Transformer and the *first*
  normalization of the *first* layer must be removed for the recipe to
  converge well (Section 3.3) — a tiny but load-bearing engineering
  note.
- **Quadratic-in-features attention is the main scaling bottleneck**;
  the paper notes that efficient attention or distillation are the
  obvious mitigations (Limitations, Section 3.3).

## Methods

The Feature Tokenizer maps each feature `j` independently:
`T_j(num) = b_j + x_j · W_j  ∈ R^d` for numerical and
`T_j(cat) = b_j + e_j^T W_j` (lookup) for categorical, stacked to
`T ∈ R^{k×d}`. A learned `[CLS]` token is prepended; `L` Pre-Norm
Transformer layers act on `(k+1)` tokens; prediction is
`Linear(ReLU(LayerNorm(T_L^[CLS])))`. Training: AdamW, cross-entropy or
MSE, no LR schedule, patience-16 early stopping, Optuna TPE for HP
search over each dataset. Continuous targets get standardization;
inputs get quantile-transform by default (with two dataset-specific
exceptions).

## Takeaways for foundation-mvp-740

- **Adopt the Feature Tokenizer + `[CLS]` head as the architectural
  spine.** Our 10 hero slots (and any per-slot side features like
  player aggregates) become `k` tokens of dim `d`; a `[CLS]` token's
  final hidden state feeds the win head. This matches the pattern that
  already works in `transformer-plus-features-740` (val_auc 0.6452 at
  77k params) and the explicit `[CLS]` design generalizes cleanly to
  the multi-head foundation setup (one `[CLS]` slot per task, or one
  shared `[CLS]` with per-task linear heads).
- **Per-feature linear projection for numerical inputs.** Don't
  concatenate scalar player-features into the input vector — wrap each
  numerical feature in its own `b + x·W` projection so the Transformer
  sees uniform-dim tokens. We already do something equivalent in
  `transformer-plus-features-740` via `Linear(8, d_model)` per slot;
  formalize this as the standard for foundation-mvp-740.
- **Use Pre-Norm and DROP the first-layer first norm.** This is the
  exact recipe the paper says was necessary for convergence; copy it
  rather than re-discovering. Cheap to verify against our existing
  MinimalTransformer (which may already do this).
- **HP search is largely exhausted on our task** (per
  [[2026-05-16-transformer-hp-sweep-740]]; 60 Optuna trials gave +0.0007
  over un-tuned), but the FT-T paper's default HP grid is the right
  starting point for a *fresh* foundation model with new features —
  don't re-run a 60-trial sweep, but copy the paper's default `d_token
  ∈ {64, 96, 128, 192}`, `n_blocks ∈ {1..4}`, dropout schedule.
- **Plan to ensemble.** FT-T benefits visibly from 5-model ensembling;
  if the multi-task foundation model wins by even +0.002, a 3-5-seed
  ensemble is the cheapest way to harvest the remaining headroom.

## Open questions / caveats

- The benchmark datasets are i.i.d. tabular tasks; none have the
  "10-token-per-row, 2-team-symmetric, partially-anonymous" structure
  of our problem. FT-T's universality claim does not by itself promise
  that the architecture beats our hand-rolled MinimalTransformer — and
  in fact the 82k MinimalTransformer at val_auc 0.6322 is already
  in the same ballpark as a generic FT-T baseline would likely be.
- The 0.6477 ceiling on this snapshot already incorporates the
  Transformer + player-feature idea; "switch to FT-Transformer" alone
  is unlikely to be the lever. The value of this paper is the
  *standard recipe* (tokenizer, CLS, Pre-Norm, ensembling), not a
  promise of further lift.
- Paper does NOT discuss pre-training. Combining FT-T with the MAE /
  contrastive objectives from [[somepalli2021saint]] and
  [[kim2024predict]] is genuinely novel territory and is the
  foundation-mvp-740 ask.
