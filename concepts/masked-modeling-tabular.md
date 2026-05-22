---
kind: concept
name: "masked-modeling-tabular"
status: seeded
added: "2026-05-22"
sources:
  - literature/papers/kim2024predict.md
  - literature/papers/somepalli2021saint.md
related_concepts:
  - tabular-foundation-model
related_experiments: []
tags: [self-supervised, mae, masked-autoencoder, pretraining, tabular]
---

# masked-modeling-tabular

## Definition

A self-supervised pre-training objective for tabular data in which a
subset of (row, column) entries is masked from the input and the
model must reconstruct or predict their values. Unlike vision MAEs
(uniform high mask rate, e.g. 75%) and NLP MAEs (uniform 15% BERT
masking), tabular MAEs face heterogeneous columns and heterogeneous
per-column missingness rates, so the right mask strategy is
*per-column* and *proportional to observed availability* (PMAE,
Kim 2024) rather than uniform. SAINT (Somepalli 2021) pairs the
masking-then-denoising objective with a contrastive InfoNCE loss
between the row and CutMix-augmented twin, and shows this hybrid
beats either signal alone in low-label regimes.

## Why it matters here

`dotaml-turbo` has ~7M+ matches in the Turbo snapshot but only ~1.5M
in the training split that defines the 0.6477 ceiling, and 66% of
player slots are anonymous (no per-player history features
available). Masked-modeling-tabular is the obvious mechanism for
(a) leveraging the extra unsupervised matches, (b) explicitly training
the encoder to be robust to anonymous-slot inputs (which look like
the same "missing" the MAE is asked to predict), and (c) giving the
foundation model an objective that survives when the supervised win
label is weakly informative for a given match (e.g. blowouts where
the win label is unsurprising). The proportional-masking insight from
PMAE — set per-column mask rate via logit-of-observed-rate — maps
directly onto our anonymity-aware setting.

## Connections

- [[tabular-foundation-model]] — the architectural family this
  objective produces a useful encoder for.
- [[draft-prediction-plateau]] — the supervised ceiling this
  pre-training objective is meant to push past.
