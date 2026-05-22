---
kind: paper
title: "To Predict or Not To Predict? Proportionally Masked Autoencoders for Tabular Data Imputation"
authors:
  - Jungkyu Kim
  - Kibok Lee
  - Taeyoung Park
year: 2024
venue: "AAAI 2025 (arXiv:2412.19152)"
url: "https://arxiv.org/abs/2412.19152"
source: "raw/papers/kim2024predict.pdf"
added: "2026-05-22"
relevance: 4
status: skimmed
related_experiments: []
related_concepts:
  - masked-modeling-tabular
  - tabular-foundation-model
tags: [tabular, mae, masked-autoencoder, imputation, mlp-mixer, proportional-masking, aaai2025]
---

# Proportionally Masked Autoencoders for Tabular Data Imputation (PMAE)

## TL;DR

PMAE argues that uniform 50%/75%-style random masking — copied from
image/text MAEs — is wrong for tabular data because columns are
heterogeneous and exhibit different empirical missingness rates.
Instead, PMAE masks each column `j` with probability proportional to
its inverse observed rate via a logit-transform mask function
`M_j(p_obs,j) = a · logit(1 − p_obs,j) + b` (defaults `a=0.05, b=0.5`),
which assigns higher prediction loss to columns that are usually
sparsely observed. It also argues MLP-Mixer token mixing beats
self-attention for the imputation task on small tables and proposes a
unified "Imputation Accuracy" metric (categorical accuracy + numerical
R²). Up to +34% over ReMasker on the General missing pattern.

## Claims

- **Uniform random masking biases the MAE training signal** away from
  the columns that need prediction most (those with high natural
  missingness) and toward columns that are usually observed
  (Motivation, Eq. 7). Proportional masking is the principled fix.
- **PMAE beats every imputation baseline tested** (Naive, KNN, EM,
  MissForest, MIWAE, GAIN, MIRACLE, HyperImpute, TDM, ReMasker) on the
  unified Imputation Accuracy metric, averaged over 9 UCI datasets
  with three semi-synthetic missing patterns (Monotone, Quasi-Monotone,
  General) under the MNAR mechanism (Table in p. 6). PMAE-transformer
  ranks 2.3 / 12, PMAE-mixer slightly better still.
- **MLP-Mixer ≥ Transformer for tabular imputation.** PMAE-mixer beats
  PMAE-trf on the harder General pattern, supporting the authors'
  argument that column interactions are "grouped" rather than
  pairwise-attention-shaped (Architecture section).
- **The logit transform matters.** Table 3 ablates the masking
  function family (linear, sigmoid, logit) and finds logit best — it
  monotonically increases the mask rate as observed rate falls, with
  the right slope behavior near 0/1.
- **Tiny number of HPs** — only `a` and `b`; grid search over both
  confirms `a=0.05, b=0.5` is a robust default across datasets (Fig 6).

## Methods

The full MAE loss decomposes into a prediction term (when entry was
already observed and the model masks it for self-supervised prediction)
and a reconstruction term (when entry was originally missing); per
column `j` the expected loss is
`E[l_ij] = M_j · l_ij^prediction + (1 − M_j) · l_ij^reconstruction`.
PMAE sets `M_j = clip([a · log((1−p_j)/p_j) + b], 0, 1)` where `p_j`
is the per-batch observed proportion in column `j`. Architectures:
either a vanilla Transformer block (self-attention over `n` column
tokens of dim `c`) or an MLP-Mixer block that swaps the token-mixing
self-attention for a small MLP over the `d`-axis. Otherwise the
encoder-decoder, optimizer, and evaluation protocol mirror ReMasker
(Du, Melis, Wang 2024).

## Takeaways for foundation-mvp-740

- **Adopt per-token mask rate proportional to observed availability,
  NOT uniform 75%.** In our setting the natural "missingness" is
  per-slot anonymity (66% of player slots are `account_id ∈ {0,
  2^32-1}`, see [[draft-prediction-plateau]]) and per-feature
  unavailability (e.g. `smoothed_winrate_hero` is NaN for any slot
  with zero prior history on that hero). For the MAE pretext task on
  the foundation model, each (slot, feature)-token's mask rate should
  scale with how often it's already missing — anonymous-account tokens
  and zero-history tokens get high mask rates (they need the model to
  predict them); fully-observed tokens get low mask rates. Cf. PMAE's
  Eq. (11) with our 8 player-feature columns + the hero-ID column.
- **Pick mask rates in the 30-50% range per-column, not 75%.** The
  paper does NOT use a single global mask rate; the effective ratio
  fluctuates per-column around `b=0.5`. For us this means: don't copy
  the BERT 15% or vision-MAE 75%. Start with per-column rates that
  reflect actual prior anonymity / sparsity (probably 30-50% effective
  mask after the logit transform on our columns).
- **MLP-Mixer is a candidate token-mixer worth a 1-shot ablation.**
  PMAE shows attention is *not* obviously better than MLP-mixing on
  small tabular tables. Our 10-slot draft is small enough that an
  MLP-Mixer encoder could match or beat self-attention at lower
  parameter cost — worth a single comparison run inside the foundation
  experiment, especially if attention's quadratic-in-tokens cost is a
  scaling concern when we add side features.
- **Use the prediction/reconstruction decomposition framing.** Our MAE
  objective should explicitly separate "predict a value that was
  observed in raw data but we masked" from "reconstruct an unmasked
  observed value" — the former is the load-bearing self-supervised
  signal; the latter is regularization. The proportional weighting
  drops out naturally.
- **Defer "imputation accuracy" as eval.** Our task is win-prediction,
  not value imputation; PMAE's metric isn't the right benchmark. The
  MAE objective in our setup is purely a pre-training auxiliary head.

## Open questions / caveats

- PMAE's 9 datasets are all small (442-48842 rows, 10-55 columns); our
  setting is the opposite (millions of rows, 10 hero slots + ~80
  per-slot features). The proportional-masking principle should
  generalize, but the MLP-Mixer-vs-Transformer ablation may not — at
  our scale and with attention already winning at
  `transformer-plus-features-740`, the Mixer is a hedge, not a
  favorite.
- The PMAE loss assumes the model gets to see the propensity (observed
  rate `p_j`) of each column at training time. We have direct access
  to per-column anonymity rates and per-feature NaN rates, so this is
  cheap. Just compute these once on the training snapshot and freeze.
- PMAE doesn't discuss multi-task heads (it's pure imputation). The
  proportional masking choice is independent of how many heads sit on
  top, so this composes cleanly with the multi-task setup from
  [[2026-05-20-rich-supervision-multitask-740]] and the UW-SO loss
  weighting from [[kirchdorfer2024analytical]].
