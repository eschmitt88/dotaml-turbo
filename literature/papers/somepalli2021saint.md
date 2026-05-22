---
kind: paper
title: "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training"
authors:
  - Gowthami Somepalli
  - Micah Goldblum
  - Avi Schwarzschild
  - C. Bayan Bruss
  - Tom Goldstein
year: 2021
venue: "arXiv:2106.01342 [cs.LG]"
url: "https://arxiv.org/abs/2106.01342"
source: "raw/papers/somepalli2021saint.pdf"
added: "2026-05-22"
relevance: 5
status: skimmed
related_experiments: []
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
tags: [tabular, transformer, intersample-attention, contrastive, cutmix, mixup, self-supervised, pretraining]
---

# SAINT: Self-Attention and Intersample Attention Transformer

## TL;DR

SAINT is a tabular Transformer that stacks two attention blocks per
layer: a standard **self-attention** over the `n+1` feature tokens of a
single row (CLS-headed, à la FT-Transformer), AND an **intersample
attention** that flattens each row into one big `(n·d)`-dim token and
runs attention across the *batch* of rows — effectively letting each
sample borrow representations from similar samples in the same batch.
It also introduces a **contrastive + denoising self-supervised
pre-training** pipeline using CutMix (input-space) and mixup
(embedding-space) augmentations with InfoNCE loss. SAINT beats prior
tabular DL methods and on-average beats XGBoost / CatBoost / LightGBM
across the benchmark suite — pre-training is the lever in low-label
regimes especially.

## Claims

- **CLS-based supervised SAINT beats both deep baselines and tuned
  GBDT** on the average benchmark (Section 5.1). It also beats
  TabTransformer because it embeds continuous features into the
  attention space rather than concatenating them post-hoc.
- **Intersample attention is the novel architectural lever.** Each
  row's `n·d` flattened embedding attends to every other row's flattened
  embedding in the batch (Algorithm 1) — analogous to an in-batch
  nearest-neighbor lookup whose distance metric is learned end-to-end.
  Helps especially when some features are missing or noisy in a row
  (Section 3.2).
- **Contrastive + denoising pre-training is the first published
  contrastive-learning result for tabular.** CutMix swaps random
  feature values from another row in the batch; mixup interpolates the
  embeddings of a (real, CutMixed) pair; the contrastive (InfoNCE)
  head pulls together representations of a row and its augmented twin,
  while a per-feature denoising MLP head reconstructs the original
  features (Section 4, Figure 1).
- **Pre-training matters most in low-data regimes.** Semi-supervised
  experiments (5%, 10%, 25% labeled) show SAINT-pretrained beats
  SAINT-supervised-only by larger margins as labeled fraction shrinks
  (Section 5.2).
- **Embedding continuous features through a per-feature MLP (rather
  than the TabTransformer's "concatenate post-attention" approach)
  alone is a substantial win** — captures categorical-continuous
  cross-correlations that TabTransformer misses (Section 5.1).

## Methods

Single SAINT stage with batch of `b` rows:
```
z_i^(1) = LN(MSA(E(x_i))) + E(x_i)          # self-attention over features of row i
z_i^(2) = LN(FF_1(z_i^(1))) + z_i^(1)
z_i^(3) = LN(MISA({z_i^(2)}_{i=1..b})) + z_i^(2)  # intersample attention across batch
r_i     = LN(FF_2(z_i^(3))) + z_i^(3)
```
MISA reshapes `(b, n, d)` to `(1, b, n·d)`, runs standard
self-attention over the `b` rows, reshapes back. Pretraining: minimize
contrastive InfoNCE between SAINT(x) and SAINT(CutMix-then-mixup(x))
projection heads + per-feature MSE/CE denoising. Then fine-tune on the
labeled set with cross-entropy / MSE head on the `[CLS]` token.

## Takeaways for foundation-mvp-740

- **Adopt SAINT's contrastive + denoising pre-training pipeline as
  the foundation-mvp-740 self-supervised stage.** The 0.6477 ceiling
  is information-bound for the win-only objective; contrastive
  pre-training on Turbo's millions of unlabeled / weakly-labeled
  matches (we have 7M+ in the snapshot vs ~1.5M used for the train
  split) is a structural way to use the extra data. Pair with the MAE
  objective from [[kim2024predict]] for the denoising-style supervision
  on a fraction of tokens.
- **CutMix in our setting = swap a player slot's features with a random
  other slot from a random other match in the batch.** This is a
  natural augmentation in the symmetric 5v5 setting — the model should
  be invariant to within-team slot order, so this is a "free"
  augmentation that doubles as a data-augmentation regularizer for the
  supervised head.
- **Intersample attention is intriguing but probably skip for the
  MVP.** The pattern — "row representation attends to a batch of other
  rows" — could let our encoder borrow representation across similar
  matchups in a batch. But it adds `O(b · n·d)` compute per layer, is
  novel, and we have no evidence it helps on this task. Defer to a
  follow-up ablation; don't gate the MVP on it.
- **Use the per-feature MLP embedding (not concat).** Same lesson as
  [[gorishniy2021revisiting]]: every numerical feature gets its own
  small linear layer to project to `d_model`. SAINT confirms this
  beats post-attention concat.
- **Mixup in embedding space, not feature space.** SAINT mixes the
  embeddings, not the raw inputs, because raw mixup of categorical IDs
  is undefined. We have the same problem (hero IDs are categorical) —
  follow SAINT's design: CutMix in raw-feature space (swap whole slots),
  mixup in embedding space (interpolate post-`Embedding` vectors).

## Open questions / caveats

- SAINT was evaluated on small tabular benchmarks (i.i.d. rows,
  thousands-to-tens-of-thousands of samples). Our setting is
  millions-of-rows and 10-tokens-per-row with strong symmetry — the
  contrastive setup will be qualitatively different (the InfoNCE
  negatives are *other matches* in the batch, but matches with similar
  drafts have near-identical labels, so the contrastive signal is
  weakly defined). Worth thinking through what "positive pair" and
  "negative pair" mean in our 10-hero token setting before
  implementing. One reasonable design: positive = same match with
  different CutMix swaps; negative = different match.
- The denoising MLP heads in SAINT predict raw feature values — for us,
  this is the MAE objective from [[kim2024predict]] applied with
  proportional masking (per that paper's recipe). The two objectives
  compose; SAINT proves contrastive + denoising works together.
- SAINT does not address multi-task supervised heads — it's contrastive
  + denoising for pre-training, then a single supervised CLS head for
  fine-tuning. For foundation-mvp-740 we want both: pre-train
  contrastive + denoising, then fine-tune jointly on win + duration +
  items + KDA with UW-SO from [[kirchdorfer2024analytical]]. No paper
  in this batch combines all three; we'd be assembling.
