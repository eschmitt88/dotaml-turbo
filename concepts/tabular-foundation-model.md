---
kind: concept
name: "tabular-foundation-model"
status: seeded
added: "2026-05-22"
sources:
  - literature/papers/gorishniy2021revisiting.md
  - literature/papers/somepalli2021saint.md
  - literature/papers/kim2024predict.md
  - literature/papers/cui2022m6.md
  - literature/papers/wang2025player.md
related_concepts:
  - masked-modeling-tabular
  - multi-query-foundation-model
  - hero-embedding-vs-onehot
related_experiments: []
tags: [architecture, transformer, ft-transformer, saint, tabular-dl, foundation]
---

# tabular-foundation-model

## Definition

The architectural family in which a **shared Transformer encoder** acts
on a sequence of per-feature (or per-entity) tokens — each numerical
feature projected via `b + x·W`, each categorical feature looked up
into an embedding table — capped with a learned `[CLS]` token whose
final hidden state drives one or more lightweight task heads.
FT-Transformer (Gorishniy 2021), SAINT (Somepalli 2021), PMAE
(Kim 2024), HIGFormer (Wang 2025) and M6-Rec (Cui 2022) are all
instances despite their differences in pre-training objective, aux
heads, and intersample / graph extensions.

## Why it matters here

For `dotaml-turbo`, this is the architectural skeleton of
`foundation-mvp-740`. We have already verified that the basic recipe —
per-slot hero-embedding tokens, optional per-slot side-feature
projection, CLS-headed Transformer — gets us from val_auc 0.6322
(hero-only) to 0.6452 (hero+features) to 0.6477 (longer training) to
0.6495 (multi-task heads) on patch-7.40 Turbo. The remaining headroom
is in the *foundation-model* practices these five papers collectively
prescribe: contrastive + denoising pre-training (SAINT), proportional
masked autoencoding (PMAE), task-prompt unification (M6-Rec),
two-stream player/team encoders (HIGFormer), and the FT-T defaults
(Gorishniy) for the spine itself.

## Connections

- [[masked-modeling-tabular]] — the self-supervised pre-training
  objective that turns the encoder into a foundation model rather than
  a single-task classifier.
- [[multi-query-foundation-model]] — the "one encoder, many heads"
  inference pattern that motivates the design.
- [[uncertainty-weighted-multitask]] — how to balance the many heads'
  losses without hand-tuned α.
- [[hero-embedding-vs-onehot]] — the field already established that
  embeddings beat one-hot on this task; the foundation-model framing
  extends that by sharing the embedding table across tasks.
- [[draft-prediction-plateau]] — the empirical reference scoreboard
  this architecture family is being asked to push above.
