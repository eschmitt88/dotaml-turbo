---
kind: concept
name: "hero-embedding-vs-onehot"
status: seedling
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaML.md
related_concepts:
  - draft-only-win-prediction
  - draft-prediction-plateau
related_experiments: []
tags: [feature-engineering, representation, dota2]
---

# hero-embedding-vs-onehot

## Definition

Two feature representations for a Dota draft tried in the prior-art DotaML
work:

- **One-hot, 300-dim.** Two binary indicators per hero (Radiant slot and
  Dire slot) over 150 hero IDs. Order-invariant by construction. Used by
  v1-v5 (LightGBM through ResidualFFN). The `max_hero_id` parameter must
  cover the full ID range: a v2 bug fixed at `=130` silently dropped 20
  heroes and biased combo rankings.
- **Learned 64-dim embeddings.** Shared per-hero embedding (same vector
  whether the hero is on Radiant or Dire) plus a learned Radiant-vs-Dire
  position embedding. Used by v6 Transformer over the 10-hero sequence,
  with 30% random `[MASK]` tokens during training to enable
  incomplete-draft scoring at inference time.

## Why it matters here

Both representations land at the same prediction plateau (~59.9% acc /
~0.635 AUC), suggesting representation is not the bottleneck for this
task — at least not at the volumes prior art tested (≤10M matches).

Implications for new experiments:

- **One-hot remains the default for the first replication on patch-7.40
  data** because it pairs cleanly with LightGBM and is cheap to debug.
- **Embeddings unlock two downstream capabilities:** incomplete-draft
  scoring (real-time use, out of scope here but relevant to the sibling
  `dotaml-serve` repo), and explicit cross-team attention as a search
  direction for breaking the plateau.
- **Watch for representation-induced confounds.** The v2→v3 jump in
  DotaML showed that bugs in coverage (max_hero_id) can shift the
  apparent meta dramatically without changing the test accuracy by
  more than 0.1pp — i.e. accuracy is insensitive but downstream
  combo-ranking is not. Whatever the new project uses for combo
  analysis must be robust to this.

## Connections

- [[draft-only-win-prediction]] — the task both representations target.
- [[draft-prediction-plateau]] — both representations hit it.
- Possible extensions: (a) hero embeddings pretrained on item-build /
  ability-build co-occurrence; (b) graph embeddings where hero-pair
  edge weights come from historical synergy; (c) lane-role inference
  feeding a structured position embedding.
