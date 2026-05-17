---
kind: concept
name: "draft-only-win-prediction"
status: growing
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaML.md
  - literature/papers/hodge2017win.md
related_concepts:
  - draft-prediction-plateau
  - radiant-side-advantage
  - hero-embedding-vs-onehot
related_experiments: []
tags: [task-definition, dota2, win-prediction]
---

# draft-only-win-prediction

## Definition

The supervised binary classification task of predicting which team wins a
Dota 2 match given only the 10 hero IDs (5 per side) and the side
assignment, with no in-game telemetry, no player identity, and no
post-draft information.

## Why it matters here

This is the single optimization target for `dotaml-turbo`. Every
experiment in this repo is judged on calibrated probability of
Radiant win on the patch-7.40 snapshot (2025-12-16 → 2026-03-23,
~19.6M Turbo matches), and on nothing else. Other tasks — item
recommendation, ability draft, live-game inference — are explicitly
out of scope (they live in sibling repos `dotaml-items` and
`dotaml-serve`).

## Connections

- [[draft-prediction-plateau]] — the empirical ceiling observed for this
  task in prior art (~60% test accuracy / ~0.635 AUC), independent of
  architecture.
- [[radiant-side-advantage]] — a confound the model must either absorb
  through the side feature or be evaluated against.
- [[hero-embedding-vs-onehot]] — the two feature representations the
  prior art exhausted at the plateau.
- [[fake-match-filtering]] — a label-noise source that has to be removed
  before this task is meaningfully evaluated.
