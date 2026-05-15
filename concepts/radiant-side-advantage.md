---
kind: concept
name: "radiant-side-advantage"
status: seedling
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaML.md
related_concepts:
  - draft-only-win-prediction
related_experiments: []
tags: [empirical-finding, side-asymmetry, dota2]
---

# radiant-side-advantage

## Definition

The empirical fact that, in Dota 2 Turbo, the Radiant team wins a
materially higher share of matches than the Dire team — reported as
+5 to +7 percentage points across the prior-art DotaML model
generations (v2-v4) regardless of model architecture.

## Why it matters here

The plain base rate of "Radiant wins" is therefore not 50% but ~55%.
This is consequential for:

- **Baseline construction.** "Always predict Radiant wins" is a
  non-trivial baseline at ~55% accuracy. A model must beat this, not
  50%, to claim signal from the draft.
- **Evaluation metrics.** Accuracy and log-loss both inherit the
  asymmetry; AUC is unaffected. Report all three.
- **Feature design.** The model must know which team is on which side.
  Hero composition alone is not enough; the side label is a load-bearing
  input.
- **Symmetry constraints.** Any architecture that artificially symmetrizes
  Radiant and Dire (e.g. sums embeddings across teams without a side
  feature) will leave this signal on the table.

The new project should verify the magnitude on the patch-7.40 snapshot
before relying on the prior-art number — meta and map changes can shift it.

## Connections

- [[draft-only-win-prediction]] — the task this asymmetry sits inside.
- Possible cause: map geometry (river camps, Roshan pit access, day-night
  bottom-rune timing). Confirmation is out of scope; mitigation via the
  side feature is in scope.
