---
kind: concept
name: "draft-prediction-plateau"
status: seedling
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaML.md
related_concepts:
  - draft-only-win-prediction
  - hero-embedding-vs-onehot
related_experiments: []
tags: [empirical-finding, capacity-vs-accuracy]
---

# draft-prediction-plateau

## Definition

The observation that, on Dota 2 Turbo draft-only win prediction, test
accuracy saturates around 59.9% and test AUC around 0.635 — independent
of model capacity or architecture family — once the dataset is large
enough (millions of matches) and basic representation issues are fixed.

## Why it matters here

In the prior-art DotaML repo, six successive model generations spanning
LightGBM, SimpleFFN (47k params), ResidualFFN (228k params), and a
Transformer with learned hero embeddings + masked-input training
(152k params) all land within ~0.04 AUC and ~1pp accuracy of each other.
The v5 README explicitly states "we may be approaching fundamental limits
of hero draft prediction."

For `dotaml-turbo`, this number is the load-bearing baseline. Any new
experiment should be evaluated against three implied tests:

1. **Sanity:** does it match the plateau on the new patch-7.40 snapshot?
2. **Patch effect:** does the plateau itself shift on a larger, more
   recent dataset?
3. **Ceiling:** does any new technique meaningfully exceed it?

A result inside ±0.01 AUC of 0.635 should be reported as "at the
plateau," not as a successful new model.

## Connections

- [[draft-only-win-prediction]] — the task whose ceiling this names.
- [[hero-embedding-vs-onehot]] — both representations hit the same
  ceiling in the prior art, suggesting representation is not the
  bottleneck.
- Hypotheses for the source of the ceiling (to be tested):
  hero-only information genuinely under-determines the outcome;
  label noise from fake matches / queue dodges; player-skill
  variance that the model cannot see; patch instability across the
  training window; calibration vs accuracy trade-off.
