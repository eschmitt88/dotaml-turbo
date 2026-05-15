---
kind: concept
name: "draft-prediction-plateau"
status: growing
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaML.md
  - experiments/2026-05-15-plateau-baseline-740/README.md
related_concepts:
  - draft-only-win-prediction
  - hero-embedding-vs-onehot
related_experiments:
  - 2026-05-15-plateau-baseline-740
tags: [empirical-finding, capacity-vs-accuracy]
---

# draft-prediction-plateau

## Definition

The observation that, on Dota 2 Turbo draft-only win prediction, test
accuracy and test AUC saturate within a narrow band — once the dataset
is large enough (millions of matches) and basic representation issues
are fixed. Empirically, the band is ≈ 0.619-0.635 AUC across the prior
art's six architectures, NOT a single 0.635 number for all of them.

**Refinement (2026-05-15):** the 0.635 figure is the v5 Transformer's
ceiling specifically. The v3 LightGBM ceiling sits lower at ≈ 0.619.
Replication on the patch-7.40 snapshot under HCE confirmed the v3
LightGBM number to within 0.003 (val_auc 0.6161 vs prior test_auc
0.6189 — see [[2026-05-15-plateau-baseline-740]]). The
**architecture-spread within the plateau** is itself a thing to model:
~0.016 AUC of headroom between the LightGBM and the Transformer
families on the prior art, not noise.

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

A result inside ±0.01 AUC of the **architecture-matched** prior-art
ceiling (LightGBM ≈ 0.619, Transformer ≈ 0.635) should be reported as
"at the plateau for that architecture," not as a successful new model.
A result that exceeds the upper end of the architecture-spread (i.e.
val_auc > ~0.645) is the genuinely interesting case.

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
