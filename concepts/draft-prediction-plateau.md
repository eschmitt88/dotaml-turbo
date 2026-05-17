---
kind: concept
name: "draft-prediction-plateau"
status: growing
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaML.md
  - literature/papers/hodge2017win.md
  - experiments/2026-05-15-plateau-baseline-740/README.md
  - experiments/2026-05-15-plateau-architectures-740/README.md
  - experiments/2026-05-16-transformer-hp-sweep-740/README.md
related_concepts:
  - draft-only-win-prediction
  - hero-embedding-vs-onehot
related_experiments:
  - 2026-05-15-plateau-baseline-740
  - 2026-05-15-plateau-architectures-740
  - 2026-05-16-transformer-hp-sweep-740
tags: [empirical-finding, capacity-vs-accuracy, hp-robust]
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

**Refinement (2026-05-15, second experiment):** the architecture-spread
is **family-driven** (Transformer beats FFN), but the **within-FFN
ordering does not reproduce** on patch-7.40 under HCE. See
[[2026-05-15-plateau-architectures-740]]. Three-architecture sweep:

- LightGBM (one-hot, prior baseline): 0.6161
- SimpleFFN (52k params, 64-dim embeds): 0.6217 (+0.006)
- ResidualFFN (225k params, 64-dim embeds): 0.6199 (+0.004) — **lower than SimpleFFN**, inverting prior art
- Transformer (82k params, attention over 11 tokens): 0.6322 (+0.016) — within 0.003 of v6's 0.6354

The Transformer-vs-FFN gap is large and reliable (≥0.011 AUC, regardless
of which FFN you pick); the FFN-internal gap inverted, suggesting
either hyperparameter sensitivity (v5's recipe was tuned on a smaller
pre-7.40 set), an artifact of switching v4 from one-hot to embeddings,
or single-seed noise. A multi-seed FFN sweep is queued to disambiguate.

**Refinement (2026-05-16, third experiment): the Transformer ceiling is
HP-robust.** A 60-trial Optuna TPE+ASHA sweep over a minimal-Transformer
baseline (9-dim search: d_model, n_heads, n_layers, ff_mult, embed_dim,
lr, weight_decay, dropout, batch_size) found best val_auc=0.6318 — within
0.001 of the un-tuned prior Transformer (0.6322). All 5 trials that ran
to convergence cluster in val_auc ∈ [0.6311, 0.6319] (0.0008 spread).
ASHA-pruned trials (55 of 60) had best ep-3 val_auc ≤ 0.6310. See
[[2026-05-16-transformer-hp-sweep-740]]. The ~0.632 Transformer ceiling
is therefore *not* an under-tuned point in a broader HP landscape — it
is a property of (architecture vocabulary × data) on this snapshot.

Implication for ceiling-breaking: further HP tuning is exhausted as a
lever. The remaining levers are structural mutation of the model (LLM-
driven program search à la [[concepts/evolutionary-expansion]]) or
new data features (draft order, lane assignment, hero-pair history,
player MMR). Anything that beats ≈ 0.632 by ≥ 0.005 must originate
from one of those, not from HP search.

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

After 2026-05-16: HP-search has been ruled out as a lever for the
Transformer architecture vocabulary on this snapshot. Any val_auc > 0.640
result must come from structural mutation, new features, or new data,
not from re-tuning existing architectures.

**Independent attestation (2026-05-17):** Hodge et al. 2017
([[literature/papers/hodge2017win]]) report hero-only Dota 2 win
prediction accuracy of 55-59% across LR and RF on mixed-rank data —
matching our `plateau-baseline-740` val_acc=0.5866 (val_auc=0.6161)
within 0.01. The same paper reports that adding in-game telemetry
(team kills, damage, gold, net worth) lifts accuracy to 75-76% — a
~17 pp gap that demonstrates the broader prediction task has
substantial headroom once feature sets richer than hero-IDs are
admitted. This is independent confirmation that the ~0.62 ceiling is
an information bottleneck, not a model bottleneck, and motivates
extending pre-game features (player identity, draft order) before
investing in further architectural sophistication.

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
