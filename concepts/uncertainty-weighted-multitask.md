---
kind: concept
name: "uncertainty-weighted-multitask"
status: seeded
added: "2026-05-22"
sources:
  - literature/papers/kirchdorfer2024analytical.md
  - literature/papers/jiang2023forkmerge.md
related_concepts:
  - tabular-foundation-model
related_experiments: []
tags: [multi-task-learning, loss-weighting, uw, uw-so, forkmerge, negative-transfer]
---

# uncertainty-weighted-multitask

## Definition

A family of methods for balancing multiple task losses in joint
training without hand-tuned per-task α weights. Kendall et al.'s
original Uncertainty Weighting (UW) learns per-task homoscedastic
uncertainty `σ_k` and weights losses as `Σ_k (1/σ_k²) L_k + log σ_k`.
**UW-SO** (Kirchdorfer 2024) closes the form analytically: at the
optimum `σ_k = L_k`, so the right weighting is the inverse loss,
passed through a softmax with tunable temperature `T`. **ForkMerge**
(Jiang 2023) is a complementary mechanism that operates at the
*parameter-update* level rather than the loss level: periodically
fork the model into target-only and target+aux branches, train
independently for `Δt` steps, then search a scalar `λ` on the target
validation set to interpolate `θ* = (1−λ)θ_tgt + λθ_joint`. UW-SO
balances within a fixed task set; ForkMerge filters out a bad
auxiliary's updates entirely.

## Why it matters here

Our [[2026-05-20-rich-supervision-multitask-740]] run shipped with
hand-tuned α (α_w=1.0, α_d=0.15, α_i=0.3, α_a=0.1) after a fragile
α_d=0.5 → 0.15 bug-fix. Adding more heads to `foundation-mvp-740`
(talents from `ability_upgrades[]`, first-blood time, tower state,
etc.) multiplies the α-tuning surface and the negative-transfer risk.
UW-SO replaces the α grid with one temperature; ForkMerge adds a
safety net that bounds the damage of a head that conflicts with the
win head. Together they make multi-head training robust enough to
add new heads without re-tuning.

## Connections

- [[tabular-foundation-model]] — the architecture this loss machinery
  supports.
- [[draft-prediction-plateau]] — the multitask result (val_auc 0.6495)
  that motivated extending the head set in the first place.
