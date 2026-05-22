---
kind: paper
title: "ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning"
authors:
  - Junguang Jiang
  - Baixu Chen
  - Junwei Pan
  - Ximei Wang
  - Dapeng Liu
  - Jie Jiang
  - Mingsheng Long
year: 2023
venue: "NeurIPS 2023 (arXiv:2301.12618)"
url: "https://arxiv.org/abs/2301.12618"
source: "raw/papers/jiang2023forkmerge.pdf"
added: "2026-05-22"
relevance: 4
status: skimmed
related_experiments: []
related_concepts:
  - uncertainty-weighted-multitask
tags: [multi-task-learning, auxiliary-task-learning, negative-transfer, parameter-averaging, validation-search, neurips2023]
---

# ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning

## TL;DR

ForkMerge analyzes negative transfer (NT) in auxiliary-task learning
and reports two surprising findings: (1) **gradient conflicts do NOT
cause NT** — even L2 weight decay conflicts with the target gradient
yet helps, while many auxiliary tasks help despite gradient conflict;
(2) **NT is best predicted by distribution shift** between the joint
train distribution and the target test distribution. Their remedy is
to periodically fork the model into two branches — one trained on
target-only, one trained on target+auxiliary — train both for `Δt`
steps, then **search a scalar `λ ∈ [0,1]` on the target validation
set** for parameter interpolation `θ* = (1−λ)θ_target + λθ_joint`, and
merge both branches to `θ*` before the next interval. This dynamically
filters out the auxiliary's harmful updates and keeps the helpful ones,
all without modifying gradients during forward training. Beats prior
NT mitigation methods (gradient surgery, gradient cosine, MGDA, etc.)
on multiple ATL benchmarks.

## Claims

- **Finding 1: gradient conflicts do not predict negative transfer.**
  Repeated experiments with auxiliary task replaced by L2 regularization
  (which always conflicts with the target gradient) show that NT
  correlates poorly with gradient cosine similarity. The previous
  literature's gradient-coordination focus is misdirected (Section 3.1,
  Figure 2).
- **Finding 2: distribution shift between joint-train and target-test
  data DOES predict NT.** When auxiliary data pulls the training
  distribution toward the test distribution, transfer is positive; when
  it pulls away, transfer is negative (Section 3.2, Figure 3). This
  reframes ATL as a generalization problem, not an optimization
  problem.
- **Weak vs. Strong Negative Transfer matters.** Weak NT
  (`TG(λ, A) < 0` for some `λ` but `max_λ TG > 0`) can be solved by
  finding the right `λ`. Strong NT (`max_λ TG < 0`) cannot — the
  auxiliary is just bad. The taxonomy itself is a useful contribution
  (Section 3, Definitions 3.2-3.3).
- **ForkMerge outperforms prior NT-mitigation methods** on a series of
  ATL benchmarks (DomainNet, CelebA, NYUv2, OpenImages,
  recommendation-CTR), beating MGDA, PCGrad, GradVac, AANG, etc.
  (Section 5). Especially effective on the strong-NT regime where
  gradient-surgery methods fail.
- **ForkMerge composes with task-weighting**: the `λ` search at merge
  time is orthogonal to per-task `α` weights inside the joint loss, so
  it stacks on top of UW / UW-SO / scalarization.

## Methods

Algorithm 1 (ForkMerge):
1. Initialize parameters `θ_0`, fork into `θ^0` (target-only) and
   `θ^1` (joint).
2. For `Δt` steps, train `θ^0` with `L_tgt` only and `θ^1` with
   `L_tgt + L_aux`.
3. Search `λ* = arg max_λ P_val((1−λ)θ^0_{t+Δt} + λ θ^1_{t+Δt})` on
   the target validation set.
4. Merge: `θ*_{t+Δt} = (1−λ*)θ^0_{t+Δt} + λ* θ^1_{t+Δt}`.
5. Synchronize both branches to `θ*_{t+Δt}`, repeat.

Cost: roughly 2× the training compute (two branches), with one
periodic validation pass to search `λ`. The paper proposes a
generalized version with N>2 branches for multiple auxiliaries; the
N-way version is much more expensive and is mostly used to *select*
which auxiliaries to keep.

## Takeaways for foundation-mvp-740

- **Use ForkMerge (or a lightweight variant) as insurance against
  negative transfer from the auxiliary heads.** Our multi-task setup
  in [[2026-05-20-rich-supervision-multitask-740]] worked (+0.0018 on
  the win head with multi-task supervision), but the proposal-target
  expansion to more heads (talents from `ability_upgrades[]`,
  first-blood time, tower state, etc.) increases the risk that some
  new head HURTS the win head. ForkMerge gives us a principled,
  parameter-averaging way to *bound* the damage of a bad auxiliary
  without removing the head entirely.
- **Concrete adoption for foundation-mvp-740**: train the joint
  foundation model (target = win, aux = duration + items + KDA + new
  heads) for `Δt = 1-2 epochs` at a time; keep a parallel target-only
  branch (same architecture, same data, just `L_win`); at each
  interval do a small `λ ∈ {0, 0.25, 0.5, 0.75, 1}` grid search on
  validation AUC; merge with the best `λ` and continue. The 2× compute
  cost is acceptable given a ~4h baseline run.
- **Frame our existing diagnostic as a negative-transfer analysis.**
  The "did the win head get better or worse vs the same-data baseline"
  check in multitask-740 is exactly the Transfer Gain definition
  (Eq. 2). Make this explicit in the foundation-mvp-740 design so
  we're measuring TG, not just absolute val_auc, when comparing
  multi-head vs single-head ablations.
- **Don't waste time on gradient-surgery methods (PCGrad, GradVac,
  CAGrad).** ForkMerge is the definitive demonstration that these
  approaches misdiagnose the problem. UW-SO from
  [[kirchdorfer2024analytical]] addresses per-batch loss balancing;
  ForkMerge addresses per-interval parameter-update filtering. The two
  compose; no need to add gradient-surgery to the stack.
- **For "head selection" (do we add the talents head or not?), use
  ForkMerge with three branches** (target-only, target+aux-set-A,
  target+aux-set-B) and let the validation `λ` answer it
  automatically, rather than running the N×{add/remove} ablation grid.

## Open questions / caveats

- ForkMerge requires a clean validation set that's not used for
  hyperparameter selection in the rest of the pipeline. We have
  exactly this (per HCE: `val_auc` is the search signal,
  `test_auc` is held out for the final scoring pass). Make sure the
  `λ` search uses the validation set, not the test set, to preserve
  the HCE separation from `~/.claude/rules/evaluation.md`.
- Parameter averaging only works if both branches stay in the same
  loss basin between syncs. The paper recommends `Δt` "small enough
  that the two branches haven't diverged into different basins" —
  typically 1-2 epochs in their experiments. For a 30-epoch run like
  multitask-740, that's 15-30 sync points, which is cheap.
- The N-branch generalization (for picking *which* auxiliaries to
  include) is expensive — `O(N)` branches each step. For
  foundation-mvp-740 with maybe 5 candidate auxiliary heads, the
  2-branch "include all aux vs target-only" is cheap; the N-branch
  "include each individually" is `O(5×)` compute and probably overkill
  for this proposal. Start with the 2-branch version.
- The paper's experimental setting is small/medium-scale supervised
  vision/NLP/CTR; it does NOT include foundation-model pre-training
  scenarios. The composition with our MAE + contrastive pre-training
  stage is untested in the literature — but architecturally, ForkMerge
  operates entirely at fine-tuning time, so the pre-training stage
  doesn't interact with it directly.
