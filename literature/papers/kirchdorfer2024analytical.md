---
kind: paper
title: "Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning"
authors:
  - Lukas Kirchdorfer
  - Cathrin Elich
  - Simon Kutsche
  - Heiner Stuckenschmidt
  - Lukas Schott
  - Jan M. Köhler
year: 2024
venue: "arXiv:2408.07985 [cs.LG]"
url: "https://arxiv.org/abs/2408.07985"
source: "raw/papers/kirchdorfer2024analytical.pdf"
added: "2026-05-22"
relevance: 5
status: skimmed
related_experiments: []
related_concepts:
  - uncertainty-weighted-multitask
tags: [multi-task-learning, loss-weighting, uncertainty-weighting, softmax-temperature, scalarization]
---

# Analytical Uncertainty-Based Loss Weighting in MTL (UW-SO)

## TL;DR

The paper diagnoses two failure modes of Kendall et al.'s Uncertainty
Weighting (UW) — initialization-sensitive "update inertia" and
overfitting — and replaces UW's learned `σ_k` with the analytically
optimal value `σ_k = L_k`, then runs the result through a softmax with
tunable temperature `T`. The resulting **UW-SO** loss is
`L = Σ_k softmax(1/sg[L_k] / T)_k · L_k`, which (a) matches the
brute-force "Scalarization" grid-search ceiling at one HP instead of
exponentially-many, (b) consistently beats six other dynamic weighting
methods (UW, DWA, IMTL-L, RLW, GLS, EW) across multi-task vision
benchmarks, and (c) reveals two practitioner findings: larger networks
flatten the gap between weighting methods, and per-method LR tuning
matters far more than weight decay.

## Claims

- **UW-SO beats UW, DWA, IMTL-L, RLW, GLS, EW, and matches
  Scalarization** on NYUv2, CityScapes, and CelebA across SegNet,
  ResNet, and Swin backbones (Section 4.2, Table 1). The single HP `T`
  is tuned with a small grid (5-7 values).
- **UW suffers from "update inertia"** — bad σ initialization can take
  ~1/4 of training to recover, regardless of the actual loss
  magnitudes (Figure 1).
- **The analytically optimal UW weight for L1 losses is `σ_k = L_k`**,
  derived in closed form (Eq. 2). After stop-gradient and the log term
  drops out, the loss reduces to `Σ_k (1/L_k) · L_k`. (For L2 and
  Cross-Entropy losses, analogous derivations in Appendix A1 give the
  same inverse-loss form with different prefactors.)
- **Softmax-with-temperature is the missing ingredient.** Pure
  "inverse loss" weighting (UW-O) is unstable for small `L_k`; the
  softmax bounds the weights, and `T` lets the practitioner pick
  anywhere on the spectrum from equal-weighting (`T→∞`) to
  one-task-dominates (`T→0`). UW-SO with tuned `T` matches
  Scalarization (Section 4.2).
- **Architecture size dampens the weighting-method gap.** On
  larger backbones, all weighting methods perform comparably; on
  smaller models (SegNet), the gap between EW and UW-SO is much wider
  (Section 4.3) — implying weighting matters most when you're
  parameter-constrained.
- **LR tuning per-weighting-method is essential.** Many of the
  contradictions in prior MTL literature trace back to one global LR
  applied across methods. UW-SO retains its advantage only when each
  method's LR is independently tuned (Section 4 caveat).

## Methods

For `K` tasks with hard-shared backbone and task-specific heads
producing losses `{L_k}`, UW-SO computes
```
ω_k = exp(1/sg[L_k] / T) / Σ_j exp(1/sg[L_j] / T)
L_total = Σ_k ω_k · L_k
```
where `sg[·]` is stop-gradient (so `ω_k` is a constant scalar per
batch, not part of the autograd graph). `T` is the only new HP. UW-SO
needs no per-task `σ_k` parameters and no learned regularizer term.
Training is otherwise standard hard-sharing MTL: shared encoder, K
heads, sum of weighted losses.

## Takeaways for foundation-mvp-740

- **Use UW-SO instead of hand-tuned α weights for the multi-task
  losses.** Our current
  [[2026-05-20-rich-supervision-multitask-740]] uses fixed
  α_w=1.0, α_d=0.15, α_i=0.3, α_a=0.1 found by trial-and-error. UW-SO
  replaces all four with one temperature `T` and adapts per-batch to
  whichever head's loss is currently dominating. The α_d=0.5 → 0.15
  bug-fix in the multitask experiment is exactly the kind of fragile
  manual tuning UW-SO eliminates.
- **Use `sg[L_k]` (stop-gradient on the weight computation).** Don't
  let the weight itself participate in the autograd graph — that was
  one of UW's failure modes per the inertia analysis. Cheap to
  implement, easy to get wrong if not explicit.
- **Tune `T` with a 5-7-point grid on the val_auc head specifically.**
  Since the win head is the target task and the others (duration, item,
  KDA) are auxiliaries, we should pick `T` that maximizes val_auc on
  the win head, not joint-loss. This composes with the ForkMerge
  insurance from [[jiang2023forkmerge]] if we want a stronger
  target-task-focused selection.
- **Don't expect UW-SO to lift val_auc much by itself** — the
  multi-task hp-fragility was real (α_d=0.5 dominated and α_d=0.15
  worked), but the current 0.6495 result already represents a
  near-optimal manual α-tune. UW-SO is more "insurance against
  α-drift when the head set changes" than a direct lift, and it's the
  right default for `foundation-mvp-740` which will have more heads
  than the multitask-740 experiment.
- **Re-tune LR alongside UW-SO.** Per the paper's main practitioner
  finding, the weighting method and LR interact; don't reuse the
  multitask-740 LR uncritically when adding UW-SO.

## Open questions / caveats

- The benchmark losses (segmentation cross-entropy, depth L1, surface
  normals cosine) are all dense pixel/voxel losses with comparable
  scale. Our losses are very different in scale — binary CE on the win
  head (~0.69), CE over 8 buckets on duration (~2.0), multi-label BCE
  over 305 items per slot (~varies), SmoothL1 on aux regression
  (~varies). UW-SO with sufficiently large `T` will damp this; small
  `T` could collapse weight onto one task. Worth a short ablation on
  `T ∈ {0.5, 1, 2, 5, 10}`.
- The paper does NOT address auxiliary-task selection (which head to
  include vs exclude). That's [[jiang2023forkmerge]]'s ForkMerge
  territory. The two compose: UW-SO weights *within* a fixed task set;
  ForkMerge decides whether to merge in the auxiliary's parameter
  updates at all.
- The "stop-gradient + softmax" trick is also what [[wang2025player]]'s
  HIGFormer uses for its MoE gating — same recipe, different
  motivation; flag for cross-reference.
