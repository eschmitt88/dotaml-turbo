---
kind: paper
title: "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts"
authors:
  - Xu Liu
  - Juncheng Liu
  - Gerald Woo
  - Taha Aksu
  - Yuxuan Liang
  - Roger Zimmermann
  - Chenghao Liu
  - Silvio Savarese
  - Caiming Xiong
  - Doyen Sahoo
year: 2024
venue: "arXiv:2410.10469 (Salesforce AI Research preprint)"
url: "https://arxiv.org/abs/2410.10469"
source: "raw/papers/liu2024moirai.pdf"
added: "2026-05-22"
relevance: 4
status: skimmed
related_experiments: []
related_concepts:
  - tabular-foundation-model
  - multi-query-foundation-model
tags: [time-series, foundation-model, mixture-of-experts, token-level-routing, salesforce, moirai, frequency-specialization]
---

# Moirai-MoE: Token-Level Mixture of Experts for Time-Series Foundation Models

## TL;DR

Moirai-MoE replaces Moirai's hand-defined frequency-level projection
heads (separate input/output layers per frequency cohort: monthly /
daily / hourly / ...) with a *single* shared projection layer plus a
sparse Mixture-of-Experts block inside each Transformer layer.
Specialization is moved from the hand-imposed frequency axis to a
**data-driven, token-level** axis, on the argument that "frequency
is not a reliable indicator of pattern" (similar patterns can recur
across frequencies; one frequency can contain many patterns). The
authors further propose a gating function that initializes routing
from k-means clusters of a pretrained dense model's token embeddings,
beating randomly-initialized linear gates across all expert counts.
Result: 17% improvement over Moirai at matched activated params,
beating TimesFM and Chronos with up to 65× fewer activated params.

## Claims

- **Token-level MoE beats frequency-level dense projections** by 17%
  MAE on Monash benchmark (29 datasets) at matched activated params;
  beats Chronos and TimesFM zero-shot on 10 held-out datasets
  (Section 4.2, Figure 3, Table 2). Aggregated zero-shot CRPS:
  Moirai-MoE_B 0.478 vs Moirai_L 0.514 vs Chronos_L 0.500.
- **Switching from masked-encoder to decoder-only objective alone
  contributes ~3-4% of the lift; the bulk (~13-14%) comes from MoE
  specialization** (Table 3 ablation: multi-projection w/ masked
  encoder 0.78, multi-projection w/ decoder-only 0.75, single
  projection + MoE w/ decoder-only 0.65).
- **Routing from k-means of pretrained token embeddings beats
  random-initialized linear gating** across all tested expert counts
  (Figure 4, right) — the cluster centroids better reflect data
  distribution than random projections do.
- **Routing is frequency-aware in shallow layers and
  frequency-invariant in deep layers.** Layers 1-2 show distinct
  expert assignments per frequency cohort; layer 6 (final) shows
  near-identical distributions across all frequencies, with only
  3/32 experts used (Figure 6). Authors call this "progressive
  denoising": shallow layers handle short-term variability per
  frequency, deep layers learn shared abstractions.
- **Per-patch separate projection layers are explicitly the wrong
  middle-ground.** The paper closes the door on hand-grouping inputs
  by *any* meta-feature (frequency, modality, patch-position): if
  specialization is the goal, do it token-level via MoE, not
  group-level via separate projections.

## Methods

Moirai-MoE inherits Moirai's patching (size P=16) and decoder-only
flattened-multivariate construction. The key change is the FFN: each
Transformer layer's FFN is replaced with `M=32` expert FFNs and a
gating function that selects K=2 experts per token. Load-balance loss
encourages uniform expert utilization. Gating variants tested: (a)
random-init linear projection (standard Switch Transformer style),
(b) random-init + load-balance loss, (c) **k-means cluster centroids
from a pretrained Moirai model's attention outputs**, with token-
expert affinity scored by Euclidean distance to centroids. Trained
on LOTSA corpus, 11M/86M activated params (117M/935M total) for
small/base. Single shared input projection layer (residual MLP).
Inference is autoregressive over patches.

## Takeaways for foundation-mvp-740

- **Do NOT design separate per-patch projection layers for our 5
  downstream queries (win / duration / items / lineup-eval /
  fun-pair).** Moirai-MoE is the explicit "we tried this, it's
  inferior to one shared projection + per-token specialization"
  evidence. The clean MVP design is: ONE input projection per token
  type (hero-slot, player-feature-slot), shared across all tasks; no
  per-task input layers.
- **Specialization belongs at the head or in the deep layers, not at
  the input projection.** Moirai-MoE's layer-6 finding (deep layers
  converge to shared experts; shallow layers diversify) matches the
  pattern we already see empirically in
  [[2026-05-20-rich-supervision-multitask-740]]: shared encoder +
  per-task linear heads worked, no per-task input layers were
  needed for the +0.0018 lift. Confirms the architectural decision.
- **MoE itself is probably premature for our MVP scale.** Moirai-MoE
  needs M=32 experts × 12 layers to outperform a dense 91M baseline.
  Our 77K-5M dense models are nowhere near the saturation point
  where MoE pays off (their 11M-activated MoE beats their 14M dense
  by 17%, suggesting MoE is most useful as a *capacity multiplier*
  at the trained-too-long boundary, which we are far from). Defer
  MoE to post-MVP if the dense encoder saturates.
- **Apply the "shallow-specializes, deep-shares" intuition as a
  diagnostic.** Once we have multi-task training going, log per-task
  per-layer attention/activation entropy: if shallow layers
  diverge between tasks and deep layers converge, that's the
  expected pattern. If shallow layers stay shared, we may be
  under-specializing. Cheap to instrument.
- **The k-means-init-from-pretrained-embeddings gating idea is
  reusable.** If we ever add MoE or any other routing/gating to our
  architecture (e.g. a per-player learned routing among player-tier
  cohorts), initialize the routing from k-means of an existing
  trained model's representations rather than from scratch — same
  ~2× sample-efficiency gain pattern Moirai-MoE shows.

## Open questions / caveats

- Time-series forecasting differs from our setting: their tokens are
  time-indexed patches with strong autoregressive structure, ours
  are unordered draft slots with permutation structure (within team).
  The "MoE beats group-projection" finding is general enough to
  transfer, but the specific architectural details (decoder-only,
  causal attention, autoregressive inference) do not.
- The 17% lift is over Moirai with the same activated params — but
  Moirai-MoE has 10× the *total* params. For a foundation model
  paying memory cost per total params (not just activated), this is
  a real cost. For our SN850X-resident inference model where memory
  is essentially free at 935M params, this would be irrelevant; for
  on-GPU training memory it matters.
- The paper does not test "per-task" specialization (e.g., one
  expert per downstream task) directly; their experts specialize on
  pattern-clusters within the same task (forecasting). For our
  multi-query foundation model, per-task heads may be a different
  axis from per-token MoE — both could coexist or one could replace
  the other. Their work doesn't tell us which.
- The k-means-from-pretrained gating requires having a pretrained
  dense model first. Bootstrap cost: train dense first, cluster,
  then init MoE. Worth it only if MoE is on the roadmap.
