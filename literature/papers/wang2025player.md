---
kind: paper
title: "Player-Team Heterogeneous Interaction Graph Transformer for Soccer Outcome Prediction"
authors:
  - Lintao Wang
  - Shiwen Xu
  - Michael Horton
  - Joachim Gudmundsson
  - Zhiyong Wang
year: 2025
venue: "KDD 2025 (arXiv:2507.10626)"
url: "https://arxiv.org/abs/2507.10626"
source: "raw/papers/wang2025player.pdf"
added: "2026-05-22"
relevance: 4
status: skimmed
related_experiments: []
related_concepts:
  - tabular-foundation-model
  - draft-only-win-prediction
tags: [sports, soccer, outcome-prediction, graph-transformer, player-features, team-features, heterogeneous-graph, kdd2025]
---

# HIGFormer: Player-Team Heterogeneous Interaction Graph Transformer

## TL;DR

HIGFormer is a two-stream pre-match outcome predictor for soccer that
separately encodes (a) per-player histories via a **heterogeneous
interaction graph** (players as nodes, in-match events — pass, shot,
foul, save, etc. — as typed edges), processed by a hybrid local-GCN +
global graph-augmented Transformer with a MoE gate; and (b) per-team
histories via a much simpler win/loss "team interaction graph"
processed with a small GCN. A final **Match Comparison Transformer**
fuses team and player embeddings to predict {win, draw, lose} on the
WyScout dataset. The paper argues — and shows ablations supporting —
that heterogeneous event types matter (collapsing them to a single
edge type degrades performance) and that combining player-level and
team-level streams beats either alone.

## Claims

- **HIGFormer outperforms prior soccer outcome predictors** (MLP,
  RNN, Hubáček feature-based, plain GNN, plain Transformer, recent
  online-game predictors HOI / OptMatch / NeuralAC) on the WyScout
  Open Access Dataset (Section 5). Numbers in main table; the
  improvement is meaningful and consistent across seasons.
- **Two-stream architecture (player + team) beats either single
  stream.** Ablation removes the team stream and the player stream
  independently; both ablations under-perform the full model
  (Section 5 ablations).
- **Heterogeneous edge types matter.** Treating all event types
  (pass / shot / foul / save / etc.) as a single edge class
  underperforms keeping them distinct (one of the headline ablations).
  This is the paper's main architectural argument: event-type
  heterogeneity is a load-bearing inductive bias.
- **Combining a local GCN (for short-range message-passing on the
  player graph) with a global graph-augmented Transformer (for
  long-range attention across all players) beats either alone**, and
  a small MoE gate adaptively balances the two streams per-input.
- **Pre-match prediction with player-history + team-history features
  can match much of the in-match-feature ceiling** — a claim
  philosophically aligned with [[hodge2017win]]'s in-game-features-
  ceiling-at-75-76% finding, but achieved here with strictly pre-match
  data.

## Methods

For match `i` with two teams of 23 players, each player `p_n^i` has a
history `H_{p_n^i} = [h_1, ..., h_{i-1}]` of past-match records, each
record being event counts `c` (10 event types: duel, foul, free-kick,
GK-leaving-line, interruption, offside, others-on-ball, pass, save,
shot) plus the player's match outcome `y`. Per past match, a
heterogeneous interaction graph is built (player-player edges typed by
event); a stack of `(local-GCN, global-graph-augmented-Transformer,
MoE-gate)` blocks produces player embeddings. Team embeddings come
from a win-rate-graph GCN. The Match Comparison Transformer concatenates
the team and player embeddings of the upcoming fixture's two teams and
runs a small Transformer over the joint set. Two-stage training:
pre-train the player stream and team stream separately, then end-to-
end fine-tune.

## Takeaways for foundation-mvp-740

- **Closest published analogue to our setting.** Sports outcome
  prediction with per-player histories, per-team histories, and a
  Transformer-style fusion head is exactly the shape of our problem;
  HIGFormer is the closest neighbor in published architecture-space.
  Read this paper as the "what does the rest of the world do for
  this" reference and cite it explicitly in the proposal.
- **Two-stream design is worth replicating.** Our current
  `transformer-plus-features-740` already kind of does this (the
  per-slot features are the player stream, the hero attention is the
  team-structure stream), but they're fused at the token level. For
  foundation-mvp-740 we should consider a more explicit two-stream
  design: one encoder over per-slot player history (8 aggregated
  features per slot), one encoder over the hero composition + hero
  pairwise interactions, fused at the head. This may help especially
  because the two signal types have very different update cadence
  (player history changes daily, hero meta changes per patch).
- **Heterogeneous edges as inductive bias is intriguing but probably
  skip for the MVP.** We do have "edge types" in our data
  (hero-hero counter/synergy in past matches, player-player coplay
  history, player-on-hero familiarity), but adding a heterogeneous
  graph layer is a substantial complication. Defer to a follow-up;
  keep the MVP at "draft tokens + per-slot features" complexity.
- **The MoE gate pattern is a useful regularizer for combining
  encoders of different scales.** If we end up with both a graph
  encoder (for player co-play) and an attention encoder (for draft
  structure), a learned per-batch gate that picks how much of each to
  use is cheaper than concatenation and aligns with the gating intuition
  in our existing aggregator design. Cross-references the temperature-
  softmax recipe from [[kirchdorfer2024analytical]].
- **Two-stage training (encoders separate, then end-to-end) is the
  right recipe for our MVP** if we add a player-history encoder
  separate from the draft encoder. It avoids the "player-encoder
  training dominates the gradient" issue and matches our existing
  "pre-compute player aggregates → run draft encoder" data pipeline.

## Open questions / caveats

- Soccer is fundamentally different from Dota 2 Turbo: 11v11 (vs 5v5),
  long-running team identities (vs. anonymous-tail Turbo players),
  rich in-match event data (we have it too via `rich_cols` but it's
  training-target-only per HCE rule). The HIGFormer player-graph
  edges are *in-match passes between players*; we have no equivalent
  fine-grained player-player interaction data in our pre-game inputs.
- The MoBA / online-game related work HIGFormer cites — OptMatch,
  NeuralAC, HOI — is closer to our setting than soccer; consider
  fetching those (especially NeuralAC) for a follow-up ingest. Wang
  et al. position HIGFormer as superior to all three, but the
  online-game baselines are the more direct neighbors of our
  draft-only-win-prediction problem.
- The paper does NOT use MAE pre-training or multi-task supervision;
  it's purely supervised three-class (win/draw/lose). The composition
  with the MAE + UW-SO + ForkMerge ideas from this batch is novel
  territory; HIGFormer doesn't pre-empt foundation-mvp-740, it sets
  the comparison bar.
- Numbers are reported on WyScout; we can't translate them to our
  val_auc scale directly. The relative ranking of architectural
  choices (two-stream beats one-stream, het-edges beats hom-edges) is
  what's transferable; the absolute lift is not.
