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
related_experiments:
  - 2026-05-15-plateau-baseline-740
  - 2026-05-15-plateau-architectures-740
  - 2026-05-16-transformer-hp-sweep-740
  - 2026-05-17-player-features-740
  - 2026-05-18-player-features-prepatch-740
  - 2026-05-18-transformer-plus-features-740
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

## Scope refinement (2026-05-17)

`player-features-740` deliberately broadened scope beyond strict
"draft-only" by adding `account_id`-derived per-player history
features (smoothed winrate, hero-specific winrate, etc.) — still
pre-game-knowable, just not strictly draft-input. The proposal
called this out as a scope expansion and the result (modest +0.0067
lift, see [[2026-05-17-player-features-740]]) confirmed the lever
exists in principle.

**This concept now applies in two flavours:**

1. **`draft-only-win-prediction` (strict, original)** — only the 10
   hero_ids + side, no account_id, no draft order. This is what
   `plateau-baseline-740`, `plateau-architectures-740`, and
   `transformer-hp-sweep-740` solve. Useful as the canonical
   information-bottleneck benchmark.

2. **`pre-game-win-prediction` (broadened)** — anything knowable
   before the match starts: 10 hero_ids + side + account_id × 10 +
   `picks_bans[]` ordering + lobby_type + cluster + start_time +
   any derived player-history features built from past matches
   (HCE-strictly leading-window). This is the scope of
   `player-features-740` and any future "add a pre-game feature
   axis" experiment.

In-game telemetry, item builds, final scores, etc. remain out of
scope for both flavours and belong to the sibling `dotaml-serve`
project.
