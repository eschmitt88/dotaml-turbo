---
kind: paper
title: "Win Prediction in Esports: Mixed-Rank Match Prediction in Multi-player Online Battle Arena Games"
authors:
  - Victoria Hodge
  - Sam Devlin
  - Nick Sephton
  - Florian Block
  - Anders Drachen
  - Peter Cowling
year: 2017
venue: "arXiv:1711.06498 [cs.AI]"
url: "https://arxiv.org/abs/1711.06498"
source: "raw/papers/hodge2017win.pdf"
added: "2026-05-17"
relevance: 4
status: skimmed
related_experiments:
  - 2026-05-15-plateau-baseline-740
  - 2026-05-15-plateau-architectures-740
  - 2026-05-16-transformer-hp-sweep-740
related_concepts:
  - draft-prediction-plateau
  - draft-only-win-prediction
tags: [moba, dota2, win-prediction, hero-features, in-game-features, mixed-rank, professional-vs-amateur]
---

# Win Prediction in Esports: Mixed-Rank Match Prediction in MOBA Games

## TL;DR

Hodge et al. (Digital Creativity Labs, University of York; AAAI-17
workshop on esports) ask whether mixed-rank (professional + extremely
high-skill amateur) match data can be used as a proxy to train models
that predict the winners of professional Dota 2 matches. They build LR
and RF classifiers on (a) hero-pick vectors and (b) in-game telemetry
sliding windows, evaluating on a held-out professional set.
**Hero-only ceilings are ~55-59% accuracy; in-game telemetry at 20 min
reaches 75-76% accuracy.** Mixed training data transfers to pro data
with only slight accuracy reduction provided the algorithm and feature
selector are tuned per regime.

## Claims

- **Hero-only win prediction saturates around 55-59% accuracy** with LR
  or RF on hero one-hot features, across both mixed-rank (1820 matches)
  and pro-only (113 matches) datasets. This is robust to algorithm
  choice and feature-selection strategy.
- **In-game telemetry at 20 min lifts accuracy to 75-76%**: ~17 pp
  improvement over hero-only, attributed to current-game-state
  information rather than draft information.
- **Mixed-rank data is a viable proxy for pro-only training data** when
  predicting pro outcomes — performance gap is small if the right
  algorithm and feature selector are used per regime.
- **The optimal predictor varies by regime**: LR best for some mixed
  configurations, RF best for some pro configurations. Generalising
  one model across regimes loses accuracy.
- **Player-on-hero identity matters but is not isolated in this paper.**
  Hodge et al. explicitly note their hero data "only consider the sets
  of heroes selected but not which players were playing those heroes"
  and cite Yang, Qin, Lei 2016 for the claim that player identity is
  important — they do not quantify it themselves.

## Methods

- **Data:** 1820 mixed-rank matches (high-skill amateurs + pros) and
  113 pro matches, ranging across an unspecified period (the paper
  flags this as a limitation re: metagame drift across patches).
- **Hero features:** one-hot encoding (113 heroes at the time × 2
  teams).
- **In-game features (Table 1):** team-aggregated Damage Dealt, Kills,
  Last Hits, Net Worth, Tower Damage, XP Gained. Each is a time series
  sampled per minute; the paper evaluates a 5-minute sliding window
  ending at the 20-minute mark.
- **Algorithms:** Logistic Regression (LR) and Random Forest (RF) via
  Weka. Feature selectors: CFS (Correlation-based Feature Selection)
  filter and Wrapper subset evaluation.
- **Evaluation:** 10-fold CV for mixed data; held-out pro test set for
  the proxy question. Accuracy reported (not AUC).

## Results

**Hero data (Table 2):**

| dataset    | LR all | LR wrapper | RF all | RF wrapper |
| ---------- | ------ | ---------- | ------ | ---------- |
| Mixed-Hero | 54.64  | **58.75**  | 53.12  | 58.30      |
| Pro-Hero   | 47.79  | 50.44      | 50.44  | **55.75**  |

**In-game data at 20 min (Table 3):**

| dataset      | LR 1-attr (Kills R-D) | LR all | LR CFS    | RF 1-attr | RF all | RF CFS    |
| ------------ | --------------------- | ------ | --------- | --------- | ------ | --------- |
| Mixed-InGame | 74.14                 | 73.36  | 74.92     | 67.76     | 73.05  | **76.17** |
| Pro-InGame   | **75.22**             | 70.80  | 71.68     | 61.06     | 66.37  | 68.14     |

**Headline gap:** in-game features outperform hero features by **~17 pp
in mixed-rank** and **~20 pp in pro** — a striking signal that the
information available *after* the game starts dwarfs the information
present in the draft alone.

## Critique / open questions

- **Patches not held fixed.** The dataset crosses multiple metagame
  patches; Hodge et al. flag this as a limitation but don't quantify
  its impact. Our own work (`splits.yaml: patch=7.40`) is more strictly
  controlled.
- **No player-identity features.** The authors explicitly acknowledge
  this gap. The cited Yang, Qin, Lei 2016 paper ("Real-time eSports
  Match Result Prediction", arXiv) is the natural next read for our
  `player-features-740` proposal.
- **Small pro set (113 matches).** The pro evaluation is noisy in
  absolute terms; some of the LR/RF gap differences (≤ 5 pp) are
  within the confidence band you'd expect at n=113.
- **Accuracy not AUC.** The paper reports accuracy rather than AUC,
  making direct comparison to our val_auc numbers indirect (we have
  `metrics.json: val_acc=0.5866 ↔ val_auc=0.6161` for the LightGBM
  baseline, which sits comfortably at the upper end of Hodge's 54-59%
  mixed-hero band).
- **Dataset gone.** No public link to the 1820+113 match set. The
  methodology is reproducible; the specific numbers are not.

## Follow-up

**Relevance:** 4 — independent and well-cited attestation of the
hero-only ceiling at 55-59% accuracy in published academic work,
matching our `plateau-baseline-740` LightGBM val_acc=0.5866 almost
exactly. Also documents (via the 75-76% in-game-feature ceiling) that
the broader prediction task has substantial headroom once richer
feature sets are admitted — supporting the information-bottleneck
framing in `[[concepts/draft-prediction-plateau]]`. Scored 4 rather
than 5 because the paper does NOT directly isolate the player-identity
contribution that motivates the next experiment
([[player-features-740 (proposed)]]); for that we'd need Yang, Qin,
Lei 2016 or a similar follow-up.

**Open angles for downstream experiments:**

- The cited Yang, Qin, Lei 2016 paper is the natural next ingest if we
  want to quantify the player-skill contribution rigorously.
- Their in-game-feature work suggests a future `dotaml-serve`-style
  live-prediction experiment could expect 75%+ accuracy with telemetry,
  even before player MMR is added.
- The mixed-rank-as-proxy finding (mixed training data ≈ pro data for
  many predictions) is reassuring for our Turbo-only training set: even
  though Turbo is its own skill regime, models trained on Turbo data
  are likely to transfer to adjacent regimes with similar accuracy
  characteristics.
