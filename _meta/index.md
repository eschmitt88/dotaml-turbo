---
name: index
description: Entry-point index for this project's knowledge graph.
---

# Index

Orientation for the project knowledge graph. Updated by `/wrap`, `/ingest`,
and `/new-experiment`.

## Maps of Content

(promote a cluster of ≥5 related concepts into `mocs/<theme>.md`)

- **MoC candidate (not yet promoted):** *Prior-art priors for draft-only
  win prediction.* Five concepts now cluster on this theme — they all came
  from one source (DotaML repo), so wait until a second independent source
  reinforces before promoting.

## Active experiments

(list of `experiments/YYYY-MM-DD-<slug>/` folders currently in flight)

*(none — all five experiments below completed; status: done.)*

## Completed experiments

- [[experiments/2026-05-15-plateau-baseline-740]] — LightGBM baseline replicates the v3 LightGBM ceiling (val_auc 0.6161, vs v3 test_auc 0.6189) on patch 7.40 under HCE. Proposal's strict band missed; proposal's spirit (plateau holds for this architecture) confirmed.
- [[experiments/2026-05-15-plateau-architectures-740]] — Three-architecture sweep mirroring DotaML v4-v6 (SimpleFFN, ResidualFFN, Transformer with 64-dim hero embeddings). Transformer 0.6322 (within 0.003 of v6's 0.6354), SimpleFFN 0.6217, ResidualFFN 0.6199 — all > LightGBM 0.6161 ceiling. Architecture-spread is real and Transformer-led, but the FFN-internal ordering inverts prior art (ResidualFFN < SimpleFFN by 0.0018). Loose hypothesis confirmed; strict rank-order hypothesis not.
- [[experiments/2026-05-16-transformer-hp-sweep-740]] — Optuna TPE+ASHA HP sweep over a minimal-Transformer baseline (60 trials, 9-dim search space). Best val_auc=0.6318, **+0.0007 above the control** trial, **-0.0004 vs the prior Transformer**. Hypothesis (≥ 0.6372) NOT confirmed. The ~0.632 ceiling is HP-robust on this snapshot — TPE found a 0.0008-wide envelope around the control point. Motivates structural mutation (islands evolution) over further HP tuning. Per-trial subprocess isolation pattern was load-bearing due to torch 2.12 + Blackwell sm_120 instability (~21% per-trial crash rate).
- [[experiments/2026-05-17-player-features-740]] — Adds ~90 per-player history features (smoothed overall + hero-specific winrate, recent form, premade-detection coplay, days-since, anonymous flag) to the LightGBM baseline via chronological leading-window aggregation over the patch-7.40 snapshot. **val_auc=0.6227, +0.0067 over baseline. Hypothesis NOT confirmed** — missed +0.020 target by 0.0134 and falls 0.0095 below the Transformer ceiling. Three sub-findings dominate; (1) per-player-per-hero winrate is the marginal-value lever (top-10 features by importance, all 10 player slots), overall winrate / recent form / co-play didn't crack top-20; (2) Turbo is 66% anonymous (12.7% of val matches have all-10-anonymous), data availability is the binding constraint; (3) coverage-bucket diagnostic monotonic (low/med/high val_auc = 0.6159/0.6230/0.6296) — cold-start IS binding, pre-patch ingest is justified BUT risky for the dominant hero-specific feature due to metagame drift.
- [[experiments/2026-05-18-player-features-prepatch-740]] — Extends the player-features-740 aggregator with ~127 days of pre-patch-7.40 history (Aug 1 → Dec 15 2025, ~100 GB additional raw under `data/history/turbo/`). **val_auc=0.6256, +0.0028 vs player-features-740 — HYPOTHESIS NOT CONFIRMED** (missed +0.005 target by 0.0022). Diagnostic story is the headline: coverage-bucket stayed monotonic and HIGH gained MOST (+0.0043), LOW gained LEAST (+0.0014) — opposite of cold-start prediction. History-source breakdown explained: low-bucket players had only 2.3% prepatch fraction, they're genuinely casual/new players (not active-but-uncached). **Cold-start is NOT the binding constraint; the casual/anonymous-player tail IS.** Big-deal observation: HIGH-coverage val_auc=0.6339 BEATS the Transformer ceiling (0.6322) for the first time — for the active 1/3 of val, player features are now the stronger lever than architecture. Whole-val ceiling still bound by anonymous tail (unchanged). Two OOM-kills forced dropping `coplay`+`unique_heroes` from the aggregator (neither in top-20 importance from prior exp; bias is bounded).

## Open questions

- The 0.635 plateau target was actually the v5 Transformer ceiling, not
  the v3 LightGBM ceiling. The "plateau across architectures" claim is
  unresolved on patch-7.40 — needs a Transformer/FFN baseline to test
  whether ~0.635 also reproduces, or whether the architecture-spread
  collapses on the new patch. See `next_candidates` in
  `experiments/2026-05-15-plateau-baseline-740/README.md`.
- Pre-flight verified: Azure file overlap is **structural**, not a
  collector bug — every day boundary in the patch-7.40 window has
  seq_num overlap, but probe of 2025-12-16/17 boundary showed 0 match_id
  intersection. Dedup-by-match_id is cheap insurance, not a hot
  mitigation. (Recorded in `concepts/match-id-vs-seq-num-ordering.md`.)
- The HCE-vs-prior-art-splits ADR was deferred. The first experiment
  ran fine without it; revisit whether the gap merits a record now that
  there is a number to compare against.
