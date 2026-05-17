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

*(none — all three experiments below completed; status: done.)*

## Completed experiments

- [[experiments/2026-05-15-plateau-baseline-740]] — LightGBM baseline replicates the v3 LightGBM ceiling (val_auc 0.6161, vs v3 test_auc 0.6189) on patch 7.40 under HCE. Proposal's strict band missed; proposal's spirit (plateau holds for this architecture) confirmed.
- [[experiments/2026-05-15-plateau-architectures-740]] — Three-architecture sweep mirroring DotaML v4-v6 (SimpleFFN, ResidualFFN, Transformer with 64-dim hero embeddings). Transformer 0.6322 (within 0.003 of v6's 0.6354), SimpleFFN 0.6217, ResidualFFN 0.6199 — all > LightGBM 0.6161 ceiling. Architecture-spread is real and Transformer-led, but the FFN-internal ordering inverts prior art (ResidualFFN < SimpleFFN by 0.0018). Loose hypothesis confirmed; strict rank-order hypothesis not.
- [[experiments/2026-05-16-transformer-hp-sweep-740]] — Optuna TPE+ASHA HP sweep over a minimal-Transformer baseline (60 trials, 9-dim search space). Best val_auc=0.6318, **+0.0007 above the control** trial, **-0.0004 vs the prior Transformer**. Hypothesis (≥ 0.6372) NOT confirmed. The ~0.632 ceiling is HP-robust on this snapshot — TPE found a 0.0008-wide envelope around the control point. Motivates structural mutation (islands evolution) over further HP tuning. Per-trial subprocess isolation pattern was load-bearing due to torch 2.12 + Blackwell sm_120 instability (~21% per-trial crash rate).

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
