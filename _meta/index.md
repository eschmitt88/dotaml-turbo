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

*(none — `2026-05-15-plateau-baseline-740` completed; status: done.)*

## Completed experiments

- [[experiments/2026-05-15-plateau-baseline-740]] — LightGBM baseline replicates the v3 LightGBM ceiling (val_auc 0.6161, vs v3 test_auc 0.6189) on patch 7.40 under HCE. Proposal's strict band missed; proposal's spirit (plateau holds for this architecture) confirmed.

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
