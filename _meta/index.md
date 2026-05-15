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

## Open questions

- Does the ~60% accuracy / 0.635 AUC plateau from prior art hold on the
  patch-7.40 19.6M-match snapshot, or does the bigger and more recent
  dataset shift it? See [[draft-prediction-plateau]].
- The prior-art experiments used chronological 80/20 splits with no held-
  out test. The new project intends HCE — that gap needs an ADR before
  the first experiment ships.
- Verify the Azure overlap-duplication issue (described in
  DotaML's `DUPLICATION_REPORT.md`) was in fact closed before
  2025-12-16, before training on the patch-7.40 window.
