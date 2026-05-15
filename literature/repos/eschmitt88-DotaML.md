---
kind: repo
name: "eschmitt88/DotaML"
url: "https://github.com/eschmitt88/DotaML"
commit: "pushed_at:2026-05-15T01:57:50Z"
source: "raw/repos/eschmitt88-DotaML.md"
added: "2026-05-15"
relevance: 5
status: scanned
related_experiments: []
related_concepts:
  - draft-only-win-prediction
  - draft-prediction-plateau
  - radiant-side-advantage
  - fake-match-filtering
  - hero-embedding-vs-onehot
tags: [prior-art, dota2, win-prediction, lightgbm, tensorflow, transformer, ucb]
---

# eschmitt88/DotaML

## Purpose

Exploratory predecessor to `dotaml-turbo`. Same author (eschmitt88), same
storage account (`dota2datalake`), same prediction target (Radiant win
probability from a 10-hero draft) — but on a **pre-patch-7.40** data window
(Aug-Oct 2025) and with code that the new project deliberately does not
reuse. Treat it as the canonical prior-art reference, not a starting
codebase.

## Shape

Layered Python project under `uv`:

- `src/azure_data_lake.py` — auth via `DefaultAzureCredential`, reads the
  partitioned Parquet layout `turbo/year=YYYY/month=MM/day=DD/*.parquet`.
- `src/features.py` — one-hot draft encoding (`create_onehot_hero_features`,
  `max_hero_id=150`).
- `src/ml_models/` — `BaseHeroPredictor` interface with `LightGBMPredictor`
  and `TensorFlowPredictor` implementations; `loader.py` auto-detects model
  type from file extension.
- `src/tf_architectures/` — registry of FFN architectures (`simple`, `deep`,
  `wide`, `residual`); a separate `transformer` lives outside the registry.
- `experiments/v1..v6/` — six promoted production models (manually copied
  from MLflow `mlruns/`).
- `docs/`, top-level design notes (`FEATURE_DESIGN.md`, `FAKE_MATCH_CRITERIA.md`,
  `DUPLICATION_REPORT.md`, `MATCH_DATA_REFERENCE.md`).

## Useful bits

- **The six-generation experiment grid** (see raw capture for full
  numbers). Plateau at ~59.9% test accuracy / ~0.635 test AUC across
  LightGBM (v1-v3), SimpleFFN (v4, 47k params), ResidualFFN (v5, 228k),
  Transformer with embeddings + 30% input masking (v6, 152k). Three
  orders of magnitude of capacity, ~0.04 spread in AUC. → seeds
  [[draft-prediction-plateau]].
- **Fake-match filter.** Forfeit detection via `tower_status_*` bits 9-10
  (both T4 towers of the losing team standing ⇒ "gg" surrender)
  combined with empty-inventory count >2. Concrete recipes in the raw
  capture. → seeds [[fake-match-filtering]].
- **Radiant-side advantage** of +5 to +7 percentage points reported
  consistently across v2/v3/v4 regardless of architecture. → seeds
  [[radiant-side-advantage]].
- **One-hot 300-dim** (150 heroes × 2 teams) vs **64-dim learned
  embeddings** (v6) with shared cross-team weights, separate Radiant/Dire
  position embedding, masked-input training for incomplete-draft scoring.
  Both representations land at the same plateau. → seeds
  [[hero-embedding-vs-onehot]].
- **UCB four-phase combo search** to rank 3-hero cores against random
  opposing drafts: 19,600 → 500 → 100 → 20 candidates at escalating
  prediction-count N. Each phase advances on UCB(β=1.0) score.
- **Documented data-hygiene findings:** 4.9% duplicate match_ids in
  early dataset traced to overlapping `match_seq_num` ranges in Azure
  filenames (4 days in Aug 2025; fix described in
  `DUPLICATION_REPORT.md`); a `max_hero_id=130` bug in v2 silently
  excluded ~20 heroes (v3 RESULTS shows the meta interpretation shifts
  substantially once fixed).
- **Splits.** Throughout: chronological 80/20 train/test by start_time
  (v6 stated explicitly; earlier versions implied by `train_fraction`
  field). No held-out test in the HCE sense.

## Follow-up

**Relevance:** 5/5 — this is the canonical prior-art reference for this
project. Every concept seeded here will be imported by the first
experiment proposal, and the ~60% accuracy plateau is the single most
important number to either replicate (as a sanity check on the new
patch-7.40 snapshot) or break (as a goal).

Open questions for downstream proposals:

- Does the ~60% / 0.635 plateau hold on the patch-7.40 19.6M-match snapshot,
  or does the bigger and more recent data shift it? Replicating v3-class
  LightGBM and v6-class Transformer on the new snapshot is the natural
  zero-th experiment.
- The prior-art experiments used chronological 80/20 splits without a held-
  out test. The new project intends HCE — that gap needs an explicit ADR.
- Patch 7.40's data window is fixed (2025-12-16 → 2026-03-23). Verify the
  Azure overlap-duplication bug described in `DUPLICATION_REPORT.md` was in
  fact closed by then, and re-check before training.
- Beyond the plateau: things the prior art did NOT try that might break it
  — (a) interaction features beyond pairs (3-hero synergies, lane-role
  inference); (b) position-aware draft order from `picks_bans[]`;
  (c) calibration analysis; (d) ensembling LightGBM with Transformer
  (the v5 README mentions this as a future direction but it was never run);
  (e) per-side specialized models given the asymmetry.
- The v6 Transformer's masked-input training is the only prior approach
  that supports scoring incomplete drafts. If a real-time use case is in
  scope later, this is the architecture to revisit.
