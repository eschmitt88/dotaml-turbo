---
kind: experiment
slug: "2026-05-22-foundation-mvp-740"
date: "2026-05-22"
status: running
hypothesis: "A ~5M-param FT-Transformer foundation model trained jointly on win, duration, items, and per-slot KDA/GPM/HD with PMAE auxiliary, permutation-equivariant within-team encoding (canonical hero-id sort + (team,team) 2x2 attention bias, NO per-slot positional), patch token, UW-SO loss weighting, and a shared 2-block decoder with task-as-token prompting, lifts whole-val win val_auc by >= 0.003 over multitask-740 (target val_auc >= 0.6525)."
result: ""
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - uncertainty-weighted-multitask
  - multi-query-foundation-model
  - attention-bias-positional
  - task-as-token-prompting
  - supervised-multitask-pretraining
related_literature:
  - gorishniy2021revisiting
  - kim2024predict
  - kirchdorfer2024analytical
  - somepalli2021saint
  - cui2022m6
  - wang2025player
  - jiang2023forkmerge
  - bi2022pangu
  - liu2024moirai
  - shoghi2023molecules
  - ghosh2024octo
  - radford2022robust
tags: [foundation-model, multitask, pmae, uw-so, ft-transformer, task-as-token]
respects:
  - "~/.claude/rules/evaluation.md"
---

# foundation-mvp-740

## Hypothesis

See frontmatter. Target `val_auc >= 0.6525` on the 2026-02-24..2026-03-09 val split, lifting >= 0.003 over the multitask-740 anchor (0.6495). Three ablations:

1. `baseline_multitask_repro` — multitask-740 design at the new ~5M scale (no PMAE, no patch token, no (team,team) bias). Anchors scaling lift.
2. `foundation_mvp` — PRIMARY. Full design.
3. `foundation_no_patch_token` — sanity ablation; on the 7.40-only window the patch token degenerates to a constant embedding, so this should track `foundation_mvp` closely.

## Setup

- Config: `config.yaml`
- Code: `data.py`, `models.py`, `loss.py`, `mae.py`, `train.py`
- Data: `data/snapshots/7.40-2025-12-16/processed/player_features_prepatch_clean/{train,val}.parquet` (validation split only during search) + `data/snapshots/7.40-2025-12-16/processed/rich_cols/{train,val}.parquet` sidecar. Item vocab: reused from multitask-740 at `../2026-05-20-rich-supervision-multitask-740/results/item_vocab.json`.
- Hardware: RTX 5080 16 GB + 96 GB DDR5 at JEDEC 4800 MT/s.

### Important deviation from proposal

The proposal called for extending training data to Aug 2025 -> Feb 2026 (~30-40M matches across patches 7.39 -> 7.40). The existing clean parquet covers only 2025-12-16 -> 2026-02-23 (~13M matches, patch 7.40 only). Two reasons to skip the extension for the MVP:

1. The rich-cols sidecar (which provides aux supervision: duration, items, KDA, GPM, HD) only covers the same 7.40-only window. Extending the clean parquet without rebuilding rich_cols would yield no aux signal on the extended rows -- the auxiliary loss components would be applied only on rows that already exist. That defeats the "more data" objective.
2. The full extension (clean + rich_cols rebuild) is the ~3-4h CPU pre-build cited in the proposal's `estimated_runtime`. With the architectural changes already large, isolating the architectural lift from the data-scale lift is cleaner.

Consequence: the patch token degenerates to a single learnable embedding on this window, and the `foundation_no_patch_token` ablation becomes a near-no-op (still useful as a tokenizer-sanity check). The proposal's cross-patch generalization diagnostic is N/A on this window. Main agent can choose to invest the data-extension build separately and re-run.

## Result

Fill in after the run. Point at `metrics.json` (validation split — search signal). `final_metrics.json` is only written by the held-out test-split pass at chain end.

## Interpretation

(post-run)

## Diagnostics

(post-run; fields per the proposal: per-task val metrics, coverage buckets, loss-component traces per epoch + UW-SO omega weights, train-val gap on win head, in-vocab item rate per slot.)

- intended_effect_confirmed: n/a
- leakage_check: n/a
- overfitting_signal: n/a
- delta_from_prior: n/a
- unexpected_findings: n/a
- next_candidates:
  - n/a
  - n/a

## Follow-up

- ...
