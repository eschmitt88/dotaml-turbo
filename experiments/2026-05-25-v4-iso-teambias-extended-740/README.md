---
kind: experiment
slug: "2026-05-25-v4-iso-teambias-extended-740"
date: "2026-05-25"
status: done
hypothesis: "Run the v2-winner architecture iso_teambias (baseline_multitask_repro + the (team_query, team_key) 2x2 attention bias only — NO PMAE, NO patch token, NO UW-SO, 8-bucket CE duration head, hand-tuned multitask alpha) on the EXTENDED cross-patch training corpus. Isolates the data-extension axis as a single factor on the architecturally-cleanest known-good design. Three outcomes attribute v3's 0.0031 regression vs iso_teambias precisely: (a) v4 >= 0.6493 -> extended data is neutral/helpful; v3 regression came from PMAE/composition. (b) v4 in [0.6470, 0.6493) -> data costs a small amount on its own. (c) v4 < 0.6470 -> extended data IS the regression cause."
result: "OUTCOME (b) CONFIRMED. val_auc=0.6471 @ epoch 16/21 (early-stop, 3.95h). Δ vs iso_teambias = -0.0022 (extended-data penalty), Δ vs v3 = +0.0009 (v4 SLIGHTLY beats v3). Attribution math: v3's -0.0031 regression vs iso_teambias = -0.0022 (extended data alone) + -0.0009 (PMAE+patch_token+dur_regression composition). ~70% of v3's loss came from the extended data itself; ~30% from component composition. Coverage buckets: LOW=0.6375, MED=0.6455, HIGH=0.6574 — all slightly above v3 across the board. Real engineering tradeoff for the foundation-model goal: train on 7.40-only for max val_auc (0.6493) or train on extended cross-patch for downstream-query generalization at ~0.002 val_auc cost."
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - draft-prediction-plateau
  - embedding-vs-features-gradient-competition
related_literature:
  - gorishniy2021revisiting
tags: [foundation-model, ablation, multi-task, data-extension]
respects:
  - "~/.claude/rules/evaluation.md"
related_prior:
  - 2026-05-25-v3-ablations-740
  - 2026-05-24-foundation-v3-740
  - 2026-05-23-foundation-component-isolation-740
  - 2026-05-20-rich-supervision-multitask-740
---

# v4-iso-teambias-extended-740

## Hypothesis

See frontmatter. Single ablation: iso_teambias architecture on the
EXTENDED cross-patch corpus.

## Setup

- Config: `config.yaml`
- Code: `data.py`, `models.py`, `train.py`, `loss.py`, `mae.py` forked
  verbatim from `experiments/2026-05-25-v3-ablations-740/`. The only
  source-code change is `train.py:828-829` (argparse `--ablation`
  choices updated to `["v4_iso_teambias_extended"]`). All ablation
  flags are config-driven (`transformer_ablations[0]`):
  - `use_features=true` — player-features ARE consumed.
  - `multitask=true` — six heads (win + dur + item + kda + gpm + hd).
  - `use_patch_token=false` — drop the patch_id token (iso_teambias
    didn't have it). PMAE-free, so no patch_token group to mask.
  - `use_team_team_bias=true` — 2x2 (team_query, team_key) attention
    bias added per attention block (~64 params total).
  - `use_pmae=false` — NO EMA-teacher, NO masking, NO reconstruction
    loss. The mae.py import is left in place but the use_pmae branch
    in train.py is skipped.
  - `use_uw_so=false` — refused at train.py:850-851; broken at all
    scopes per prior experiments.
  - `dur_loss_mode=ce` — 8-bucket CrossEntropy on duration
    (`nn.Linear(d_model, 8)` head; `F.cross_entropy` loss).
  - `use_player_embedding=false` — no player_embed module
    instantiated. Account_idx tensor still flows through dataloader
    but is unused at the model.
  - Multitask alpha (matches multitask-740 / iso_teambias): win=1.0,
    dur=0.15, item=0.3, kda=0.1, gpm=0.1, hd=0.1. No alpha_mae since
    PMAE is off.
- Data: `data/snapshots/7.40-2025-12-16/processed/{player_features_extended,rich_cols_extended}/{train,val}.parquet`
  REUSED verbatim from foundation-v3-740 (no rebuild). Item vocab from
  `experiments/2026-05-20-rich-supervision-multitask-740/results/item_vocab.json`.
- HCE-strict: `data.py:assert_no_test_dates` refuses test window
  [2026-03-10, 2026-03-23] per `splits.yaml`.
- Hardware: RTX 5080 16 GB + 96 GB DDR5 at JEDEC 4800 MT/s.

## Result

Fill in after the run. Point at `metrics.json` (validation split).

## Interpretation

Outcome (b) per the proposal's decision tree. The attribution
arithmetic resolves cleanly:

```
iso_teambias (7.40-only):       0.6493
v4 = iso_teambias (extended):   0.6471   → extended-data penalty: -0.0022
v3 (full stack + extended):     0.6462   → composition penalty:   -0.0009
                                            (vs v4, same data)
total v3 regression vs iso_teambias =       -0.0031  ✓
```

So roughly 70% of v3's loss came from the extended cross-patch data
itself, and 30% from component composition (PMAE + patch token + dur
regression interactions on top of the simpler stack). The patch token
in v3 was supposed to compensate for cross-patch distribution shift;
it did not fully — a single learned scalar+token can't bridge the
multi-patch train ↔ single-patch val gap on its own.

Real engineering tradeoff for the foundation-model goal:
- Train on 7.40-only: max val_auc (0.6493, iso_teambias) but no
  cross-patch info for downstream queries (item rec, hero meta,
  lineup-vs-lineup) that depend on multi-patch coverage.
- Train on extended: ~0.002 val_auc penalty on the win head but
  cross-patch knowledge in the encoder for downstream queries.

Foundation-model framing favors extended; per-game-prediction
framing favors 7.40-only.

## Diagnostics

- intended_effect_confirmed: yes — outcome (b) resolved, attribution math closes cleanly
- leakage_check: HCE strict, splits.yaml-driven date filter in data.py:assert_no_test_dates
- overfitting_signal: train_win=0.6492 val_win=0.6549 gap=+0.0057 healthy
- delta_from_prior: vs v3 +0.0009 (better); vs iso_teambias -0.0022; vs cleanup_anchor -0.0006
- unexpected_findings: extended-data penalty (-0.0022) was larger than the composition penalty (-0.0009) — naive intuition would have flipped that priority. Suggests patch-token / cross-patch-mixing is the harder problem.
- seeds_run: 1 (single run)
- metric_aggregation: single-run
- next_candidates:
  - v5-rich-skill-features-740 (recommended): extend per-player input feature block from 8 → ~14 features with item-derived skill proxies (last20_gpm, last20_hd, last20_kda, hero-specific variants) and richer hero-novelty signal. Tests whether richer engineered features (no embeddings) can close the v4 → iso_teambias gap. Needs a ~3-4h CPU data rebuild + ~6h training.
  - v5-pmae-only-on-v4-740 (cheap diagnostic): add only PMAE EMA-teacher to v4 (single component-add). Isolates the PMAE-extended-data interaction. ~6h training, no data rebuild.
  - v6-train-val-distribution-align-740: subsample / reweight extended train to match val's single-patch distribution. Tests whether the 0.0022 data-penalty can be neutralized by smarter sampling rather than a richer architecture. ~6h.

## Follow-up

- Main agent runs the full pipeline via
  `nohup bash experiments/2026-05-25-v4-iso-teambias-extended-740/run_all.sh > experiments/2026-05-25-v4-iso-teambias-extended-740/full_run.log 2>&1 &`
- Live-monitor per `~/.claude/CLAUDE.md` (poll every 30-45 min, halt on
  PATTERN of 3+ bad epochs).
