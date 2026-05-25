---
kind: experiment
slug: "2026-05-25-v4-iso-teambias-extended-740"
date: "2026-05-25"
status: running
hypothesis: "Run the v2-winner architecture iso_teambias (baseline_multitask_repro + the (team_query, team_key) 2x2 attention bias only — NO PMAE, NO patch token, NO UW-SO, 8-bucket CE duration head, hand-tuned multitask alpha) on the EXTENDED cross-patch training corpus. Isolates the data-extension axis as a single factor on the architecturally-cleanest known-good design. Three outcomes attribute v3's 0.0031 regression vs iso_teambias precisely: (a) v4 >= 0.6493 -> extended data is neutral/helpful; v3 regression came from PMAE/composition. (b) v4 in [0.6470, 0.6493) -> data costs a small amount on its own. (c) v4 < 0.6470 -> extended data IS the regression cause."
result: ""
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

(post-run — match the outcome decision tree in the proposal).

## Diagnostics

(post-run)

## Follow-up

- Main agent runs the full pipeline via
  `nohup bash experiments/2026-05-25-v4-iso-teambias-extended-740/run_all.sh > experiments/2026-05-25-v4-iso-teambias-extended-740/full_run.log 2>&1 &`
- Live-monitor per `~/.claude/CLAUDE.md` (poll every 30-45 min, halt on
  PATTERN of 3+ bad epochs).
