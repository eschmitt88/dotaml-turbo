---
kind: experiment
slug: "2026-05-25-v3-ablations-740"
date: "2026-05-25"
status: done
hypothesis: "Two factor-isolation ablations on the foundation-v3 stack (val_auc=0.6462). A1 (v3_dur_ce) reverts the duration head from log-seconds SmoothL1 regression back to 8-bucket CE; tests whether the duration loss-form switch caused the regression vs iso_teambias=0.6493. A2 (v3_player_emb) adds a ~4M-param per-player identity embedding (128-dim) on top of the v3 stack; tests whether the embedding-prelim-740 NULL result was data-size-limited (extended corpus has 5x more rows). Either ablation reaching val_auc >= 0.6485 is a real lift; >=0.6493 beats iso_teambias."
result: "BOTH NEGATIVE, informative. A1 v3_dur_ce val_auc=0.6349 @ epoch 11/16 (early-stop, Δ=-0.0113 vs v3, Δ=-0.0144 vs iso_teambias) — reverting duration to CE HURTS in the v3 stack; duration regression was NOT the v3 regression cause. A2 v3_player_emb val_auc=0.6290 @ epoch 2/7 (catastrophic overfit, Δ=-0.0172 vs v3, Δ=-0.0186 vs embedding-prelim-740 NULL) — train_win went DOWN while vl_win went UP; coverage-bucket val_auc hurt uniformly (low -0.015, medium -0.017, high -0.019). Two failure modes of player embeddings now documented in [[concepts/embedding-vs-features-gradient-competition]]: starvation (7.40-only) vs overfit (extended cross-patch). Remaining v3-regression suspect: extended cross-patch data itself OR PMAE-on-extended interaction. Next: v4-iso-teambias-extended isolates the data-extension effect on the v2-winner architecture."
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - embedding-vs-features-gradient-competition
related_literature:
  - gorishniy2021revisiting
  - kim2024predict
  - bi2022pangu
tags: [foundation-model, ablation, multi-task, player-embedding]
respects:
  - "~/.claude/rules/evaluation.md"
related_prior:
  - 2026-05-24-foundation-v3-740
  - 2026-05-23-foundation-component-isolation-740
  - 2026-05-20-rich-supervision-multitask-740
  - 2026-05-19-player-embedding-prelim-740
---

# v3-ablations-740

## Hypothesis

See frontmatter. Two single-axis ablations on the foundation-v3
architecture, both reusing the existing extended cross-patch parquets:

- **A1 (`v3_dur_ce`)**: single-line revert of duration head from
  log-seconds SmoothL1 regression back to 8-bucket CrossEntropy.
- **A2 (`v3_player_emb`)**: add per-player identity embedding lookup
  (~4M params at 128 dim, 0=anonymous, 1..30k=top-frequent,
  30001..31024=1024 hash buckets for rare players) on top of v3.
  AdamW with no weight decay on the embedding table.

## Setup

- Config: `config.yaml`
- Code: `data.py`, `models.py`, `train.py`, `loss.py`, `mae.py` forked
  from `experiments/2026-05-24-foundation-v3-740/` with:
  - `data.py`: new optional account_id sidecar join + canonical-sort
    lockstep reorder. Account_idx tensor is always emitted (all-anon
    when A2 is off).
  - `models.py`: `dur_loss_mode` flag (regression|ce); when ce,
    `dur_head = nn.Linear(d_model, n_dur_buckets)`. Optional
    `player_embed = nn.Embedding(vocab_size, 128)` + `player_proj` to
    d_model, added to the per-slot token after hero+team+feat sum.
  - `train.py`: AdamW with no-weight-decay group for embedding params.
    Dur loss F.cross_entropy or SmoothL1 per mode. EMA teacher
    forwards account_idx. New A2 diagnostics: embedding L2-norm
    distribution + sample cosine similarity + topk-stratified val_auc.
- Vocab build: `build_vocab.py` streams account_id sidecar(s) per row
  group and writes `vocab/player_id_vocab.json`.
- Sidecar build: `build_account_sidecar_extended.py` walks raw history
  for the ~19M extended train rows that the prior 7.40-only sidecar
  doesn't cover (val sidecar covers extended val 100%).
- Data: extended parquets reused verbatim from foundation-v3-740
  (player_features_extended + rich_cols_extended). No rebuild.
- Hardware: RTX 5080 16 GB + 96 GB DDR5 at JEDEC 4800 MT/s.

## Result

Fill in after the run. Point at `metrics.json` (validation split). The
better of the two ablations writes `metrics.json` via the run_all.sh
copy step; per-ablation files are `metrics_v3_dur_ce.json` and
`metrics_v3_player_emb.json`.

## Interpretation

(post-run)

## Interpretation

- **A1 (v3_dur_ce)**: 0.6349 (best ep 11, early-stop ep 16). Reverting
  duration to CE on the v3 stack actively HURT by 0.0113. The duration
  regression switch was NOT the v3 regression cause — it was either
  neutral or slightly positive. The CE head learned normally
  (dur_top1=0.176 vs random 0.125); it just doesn't help the win
  head more than regression did.
- **A2 (v3_player_emb)**: 0.6290 (best ep 2, early-stop ep 7). The
  embedding table OVERFIT: train_win 0.6812→0.6550 (down) vs vl_win
  0.6682→0.6840 (up) — opposite directions. Coverage-bucket val_auc
  hurt uniformly, and the HIGH bucket (where regular players
  concentrate) was hurt the MOST (Δ=-0.019), the opposite of what
  "embeddings help frequent players" would predict.
- **Cumulative**: v3's 0.6462 ceiling is not unlocked by simpler
  component swaps (duration form or per-player identity). Remaining
  suspects for v3's regression vs iso_teambias (0.6493): the extended
  cross-patch data itself, OR PMAE-on-extended-data interaction.

## Diagnostics

- intended_effect_confirmed: yes (both ablations cleanly attribute) — A1 metrics_v3_dur_ce.json:val_auc=0.6349; A2 metrics_v3_player_emb.json:val_auc=0.6290
- leakage_check: HCE strict, splits.yaml-driven date filter in data.py:assert_no_test_dates
- overfitting_signal: A1 train=0.6712 val=0.6611 gap≈0.0101 healthy; A2 train=0.655 val=0.684 (REVERSED — classic embedding overfit); both visible in full_run.log epoch-by-epoch
- delta_from_prior: A1 vs v3=-0.0113, vs iso_teambias=-0.0144; A2 vs v3=-0.0172, vs embedding_prelim=-0.0186
- unexpected_findings: A2 overfit mode (vs the expected starvation mode from concept note) — extended-data per-player signal is strong enough to learn, but train (Aug2025-Feb2026 multi-patch) and val (single-patch 7.40) distributions differ enough that learned per-player vectors don't transfer. Concept note updated with the two failure modes.
- seeds_run: 1 (single run per ablation)
- metric_aggregation: single-run
- next_candidates:
  - v4-iso-teambias-extended-740: run v2-winner (iso_teambias) architecture on extended data — directly tests whether the data extension itself is the regression cause (vs v3 component composition).
  - v4-pmae-on-extended-isolation-740: keep iso_teambias's simpler stack but add PMAE EMA-teacher; tests whether PMAE-on-extended-data interaction is the regression cause.
  - anonymous-aware-modeling-740 (carryover): orthogonal axis; addresses the LOW-bucket binding constraint (anonymous tail is 0.6364 vs HIGH=0.6565 — biggest delta in the project).

## Follow-up

- Main agent runs full pipeline via
  `nohup bash experiments/2026-05-25-v3-ablations-740/run_all.sh > experiments/2026-05-25-v3-ablations-740/full_run.log 2>&1 &`
- Live-monitor per `~/.claude/CLAUDE.md` (poll every 30-45 min).
- A2 needs the embedding-diagnostics check from
  `concepts/embedding-vs-features-gradient-competition.md`: if
  `embedding_diagnostics.l2_norm_quantiles.p50` stays near init
  (~0.02 * sqrt(128) ~= 0.23), the embedding lost the gradient
  competition.
