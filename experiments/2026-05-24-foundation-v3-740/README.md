---
kind: experiment
slug: "2026-05-24-foundation-v3-740"
date: "2026-05-24"
status: done
hypothesis: "v3 = v2 component-isolation code MINUS UW-SO PLUS hand-tuned alpha weights (multitask-740's 1.0/0.15/0.3/0.1) PLUS duration switched from 8-bucket CE to log-seconds SmoothL1 regression PLUS extended cross-patch training data (Aug 2025 - Feb 2026, ~3 patches with patch_id token now meaningful). Target val_auc >= 0.6508 (+0.0015 over iso_teambias 0.6493 anchor)."
result: "MISSED TARGET. val_auc=0.6462 @ epoch 25/30 (clean convergence, 6.08h wall). Δ vs target -0.0046; Δ vs iso_teambias -0.0031; Δ vs multitask_740 -0.0033. Tied with iso_pmae (0.6464). Coverage buckets: HIGH=0.6565, MED=0.6450, LOW=0.6364 — anonymous tail still the binding constraint. Training healthy (not a crash like foundation-mvp); the composite design just doesn't compose additively. Ablations in v3-ablations-740 (A1 dur_CE, A2 player_emb) both NEGATIVE, ruling out duration form and player identity as v3-regression causes; remaining suspect = extended cross-patch data itself OR PMAE-on-extended-data interaction."
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - attention-bias-positional
  - task-as-token-prompting
related_literature:
  - gorishniy2021revisiting
  - kim2024predict
  - bi2022pangu
  - shoghi2023molecules
  - wang2025player
tags: [foundation-model, multi-task, regression, cross-patch]
respects:
  - "~/.claude/rules/evaluation.md"
related_prior:
  - 2026-05-23-foundation-component-isolation-740
  - 2026-05-22-foundation-mvp-740
  - 2026-05-20-rich-supervision-multitask-740
  - 2026-05-19-upstream-data-cleanup-740
---

# foundation-v3-740

## Hypothesis

See frontmatter. v3 combines the v2 evidence (canonical hero sort safe,
team-team bias helpful +0.0023, PMAE + EMA-teacher safe, UW-SO is the
saboteur) with two long-pending fixes (duration regression; cross-patch
data extension) and the multitask-740 anchor's hand-tuned alpha weights.

## Setup

- Config: `config.yaml`
- Code: `data.py`, `models.py`, `train.py`, `loss.py`, `mae.py` forked from
  `experiments/2026-05-23-foundation-component-isolation-740/` with:
  - `loss.py`: UW-SO dropped (file kept as no-op import for compatibility);
    plain alpha-weighted sum.
  - `models.py`: `dur_head = nn.Linear(d_model, 1)` (regression scalar) in
    place of 8-bucket CE.
  - `train.py`: duration loss switched to SmoothL1 on log(seconds+1);
    post-hoc bucket-top1-acc computed for anchor comparison.
  - `data.py`: duration target switched from bucket index to
    log(duration_seconds + 1.0) float scalar; patch_id derived from
    start_time_date (multi-patch corpus).
- Data: extended player_features at
  `data/snapshots/7.40-2025-12-16/processed/player_features_extended/`
  (train 2025-08-15 - 2026-02-23, val 2026-02-24 - 2026-03-09). Extended
  rich_cols sidecar at `.../rich_cols_extended/`. Built via
  `build_features_extended.py` + `build_rich_cols_extended.py` (forked
  from cleanup-740 + multitask-740 with the multi-checkpoint defense and
  pyarrow row-group-stats verify — no full re-read).
- Hardware: RTX 5080 16 GB + 96 GB DDR5 at JEDEC 4800 MT/s.

## Result

Fill in after the run. Point at `metrics.json` (validation split — search
signal). `final_metrics.json` is only written by an explicit held-out
test-split pass at chain end.

## Interpretation

(post-run)

## Diagnostics

(post-run)

- intended_effect_confirmed: n/a
- leakage_check: n/a
- overfitting_signal: n/a
- delta_from_prior: n/a
- unexpected_findings: n/a
- next_candidates:
  - n/a
  - n/a

## Follow-up

- Main agent runs full pipeline via
  `nohup bash experiments/2026-05-24-foundation-v3-740/run_all.sh > experiments/2026-05-24-foundation-v3-740/full_run.log 2>&1 &`
- Live-monitor per `~/.claude/CLAUDE.md` (poll every 30-45 min).
