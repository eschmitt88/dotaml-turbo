---
kind: experiment
slug: "2026-05-23-foundation-component-isolation-740"
date: "2026-05-23"
status: running
hypothesis: "Of the four new components added in foundation-mvp-740 (PMAE, UW-SO, (team,team) attention bias, patch token) on top of the working multitask baseline, at least one introduces the training instability that caused val_auc to collapse to 0.5058. Three single-component isolation ablations on top of baseline_multitask_repro (val_auc=0.6470 anchor) attribute the failure: a component is safe iff its ablation lands within [0.6440, 0.6500]."
result: ""
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - uncertainty-weighted-multitask
  - attention-bias-positional
  - task-as-token-prompting
related_literature:
  - gorishniy2021revisiting
  - kim2024predict
  - kirchdorfer2024analytical
  - bi2022pangu
tags: [diagnostic, ablation, foundation-model, bug-fix]
respects:
  - "~/.claude/rules/evaluation.md"
related_prior:
  - 2026-05-22-foundation-mvp-740
  - 2026-05-20-rich-supervision-multitask-740
---

# foundation-component-isolation-740

## Hypothesis

See frontmatter. Three single-component isolation ablations attribute
which of foundation-mvp-740's four added components caused the
val_auc=0.5058 collapse. Anchor: baseline_multitask_repro=0.6470
(reproduced cleanly in foundation-mvp-740).

| ablation | new component added on top of baseline |
| --- | --- |
| iso_uwso | UW-SO loss weighting (with per-task initial-loss normalization fix) |
| iso_pmae | PMAE auxiliary objective (with EMA-teacher bug-fix) |
| iso_teambias | (team_q, team_k) 2x2 per-head per-layer attention bias |

Patch token is NOT isolated separately: on the 7.40-only window it
degenerates to a single learned embedding and was already confirmed
harmless in foundation-mvp-740 (`foundation_no_patch_token` also collapsed).

## Bug-fixes applied

### Bug A: PMAE collapse (mae_loss -> 0 mid-training)

Foundation-mvp-740's mae.py + train.py implemented PMAE with the teacher
pass using the SAME model weights as the student (just under `no_grad`).
This admits the classic BYOL/JEPA representational-collapse mode: the
model learns to make encoder outputs at masked positions invariant to
the mask (propagating only from un-masked neighbors), driving SmoothL1
to 0 without learning useful representations.

**Fix.** `EMATeacher` in mae.py wraps a deep-copied teacher with
stop-gradient, updated via EMA (momentum=0.996 default) each step.
Teacher always sees un-masked input; student sees masked input;
reconstruction target is the lagged-teacher's encoder output. The
EMA lag prevents the trivial-collapse mode.

### Bug B: UW-SO loss-scale misapplication

Foundation-mvp-740's UWSO applied softmax(1/sg[L_k]/T) directly to RAW
per-task losses whose magnitudes spanned ~30x (items ~0.07, dur ~2.1,
win ~0.69). The result over-weighted low-magnitude tasks by ~30x,
drowning the win head's gradient. Observed symptom: train_win loss
INCREASED across epochs 1-5.

**Fix.** UWSO now tracks per-task initial-loss running mean L_k_init
over the first 100 batches, then computes omega = softmax(1 / (L_k /
L_k_init) / T). Before init_window_batches: omega defaults to uniform.
Per-task omega logged every epoch.

## Setup

- Config: `config.yaml` (three iso_* ablations + retained reference entries).
- Code: `data.py`, `models.py` verbatim from foundation-mvp-740;
  `loss.py`, `mae.py`, `train.py` with bug-fixes applied.
- Data: reuses foundation-mvp-740's data sources (clean parquet +
  rich-cols sidecar + item_vocab from multitask-740).
- Hardware: RTX 5080 16 GB + 96 GB DDR5 at JEDEC 4800 MT/s.

## Result

Fill in after the run. Point at `metrics.json` (validation split — search signal).
`final_metrics.json` is only written by the held-out test-split pass at chain end.

## Interpretation

(post-run)

## Diagnostics

(post-run; per-ablation: val_auc trajectory, per-task UW-SO omega weights
(iso_uwso only meaningful), mae_loss + mask_count + mask_fraction +
teacher/student L2 norms at mask positions (iso_pmae), coverage buckets,
final delta_vs_baseline_multitask_repro_anchor.)

- intended_effect_confirmed: n/a
- leakage_check: n/a
- overfitting_signal: n/a
- delta_from_prior: n/a
- unexpected_findings: n/a
- next_candidates:
  - n/a
  - n/a

## Follow-up

- Main agent runs full pipeline via `nohup bash run_all.sh > full_run.log 2>&1 &`.
- Live-monitor each ablation; halt early on pattern of 3+ consecutive bad epochs.
- v3 design uses this experiment's pass/fail attribution.
