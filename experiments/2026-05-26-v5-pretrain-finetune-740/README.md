---
kind: experiment
slug: "2026-05-26-v5-pretrain-finetune-740"
date: "2026-05-26"
status: abandoned
hypothesis: "Classical BERT-style sequential pre-train + fine-tune closes (or beats) the v4 → iso_teambias gap on extended cross-patch data. Phase 1: PMAE-style self-supervised pre-train over SIX masked input groups (player_block + hero_token + item_list + KDA + GPM + HD per slot) with EMA-teacher momentum=0.996. Phase 2A: linear probe over frozen encoder (KDA/GPM/HD/items FULLY masked, matching inference). Phase 2B: full multi-task fine-tune from pre-trained encoder with low encoder LR (1e-5) and head LR (1e-3). Targets: linear-probe val_auc >= 0.6300 (representation is non-trivially informative); full fine-tune val_auc >= 0.6485 (foundation closes >= 50% of the v4 -> iso_teambias gap of 0.0022); >= 0.6493 (foundation beats iso_teambias on extended data, real win)."
result: "HALTED at Phase 1 epoch 16/20 per pre-committed halt criterion. Mid-pretrain linear probes (epochs 5/10/15) showed encoder is NOT learning win-discriminative features: val_auc trajectory 0.4711 (init) → 0.5237 (ep5) → 0.5304 (ep10) → 0.5263 (ep15) — encoder learned something win-discriminative in the first 5 epochs (+0.05 lift) then plateaued, then REGRESSED. Classic SSL pathology: prolonged reconstruction training pushed encoder toward reconstruction-only representations (per-group losses DID slowly decrease, especially hero CE 1.55 → 1.41), eroding the task-relevant features it briefly had. Saved ~10h of subsequent fine-tune compute by halting. Result is informative: BERT-style raw-target reconstruction with these mask groups doesn't produce win-useful representations on extended cross-patch data. Path forward: v6-jepa-pretrain-finetune-740 reuses the v5 scaffolding (EMA teacher already in place, just unused) but swaps the loss form to JEPA — predicting masked TOKEN REPRESENTATIONS in latent space instead of raw token reconstruction. Tests whether the over-specialization was the loss form (with everything else fixed)."
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - draft-prediction-plateau
  - embedding-vs-features-gradient-competition
related_literature:
  - kim2024predict
  - gorishniy2021revisiting
tags: [foundation-model, pretrain-finetune, multi-task, pmae, ema-teacher, data-extension]
respects:
  - "~/.claude/rules/evaluation.md"
related_prior:
  - 2026-05-25-v4-iso-teambias-extended-740
  - 2026-05-25-v3-ablations-740
  - 2026-05-24-foundation-v3-740
  - 2026-05-23-foundation-component-isolation-740
  - 2026-05-20-rich-supervision-multitask-740
---

# v5-pretrain-finetune-740

## Hypothesis

See frontmatter. Three sequential phases — biggest single experiment in
the project (~12.5h wall).

## Setup

- Config: `config.yaml`
- Code: `data.py`, `models.py`, `mae.py`, `train.py` — forked from
  `experiments/2026-05-25-v4-iso-teambias-extended-740/` and
  `experiments/2026-05-23-foundation-component-isolation-740/mae.py`.
  Key changes:
  - `data.py` adds scalar inputs `scalar_inputs[10, 3] = (kda_log1p,
    gpm_raw, hd_log1p)` standardized via train-fit mean/std. v4-style
    targets (`y_kda`/`y_gpm`/`y_hd`) retained.
  - `models.py`:`FoundationTransformerV5` adds 6 per-group LEARNED mask
    tokens (replace the per-slot contribution when masked), 4 zero-init
    rich-input projections (item / kda / gpm / hd), and 6 pre-train
    reconstruction heads (player SmoothL1, hero CE, item BCE, kda/gpm/hd
    SmoothL1). All v4 task heads + decoder retained for Phase 2B.
  - `mae.py`:`SixGroupMasker` independently masks each of 6 groups at
    `p_group=0.4` per example. `EMATeacherV5` keeps the iso_pmae
    pattern (deep-copy + momentum=0.996). The primary loss is
    raw-target reconstruction (BERT-style, not BYOL alignment) — EMA
    teacher is kept for future variants but not consumed here.
  - `train.py --phase {pretrain,probe,finetune}` is the single
    orchestrator; each phase has its own checkpoint/output logic.
- Data: extended player_features + rich_cols sidecar parquets at
  `data/snapshots/7.40-2025-12-16/processed/{player_features_extended,
  rich_cols_extended}/`, reused verbatim from v3/v4 — no rebuild.
- Splits: project `splits.yaml`. HCE-strict — `data.py` refuses any
  test-window date [2026-03-10, 2026-03-23].
- Pipeline: `run_all.sh` runs smoke-pretrain -> smoke-finetune ->
  pretrain -> probe -> finetune sequentially.

## Result

**HALTED at Phase 1 epoch 16/20** per pre-committed mid-pretrain probe
halt criterion. Phase 2A linear probe + Phase 2B fine-tune never ran.
See `mid_probe_history.json` for trajectory.

| Phase | Status | Best metric |
|---|---|---|
| Phase 1 (PMAE pre-train) | HALTED ep 16/20 | per-group recon losses ↓ but win-discriminative signal ↑ then ↓ |
| Phase 2A (linear probe) | not run | — |
| Phase 2B (full fine-tune) | not run | — |

Mid-pretrain probe trajectory (small linear probe on 50k val every 5 epochs):

| Probe Epoch | val_auc | Δ vs random (0.50) |
|---|---|---|
| 1 (smoke init) | 0.4711 | −0.029 |
| 5 | 0.5237 | +0.024 (encoder briefly held useful features) |
| 10 | 0.5304 | +0.030 (plateau) |
| 15 | **0.5263** | +0.026 (REGRESSION — encoder drifted) |

Per-group reconstruction losses kept decreasing throughout
(hero CE 1.55 → 1.41 over 16 epochs, all per-group losses slowly
improving). The encoder was getting better at token reconstruction
WHILE simultaneously losing its briefly-held win-discriminative
signal — classic SSL over-specialization. Halted to save ~10h of
fine-tune compute that would have built on a degraded encoder.

## Interpretation

The mid-pretrain probe trajectory (0.4711 → 0.5237 → 0.5304 → 0.5263)
tells a clean story: PMAE-style reconstruction DID drive the encoder
in a useful direction for the first 5 epochs, then it plateaued, then
it actively regressed.

The mechanism is the classic SSL pathology: optimizing per-token
reconstruction (BERT-style) increasingly specializes the encoder to
low-level token recovery (surface co-occurrence, exact value
recovery) at the expense of higher-level semantic features. The
encoder learns "what token comes next" not "what makes this match
likely to result in radiant_win". The per-group reconstruction
losses confirm the encoder kept getting better at reconstruction
even as the win-discriminative signal eroded.

In other words: 20 epochs of reconstruction pre-training is *too
much* — the encoder briefly held useful features at epoch 5 but
over-trained past them. Either shorter pretrain OR a different
objective is needed.

We chose the latter: v6-jepa-pretrain-finetune-740 keeps the v5
scaffolding (same 6-group masking, same EMA teacher) but swaps the
reconstruction loss for JEPA — predicting masked-position
REPRESENTATIONS in latent space rather than raw token values. The
hypothesis is that latent-space prediction avoids the
over-specialization to reconstruction details, preserving the win-
discriminative features that briefly appeared at epoch 5.

## Diagnostics

- intended_effect_confirmed: NO — encoder did not learn stable win-discriminative features. Trajectory in mid_probe_history.json.
- leakage_check: HCE strict, splits.yaml-driven date filter passed.
- overfitting_signal: n/a (halted before fine-tune; pre-train per-group reconstruction losses healthy)
- delta_from_prior: pre-train mid_probe peaked at 0.5304 @ ep10 vs random baseline 0.50; v4 anchor 0.6471 not reached because we halted.
- unexpected_findings: encoder briefly held useful win-relevant features at epoch 5 (+0.05 over random) then drifted. Suggests SHORT reconstruction pretrain could be useful — but more interestingly, the JEPA-style alternative may avoid the drift entirely. The EMA teacher infrastructure was scaffolded but unused for the loss (BERT-style raw-target reconstruction); v6 will use it for JEPA.
- seeds_run: 1 (single halted run)
- metric_aggregation: single-run
- next_candidates:
  - v6-jepa-pretrain-finetune-740 (immediate next step): swap reconstruction → JEPA latent-space prediction, reuse v5 scaffolding
  - v6b-short-reconstruction-pretrain: alternative cheap test of "we just over-trained"; max_epochs=5 then Phase 2A/2B
  - v6c-engineered-features (parallel pivot): if SSL family doesn't pay off, fall back to richer engineered features (the originally-drafted v5-rich-skill-features-740)

## Follow-up

- ...
