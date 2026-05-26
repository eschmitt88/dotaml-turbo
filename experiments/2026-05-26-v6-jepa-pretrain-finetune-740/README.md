---
kind: experiment
slug: "2026-05-26-v6-jepa-pretrain-finetune-740"
date: "2026-05-26"
status: abandoned
hypothesis: "JEPA-style self-supervised pre-training (predicting masked TOKEN REPRESENTATIONS in latent space, rather than reconstructing raw token values) avoids the over-specialization pathology that halted v5. v5 mid-pretrain probe trajectory (0.4711 init → 0.5237 @ ep5 → 0.5304 @ ep10 → 0.5263 @ ep15) showed the encoder briefly held win-discriminative features at epoch 5 then drifted toward reconstruction-only representations as per-group losses kept decreasing. JEPA optimizes for semantic prediction (latent-content alignment) not token-level fidelity, so the encoder should not drift away from useful features as training continues. Targets — same as v5: mid-probe trajectory must show val_auc monotone-increasing past 0.55 by epoch 10 (else halt); Phase 2A linear probe ≥ 0.6300; Phase 2B full fine-tune ≥ 0.6485 (closes 50% of v4 → iso_teambias gap), ≥ 0.6493 beats iso_teambias on extended."
result: "HALTED at Phase 1 epoch 11/20 — classic JEPA representation collapse. jepa_loss collapsed from 0.0284 (ep1) → 0.0014 (ep10, plateau), rep_l2_mean shrunk from 14.4 → 2.5 (still decreasing), mid_probe val_auc STUCK at random (0.5013 @ ep5, 0.5017 @ ep10). Encoder found the trivial solution: produce small-magnitude reps that are easily predictable by the predictor MLP. Pairwise cosine similarity DID decrease (0.972 → 0.911) — slot differentiation worked — but reps differentiated INTO a low-information manifold the linear probe couldn't extract win signal from. Two of three pre-committed halt criteria fired (mid_probe ≤ 0.51 AND rep L2 shrinking). v6 worse than v5 on the mid_probe diagnostic (v5 peaked at 0.5304, v6 stayed 0.5017). Implementation kept EMA teacher as designed but no VICReg-style variance/covariance regularization or normalized predictor target — both standard JEPA collapse mitigations. Saved ~10h. Foundation-SSL effort paused; v4 diagnostic confirmed the encoder is sound and the val_auc ceiling is data-bound (see [[_meta/deferred-foundation-paths]] for the 5 architectural variants we considered but deferred). Pivoting to downstream queries on v4."
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - draft-prediction-plateau
  - embedding-vs-features-gradient-competition
related_literature:
  - kim2024predict
  - gorishniy2021revisiting
tags: [foundation-model, pretrain-finetune, multi-task, jepa, ema-teacher, latent-space, data-extension]
respects:
  - "~/.claude/rules/evaluation.md"
related_prior:
  - 2026-05-26-v5-pretrain-finetune-740
  - 2026-05-25-v4-iso-teambias-extended-740
  - 2026-05-25-v3-ablations-740
  - 2026-05-24-foundation-v3-740
  - 2026-05-23-foundation-component-isolation-740
  - 2026-05-20-rich-supervision-multitask-740
---

# v6-jepa-pretrain-finetune-740

## Hypothesis

See frontmatter. Single-change ablation of v5 — swap Phase 1 loss form
from BERT-style raw-target reconstruction to JEPA-style latent-space
prediction. Everything else (architecture, data, mask schedule, EMA
teacher infrastructure, Phase 2A/2B logic) is verbatim from v5.

## Setup

- Config: `config.yaml` (single new field `pretrain.loss_form: jepa`;
  removes the now-unused `pretrain.loss_weights` block; adds collapse-
  detector thresholds).
- Code: `data.py`, `models.py`, `mae.py`, `train.py` — forked from
  `experiments/2026-05-26-v5-pretrain-finetune-740/`. Diff vs v5:
  - `data.py`: UNCHANGED (byte-identical copy).
  - `models.py`: `FoundationTransformerV5` → `FoundationTransformerV6`
    rename ONLY. Architecture identical. The six pretrain reconstruction
    heads remain on the module but receive NO gradient during v6 Phase
    1 (the JEPA loss does not flow through them). They are kept so the
    state_dict matches v5's verbatim.
  - `mae.py`: drops `per_group_reconstruction_losses`; adds
    `slot_mask_from_groups`, `jepa_loss`, `representation_diagnostics`,
    and `JEPAPredictor` (small 2-layer MLP applied to student per-slot
    reps). `EMATeacherV5` → `EMATeacherV6` (identical behavior).
  - `train.py`: Phase 1 forward pass replaced. Student encodes the
    masked input; teacher (EMA copy, `torch.no_grad`) encodes the
    UN-masked input; loss = SmoothL1 between predictor(student_reps)
    and teacher_reps at masked slot positions. Per-step EMA update of
    teacher (unchanged from v5). New per-epoch logging: JEPA loss,
    per-slot rep L2-norm mean/std, pairwise cosine across slots in a
    batch (collapse detector). Phases 2A/2B unchanged from v5.
- Data: extended player_features + rich_cols sidecar parquets at
  `data/snapshots/7.40-2025-12-16/processed/{player_features_extended,
  rich_cols_extended}/`, reused verbatim from v3/v4/v5 — no rebuild.
- Splits: project `splits.yaml`. HCE-strict — `data.py` refuses any
  test-window date [2026-03-10, 2026-03-23].
- Pipeline: `run_all.sh` runs smoke-pretrain → smoke-finetune →
  pretrain → probe → finetune sequentially.

## JEPA loss form (the single conceptual change)

For each row, with per-row per-group masking p_group=0.4:

```
mask_dict      = SixGroupMasker(B)              # {group: [B] bool}
slot_mask[b,:] = OR_over_groups(mask_dict[g][b])  # [B, 10] bool, uniform per-row
student_reps   = encoder.encode(masked_inputs, mask_dict=mask_dict)   # [B,10,D]
with torch.no_grad():
    teacher_reps = ema_encoder.encode(unmasked_inputs, mask_dict=None) # [B,10,D]
loss = F.smooth_l1_loss(
    predictor(student_reps)[slot_mask],
    teacher_reps.detach()[slot_mask],
    reduction='mean',
)
ema.update(student)
```

The predictor is a small Linear(D)->GELU->Linear(D) MLP, Phase-1-only,
NOT saved with the encoder checkpoint. With p_group=0.4 over 6 groups,
the expected fraction of rows with at least one group masked is
1 - 0.6^6 ~= 0.953, so roughly 95% of rows (all 10 slots each) contribute
to the loss per batch.

### Mask-position semantics (decision recorded)

A "masked" slot is defined as: ANY of the 6 groups (player_block,
hero_token, item_list, kda, gpm, hd) is masked for that row. Because
each group's mask is per-row ([B] bool, not [B, 10]), the resulting
per-slot mask is uniform within a row. Each row contributes either
all 10 slots or 0 slots to the loss. This matches the v5 masking
convention and is the simplest semantics consistent with the
proposal.

### Collapse detector

Per training batch we log:
- `rep_l2_mean` / `rep_l2_std`: L2 norm of each per-slot rep,
  aggregated across batch + slots.
- `pairwise_cos_mean`: mean off-diagonal cosine similarity across the
  10 slots within each example, averaged over the batch.

`collapse_warning = (rep_l2_mean < 1e-3) OR (pairwise_cos_mean > 0.95)`.
A persistent True across an epoch is a hard halt signal.

## Halt criteria

- **Phase 1**: mid_probe val_auc still <= 0.51 (random) at epoch 10; OR
  per-slot rep L2-norm shrinks toward zero (collapse); OR pairwise
  cosine across slots >= 0.95 (collapse); OR loss explodes / NaN.
- **Phase 2A/2B**: standard - val_auc at random for 3+ epochs, NaN,
  kernel events.

## Result

Fill in after the run. `metrics_finetune.json` (best val on the
multi-task fine-tune; mirrored to `metrics.json`). Linear probe in
`metrics_linear_probe.json`. Pre-train trajectories in
`results/pretrain_history.json` and `results/mid_probe_history.json`.

## Interpretation

TBD.

## Diagnostics

TBD - filled in by the implementer post-run.
