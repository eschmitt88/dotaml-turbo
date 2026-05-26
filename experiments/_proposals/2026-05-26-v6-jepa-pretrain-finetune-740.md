---
kind: proposal
slug: v6-jepa-pretrain-finetune-740
date: 2026-05-26
status: proposed
hypothesis: "JEPA-style self-supervised pre-training (predicting masked TOKEN REPRESENTATIONS in latent space, rather than reconstructing raw token values) avoids the over-specialization pathology that halted v5. The v5 mid-pretrain probe trajectory (0.4711 init → 0.5237 @ ep5 → 0.5304 @ ep10 → 0.5263 @ ep15) showed the encoder briefly held win-discriminative features at epoch 5 then drifted toward reconstruction-only representations. JEPA optimizes for semantic prediction (what does the masked content MEAN in context) instead of token-level fidelity (what exact values were there), so the encoder should not drift away from useful features as training continues. Targets — same as v5: mid-probe trajectory must show val_auc monotone-increasing past 0.55 by epoch 10 (else halt); Phase 2A linear probe ≥ 0.6300; Phase 2B full fine-tune ≥ 0.6485 (closes 50% of v4 → iso_teambias gap), ≥ 0.6493 beats iso_teambias on extended."
rationale: >
  v5-pretrain-finetune-740 (2026-05-26) HALTED at Phase 1 epoch 16/20
  per the pre-committed mid-pretrain halt criterion. The mid-probe
  trajectory was diagnostic, not catastrophic: the encoder learned
  something win-discriminative in the first 5 epochs (mid_probe
  val_auc 0.4711 → 0.5237 = +0.05) then plateaued, then regressed.
  The per-group reconstruction losses kept slowly decreasing (hero CE
  1.55 → 1.41) while the win signal eroded — classic SSL
  over-specialization to low-level reconstruction.

  Two reasonable lessons from v5:
  - (1) Pre-train was too long. Maybe a 5-epoch reconstruction
    pre-train would have produced a useful encoder.
  - (2) The reconstruction objective itself is poorly aligned with
    "produce win-discriminative representations". Per-token
    reconstruction optimizes for surface fidelity, not semantic
    structure.

  This experiment tests (2) directly while reusing all the v5
  scaffolding. The implementer already set up:
  - 6 input groups (player_block + hero_token + item_list + KDA +
    GPM + HD per slot) — keep as-is.
  - Per-group learned mask tokens — keep as-is.
  - Mask schedule (each group masked independently with p_group=0.4) —
    keep as-is.
  - EMA teacher (deep-copy encoder, momentum=0.996, stop-gradient) —
    keep as-is, BUT ACTUALLY USE IT FOR THE LOSS this time. v5
    scaffolded the EMA teacher but used raw-target reconstruction
    (BERT-style); v6 swaps the loss form to JEPA.

  JEPA loss (the single conceptual change vs v5):
  - Student: encoder( masked input with mask tokens ) → per-slot latent reps
  - Teacher: ema_encoder( unmasked input ).detach() → per-slot latent reps
  - Loss: SmoothL1 between student's per-slot reps and teacher's per-slot
    reps, ONLY at positions where the group was masked. Averaged over
    masked slots.
  - No decoder, no per-group reconstruction heads. The encoder simply
    predicts what the teacher would produce at masked positions.

  Why this might help — informed take, not certainty:
  - JEPA optimizes representation alignment, not exact value recovery.
    The encoder can produce a representation that "knows the masked
    position means a high-skill player on a meta hero" without having
    to predict the exact GPM scalar value.
  - This is exactly the discriminative-vs-reconstructive distinction
    that matters for win prediction. Win is a downstream discriminative
    task; JEPA's training objective is itself discriminative-like
    (predict the right latent, distinguish from wrong latent).
  - Empirically: I-JEPA and V-JEPA (Assran et al. 2023, 2024) outperform
    MAE on downstream tasks in vision. The "predict in latent space"
    pattern transfers across modalities (vision, audio).

  Why this might NOT help — risks to be honest about:
  - Representation collapse: student degenerates to predicting a
    constant vector (loss → 0 trivially). Mitigation: EMA teacher
    with stop-gradient (proven safe in iso_pmae); asymmetric
    architecture (student sees masked input, teacher sees full).
    Diagnostic: log per-slot rep L2-norm; if it shrinks toward zero
    or all slots have near-identical reps (high cosine similarity),
    collapse is happening.
  - Win prediction might require token-level signal that JEPA throws
    away. E.g., "what specific hero is in slot 3" might matter for
    win, and JEPA doesn't preserve exact identity. The fine-tune
    stage should still be able to recover this from the unmasked
    inputs at fine-tune time, but the encoder weights are tuned for
    semantic prediction, not identity.
  - Loss is in continuous latent space — no per-group interpretable
    losses to monitor; just one scalar (with the mid-probe and the
    rep-L2-norm as the only convergence signals).

  This experiment is the cheapest direct test of "was the loss form
  the problem in v5?" because everything else stays fixed.
reads:
  - "[[experiments/2026-05-26-v5-pretrain-finetune-740]]"
  - "[[experiments/2026-05-25-v4-iso-teambias-extended-740]]"
  - "[[experiments/2026-05-23-foundation-component-isolation-740]]"
  - "[[concepts/tabular-foundation-model]]"
  - "[[concepts/masked-modeling-tabular]]"
  - "[[concepts/embedding-vs-features-gradient-competition]]"
  - "[[mocs/foundation-models]]"
expected_metric:
  name: val_auc
  target: 0.6485
  direction: "higher-is-better (mid-probe must rise past 0.55 by ep10; linear probe ≥ 0.6300; full fine-tune ≥ 0.6485 lift, ≥ 0.6493 beats iso_teambias)"
design_sketch:
  - "**Fork v5 codebase verbatim**. Copy `experiments/2026-05-26-v5-pretrain-finetune-740/` into `experiments/2026-05-26-v6-jepa-pretrain-finetune-740/`. Three targeted code changes; everything else is identical."
  - ""
  - "## Code changes (single conceptual swap)"
  - "**1. mae.py — switch loss form to JEPA**:"
  - "  - Keep `SixGroupMasker` and `EMATeacherV6` (renamed from V5) unchanged."
  - "  - Drop the per-group reconstruction heads + losses (`player_recon_head`, `hero_recon_head`, `item_recon_head`, `kda_recon_head`, `gpm_recon_head`, `hd_recon_head`)."
  - "  - Add a single small `predictor` MLP head: `nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))`. Operates on student per-slot reps to better match teacher's latent space."
  - "  - JEPA loss: `loss = F.smooth_l1_loss(predictor(student_slot_reps[mask]), teacher_slot_reps[mask].detach(), reduction='mean')`. ONE scalar loss replaces 6 per-group losses."
  - ""
  - "**2. train.py — pretrain phase swaps reconstruction call → JEPA call**:"
  - "  - Forward pass: `student_reps = encoder(masked_inputs, mask_tokens=mask)`; `with torch.no_grad(): teacher_reps = ema_encoder(unmasked_inputs)`."
  - "  - Compute mask at the slot/group level (which slots have which groups masked) and gather per-slot teacher representations at corresponding positions."
  - "  - Per-step EMA update of teacher: `teacher_params = momentum * teacher_params + (1 - momentum) * student_params` (unchanged from v5)."
  - "  - Logging: per-epoch JEPA loss, mid-pretrain probe trajectory (unchanged from v5), AND new representation-quality diagnostics: per-slot rep L2-norm mean/std, pairwise cosine similarity across slots in a batch (collapse-detector: if all pairs have cosine ≥ 0.95, collapse is happening)."
  - ""
  - "**3. config.yaml — single line, p_group stays 0.4**:"
  - "  - Add `pretrain.loss_form: jepa` (replaces implicit `reconstruction`)."
  - "  - Keep `p_group: 0.4`, `ema_momentum: 0.996`, `max_epochs: 20`, `mid_probe_epochs: [5, 10, 15, 20]`."
  - ""
  - "## Pipeline orchestration (identical to v5)"
  - "- `run_all.sh` runs: smoke-pretrain → smoke-finetune → pretrain → probe → finetune."
  - "- Per-trial subprocess retry wrapper."
  - "- `python -u` mandatory."
  - "- HCE strict: `assert_no_test_dates` in data.py."
  - "- Data: REUSE extended parquets from v3/v4/v5 verbatim. No rebuild."
  - ""
  - "## Phase 2A linear probe (unchanged from v5)"
  - "- Load `pretrain_encoder.pt`, freeze all encoder params."
  - "- KDA/GPM/HD/items FULLY masked at probe time (matches inference distribution)."
  - "- Single linear win head on pooled CLS-like representation."
  - "- 5 epochs at lr=1e-2, eval val_auc each epoch, keep best."
  - "- Output: `metrics_linear_probe.json`."
  - ""
  - "## Phase 2B full fine-tune (unchanged from v5)"
  - "- Load `pretrain_encoder.pt`, unfreeze."
  - "- All 6 v4 multi-task heads (win + dur + items + kda + gpm + hd), same α weights."
  - "- AdamW with encoder lr=1e-5 (wd=0) + heads lr=1e-3 (wd=1e-4)."
  - "- max_epochs=20, patience=5 on val_win_log_loss."
  - "- Output: `metrics_finetune.json` with anchors block + delta_vs_* fields."
  - ""
  - "## Halt criteria (per ~/.claude/CLAUDE.md monitoring rule)"
  - "- **Phase 1 halt**: mid_probe val_auc still ≤ 0.51 (random) at epoch 10 (encoder not learning); OR per-slot rep L2-norm shrinks toward zero (collapse); OR pairwise cosine similarity across slots ≥ 0.95 (collapse); OR loss explodes / NaN."
  - "- **Phase 2A/2B halt**: standard — val_auc at random for 3+ epochs, NaN, kernel events."
risks:
  - "**Representation collapse** (the JEPA-specific risk): student predicts a constant vector, satisfying the loss trivially. Mitigation: EMA teacher with stop-gradient (proven safe in iso_pmae); asymmetric inputs (student sees masked, teacher sees unmasked); explicit collapse diagnostics (per-slot rep L2-norm + pairwise cosine in mid_probe_history). If detected, halt and try (a) higher EMA momentum (0.999), (b) drop the predictor MLP and predict directly, or (c) add a small variance-regularization term (VICReg-style)."
  - "**Same v5 pathology but in latent space**: encoder briefly learns useful features then drifts toward 'predict the constant latent' or 'predict the mean latent'. Mid-pretrain probes will catch this. Same halt criteria as v5 apply (probe val_auc ~0.50 by epoch 10 → halt)."
  - "**JEPA may need different hyperparameters than reconstruction**: EMA momentum, mask rate, predictor MLP architecture, learning rate schedule may all need tuning. We're testing the CHEAP-CHANGE version (same hyperparameters as v5, swap only the loss form). If this fails, a tuned JEPA might still work — but that's a v7 question."
  - "**Compute risk**: 12.5h end-to-end, same as v5. Mid-probe halt criterion at epoch 10 limits risk to ~3h if pretrain is broken. Worst case: pretrain finishes (6h), linear probe (0.5h), fine-tune underperforms v4 (6h) = full 12.5h burn with no lift."
  - "**Fine-tune may underperform v4 even if pre-train works**: the encoder is tuned for semantic prediction, not exact token identity. Fine-tune needs to recover identity-level signal. Might happen, might not."
related_prior:
  - 2026-05-26-v5-pretrain-finetune-740
  - 2026-05-25-v4-iso-teambias-extended-740
  - 2026-05-23-foundation-component-isolation-740
estimated_runtime: "≈12.5h end-to-end on RTX 5080: 6h pretrain + 0.5h linear probe + 6h fine-tune. No new data builds (reuses v3-built extended parquets). Same budget as v5. Halt criterion at epoch-10 mid-probe limits worst-case risk to ~3h if pretrain is broken."
---

# v6-jepa-pretrain-finetune-740 — swap reconstruction loss for JEPA

## Where this fits

v5 halted cleanly at Phase 1 epoch 16/20: the encoder learned
something win-discriminative at epoch 5 (mid_probe 0.5237) then
drifted (epoch 10: 0.5304, epoch 15: 0.5263). Classic SSL
over-specialization to reconstruction. Per-group losses kept
decreasing while the win signal eroded.

v6 swaps the loss form: instead of predicting masked token VALUES
(BERT-style raw-target reconstruction), the encoder predicts
masked token REPRESENTATIONS in latent space (JEPA-style). Same
scaffolding, same masking, same EMA teacher (finally used for its
intended purpose). Single-change ablation of v5.

## Decision tree

```
Phase 1 mid-probe trajectory?
├── monotone-increasing past 0.55 by ep10 → JEPA learned useful features
│   ├── Phase 2A val_auc ≥ 0.6300 → strong representation
│   └── Phase 2B val_auc?
│       ├── ≥ 0.6500 → foundation pattern pays off; v7 builds on this
│       ├── ≥ 0.6493 → beats iso_teambias on extended; worth the cost
│       ├── ≥ 0.6471 → matches v4; foundation neutral-to-positive
│       └── < 0.6471 → pre-train + fine-tune actively hurts; v4 ceiling
├── plateau-and-drift (same as v5 in latent space) → halt at ep10
│   → SSL family is not the lever; pivot to engineered features (v7-rich-skill)
└── representation collapse (L2 → 0 or all-slots cosine ≥ 0.95)
    → halt at any epoch; try collapse mitigations (VICReg, higher EMA momentum)
```

## Anchor table

| Reference | val_auc | Pattern |
|---|---|---|
| iso_teambias (7.40-only) | 0.6493 | multi-task joint, 7.40-only |
| **v4 (PRIMARY)** | **0.6471** | multi-task joint, extended |
| cleanup-740 | 0.6477 | transformer + features, 7.40-only |
| baseline_multitask_repro | 0.6470 | foundation-mvp baseline |
| iso_pmae | 0.6464 | + PMAE (joint), 7.40-only |
| v3 (joint PMAE+multitask) | 0.6462 | full stack joint, extended |
| **v5 (HALTED)** | mid_probe peak 0.5304 | reconstruction pretrain, never fine-tuned |
| **v6 linear probe** | ? | JEPA pretrain + frozen probe |
| **v6 fine-tune** | ? | JEPA pretrain + multi-task fine-tune |

## Out of scope (deferred to v7+)

- Tuning JEPA hyperparameters (EMA momentum, mask rate, predictor MLP depth) — if v6 is close but not quite, v7 tunes.
- Player-centric contrastive (Design C from the decision discussion) — if JEPA fails too, contrastive is the next SSL family to try.
- Engineered features (the originally-drafted v5-rich-skill) — pragmatic pivot if SSL family universally fails.
- HCE final-scoring pass on the sealed test window — never touched during search.
