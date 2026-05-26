---
kind: proposal
slug: v5-pretrain-finetune-740
date: 2026-05-26
status: implemented
experiment: experiments/2026-05-26-v5-pretrain-finetune-740/
hypothesis: "A genuine foundation-model pattern — self-supervised pre-training followed by task fine-tuning — closes (or beats) the v4 → iso_teambias gap on extended cross-patch data. Phase 1 pre-trains an FT-Transformer encoder on extended data via PMAE-style masked reconstruction over SIX input groups (player_features, hero_token, item_list, KDA-scalar, GPM-scalar, HD-scalar) per slot, with EMA-teacher (momentum=0.996). Phase 2 attaches multi-task heads (win + dur + items + kda + gpm + hd, same as v4) and fine-tunes from the pre-trained encoder with low encoder LR. A linear-probe arm is run first as a diagnostic. Targets: linear-probe val_auc ≥ 0.6300 (representation is non-trivially informative); full fine-tune val_auc ≥ 0.6485 (foundation closes ≥50% of the v4 → iso_teambias gap of 0.0022); ≥ 0.6493 (foundation beats iso_teambias on extended data, real win). Closes the open question 'does pre-train + fine-tune help, or is v4-style joint training already optimal?'"
rationale: >
  v4-iso-teambias-extended-740 (val_auc=0.6471 @ epoch 16) resolved
  the v3 regression cleanly: extended-data penalty -0.0022 + composition
  penalty -0.0009 = -0.0031. v4 is the strongest joint-trained baseline
  on extended data, but it is NOT a foundation model in the strict sense:
  it lacks any self-supervised / pre-training stage and depends entirely
  on labelled supervision for every task.

  The user's framing is foundation-model: trained on as much data as
  possible, producing reusable representations that can be queried for
  many downstream tasks (item rec across patches, hero-pair synergy,
  lineup-vs-lineup, fun-pair max-kills). v3 attempted this via JOINT
  multi-task + PMAE training but the composition cost outweighed the
  PMAE benefit on win val_auc (-0.0009).

  This experiment tests the classical alternative: SEQUENTIAL pre-train
  → fine-tune. Two arms in fine-tune isolate where any lift comes from:

  - **Linear probe** answers "is the PMAE-pretrained encoder
    representation alone useful for win prediction?" If yes → the
    foundation pattern is producing genuine signal. If near-random
    → pre-training didn't learn discriminative features.
  - **Full fine-tune** answers "does pre-train + fine-tune beat v4's
    joint training?" If yes → foundation pattern beats joint training;
    extended-data + foundation framing is justified. If no → joint
    training is already optimal and pre-train adds no value (informative
    null).

  KDA, GPM, HD are included as masked input groups during pre-train
  (the user's explicit choice). The encoder learns to predict per-player
  post-game metrics from pre-game features alone (when those groups
  are fully masked, mimicking the inference distribution). This is the
  most "foundation-like" framing — the encoder learns ALL the structure
  of a match, not just hero compositions and player history. HCE is
  preserved because at fine-tune AND inference time KDA/GPM/HD are
  fully masked / absent.

  This is also a real-cost experiment (~12.5h) compared to a cheap
  feature-engineering ablation. The user explicitly chose the highest-
  fidelity option to test the foundation hypothesis. The compute cost
  is justified by the binary nature of the result: either foundation
  pays off (and v6+ builds on it) or joint training is optimal
  (and we should stop calling v4 anything other than a multi-task
  supervised model).
reads:
  - "[[experiments/2026-05-25-v4-iso-teambias-extended-740]]"
  - "[[experiments/2026-05-25-v3-ablations-740]]"
  - "[[experiments/2026-05-24-foundation-v3-740]]"
  - "[[experiments/2026-05-23-foundation-component-isolation-740]]"
  - "[[experiments/2026-05-20-rich-supervision-multitask-740]]"
  - "[[concepts/tabular-foundation-model]]"
  - "[[concepts/masked-modeling-tabular]]"
  - "[[concepts/embedding-vs-features-gradient-competition]]"
  - "[[literature/papers/kim2024predict]]"
  - "[[literature/papers/gorishniy2021revisiting]]"
  - "[[mocs/foundation-models]]"
expected_metric:
  name: val_auc
  target: 0.6485
  direction: "higher-is-better (linear probe ≥ 0.6300; full fine-tune ≥ 0.6485 meaningful lift; ≥ 0.6493 beats iso_teambias on extended data)"
design_sketch:
  - "**Three sequential phases**, single experiment folder:"
  - ""
  - "## Phase 1 — Self-supervised pre-train (~6h)"
  - "- **Architecture**: FT-Transformer skeleton matching v4 (d_model=256, n_heads=8, n_layers=6, FFN=4×d_model, Pre-Norm, first-layer first-LN removed, (team_query, team_key) 2×2 attention bias, canonical hero sort at load time)."
  - "- **Data**: extended player_features + rich_cols sidecar parquets (REUSED from v3/v4 — no rebuild)."
  - "- **Inputs per slot during pre-train**:"
  - "  - player_block (8-dim features, as in v4)"
  - "  - hero_token (categorical, 130 hero vocab)"
  - "  - item_list (305-dim multi-label, from rich_cols)"
  - "  - kda_scalar = (K+A)/max(D,1) — 1-dim continuous"
  - "  - gpm_scalar — 1-dim continuous"
  - "  - hd_scalar (log1p) — 1-dim continuous"
  - "  - Total per slot: ~448-dim equivalent. 10 slots + (team,team) bias + maybe a CLS token."
  - "- **Mask schedule (group-level random masking)**: for each training example, independently mask each of the 6 groups with probability `p_group=0.4`. Masked groups are replaced with a per-group learned mask token (NOT zero). Reproduces the inference distribution where item_list/kda/gpm/hd are FULLY absent."
  - "- **EMA-teacher** (proven safe in iso_pmae): deep-copy encoder, momentum=0.996, stop-gradient on teacher path. Student sees masked input, predicts teacher's representations of unmasked input."
  - "- **Per-group reconstruction losses**:"
  - "  - player_block: SmoothL1"
  - "  - hero_token: cross-entropy (130-way)"
  - "  - item_list: BCE per-class (305 sigmoids)"
  - "  - kda/gpm/hd scalars: SmoothL1"
  - "  - Total loss = sum of per-group losses, weighted equally initially. Implementer may tune if any group dominates."
  - "- **Training recipe**: Adam lr=1e-3, 1k-step warmup → cosine to 1e-5, batch_size=512, max_epochs=20 (fixed — no val_auc to early-stop on), bf16 autocast. Track val PMAE reconstruction loss per group as the convergence signal."
  - "- **Mid-pretrain linear probe (NEW)**: every 5 epochs (epochs 5, 10, 15, 20), pause pre-training and run a fast linear-probe diagnostic: take encoder snapshot, freeze it, train a fresh linear win head on a SMALL train subset (50k rows, 3 epochs, lr=1e-2), eval on a SMALL val subset (20k rows). Report mid_probe_val_auc per snapshot. Adds ~60-120s per probe (~4-8 min total over 4 probes). Diagnostic value: if mid_probe_val_auc trajectory is monotone-increasing toward 0.60+, encoder is learning win-discriminative features; if it stays at ~0.50, pre-training is misaligned with win prediction and we can halt early instead of burning the full 6h. The fresh linear head is discarded after each probe — the encoder continues pre-training unmodified."
  - "- **Output**: encoder checkpoint `pretrain_encoder.pt` at last epoch, plus a `pretrain_history.json` with per-epoch per-group losses AND `mid_probe_history.json` with the 4 mid-pretrain probe val_aucs."
  - ""
  - "## Phase 2A — Linear probe (~0.5h)"
  - "- **Load** encoder from `pretrain_encoder.pt`, freeze ALL encoder parameters."
  - "- **Inputs at probe time** match v4/iso_teambias: only pre-game info (player_block, hero_token). Item_list/kda/gpm/hd are FULLY MASKED (replaced with the per-group mask tokens learned during pre-train)."
  - "- **Single linear win head** on the encoder's pooled CLS-like representation. No multi-task heads."
  - "- **Training**: 5 epochs at lr=1e-2 on win loss only. Eval on val_auc each epoch, keep best."
  - "- **Output**: `metrics_linear_probe.json` with val_auc trajectory + best."
  - ""
  - "## Phase 2B — Full multi-task fine-tune (~6h)"
  - "- **Load** encoder from `pretrain_encoder.pt`, UNFREEZE everything."
  - "- **Inputs match v4**: player_block + hero_token only (item_list/kda/gpm/hd fully masked at the encoder, but their PREDICTION heads are still trained from rich_cols targets)."
  - "- **Multi-task heads** (same as v4): win + dur (8-bucket CE) + items (305-way BCE) + kda (SmoothL1) + gpm (SmoothL1) + hd (SmoothL1)."
  - "- **Multitask α** (same as v4): win=1.0, dur=0.15, item=0.3, kda=0.1, gpm=0.1, hd=0.1. No α_mae (PMAE not used during fine-tune — classical BERT recipe)."
  - "- **Optimizer**: AdamW with parameter groups: encoder lr=1e-5, heads lr=1e-3, weight_decay=0 on encoder (preserve pre-trained representations), weight_decay=1e-4 on heads. 1k-step warmup → cosine to 1e-7 (encoder) / 1e-5 (heads)."
  - "- **Training**: max_epochs=20, early-stop patience=5 on val_win_log_loss, bf16 autocast, batch_size=512."
  - "- **Output**: `metrics_finetune.json` with full v4-style metrics block (val_auc, per-epoch trajectory, val_metrics_at_best, coverage_bucket_val_auc, patch_id distribution, deltas vs all anchors)."
  - ""
  - "## Pipeline orchestration"
  - "- `run_all.sh` runs Phase 1 → Phase 2A → Phase 2B sequentially. Each phase gates the next on rc=0."
  - "- Smoke (1 epoch, 50k rows) on Phase 1 + Phase 2B before launching full."
  - "- Per-trial subprocess retry wrapper (defensive)."
  - "- `python -u` mandatory throughout."
  - "- Wall budget: 6h pretrain + 0.5h probe + 6h finetune = ~12.5h. Within budget.yaml's 24h ceiling."
  - ""
  - "## Diagnostics (in addition to standard)"
  - "- Per-epoch per-group PMAE reconstruction loss (Phase 1 — proves encoder is learning each group, not just collapsing)."
  - "- Encoder weight L2-norm change pre/post-finetune (Phase 2B — catastrophic forgetting check; if L2 changes by > 50%, encoder LR may be too high)."
  - "- coverage_bucket_val_auc on linear probe AND fine-tune (do the buckets respond differently to the foundation pattern?)."
  - "- Anchors block in each metrics.json: v4=0.6471, iso_teambias=0.6493, v3=0.6462, cleanup=0.6477, baseline_multitask_repro=0.6470, iso_pmae=0.6464."
risks:
  - "**Pre-train doesn't converge / PMAE losses plateau at high values** — encoder didn't learn anything useful. Symptoms: per-group PMAE val losses don't decrease after epoch 5. Mitigation: implementer logs per-group losses each epoch; if plateau detected, halt and investigate (probably mask schedule is too aggressive or per-group loss weighting is unbalanced)."
  - "**Linear probe near random (val_auc ≤ 0.55)** — encoder didn't learn win-discriminative features even though PMAE reconstruction worked. Informative but unfortunate: tells us PMAE objective doesn't align with win prediction on this data. Full fine-tune may still recover via the unfrozen encoder, so still run Phase 2B."
  - "**Full fine-tune catastrophic forgetting** — encoder LR too high, loses pre-trained representation, ends up like v4. Mitigation: encoder lr=1e-5 (much lower than head lr=1e-3); weight_decay=0 on encoder; log L2-norm change pre/post-finetune."
  - "**Full fine-tune ≤ v4 (≤ 0.6471)** — pre-train + fine-tune doesn't beat joint training. INFORMATIVE NULL: tells us that the foundation pattern, with our specific PMAE setup on extended data, doesn't add value over v4. Next experiment would be either (a) richer pre-train (more groups, longer training, different mask schedule), (b) different self-supervised objective (contrastive, JEPA), or (c) accept v4 as the ceiling and pivot to richer engineered features."
  - "**Train-val distribution shift (the v3/v4 concern, repeated)** — pre-train sees multi-patch corpus including KDA/GPM/HD; val is single-patch. The patch_id token is NOT included this time (matches v4). Pre-train representations may overfit to multi-patch metas, hurting val transfer. Risk is real but not addressable in this experiment; if v5 underperforms v4, this is the leading hypothesis for v6 (patch-aware pre-training)."
  - "**Compute risk**: HIGH (~12.5h is the biggest single experiment in the project). Mitigation: run sequentially with checkpointing between phases (so if Phase 2B crashes, we don't lose Phase 1's encoder). Live-monitor per `~/.claude/CLAUDE.md`."
related_prior:
  - 2026-05-25-v4-iso-teambias-extended-740
  - 2026-05-25-v3-ablations-740
  - 2026-05-24-foundation-v3-740
  - 2026-05-23-foundation-component-isolation-740
  - 2026-05-20-rich-supervision-multitask-740
estimated_runtime: "≈12.5h end-to-end on RTX 5080: 6h pre-train + 0.5h linear probe + 6h full fine-tune. No new data builds (reuses v3-built extended parquets). Within budget.yaml's 24h ceiling. Largest single experiment in the project; checkpoint between phases for crash recovery."
---

# v5-pretrain-finetune-740 — classical foundation-model pattern, finally

## Where this fits

v4 closed the diagnostic arc on v3's regression (outcome b: 70% data
+ 30% composition). v4 = 0.6471 is the strongest joint-trained
baseline on extended data, but it is NOT a foundation model in the
strict sense — it lacks any self-supervised / pre-training stage.

This experiment tests the classical foundation pattern (BERT-style
pre-train then fine-tune) on the multi-patch extended corpus. The
user explicitly chose the highest-fidelity option:
- Pre-train PMAE over SIX input groups (player_block + hero_token +
  item_list + KDA + GPM + HD per slot), masking each independently
  at p=0.4 to reproduce the inference distribution.
- Two fine-tune arms: linear probe (frozen encoder, win-only head)
  and full multi-task fine-tune (unfrozen, all v4 heads).

## Decision tree

```
Phase 2A (linear probe) val_auc?
├── ≥ 0.6400 → encoder learned strong representation; foundation viable
├── [0.6300, 0.6400) → encoder learned some signal; full fine-tune is
│                       where the comparison happens
├── [0.5500, 0.6300) → weak signal; foundation pattern marginal
└── < 0.5500 → encoder didn't learn discriminative win features;
               possible PMAE objective misalignment

Phase 2B (full fine-tune) val_auc?
├── ≥ 0.6500 → foundation pattern clearly pays off; v6 builds on this
├── [0.6493, 0.6500) → beats iso_teambias on extended data; foundation
│                       worth the compute cost
├── [0.6471, 0.6493) → matches or slightly beats v4; foundation
│                       neutral-to-positive; downstream-query benefit
│                       is the marginal value (need separate eval)
└── < 0.6471 → pre-train + fine-tune actively hurts; joint training
               (v4-style) is the ceiling on extended data
```

## Anchor table

| Reference | val_auc | Pattern |
|---|---|---|
| iso_teambias (7.40-only) | 0.6493 | multi-task joint, 7.40-only |
| multitask-740 | 0.6495 | multi-task joint, 7.40-only |
| cleanup-740 | 0.6477 | transformer + features, 7.40-only |
| baseline_multitask_repro | 0.6470 | foundation-mvp's clean baseline |
| **v4 (target)** | **0.6471** | multi-task joint, extended |
| iso_pmae | 0.6464 | + PMAE, 7.40-only |
| **v3 (joint PMAE)** | **0.6462** | full stack joint, extended |
| v5 linear probe | ? | PMAE pre-train + frozen probe |
| v5 fine-tune | ? | PMAE pre-train + multi-task fine-tune |

## Live monitoring

Per `~/.claude/CLAUDE.md`. Phase 1 has limited but real val signal via
the mid-pretrain linear probes (epochs 5/10/15/20):
- **Phase 1 halt signals**: per-group reconstruction loss INCREASING
  for 3+ epochs (anti-learning); NaN/Inf; kernel events; GPU stall;
  **mid_probe_val_auc still at ~0.50 random after the epoch-10 probe**
  (encoder is not learning win-discriminative features — pre-training
  is misaligned, halt and rethink mask schedule / per-group loss
  weights instead of burning the full 6h).
- **Phase 2A/2B halt signals**: standard — val_auc at random for 3+
  epochs, NaN/Inf, train_win INCREASING while vl_win UP (overfit),
  kernel events.

## Out of scope (deferred to v6+)

- Rich engineered features (the originally-drafted v5-rich-skill — if
  v5-pretrain-finetune wins, v6 layers rich features on top).
- Patch-aware pre-training (separate patch-conditioned mask tokens).
- Pre-training on a longer time window (no test-window data; same
  extended corpus).
- HCE final-scoring pass (sealed test window — never touched).
