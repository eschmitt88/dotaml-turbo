---
kind: proposal
slug: v7-unified-masked-multitask-740
date: 2026-05-26
status: implemented
experiment: experiments/2026-05-26-v7-unified-masked-multitask-740/
hypothesis: "A unified masked-multi-task foundation that supports ALL specified downstream queries (personal win prob, hero pick rec with partial drafts, item rec with win-conditional comparison, win rate vs duration, kills/min pairs, lineup matchup) in a single model. Architecture: v4's FT-Transformer encoder + decoder, extended so every input group (heroes, player_feats, items, kills/deaths/assists separately, gpm, hd, duration scalar, win) is MASKABLE. Trained via 9-scenario sampling (each batch picks one scenario; scenarios are designed to match inference use cases — pure_pregame, partial_draft, partial_items, duration_cond, items_cond, outcome_cond, kills_pair_probe, random_uniform, everything_visible). Per-scenario loss weights and a per-scenario mid-pretrain probe suite (9 probes) provide an adaptive training signal. The v4-style multi-task supervised loss is ALWAYS computed (the strong anchor that v5/v6 lacked). Target: a single foundation that natively answers every query in the downstream menu without architectural hacks. val_auc on pure_pregame inference path ≥ 0.6471 (matches v4); val_auc on items_cond path ≥ 0.80 (items as input is highly informative)."
rationale: >
  Step-back analysis of the downstream-query menu revealed v4 is not
  the right foundation for the queries the user actually wants:
  - Hero pick rec with partial drafts needs trained mask tokens
    (v4 only ever saw full drafts).
  - Item rec for win optimization needs items as INPUT (v4 only has
    items as output).
  - Item rec conditional on winning needs win as INPUT (v4 only has
    win as output).
  - Win rate vs duration needs duration as INPUT (v4 only has duration
    as output).
  - Kills/min queries need raw kills (v4 has only KDA composite).

  Common pattern: every query reduces to "given some subset of inputs
  + outputs as KNOWN, predict the rest." This is literally
  masked-prediction language modeling on tabular data.

  The v4 diagnostic confirmed the FT-Transformer encoder + decoder
  architecture is sound (hero embeddings cluster by role; encoder
  organizes matches by predicted outcome along PCA-1 = +0.98 corr
  with win_pred). The 0.647 ceiling is data-bound, not architecture-
  bound. v5/v6 SSL pretraining failed because they tried to use ONLY
  the masked-prediction signal (no strong anchor) — v5 over-specialized
  to per-token reconstruction; v6 collapsed to predictable
  low-magnitude latent reps.

  v7 fixes this by combining BOTH:
  - v4's multi-task supervised heads (the strong anchor, always
    computed) — anchors the encoder to win-discriminative features.
  - Masking augmentation (random subset of inputs masked per batch) —
    teaches the model to handle partial inputs at inference time.

  The 9-scenario sampling distribution explicitly matches training to
  inference use cases: each downstream query corresponds to a
  specific scenario, and the model gets distinct training signal for
  each. Per-scenario mid-pretrain probes (run every 2 epochs) provide
  the adaptive feedback loop: scenarios where the probe is weak get
  their sampling probability up-weighted in subsequent epochs (capped
  at 2× initial).

  Per-scenario loss weights additionally bias optimization toward the
  most-important queries (pure_pregame for personal win prob is
  weighted 2× on the win head).

  Three architectural changes vs v4:
  1. K, D, A as separate heads (not composite KDA) — enables true
     kills/min queries.
  2. Duration as scalar regression (not 8-bucket CE) — v3 showed
     scalar works better in more-complex architectures; v7 is more
     complex than v4.
  3. Items, duration, win, and per-slot stats become INPUTS as well
     as outputs (with learned mask tokens for the "unknown" state) —
     enables the conditional and partial-draft queries.

  Risks (real, learned from v5/v6):
  - Masking augmentation could weaken the encoder on full-input
    queries. Mitigation: 15% of batches are everything_visible
    (pure supervised, the v4 baseline). Per-scenario probe catches
    drift on pure_pregame.
  - Scenario-sampling could over-emphasize easy scenarios. Mitigation:
    adaptive sampling probability based on probe shortfall.
  - V5/v6 pattern (encoder finds degenerate shortcut) could repeat.
    Mitigation: ALWAYS-computed supervised heads anchor representations
    to be useful for win prediction; mid-pretrain probe suite halts
    at first sign of stagnation.
  - Items-as-input might leak post-game info during training. Honest:
    yes, that's the design — items are sometimes given as input
    (items_cond scenario) so the model learns the conditional
    P(win | items). At inference time, queries that don't know items
    mask them; queries that DO want to condition on items provide them.
    HCE is preserved because all of this is train+val only, never test.
reads:
  - "[[experiments/2026-05-25-v4-iso-teambias-extended-740]]"
  - "[[experiments/2026-05-26-v6-jepa-pretrain-finetune-740]]"
  - "[[experiments/2026-05-26-v5-pretrain-finetune-740]]"
  - "[[experiments/2026-05-24-foundation-v3-740]]"
  - "[[experiments/2026-05-20-rich-supervision-multitask-740]]"
  - "[[_meta/deferred-foundation-paths]]"
  - "[[concepts/tabular-foundation-model]]"
  - "[[concepts/masked-modeling-tabular]]"
  - "[[concepts/embedding-vs-features-gradient-competition]]"
  - "[[mocs/foundation-models]]"
expected_metric:
  name: val_auc_pure_pregame
  target: 0.6471
  direction: "matches-or-exceeds-v4 on pure_pregame path AND items_cond path ≥ 0.80 AND duration_cond path ≥ 0.68 — all three required for v7 to be considered a successful foundation"
design_sketch:
  - "**Reuse v4 codebase.** Fork `experiments/2026-05-25-v4-iso-teambias-extended-740/` as the base. Significant architectural changes below."
  - ""
  - "## Architecture changes vs v4"
  - ""
  - "### Per-slot input groups (each MASKABLE; ~10 slots each)"
  - "- `hero_token` — categorical (vocab=130 heroes + 1 mask). Embedding lookup as in v4 + add a learned `hero_mask_embed` of shape (d_model,)."
  - "- `player_feat_block` — 8-dim continuous (the existing player features). Project via `Linear(8, d_model)` as in v4 + add a learned `player_feat_mask_embed`."
  - "- `items_set` — 305-dim sparse multi-hot (up to 6 items per slot). Use `item_embed = nn.Embedding(305, d_model)` and SUM embeddings for items in bag, scaled by 1/sqrt(K) for K items. Add a learned `items_mask_embed`."
  - "- `kills` — 1-dim continuous scalar. Project via `Linear(1, d_model)`. Add a learned `kills_mask_embed`."
  - "- `deaths` — same pattern. Separate head and mask token."
  - "- `assists` — same pattern."
  - "- `gpm` — same pattern (log1p-transform target during training for stability)."
  - "- `hd` — same pattern (log1p-transform target)."
  - ""
  - "### Per-match input groups (each MASKABLE; 1 each)"
  - "- `duration` — 1-dim continuous scalar (log-seconds). Project via `Linear(1, d_model)`. Add a learned `duration_mask_embed`."
  - "- `win` — categorical (vocab=2 + 1 mask). Embedding lookup + add a learned `win_mask_embed`."
  - ""
  - "### Per-slot final input embedding"
  - "  `slot_token[s] = hero_embed[s] + player_feat_proj[s] + items_pooled[s] + kills_proj[s] + deaths_proj[s] + assists_proj[s] + gpm_proj[s] + hd_proj[s]`"
  - "  (where each term is replaced by the corresponding `*_mask_embed` if that group was masked for that slot)."
  - ""
  - "### Per-match additional tokens"
  - "  Two extra tokens (positions 10 and 11): `duration_proj` (or mask), `win_embed` (or mask). Prepended/appended to the 10 slot tokens. Total sequence length = 12 tokens."
  - ""
  - "### Decoder + task heads (extends v4)"
  - "- `win_head`: Linear(d_model, 1) → BCEWithLogits. ONE task token (TASK_WIN)."
  - "- `dur_head`: Linear(d_model, 1) → SmoothL1 on log-seconds. ONE task token (TASK_DUR). (Note: regression, NOT 8-bucket CE — change from v4.)"
  - "- `items_head`: Linear(d_model, 305) → BCEWithLogits per-class. 10 task tokens (TASK_ITEMS_0..9)."
  - "- `kills_head`: Linear(d_model, 1) → SmoothL1. 10 task tokens (TASK_KILLS_0..9). (NEW vs v4: separate from D and A.)"
  - "- `deaths_head`: Linear(d_model, 1) → SmoothL1. 10 task tokens (TASK_DEATHS_0..9). (NEW.)"
  - "- `assists_head`: Linear(d_model, 1) → SmoothL1. 10 task tokens (TASK_ASSISTS_0..9). (NEW.)"
  - "- `gpm_head`: same as v4 (10 task tokens, SmoothL1)."
  - "- `hd_head`: same as v4 (10 task tokens, SmoothL1)."
  - "- Total task tokens: 1 + 1 + 10×6 = 62 (up from v4's 42)."
  - ""
  - "### Total new params vs v4"
  - "- ~10 mask embeddings × 256 dim = ~2.5k params"
  - "- 4 new continuous-input projections (kills, deaths, assists, duration) × 256 = ~1k each"
  - "- 305-dim item embedding table: 305 × 256 = ~78k params"
  - "- 3 new task tokens (kills, deaths, assists) × 10 × 256 = ~8k"
  - "- 3 new heads (kills, deaths, assists separated from KDA composite) × 256 = ~800"
  - "- Total: ~95k new params on top of v4's 6.5M. Negligible."
  - ""
  - "## 9-scenario sampling (the heart of v7 training)"
  - ""
  - "Each batch samples ONE scenario from the categorical distribution. The same masking pattern applies to all examples in the batch. This makes per-scenario probe metrics clean (each scenario gets distinct batches; loss + probe trajectories are separable)."
  - ""
  - "Initial sampling distribution + loss weights (these are TUNABLE; adaptive update every 2 epochs):"
  - ""
  - "| Scenario | What's masked | Init sample prob | Init loss weights |"
  - "|---|---|---|---|"
  - "| `everything_visible` | nothing | 15% | win 2.0, others 1.0 |"
  - "| `pure_pregame` | items, k, d, a, gpm, hd, dur, win | 30% | win 2.0, others 1.0 |"
  - "| `partial_draft` | 1-5 random hero slots + all post-game | 15% | win 1.5, others 1.0 |"
  - "| `partial_items` | 1-3 items per slot + post-game | 10% | items 2.0, others 1.0 |"
  - "| `duration_cond` | items, k, d, a, gpm, hd, win | 8% | win 1.5, others 1.0 |"
  - "| `items_cond` | k, d, a, gpm, hd, dur, win | 8% | win 1.5, others 1.0 |"
  - "| `outcome_cond` | items, k, d, a, gpm, hd, dur (UNMASK win at true value) | 5% | items 2.0, others 1.0 |"
  - "| `kills_pair_probe` | all heroes except 1-2 ally pair + all post-game | 5% | kills 1.5, dur 1.5, others 1.0 |"
  - "| `random_uniform` | each group independently masked at rate ~ Beta(2, 4), median ~0.33 | 4% | all 1.0 |"
  - ""
  - "All scenarios ALWAYS compute losses on all heads (multi-task supervised anchor). Loss weights modulate the per-head loss before summation. Some scenarios HIDE the corresponding input from the encoder (mask token replaces the value) but the supervision target for the corresponding output head is still the true value."
  - ""
  - "## Mid-pretrain probe suite (per-scenario, run every 2 epochs)"
  - ""
  - "Each probe is a small held-out eval on a 50k-row val subset, runs in ~30 seconds. Probe set runs in ~5 min total. Per-scenario probes:"
  - ""
  - "| Probe | Metric | Initial target | Halt-if-stuck-at |"
  - "|---|---|---|---|"
  - "| `pure_pregame_probe` | val_auc on win head with pure_pregame masking | 0.6471 (v4) | ≤ 0.55 at ep10 |"
  - "| `partial_draft_probe` | top-5 hero rec accuracy (true hero in model's top-5 for a randomly-masked slot) | 30% | ≤ 15% at ep10 |"
  - "| `duration_cond_probe` | val_auc on win head with true duration as input | 0.68 (duration is post-hoc) | ≤ 0.60 at ep10 |"
  - "| `items_cond_probe` | val_auc on win head with true items as input | 0.80 (items strongly predictive) | ≤ 0.70 at ep10 |"
  - "| `outcome_cond_probe` | item mAP@10 conditioning on true win | 0.40 | ≤ 0.30 at ep10 |"
  - "| `partial_items_probe` | held-out item BCE on masked items | converges | not improving |"
  - "| `kills_pair_probe_probe` | kills SmoothL1 MAE for kills_pair_probe scenario val rows | < 2.5 kills | > 4.0 at ep10 |"
  - "| `gpm_probe` | gpm SmoothL1 MAE for full-input val | < 30 GPM | > 50 at ep10 |"
  - "| `hd_probe` | hd_log1p SmoothL1 MAE for full-input val | < 0.30 (log scale) | > 0.50 at ep10 |"
  - ""
  - "Halt criterion: any probe below its halt threshold at epoch 10 → halt the run, log the diagnostic, surface to user for design adjustment."
  - ""
  - "## Adaptive sampling probability update (every 2 epochs)"
  - ""
  - "After each probe suite run:"
  - "- For each scenario, compute `gap = max(0, target - probe_value)` (positive if below target)."
  - "- If `gap > 0.02` (below target by >2pp): multiply sampling prob by 1.2 (cap at 2× initial)."
  - "- If `gap < -0.02` (above target by >2pp): multiply sampling prob by 0.95 (floor at 0.5× initial)."
  - "- Re-normalize so all sampling probs sum to 1.0."
  - "- Log the updated distribution to `adaptive_sampling_history.json` for transparency."
  - ""
  - "Initial loss weights are NOT adaptively tuned — just sampling probability. (Keep one knob; otherwise dynamics get muddy.)"
  - ""
  - "## Training recipe"
  - "- Adam lr=1e-3, 1k-step warmup → cosine to 1e-5"
  - "- batch_size=512, max_epochs=25, early-stop patience=8 on pure_pregame_probe val_auc"
  - "- bf16 autocast"
  - "- Per-trial subprocess isolation retry wrapper (defensive)"
  - "- `python -u` mandatory"
  - "- Wall budget: 10-15h (heavier per-batch than v4; longer training to cover all scenarios)"
  - ""
  - "## Data"
  - "- REUSE extended player_features + rich_cols sidecar parquets verbatim from v3/v4 (no data rebuild needed)."
  - "- Train: 2025-08-15 → 2026-02-23 (extended cross-patch)."
  - "- Val: 2026-02-24 → 2026-03-09 (held out)."
  - "- Test: [2026-03-10, 2026-03-23] SEALED — never touched."
  - "- Item vocab: existing 305-item table from rich-supervision-multitask-740."
  - ""
  - "## Outputs / files in experiment folder"
  - "- `pretrain_encoder.pt` — final trained model checkpoint"
  - "- `pretrain_history.json` — per-epoch per-scenario loss + sampling distribution + adaptive updates"
  - "- `probe_suite_history.json` — per-2-epoch probe metrics, all 9 probes"
  - "- `metrics_v7.json` — final canonical metrics on the pure_pregame inference path (for comparison to v4 anchor)"
  - "- `serve/v7_inference.py` — load model + maskable forward function. The core wrapper."
  - "- `serve/lookups.py` — account_id → player_features, hero ID ↔ name, item ID ↔ name (OpenDota constants)."
  - "- `serve/queries.py` — concrete query functions (personal_winprob, hero_pick_rec, item_rec, win_vs_duration, kills_per_minute_pair, lineup_matchup)."
  - "- `serve/notebook.qmd` — interactive notebook with example queries on the user's account (3303652)."
risks:
  - "**Same v5/v6 pattern in disguise**: encoder finds a degenerate shortcut. PRIMARY mitigation: the multi-task supervised heads are ALWAYS computed and weighted ≥ 1.0 — the strong anchor. Secondary mitigation: 15% of batches are everything_visible (pure supervised, exactly v4). Tertiary: per-scenario probe suite catches drift early."
  - "**Mask augmentation degrades full-input performance**: model becomes robust to missing inputs but weaker on the full-input pure_pregame query. Halt criterion: pure_pregame_probe ≤ 0.55 at epoch 10 triggers halt. Acceptable lower bound: pure_pregame_probe within 0.005 of v4=0.6471 at end of training (otherwise the foundation is worse than v4 on its core query)."
  - "**Scenario imbalance**: easy scenarios (everything_visible) over-dominate, hard scenarios (partial_draft) under-train. Mitigation: adaptive sampling re-weighting based on probe shortfall."
  - "**Items-as-input is hard to learn**: 305-dim multi-hot input from sparse data may not flow useful gradients. Mitigation: item embedding table (305 × 256) is the same architecture as our existing hero embedding (130 × 256) which we KNOW learns useful structure. Items_cond probe targets val_auc ≥ 0.80 — well-separated from random (0.5), so if it's working we'll see it."
  - "**K, D, A heads might be harder than KDA composite**: composite is smoother (averages out noise); separates expose noise. Mitigation: per-head SmoothL1 (robust to outliers); kills/deaths/assists are integer-valued so we can also try MSE."
  - "**Scenario probe metrics could be noisy**: 50k-row probe with random masking has variance. Mitigation: use FIXED probe val subset + seed (no per-call randomness on the probe data). Adapt sampling probs only if gap > 2pp (avoid noise-driven updates)."
  - "**Compute risk**: 10-15h, longer than v4. If pure_pregame_probe shows good trajectory at ep6, the run is on track; if not, halt at ep10 saves the rest. Worst case: ~3h wasted if early halt."
  - "**Live-monitoring discipline**: probe suite every 2 epochs is the diagnostic. Halt early on ANY of the per-scenario halt criteria."
related_prior:
  - 2026-05-25-v4-iso-teambias-extended-740
  - 2026-05-26-v6-jepa-pretrain-finetune-740
  - 2026-05-26-v5-pretrain-finetune-740
  - 2026-05-24-foundation-v3-740
  - 2026-05-20-rich-supervision-multitask-740
estimated_runtime: "≈12h end-to-end on RTX 5080. No new data builds (reuses v3-built extended parquets). Within budget.yaml's 24h ceiling. Halt criterion at probe-suite epoch 10 limits worst-case risk to ~3h if any scenario probe is stuck."
---

# v7-unified-masked-multitask-740 — the unified foundation

## Where this fits

Downstream-query requirements analysis showed v4 is not the right
foundation: it can't handle partial drafts, items as input, win as
input, duration as input, or separate K/D/A. The user explicitly
asked for "a foundation that will work for all of the specified use
cases."

v7 IS that foundation. Three architectural changes from v4:

1. **Separate K, D, A heads** (not composite KDA) — enables true
   kills/min queries.
2. **Duration as scalar regression** (not 8-bucket CE) — v3 evidence
   shows scalar works better in complex architectures.
3. **All input groups become MASKABLE** with learned mask tokens —
   enables every query as "mask what we don't know, query what we
   want."

Training combines v4's supervised anchor (proven to produce
meaningful representations per the diagnostic) with masking
augmentation (the SSL-style coverage that v5/v6 attempted without
anchor). Per-scenario sampling explicitly maps training to inference
use cases. Per-scenario probe suite catches v5/v6-style failure
modes early.

## Anchor table

| Reference | val_auc | What it represents |
|---|---|---|
| v4 (PRIMARY ANCHOR) | 0.6471 | pure_pregame query, full-input multi-task supervised |
| iso_teambias | 0.6493 | 7.40-only ceiling |
| baseline_multitask_repro | 0.6470 | foundation-mvp baseline |
| **v7 pure_pregame target** | **≥ 0.6471** | matches v4 on its core query |
| **v7 items_cond target** | **≥ 0.80** | items-as-input gives massive lift (items predictive of win) |
| **v7 duration_cond target** | **≥ 0.68** | duration-as-input modest lift |

## Decision tree

```
Probe suite at epoch 6:
├── pure_pregame ≥ 0.62 AND items_cond ≥ 0.70 AND others above halt threshold
│   → on track; continue training
├── pure_pregame ≤ 0.55 OR items_cond ≤ 0.55
│   → halt; encoder isn't learning core queries; review scenario weights
└── most probes above halt but partial_draft stuck
    → halt + adjust partial_draft sampling weight upward

Final result at epoch 20-25:
├── pure_pregame ≥ 0.6471 (matches v4)
│   AND items_cond ≥ 0.80
│   AND duration_cond ≥ 0.68
│   → v7 IS the foundation; ship downstream queries
├── pure_pregame matches v4 but conditional probes weak
│   → foundation works for queries that don't need conditioning;
│     train auxiliary heads for the conditional queries
└── pure_pregame < v4
    → masking augmentation hurt the core query; either drop down to
      v4 with hero-mask augmentation only (smaller scope), or
      retrain with higher everything_visible weight
```

## Out of scope (deferred)

- Item progression (early vs late item recommendation): need parsed
  replay data with timing, which we don't have. v7 enables a heuristic
  via item-cost + predicted-gpm + counterfactual-win-prob, but true
  progression remains future work.
- Formal causal inference (DML, propensity weighting): post-foundation
  analysis work, separate from v7.
- HCE final-scoring pass on the sealed test window — never touched
  during search.
