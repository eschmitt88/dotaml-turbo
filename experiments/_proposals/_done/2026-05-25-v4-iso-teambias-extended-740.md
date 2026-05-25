---
kind: proposal
slug: v4-iso-teambias-extended-740
date: 2026-05-25
status: implemented
experiment: experiments/2026-05-25-v4-iso-teambias-extended-740/
hypothesis: "Run the v2-winner architecture `iso_teambias` (baseline_multitask_repro + only the (team_query, team_key) attention bias added — NO PMAE, NO patch token, NO UW-SO, 8-bucket CE duration, multitask α=multitask-740's hand-tuned weights) on the EXTENDED cross-patch training corpus (Aug 2025 → Feb 2026, ~32M rows across 3 patches). This isolates the cross-patch data extension as a single factor on the architecturally-cleanest known-good design. Three possible outcomes attribute v3's 0.0031 regression vs iso_teambias precisely: (a) v4 ≥ 0.6493 → extended data is neutral or helpful; v3's regression came from PMAE-on-extended-data interaction or composition. (b) v4 in [0.6470, 0.6493) → extended data costs a small amount on its own; v3's regression is partly data + partly composition. (c) v4 < 0.6470 → extended data is itself the regression cause; iso_teambias was the best the cross-patch corpus can support and v3's architecture isn't to blame."
rationale: >
  v3-ablations-740 (2026-05-25) ruled out the two most plausible
  alternative causes for v3's regression:

  - **A1 (v3_dur_ce)** showed reverting duration to 8-bucket CE on
    the v3 stack actively HURTS by 0.0113. The duration loss-form
    switch was not the regression cause.
  - **A2 (v3_player_emb)** showed adding 4M player embeddings to v3
    causes catastrophic overfit (val_auc=0.6290, Δ=-0.0172). The
    identity axis remains closed AND embeddings on extended data
    overfit hard (new failure mode documented in
    `[[concepts/embedding-vs-features-gradient-competition]]`).

  Remaining suspects: (1) extended cross-patch data itself; (2)
  PMAE-on-extended-data interaction; (3) composition of multiple
  v3 components. v4 isolates (1) by running the cleanest known-good
  architecture (iso_teambias, the v2 winner at 0.6493 on 7.40-only)
  on the extended corpus.

  iso_teambias was chosen because:
  - It is the highest-val_auc architecture in the project (0.6493),
    beating multitask-740 (0.6495) within noise.
  - It has the smallest delta vs the cleanup anchor (just ~64 params
    of (team,team) bias added), so the comparison vs cleanup-anchor
    on the same data scope is clean.
  - It does NOT include the components most likely to interact with
    extended data: PMAE (auxiliary objective that learns from masked
    multi-patch features), patch_id token (may not generalize to a
    single-patch val), UW-SO (failed at all data scopes).

  Single ablation, ~6h wall, reuses the v3-built extended parquets
  verbatim. No new data build phase.
reads:
  - "[[experiments/2026-05-25-v3-ablations-740]]"
  - "[[experiments/2026-05-24-foundation-v3-740]]"
  - "[[experiments/2026-05-23-foundation-component-isolation-740]]"
  - "[[experiments/2026-05-20-rich-supervision-multitask-740]]"
  - "[[concepts/embedding-vs-features-gradient-competition]]"
  - "[[concepts/tabular-foundation-model]]"
expected_metric:
  name: val_auc
  target: 0.6493
  direction: "higher-is-better (treat outcomes (a)/(b)/(c) above as the diagnostic fork — any clean attribution is success)"
design_sketch:
  - "**Reuse v3-ablations codebase.** Fork `experiments/2026-05-25-v3-ablations-740/` (models.py, train.py, loss.py, mae.py, data.py). Single ablation `v4_iso_teambias_extended` with flags: use_features=true, multitask=true, use_patch_token=FALSE, use_team_team_bias=TRUE, use_pmae=FALSE, use_uw_so=FALSE, dur_loss_mode=ce, use_player_embedding=FALSE."
  - "**Reuse extended data parquets** (no rebuild):"
  - "  • `data/snapshots/7.40-2025-12-16/processed/player_features_extended/{train,val}.parquet`"
  - "  • `data/snapshots/7.40-2025-12-16/processed/rich_cols_extended/{train,val}.parquet`"
  - "  Item vocab from `experiments/2026-05-20-rich-supervision-multitask-740/results/item_vocab.json`."
  - "  No player-embedding sidecar needed (use_player_embedding=False)."
  - "**Loss recipe** (matches multitask-740 / iso_teambias):"
  - "  • α_win = 1.0, α_dur = 0.15, α_item = 0.3, α_kda = 0.1, α_gpm = 0.1, α_hd = 0.1"
  - "  • No PMAE → no α_mae."
  - "  • No patch token → patch_id ignored at input."
  - "  • Duration head = nn.Linear(d_model, 8), F.cross_entropy on bucket-index targets."
  - "**Architecture**: same FT-Transformer skeleton (d_model=256, n_heads=8, n_layers=6, FFN=4×d_model, Pre-Norm, first-layer first-LN removed, canonical hero sort at load time). Single (team_query, team_key) 2×2 attention bias added per attention block."
  - "**Training recipe**: Adam lr=1e-3, 1k-step warmup → cosine to 1e-5, batch_size=512, max_epochs=30, early-stop patience=5 on val_win_log_loss, bf16 autocast. Per-trial subprocess isolation retry wrapper. `python -u` mandatory."
  - "**Wall budget**: ~6h training + 5 min smoke ≈ 6h total. Within budget.yaml's 24h ceiling."
  - "**Diagnostics**:"
  - "  • Per-epoch val trajectory (val_auc, vl_win, dur_top1, item_mAP_at_10, per-task tr losses)."
  - "  • val_metrics_at_best (auc, log_loss, acc, brier, calibration)."
  - "  • coverage_bucket_val_auc (low/medium/high tercile by n_games_log1p)."
  - "  • patch_id distribution (train + val) — confirms cross-patch corpus is multi-patch."
  - "  • delta_vs_v3, delta_vs_iso_teambias, delta_vs_cleanup_anchor, delta_vs_target."
risks:
  - "**Outcome (a): v4 ≥ 0.6493** (extended data is neutral/helpful). Implies v3's regression came from PMAE-on-extended-data interaction OR multi-component composition. Next experiment would isolate PMAE: `v5-pmae-on-iso-teambias-extended-740` adds only PMAE EMA-teacher to v4 on the same data. If v5 < v4, PMAE-extended interaction is the culprit. If v5 ≈ v4, composition with other v3 knobs is."
  - "**Outcome (b): v4 in [0.6470, 0.6493)** (extended data costs a little). Both data extension AND component composition contribute. Probably means accepting a small data-extension penalty is the price of training on a more meaningful corpus, and the downstream-query benefits (item rec, lineup synergy, duration calibration across patches) justify it."
  - "**Outcome (c): v4 < 0.6470** (extended data is the regression cause). iso_teambias's 0.6493 on 7.40-only is the project ceiling, and extending data costs us more than the patch-token's compensatory power can recover. Next: either restrict training back to 7.40-only OR redesign the patch token to actively reweight cross-patch examples."
  - "**Live-monitoring discipline**: per `~/.claude/CLAUDE.md`. Halt early on PATTERN of 3+ consecutive bad epochs (train loss increasing, val_auc at random, NaN, multi-task collapse, kernel events). v3's epoch 6-10 bumpy plateau then recovered to 0.6462; A1's plateau didn't recover (early-stopped at 16). For v4, be patient through epoch 10 then assess."
  - "**Compute risk: low** — single ablation, ~6h. No new data build. No new external dependencies."
related_prior:
  - 2026-05-25-v3-ablations-740
  - 2026-05-24-foundation-v3-740
  - 2026-05-23-foundation-component-isolation-740
  - 2026-05-20-rich-supervision-multitask-740
estimated_runtime: "≈6h on RTX 5080. Reuses v3-built extended parquets — no new data builds. Within budget.yaml's 24h ceiling."
---

# v4-iso-teambias-extended-740 — does the extended data itself cost val_auc?

## Where this fits

The v3 → v3-ablations arc has narrowed v3's 0.0031 regression vs
iso_teambias to one of: extended cross-patch data, PMAE-on-extended
interaction, or composition of multiple v3 components.

v4 isolates the data-extension axis by running the v2-winner
architecture (iso_teambias: simplest, cleanest, highest val_auc on
7.40-only) on the extended corpus. Single ablation. Cheap. Cleanly
attributes.

## Anchor table

| Reference | val_auc | Data | Architecture |
|---|---|---|---|
| iso_teambias (target) | 0.6493 | 7.40-only | multitask + (team,team) bias |
| multitask-740 | 0.6495 | 7.40-only | multitask, no (team,team) bias |
| cleanup-740 | 0.6477 | 7.40-only | Transformer + features only |
| baseline_multitask_repro | 0.6470 | 7.40-only | foundation-mvp's clean baseline |
| iso_pmae | 0.6464 | 7.40-only | + PMAE EMA-teacher |
| **v3** | 0.6462 | **extended** | full foundation stack |
| **v3_dur_ce** | 0.6349 | extended | v3 with duration as CE |
| **v3_player_emb** | 0.6290 | extended | v3 + 4M player embeddings |

## Outcome decision tree

```
v4 result?
├── ≥ 0.6493 → extended data is NEUTRAL or HELPFUL
│              → v3 regression came from PMAE/composition
│              → next: v5-pmae-on-v4 isolates PMAE-extended interaction
│
├── [0.6470, 0.6493) → extended data costs a small amount
│                     → v3 regression is BOTH data + composition
│                     → next: accept the cost and focus on downstream queries
│                       OR pivot to anonymous-aware-modeling (orthogonal axis)
│
└── < 0.6470         → extended data IS the regression cause
                       → iso_teambias's 0.6493 is the project ceiling
                       → next: restrict training to 7.40-only, OR redesign
                         patch token to reweight cross-patch examples
```

## Out of scope

- HCE test/ access (never).
- Rebuilding extended parquets (reuse v3's).
- Adding PMAE, patch token, UW-SO, or player embeddings (those are
  isolated by future experiments, not this one).
- Final-scoring pass on the sealed test window.
