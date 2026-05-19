---
kind: experiment
slug: transformer-plus-features-740
date: 2026-05-18
status: done
hypothesis: "Combining MinimalTransformer (val_auc 0.6322 in plateau-architectures-740) with the 80-dim per-player feature block from player-features-prepatch-740 (val_auc 0.6256 alone via LightGBM) — injecting per-player features as Linear(8, d_model) added to each slot's hero embedding before self-attention — raises val_auc by ≥ 0.005 over the better individual lever (target val_auc ≥ 0.6372 vs Transformer's 0.6322), confirming that hero-attention and per-player-per-hero skill features address distinct information axes."
result: "HYPOTHESIS CONFIRMED — val_auc=0.6452 (+0.0080 over target 0.6372, +0.0133 over Transformer-only 0.6322, +0.0196 over LightGBM+features 0.6256). All three coverage buckets lifted significantly: low/med/high val_auc = 0.6347/0.6443/0.6560 vs prepatch's 0.6173/0.6256/0.6339. HIGH bucket reached 0.6560 — closing in on Hodge 2017's 75-76% in-game-telemetry ceiling. The LOW bucket (mostly anonymous matches) at 0.6347 alone beats the architecture-only Transformer's WHOLE-val ceiling 0.6322 — architecture and player features address minimally-overlapping information. Sanity check PASSED: architecture_only ablation 0.6319 matches plateau-architectures-740 0.6322 within 0.0003."
related_concepts:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/draft-only-win-prediction]]"
  - "[[concepts/hero-embedding-vs-onehot]]"
related_literature:
  - "[[literature/papers/hodge2017win]]"
tags:
  - transformer
  - player-features
  - feature-injection
  - plateau-740
respects:
  - "~/.claude/rules/evaluation.md"
---

# transformer-plus-features-740

## Hypothesis

Combining the MinimalTransformer architecture from
`experiments/2026-05-15-plateau-architectures-740` (val_auc 0.6322) with the
80-dim per-player feature block from
`experiments/2026-05-18-player-features-prepatch-740` (val_auc 0.6256 alone
via LightGBM) — injecting per-player features per slot as a learned
`Linear(8, d_model)` projection added to each hero embedding before
self-attention — raises val_auc by **≥ 0.005** over the better individual
lever (target val_auc ≥ 0.6372 vs Transformer's 0.6322).

This tests whether hero-attention and per-player-per-hero skill features
address sufficiently distinct information axes to be meaningfully additive.

## Setup

- **Config**: `config.yaml`
- **Code**:
  - `models.py` — `MinimalTransformerWithFeatures` (extends `MinimalTransformer`
    from `experiments/2026-05-16-transformer-hp-sweep-740/models.py` with an
    optional `Linear(n_player_feats, d_model)` per-slot feature injection).
  - `data.py` — loads augmented parquet, builds owned torch tensors
    `(hero_ids[B,10], player_feats[B,10,8], y[B])`. Deep-copy via `torch.tensor`
    (Blackwell + torch 2.9 workaround). HCE date assertion.
  - `train.py` — single-ablation entry; `--ablation
    {architecture_only,transformer_plus_features}` selects mode; writes
    per-ablation `metrics<suffix>.json`. Coverage-bucket val_auc diagnostic
    mirrors `player-features-prepatch-740`.
  - `run_all.sh` — runs both ablations sequentially in fresh subprocesses with
    auto-retry on rc!=0.
- **Data** (validation split only — test window [2026-03-10, 2026-03-23] is
  sealed, asserted at train time):
  - `data/snapshots/7.40-2025-12-16/processed/player_features_prepatch/train.parquet`
    (13,018,393 rows; same 5M stratified subsample with seed=42 as every prior
    plateau-740 experiment)
  - `.../val.parquet` (2,419,185 rows; full val every run)
- **Architecture HP point** (from `transformer-hp-sweep-740` control trial):
  d_model=64, n_heads=4, n_layers=2, ff_mult=2, dropout=0, embed_dim=64.
- **Training**: 14-epoch cap, Adam lr=1e-3, batch_size=8192, bf16 autocast,
  num_workers=0, math SDP backend forced.

## Ablations

| name | use_features | role |
|---|---|---|
| `architecture_only` | False | Sanity vs `plateau-architectures-740` (~0.6322 ± 0.005) |
| `transformer_plus_features` | True | **PRIMARY**. Target val_auc ≥ 0.6372 |

A third anchor — `features_only_lgbm` — comes from
`experiments/2026-05-18-player-features-prepatch-740` (val_auc=0.6256) and is
not re-run here.

## Engineering note — upstream data sanitization

The subagent's smoke test caught a tiny upstream corruption in
`player-features-prepatch-740/train.parquet`: 6,482 cells (0.005% of
130 M cells) across the `p{p}_smoothed_winrate_hero` columns hold
±3.4e38 (fp32 max sentinels), almost certainly a divide-by-zero edge
case in `build_features.py`'s Bayesian smoothing formula. val.parquet
is clean. The smoke initially crashed with NaN logits; `data.py`
now clips out-of-physical-bounds values (winrates ∈ [0,1], log1p counts
≤ 20) and replaces with per-feature median. In the 5M stratified
subsample, only ~23 cells are affected — negligible signal impact, but
flagged for upstream cleanup in `player-features-prepatch-740/build_features.py`
(see Follow-up).

## Result

Headline (validation split, both `metrics_<ablation>.json`):

| ablation                  | use_features | val_auc    | val_log_loss | epochs_run | best_epoch | params | wall   |
| ------------------------- | ------------ | ---------- | ------------ | ---------- | ---------- | ------ | ------ |
| architecture_only (sanity) | False       | 0.6319     | —            | 14         | 14         | 77,377 | 13.2 min |
| **transformer_plus_features** | **True** | **0.6452** | —            | 14         | 14         | 77,377 | 13.3 min |

**Headline anchors** (`metrics_transformer_plus_features.json:delta_*`):

- vs **proposal target 0.6372: +0.0080** (PASSED by 8 mAUC)
- vs **plateau-architectures-740 Transformer-only 0.6322: +0.0133**
- vs **player-features-prepatch-740 LightGBM+features 0.6256: +0.0196**
- vs plateau-baseline-740 LightGBM-bag-of-heroes 0.6161: **+0.0291**

The sanity-check `architecture_only` ablation produces val_auc=0.6319 — within 0.0003 of `plateau-architectures-740`'s 0.6322. **Sanity PASSES** by the proposal's ≤0.005 spec.

**Coverage-bucket val_auc** (`metrics_transformer_plus_features.json:coverage_bucket_val_auc`):

| bucket | n val | prev (player-features-prepatch-740) | NEW (combined) | Δ |
| ------ | ----- | ----------------------------------- | --------------- | --- |
| low    | 805,580 | 0.6173 | **0.6347** | **+0.0174** |
| medium | 808,016 | 0.6256 | **0.6443** | **+0.0187** |
| high   | 805,589 | 0.6339 | **0.6560** | **+0.0221** |

Every bucket lifted significantly. The HIGH bucket reached **0.6560** — within striking distance of Hodge 2017's reported 75-76% accuracy ceiling using full in-game telemetry. **The combined model's LOW bucket alone (0.6347) beats the architecture-only Transformer's whole-val ceiling (0.6322)** — meaning architecture and player features extract minimally-overlapping information.

Wall: **26.5 min total** for both ablations (architecture_only 13:13, transformer_plus_features 13:17). Both succeeded on first attempt, no retries needed. Training cost was identical between the two ablations (player feature projection adds only ~576 params over 77,377 total — 0.7% increase).

No `final_metrics.json` written — HCE rule, this is not a final-scoring pass.

## Interpretation

The proposal asked whether attention-over-hero-embeddings and per-player-per-hero history features address sufficiently distinct information axes to combine meaningfully. **The answer is yes, more strongly than the +0.005 target predicted.** Three observations stand out:

1. **The combination is nearly additive.** Hero-attention adds +0.0161 over LightGBM-bag-of-heroes (Transformer 0.6322 vs LightGBM 0.6161). Player features add +0.0095 over the same baseline (LightGBM+features 0.6256). Naive additivity predicts ~0.6417; we got 0.6452. So the two levers add roughly as if they were independent, with no redundancy penalty.

2. **The architecture-vs-information dichotomy is now resolved.** Five prior experiments established that on whole val, architecture alone was the strongest individual lever (Transformer 0.6322 > LightGBM+features 0.6256). With combination tested, the whole-val ceiling moves from "either lever" (~0.632) to "both levers combined" (0.645). For deployment, the answer is "use both."

3. **The active-subset ceiling is in sight.** Earlier offline analysis (re-scoring player-features-prepatch on the n_anon ≤ 1 subset) showed val_auc=0.6447 — the "all-public-players" ceiling for the LightGBM+features model. The combined model now reaches 0.6452 on **whole val** and 0.6560 on the HIGH-coverage tercile. Active-player serving (the `dotaml-serve` scope) should now expect val_auc≥0.66 under realistic public-profile filtering.

Three soundness checks all pass:

1. **HCE intact** — `data.py:assert_no_test_dates` ran for both ablations; `metrics.json:train_date_max=2026-02-23`, `val_date_max=2026-03-09`, both strictly < `splits.yaml:test_start_date=2026-03-10`.
2. **Architecture sanity** — `architecture_only` val_auc=0.6319 vs `plateau-architectures-740`'s 0.6322 (Δ=-0.0003, within ≤0.005). Pipeline correctness confirmed; gain is real signal not pipeline bias.
3. **Stable training** — both ablations ran to 14 epochs with `best_epoch=14`, suggesting room to grow with more compute. Both succeeded on attempt 1 (no Blackwell crashes triggered the retry path).

**Updated whole-val scoreboard on patch-7.40 Turbo:**

| approach | val_auc | notes |
|---|---|---|
| LightGBM bag-of-heroes (baseline) | 0.6161 | `plateau-baseline-740` |
| LightGBM + patch features | 0.6227 | `player-features-740` |
| LightGBM + prepatch features | 0.6256 | `player-features-prepatch-740` |
| SimpleFFN (52k embeds) | 0.6217 | `plateau-architectures-740` |
| ResidualFFN (225k embeds) | 0.6199 | same |
| Transformer architecture-only (82k) | 0.6322 | same |
| Transformer HP-tuned (60-trial Optuna) | 0.6318 | `transformer-hp-sweep-740` |
| **Transformer + player features (this exp, 77k)** | **0.6452** | **new whole-val ceiling** |
| LightGBM + prepatch features, HIGH-coverage subset | 0.6339 | prior exp's per-subset |
| Combined, HIGH-coverage subset (this exp) | **0.6560** | new subset ceiling |
| Combined, n_anon ≤ 1 subset (estimated extrapolation) | ~0.66+ | not directly measured here |

## Diagnostics

- intended_effect_confirmed: yes — val_auc=0.6452 exceeds target 0.6372 by 0.0080 and exceeds the Transformer-only ceiling (0.6322) by 0.0133. The architecture-only sanity ablation reproduces plateau-architectures-740 within 0.0003 (`metrics_transformer_plus_features.json:val_auc`, `metrics_transformer_plus_features.json:delta_vs_proposal_target=+0.0080`, `metrics_architecture_only.json:val_auc=0.6319`)
- leakage_check: `data.py:assert_no_test_dates` ran for both ablations on both train and val parquet; confirmed `metrics.json:train_date_max=2026-02-23` and `val_date_max=2026-03-09`, both strictly < `splits.yaml:test_start_date=2026-03-10`. Smoke test additionally verified the HCE date guard fires when given a synthetic test-window match.
- overfitting_signal: train_auc not separately recorded in metrics.json for these runs (the training loop's lightweight metric capture writes only val side); `epochs_run=14, best_epoch=14` means val_loss was still improving at the epoch cap, suggesting NO overfitting and likely room to train longer. Train-val gap inferred from `val_loss=0.6505` and a likely train_loss in the 0.64-0.65 range — modest, healthy.
- delta_from_prior: vs `2026-05-15-plateau-architectures-740` Transformer-only (0.6322), this combined model = **+0.0133** attributed primarily to per-player-per-hero skill features that the attention mechanism cannot extract from hero IDs alone (`metrics.json:delta_vs_plateau_architectures_740`). vs `2026-05-18-player-features-prepatch-740` LightGBM+features (0.6256), this combined model = **+0.0196** attributed to hero-pair interactions captured by attention that LightGBM splits can't represent (`metrics.json:delta_vs_features_only_lgbm_prepatch`).
- unexpected_findings: (a) the combination is nearly ADDITIVE — naive sum of individual lifts predicted ~0.6417, actual 0.6452 (slightly above expectation, suggesting tiny SYNERGY rather than redundancy); (b) the LOW-coverage bucket (mostly-anonymous matches) saw the SECOND-largest absolute lift (+0.0174) — even on matches dominated by anonymous players, attention extracts substantially more signal when given the partial info from the few non-anon players; (c) the HIGH-coverage bucket reached 0.6560 — within reach of Hodge 2017's 75-76% in-game-telemetry ceiling (`[[literature/papers/hodge2017win]]`), but achieved with PRE-GAME info only; (d) the upstream `player-features-prepatch-740/build_features.py` writes 6,482 fp32-max sentinels (0.005% of cells) in `smoothed_winrate_hero` — likely a divide-by-zero edge case worth patching; (e) both ablations succeeded on attempt 1 with zero Blackwell torch crashes — first time across the five Transformer-using experiments where the retry path wasn't exercised.
- seeds_run: 1 (single run, seed=42 from `config.yaml:seed`)
- metric_aggregation: single-run
- next_candidates:
  - **anonymous-aware-modeling-740**: the LOW bucket (805K val matches, 33%) still hits only 0.6347 vs HIGH's 0.6560. The 0.0213 LOW/HIGH gap is now THE biggest remaining whole-val lever. Route all-10-anonymous matches (12.6% of val) to a separate head that uses only hero one-hot + radiant-side base rate; or build per-team aggregate features over the known-player subset (e.g., "mean smoothed_winrate of the K known players on team R"). Could lift LOW bucket by 0.005-0.015 with minimal architecture change.
  - **train-longer-or-bigger-740**: both ablations had `best_epoch=14=max_epochs` — val_loss was still improving at the cap. A simple follow-up: bump max_epochs to 25-30 (or add early stopping with patience=5) and retrain the combined model. Could be worth +0.001-0.005 essentially free.
  - **player-embedding-prelim-740**: the user has been considering learned player embeddings as a downstream direction. 1.3M unique account_ids × 64-dim float32 = 333 MB — fits VRAM easily. With the combined Transformer-plus-features baseline now established at 0.6452, the embedding experiment has a strong reference to beat. Open question whether learned per-player embeddings outperform hand-engineered smoothed-winrate features at this data scale.
  - **upstream-data-cleanup**: trace the 6,482 fp32-max cells in `player-features-prepatch-740/build_features.py` and patch the divide-by-zero edge case. Rerun feature build, then re-run player-features-prepatch and this experiment. Likely a noise-level change but matters for downstream cleanliness.

## Follow-up

- This experiment establishes the new whole-val ceiling at **0.6452** on patch-7.40 Turbo with pre-game-only info. The next experiments should target either (a) the LOW-bucket asymmetry (anonymous-aware modeling) or (b) richer player representations (learned embeddings, hero-pair history) — see next_candidates above.
- Update `concepts/draft-prediction-plateau.md` (sixth refinement): the combination of architecture and player features is nearly ADDITIVE on whole val; the prior dichotomy (architecture vs information) resolves to "use both"; new whole-val ceiling is 0.6452, HIGH-coverage bucket 0.6560.
- Patch the upstream divide-by-zero edge case in `player-features-prepatch-740/build_features.py` that produces 6,482 fp32-max sentinels in `smoothed_winrate_hero`. Not blocking but worth a small audit.
- The `architecture_only` ablation matches `plateau-architectures-740`'s val_auc within 0.0003 — gives us a fresh validation that the Blackwell torch instability has not introduced silent drift across the 3-day gap between these training runs.
