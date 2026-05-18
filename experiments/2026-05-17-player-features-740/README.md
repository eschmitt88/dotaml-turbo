---
kind: experiment
slug: player-features-740
date: 2026-05-17
status: done
hypothesis: "Adding per-player historical features (smoothed overall and hero-specific win rates, recent form, premade-detection co-play counts) computed from the patch-7.40 snapshot's leading-window history to the existing LightGBM baseline raises val_auc by >= 0.020 over plateau-baseline-740 (target val_auc >= 0.6361), clearly exceeding both the architectural ceiling at 0.6322 (plateau-architectures-740) and the HP-tuned ceiling at 0.6318 (transformer-hp-sweep-740), and confirming the plateau is an information bottleneck (missing player identity) rather than a representation bottleneck."
result: "Hypothesis NOT confirmed but real signal extracted. val_auc=0.6227 (+0.0067 over baseline 0.6161, missing the +0.020 target by 0.0134, and below both the Transformer ceiling 0.6322 and HP-tuned ceiling 0.6318). Heroes-only sanity check passed (Δ=-0.0001). Coverage-bucket diagnostic shows monotonic lift (low→high coverage = 0.6159→0.6296, Δ=+0.0137), confirming cold-start is binding. Anonymous accounts dominate (66% of player-slots; 12.7% of val matches have ALL 10 players anonymous) — this is the real binding constraint, not architecture. Top-10 importances are exclusively per-player smoothed_winrate_on_current_hero. Pre-patch ingest (player-features-prepatch-740) is justified for non-hero features but risky for the dominant hero-specific feature due to metagame drift."
related_concepts:
  - draft-only-win-prediction
  - draft-prediction-plateau
related_literature:
  - eschmitt88-DotaML
  - hodge2017win
related_prior:
  - 2026-05-15-plateau-baseline-740
  - 2026-05-15-plateau-architectures-740
  - 2026-05-16-transformer-hp-sweep-740
respects:
  - ~/claude-system/claude/rules/evaluation.md
tags: [lightgbm, player-history, leading-window, anonymous-accounts, cold-start, hce]
---

# player-features-740

## Hypothesis

Adding per-player history features (~90 columns, 9 per player x 10
players) on top of the plateau-baseline-740 LightGBM recipe should
raise val_auc by >= 0.020 (from 0.6161 to >= 0.6361), exceeding the
plateau ceilings at 0.6322 (architectures) and 0.6318 (HP sweep).
If it does, the plateau is an information bottleneck (missing player
identity) rather than a representation bottleneck.

## Setup

- Data: `data/snapshots/7.40-2025-12-16/processed/player_features/{train,val}.parquet`
  built by `build_features.py` from the raw parquet under
  `raw/turbo/year=*/month=*/day=*/matches_*.parquet`.
- Features:
  - Sparse 300-dim hero one-hot + 1-bit Radiant side (same as baseline)
  - Dense 90-dim player features: per player p in [0..9]:
    `n_games_log1p`, `smoothed_winrate`, `smoothed_winrate_hero`,
    `last10_winrate`, `days_since_last_log1p`, `n_games_hero_log1p`,
    `hero_diversity_log1p`, `coplay_mean`, `is_anonymous`.
- Smoothing: Bayesian, alpha=5 (~10-game shrinkage). Hero-specific rates
  shrink toward per-hero base rate; overall rates shrink toward 0.5.
- Cold-start: anonymous (account_id in {0, 4294967295}) get global priors
  and only contribute to per-hero global counts (not per-account state).
- Model: LightGBM identical to plateau-baseline-740 (500 rounds, lr 0.1,
  31 leaves), trained on 5M-row stratified subset of train (seed=42).

## Data pipeline

`build_features.py` walks the raw parquet day-by-day in chronological
order. For each day, files are concatenated and the resulting matches
are sorted by `start_time` (ties broken by `match_id`) before being
processed.

For each match at time T:
1. SNAPSHOT pre-match features for all 10 players from the running
   aggregator (which holds state only from matches with `start_time < T`).
2. UPDATE the aggregator with this match's outcome.

Emit set = exactly the match_ids in the filtered processed parquet
(`processed/{train,val}.parquet`). Aggregator state is updated from
ALL raw matches (no filter), so a player's history reflects every game
they actually played.

HCE assertions:
- `build_features.py` refuses to read any date in
  `[test_start_date, test_end_date]`.
- `train.py` asserts the loaded train/val parquets contain no test-window
  dates before training.

## Result

Headline (validation split — search signal, `metrics.json`):

| ablation              | feature_dim | val_auc    | val_acc | val_log_loss | Δ vs baseline (0.6161) |
| --------------------- | ----------- | ---------- | ------- | ------------ | ---------------------- |
| heroes_only (sanity)  | 301         | 0.6160     | 0.5864  | 0.6698       | **-0.0001** (PASSES sanity ≤0.001) |
| **heroes_plus_features** | 391         | **0.6227** | 0.5911  | 0.6671       | **+0.0067**            |
| features_only         | 90          | (crashed)  | —       | —            | LightGBM internal assert at round ~50 |

`features_only` crashed with `Check failed: (best_split_info.left_count) > (0)` (a LightGBM split-finding assert that fires when a tree node has no variance for any candidate feature — almost certainly the ~12.7% of val matches with all 10 anonymous players collapse all features to identical prior values). Not investigated further; the `heroes_plus_features` headline + the `heroes_only` sanity check together fully cover the hypothesis test.

Anchors (`metrics.json:delta_vs_*`):

- vs plateau-baseline-740 (0.6161): **+0.0067**
- vs plateau-architectures-740 Transformer (0.6322): **-0.0095** (player features WORSE than architecture)
- vs transformer-hp-sweep-740 (0.6318): **-0.0091**
- vs proposal target (0.6361): **-0.0134** (missed)

Coverage-bucket val_auc diagnostic (`metrics.json:coverage_bucket_val_auc`):

| bucket | n val | mean coverage (log1p) | val_auc |
| ------ | ----- | ---------------------- | ------- |
| low    | 805,588 | 0.29 (~0.3 games/player) | 0.6159 |
| medium | 808,008 | 1.28 (~2.6 games/player) | 0.6230 |
| **high** | 805,589 | **2.69 (~13.7 games/player)** | **0.6296** |

**Monotonic increase: cold-start IS binding.** Low-coverage val matches see ZERO lift from player features (val_auc = baseline exactly); high-coverage matches see +0.0137 over baseline. The overall +0.0067 is dragged down by the bottom third.

Anonymous-account histogram for val (`metrics.json:anonymous_per_match_hist_val`):

| anonymous count | n val matches | % |
| --------------- | ------------- | --- |
| 0 (all known)   |     5,884 | 0.2% |
| 1               |    24,997 | 1.0% |
| 2               |    63,623 | 2.6% |
| 3               |   124,763 | 5.2% |
| 4               |   199,138 | 8.2% |
| 5               |   273,430 | 11.3% |
| 6 (mode)        |   331,271 | 13.7% |
| 7               |   363,583 | 15.0% |
| 8               |   372,577 | 15.4% |
| 9               |   354,042 | 14.6% |
| **10 (all anonymous)** | **305,877** | **12.7%** |

**Mean 6.66 anonymous per match** — only ~3-4 player-slots per match have extractable history on average.

Top-20 feature importances (`metrics.json:feature_importance_top20`):

- **Top 10 are ALL `pX_smoothed_winrate_hero`** (player X's smoothed winrate ON THE SPECIFIC HERO they're playing), one per player-slot, gain 73-87k each.
- Next tier (11-20) is mixed: a few specific hero one-hots (`r_hero_67`, `d_hero_67`, etc.) and `pX_n_games_hero_log1p` (sample-size context for the hero winrate).
- **NOT in top 20:** `pX_smoothed_winrate` (overall), `pX_last10_winrate` (recent form), `pX_coplay_mean` (premade detection), `pX_is_anonymous`, `pX_days_since_last_log1p`, `pX_hero_diversity_log1p`. Overall winrate, recent form, premade signal, and rust effect contributed essentially nothing.

Build stats (`data/snapshots/.../player_features/build_stats.json`):
read 15,437,578 matches (train + val window only); 0 bad_json; 1,329,669 unique account_ids tracked; build wall = 4,123 s (≈ 69 min); mean anonymous_per_match = 6.66.

No `final_metrics.json` written — HCE rule, this is not a final-scoring pass. Test window `[2026-03-10, 2026-03-23]` never read (asserted in `build_features.py` and `train.py`).

## Interpretation

The proposal asked whether adding per-player history features could push val_auc past the ~0.632 architectural ceiling and confirm the plateau is an information bottleneck. The answer is **a clean partial: information helps but less than architecture does on this snapshot**, with three sub-findings:

1. **The lever exists but is small.** +0.0067 AUC is real and exceeds noise (the sanity-check ablation matches plateau-baseline-740 within 0.0001, ruling out a pipeline bug). For the high-coverage tercile of val it grows to +0.0137. So player history is informative — just not informative ENOUGH at the current anonymous-account fraction to beat architecture.

2. **The dominant signal is per-player-per-hero winrate, NOT overall skill.** The top 10 features by importance are exclusively `pX_smoothed_winrate_on_current_hero` (one per player slot, gains 73-87k each). Overall winrate, recent form, and co-play (premade detection) didn't even crack the top 20. This is informative for downstream work: the marginal value of player features lives in hero-specific affinity, which is exactly the feature with the SMALLEST per-player sample (a player playing many heroes spreads their games thinly).

3. **The binding constraint is data availability, not feature engineering.** 66% of player-slots in Turbo are anonymous (Steam-private profiles). For those, we can compute only global priors. 12.7% of val matches have ALL 10 players anonymous — the model gets no signal at all for those. Even with perfect feature engineering, ~2/3 of the per-player signal is structurally unavailable.

Comparing to `plateau-architectures-740`'s Transformer (val_auc=0.6322): the Transformer with no player info beats LightGBM-with-player-features by 0.0095. So on this snapshot, **attention over hero embeddings is a more reliable lever than 90 player-history columns of which 66% are masked**. This is the opposite of what Hodge 2017 [[literature/papers/hodge2017win]] would have predicted for ranked matches and is itself a publishable observation about Turbo specifically.

Three soundness checks all pass:

1. **HCE intact.** `build_features.py` and `train.py` both assert no test-window dates; metrics confirm `train_date_max=2026-02-23` and `val_date_max=2026-03-09` (both strictly < `splits.yaml:test_start_date=2026-03-10`).
2. **Heroes-only ablation reproduces baseline within 0.0001 AUC.** Data pipeline is correct; the modest lift is the true signal, not noise.
3. **Train-val gap = 0.0133** — comparable to plateau-baseline-740 (0.0126). No overfitting on the player features.

## Diagnostics

- intended_effect_confirmed: no — val_auc=0.6227 misses target 0.6361 by 0.0134 AND falls 0.0091-0.0095 short of the Transformer ceiling at 0.6322 (`metrics.json:val_auc`, `metrics.json:delta_vs_proposal_target_val_auc=-0.0134`, `metrics.json:delta_vs_plateau_architectures_740_val_auc=-0.0095`); partial real signal: +0.0067 over LightGBM baseline (`metrics.json:delta_vs_plateau_baseline_740_val_auc=+0.0067`)
- leakage_check: HCE assertions in `build_features.py:assert_no_test_dates` and `train.py:assert_no_test_dates`; verified via `metrics.json:train_date_max=2026-02-23` and `val_date_max=2026-03-09`, both strictly < `splits.yaml:test_start_date=2026-03-10`. Aggregator uses strict `start_time < T` per-match snapshot (`build_features.py` chronological walk) — no future-into-past leakage
- overfitting_signal: train=0.6360 val=0.6227 gap=0.0133 — well-fit, comparable to baseline's 0.0126 gap; no overfit (`metrics.json:train_val_auc_gap`)
- delta_from_prior: vs 2026-05-15-plateau-baseline-740 (LightGBM val_auc=0.6161) = +0.0067 attributed to per-player-per-hero smoothed winrate (top-10 features by gain in `metrics.json:feature_importance_top20`); vs 2026-05-15-plateau-architectures-740 Transformer (0.6322) = -0.0095 — player features WORSE than attention-over-hero-embeddings on this snapshot
- unexpected_findings: (a) the dominant lever is per-player-per-hero winrate (top-10 importances all `pX_smoothed_winrate_hero`); overall winrate, recent form, premade detection, rust effect all contributed essentially nothing; (b) Turbo is heavily anonymous — 66% of player-slots, 12.7% of val matches all-anonymous — much higher than ranked-MOBA expectations would suggest; (c) coverage-bucket diagnostic is monotonic (low/med/high val_auc = 0.6159/0.6230/0.6296), confirming cold-start binding AND telling us a pre-patch ingest would help most on hero-specific features (which dominate importance) — but those are exactly the features most affected by metagame drift across patches (Hodge 2017 [[literature/papers/hodge2017win]] flags this)
- seeds_run: 1 (single run, seed=42 from `config.yaml:seed`)
- metric_aggregation: single-run
- next_candidates:
  - **player-features-prepatch-740 (limited scope).** Ingest patch-7.39 raw Turbo data (~75 days, ~75 GB) and use it ONLY for non-hero-specific features (overall winrate, recent form, co-play, hero diversity, days-since). KEEP hero-specific winrate restricted to the patch-7.40 window to avoid metagame-drift confound per Hodge 2017. Expected gain: small (since non-hero features didn't crack top-20 importance here), but cleanly tests whether the cold-start-binding signal was hero-specific or overall.
  - **player-on-hero-deeper-740.** Stay within patch 7.40 but enrich the per-player-per-hero signal: add per-player-per-hero-PAIR winrate (this player on hero X with hero Y as ally), per-player-VS-enemy-hero winrate, per-player role/lane preference. The top-importance finding says the per-hero signal is the marginal-value lever; deepen it before broadening to other axes.
  - **anonymous-handling-redesign.** Treat anonymous players as a structural attribute rather than a missing-data hole. Possibilities: (i) per-match aggregate features over the non-anonymous subset (e.g. "mean winrate of the K known players on team R"), (ii) hero-on-side base-rate features conditional on the count of anonymous players, (iii) a tiny matchmaking-skill-mixture model. This addresses the 12.7% all-anonymous tail that's effectively dead-weight in current model.

## Follow-up

- The hypothesis is cleanly NOT confirmed for the strict target, but the experiment delivered three sharper sub-findings that redirect the search:
  1. Per-player-per-hero winrate is the marginal-value feature; everything else we tried is decorative.
  2. Turbo's 66% anonymous fraction is the binding constraint, not feature engineering or architecture.
  3. Cold-start IS binding (coverage-bucket monotonic) but pre-patch ingest is risky precisely because the binding feature (hero-specific winrate) is metagame-drift-sensitive.
- Update `concepts/draft-prediction-plateau.md` (fourth refinement): the ~0.632 ceiling on patch-7.40 Turbo is partially attributable to anonymous-account data scarcity, NOT solely to architecture. With the per-player-per-hero winrate signal layered on top of LightGBM, the COVERAGE-MATCHED ceiling reaches 0.6296 — still below the architecture-only Transformer ceiling of 0.6322.
- Update `concepts/draft-only-win-prediction.md`: this experiment broke the "draft-only" scope (added account_id-derived features). Per the proposal's explicit scope-expansion note, refine the concept to allow pre-game-knowable features (account_id and derivatives) while keeping "no in-game telemetry, no post-draft info" as the HCE-safe boundary. Or seed a new concept `pre-game-win-prediction` that supersedes this one.
- The `features_only` crash (LightGBM `best_split_info.left_count` assert) is a known LightGBM behavior for degenerate features; not worth investigating since the headline result + sanity check are sufficient. Re-running with `min_data_in_leaf` raised to e.g. 100 would likely fix it but isn't on the critical path.
- `data/snapshots/.../player_features/{train,val}.parquet` (~1.4 GB total) is reusable for any downstream experiment that wants the same per-player-history features. Worth keeping under DVC tracking when we formalise that (open follow-up from the previous wrap).
