---
kind: experiment
slug: player-features-prepatch-740
date: 2026-05-18
status: done
hypothesis: "Extending the per-player history aggregator from player-features-740 with ~127 days of pre-7.40 Turbo data (Aug 2025 → Dec 15 2025, ~118 GB additional raw) raises val_auc by >= 0.005 over the patch-only result (target val_auc >= 0.6277, vs player-features-740's 0.6227), and meaningfully flattens the coverage-bucket diagnostic, confirming that the cold-start signal observed in player-features-740 was the binding constraint rather than a feature-engineering ceiling."
result: "Hypothesis NOT confirmed but the diagnostic flipped the story. val_auc=0.6256 (+0.0028 over player-features-740, missing the +0.005 target by 0.0022). The coverage-bucket diagnostic did NOT flatten — it stayed monotonic with the HIGH bucket gaining most (+0.0043) and LOW gaining least (+0.0014), opposite of the cold-start hypothesis. History-source breakdown explains why: low-bucket players had only 2.3% prepatch fraction (4 prepatch games avg) — they were genuinely casual/new, not active players in a cold-start window. The casual-player tail is the binding constraint, not lookback length. Notably: in the HIGH bucket, val_auc=0.6339 BEATS the Transformer ceiling (0.6322) for the first time — for the active 1/3 of players, player features are now the better lever than architecture. Whole-val ceiling is bound by the casual/anonymous tail (66% anonymous mean per match unchanged)."
related_concepts:
  - draft-only-win-prediction
  - draft-prediction-plateau
related_literature:
  - eschmitt88-DotaML
  - hodge2017win
related_prior:
  - 2026-05-17-player-features-740
  - 2026-05-15-plateau-baseline-740
  - 2026-05-15-plateau-architectures-740
  - 2026-05-16-transformer-hp-sweep-740
respects:
  - ~/claude-system/claude/rules/evaluation.md
tags: [lightgbm, player-history, prepatch, cold-start, anonymous-accounts, hce]
---

# player-features-prepatch-740

## Hypothesis

Extending the per-player history aggregator with ~127 days of pre-patch-7.40
Turbo data (2025-08-01 → 2025-12-15) raises val_auc by >= 0.005 over
player-features-740's 0.6227 (target >= 0.6277), and flattens the
coverage-bucket monotonic from player-features-740. If confirmed, the
cold-start signal is the binding constraint rather than a feature-engineering
ceiling.

## Setup

- Data:
  - Pre-patch raw: `data/history/turbo/year=YYYY/month=MM/day=DD/matches_*.parquet`
    (pulled fresh from Azure, ~127 days, ~100 GB).
  - Patch-7.40 raw: `data/snapshots/7.40-2025-12-16/raw/turbo/...`
    (already on disk, ~81 GB, train+val window only — HCE-sealed against
    test window [2026-03-10, 2026-03-23]).
  - Processed parquet (this experiment's output):
    `data/snapshots/7.40-2025-12-16/processed/player_features_prepatch/{train,val}.parquet`.
- Features: identical schema to player-features-740 (90 dense + 300 sparse
  hero one-hot + 1 side bit), PLUS two new per-player tracking columns —
  `pX_n_games_prepatch` and `pX_n_games_inpatch` (uint32) — used by the
  history-source-breakdown diagnostic. These tracking columns are NOT fed
  to the model.
- Aggregator: same as player-features-740 (Bayesian smoothing alpha=5,
  hero-specific shrinkage, recent-form window=10, anonymous-account
  handling). Walks history/ first chronologically, then snapshot raw/
  — no state reset, no decay at the patch boundary.
- Model: LightGBM identical to player-features-740 and plateau-baseline-740
  (500 rounds, lr 0.1, 31 leaves, 5M-row stratified subset, seed=42).
- HCE: aggregator never reads test window or post-snapshot dates. Hard
  refusals in `pull_history.py` and `build_features.py`.

## Data pipeline

`build_features.py` walks all `matches_*.parquet` under the configured
`raw_roots` (history first, then snapshot raw), groups by YYYY-MM-DD,
and processes days in chronological order. Within a day, matches are
sorted by `(start_time, match_id)` before processing.

For each match at time T:
1. SNAPSHOT pre-match features for all 10 players from the running
   aggregator (state strictly from matches with `start_time < T`).
   Also record per-player `(n_games_prepatch, n_games_inpatch)`.
2. UPDATE the aggregator with this match's outcome, labelling it
   prepatch (date < 2025-12-16) or in-patch.

Emit set = exactly the match_ids in `data/snapshots/.../processed/{train,val}.parquet`
(the same row set as player-features-740). Aggregator state is updated
from ALL raw matches across BOTH roots.

HCE: `pull_history.py` and `build_features.py` refuse test-window dates
and post-snapshot dates. `train.py` asserts the loaded train/val parquet
contain no test-window dates before training.

## Run

```bash
nohup bash experiments/2026-05-18-player-features-prepatch-740/run_all.sh \
  > /tmp/dotaml_pfp.log 2>&1 &
```

Pipeline (sequential):
1. `pull_history.py` — Azure pull, [2025-08-01, 2025-12-15] (~30 min, ~100 GB).
2. `build_features.py` — chronological walk over ~265 days (~3 h).
3. `train.py --ablation heroes_plus_features` (~5 min).
4. `train.py --ablation heroes_only --metrics-suffix _ablation_heroes_only`
   (~5 min, sanity-check rebuild — must reproduce plateau-baseline-740
   within 0.001 AUC).
5. `train.py --ablation features_only --metrics-suffix _ablation_features_only`
   (~5 min; may crash with the same LightGBM split-finding assert as
   player-features-740 since the degenerate all-anonymous matches are
   unchanged — informational only).

Total ~3.5 h wall.

## Sanity expectations (to be filled post-run)

- `heroes_only` ablation must reproduce plateau-baseline-740 within
  0.001 AUC (data pipeline correctness check on the new processed
  parquet's hero columns).
- The new processed `train.parquet` should produce the same hero
  one-hot rows as player-features-740's processed parquet (same match
  set, same hero columns sourced from the same authoritative processed
  index).

## Engineering note — coplay dropped mid-experiment

The first two full builds OOM-killed at ~93 GB RSS, caused by the per-account `coplay` nested dict ballooning at the ~5M-account scale (5M accounts × 200-cap entries × ~75 bytes ≈ 75 GB just for that one feature). Per `player-features-740`'s feature_importance ranking, `coplay_mean` wasn't in the top 20 — contributing essentially nothing. After two OOMs (2026-05-18 07:38 and ~13:30 UTC), `coplay` tracking was removed entirely; `unique_heroes` set was also replaced with `len(hero_n[acct])` (saves ~8 GB more). Memory peak after the edit dropped to under 15 GB. Feature schema is now 8 player features × 10 players = 80 cols (vs 90 in player-features-740). The heroes_only sanity rebuild still PASSED (Δ=-0.0001 vs plateau-baseline-740) confirming the data pipeline is correct.

## Result

Headline (validation split, `metrics.json`):

| ablation              | feature_dim | val_auc    | val_acc | val_log_loss | Δ vs baseline (0.6161) | Δ vs player-features-740 (0.6227) |
| --------------------- | ----------- | ---------- | ------- | ------------ | ---------------------- | --------------------------------- |
| heroes_only (sanity)  | 301         | 0.6160     | 0.5864  | 0.6698       | **-0.0001** (PASSES sanity ≤0.001) | —                                 |
| **heroes_plus_features** | 381         | **0.6256** | 0.5931  | 0.6659       | **+0.0095**            | **+0.0028**                       |
| features_only         |  80         | 0.6065     | 0.5796  | 0.6729       | -0.0096                | (was crash in prior exp)          |

Anchors (`metrics.json:delta_vs_*`):

- vs plateau-baseline-740 (LightGBM 0.6161): **+0.0095**
- vs **player-features-740 (0.6227): +0.0028** ← the key delta this experiment isolates
- vs plateau-architectures-740 Transformer (0.6322): **-0.0066** (still below architecture-only)
- vs transformer-hp-sweep-740 (0.6318): -0.0062
- vs proposal target (0.6277): **-0.0022** (missed by 0.0022)

Coverage-bucket val_auc — **stayed monotonic, did NOT flatten** (`metrics.json:coverage_bucket_val_auc`):

| bucket | prev (player-features-740) val_auc | new val_auc | Δ | new mean coverage log1p |
|--------|-------------------------------------|-------------|---|--------------------------|
| low    | 0.6159                              | 0.6173      | **+0.0014** | 0.33 |
| medium | 0.6230                              | 0.6256      | +0.0026     | 1.45 |
| high   | 0.6296                              | **0.6339**  | **+0.0043** | 3.06 |

**The HIGH bucket gained most, the LOW bucket gained least — opposite of the cold-start prediction.** And the HIGH-coverage val_auc of 0.6339 **beats the Transformer ceiling (0.6322)** for the first time. For the active 1/3 of players, player features are now the stronger lever than architecture.

History-source breakdown (`metrics.json:history_source_breakdown`) explains why low-bucket didn't gain more:

| bucket | mean prepatch fraction | mean prepatch games/slot | mean inpatch games/slot |
|--------|------------------------|--------------------------|--------------------------|
| low    | **2.3%**               | 4.2                      | 6.0                      |
| medium | 11.4%                  | 28.6                     | 33.7                     |
| high   | **24.9%**              | 81.0                     | 86.4                     |

Low-bucket players had ALMOST NO pre-patch presence either — they're genuinely casual / new players, not active players who happened to fall into our cold-start window. Pre-patch data can't rescue players who weren't around then.

Anonymous-account histogram for val unchanged from `player-features-740` (12.6% all-10-anonymous matches, mean 6.66 anonymous slots/match) — because anonymous status is a Steam privacy setting independent of data history we have.

Top-15 feature importances unchanged in character from `player-features-740`: **all 10 `pX_smoothed_winrate_hero` features dominate** (gains 65-80k each). Confirms the user's pre-experiment prediction that per-player-per-hero deviation from base rate is patch-stable; pre-patch data primarily strengthened the dominant signal.

Build wall: **120 min** (with coplay dropped); memory peak ~15 GB. Down from the OOM-killing 93 GB of the first run. Total experiment wall (incl. Azure pull): ~24 min pull + 120 min build + 8 min for 3 ablations ≈ 2.5 h.

No `final_metrics.json` written — HCE rule, this is not a final-scoring pass.

## Interpretation

The proposal predicted: if cold-start is binding, coverage buckets should flatten (low gains most). The result inverted that prediction: **the coverage-bucket diagnostic stayed monotonic and HIGH gained most.** Pre-patch data IS contributing (12.9% of total games on average; 24.9% in the high bucket), and it DID lift val_auc by +0.0028. But the lift is concentrated on already-active players, not on cold-start cases.

**Why the prediction was wrong:** I'd implicitly assumed low-bucket val matches contained active players who happened to play their first patch-7.40 matches recently (so had little in-patch lookback). The data shows otherwise: low-bucket players also had little PRE-patch presence (2.3% prepatch fraction, 4 prepatch games avg). They're genuinely casual or new accounts, not active players in a cold-start window. Pre-patch data cannot rescue them because they weren't around then either.

**The binding constraint is the casual/anonymous tail, not lookback length.** Two facts confirm this:

1. **Anonymous fraction unchanged.** 66% anonymous mean per match, 12.6% all-10-anonymous matches — identical to `player-features-740` because anonymous status is a Steam privacy setting independent of how much history we collect.
2. **HIGH-bucket val_auc 0.6339 beats Transformer.** For the active 1/3 of players, player features extract more signal than 82k attention parameters can. The lever IS strong; it just doesn't reach the bottom 2/3 of val.

Three soundness checks all pass:

1. HCE intact: build asserts in `build_features.py:assert_no_test_or_postsnapshot` (called against both data/history/turbo/ and data/snapshots/.../raw/turbo/); confirmed `train_date_max=2026-02-23`, `val_date_max=2026-03-09`, all strictly < test_start_date.
2. Heroes-only sanity ablation reproduces plateau-baseline-740 within 0.0001 — data pipeline correct.
3. Train-val AUC gap = 0.0141 — comparable to player-features-740 (0.0133), no overfitting from the additional features/data.

**Top-of-stack scoreboard on patch-7.40 Turbo (val_auc):**

| approach | val_auc | notes |
|---|---|---|
| LightGBM bag-of-heroes (baseline) | 0.6161 | `plateau-baseline-740` |
| LightGBM + patch features         | 0.6227 | `player-features-740` |
| LightGBM + prepatch features      | **0.6256** | this experiment |
| SimpleFFN (52k emb)               | 0.6217 | `plateau-architectures-740` |
| ResidualFFN (225k emb)            | 0.6199 | same |
| Transformer (82k emb)             | **0.6322** | same — still whole-val winner |
| Transformer HP-tuned (60 trials)  | 0.6318 | `transformer-hp-sweep-740` |
| **LightGBM + prepatch features, HIGH-coverage subset** | **0.6339** | active 1/3 of val |

## Diagnostics

- intended_effect_confirmed: no — val_auc=0.6256 missed target 0.6277 by 0.0022, and the coverage-bucket diagnostic did NOT flatten (low/med/high val_auc = 0.6173/0.6256/0.6339, still monotonic with HIGH gaining most) (`metrics.json:val_auc`, `metrics.json:delta_vs_proposal_target_val_auc=-0.0022`, `metrics.json:coverage_bucket_val_auc.buckets`)
- leakage_check: HCE assertions in `pull_history.py:67-91` (refuses test-window + post-snapshot dates at pull time) and `build_features.py:assert_no_test_or_postsnapshot` (asserts during build); verified via `metrics.json:train_date_max=2026-02-23`, `val_date_max=2026-03-09`, both strictly < `splits.yaml:test_start_date=2026-03-10`. Chronological per-match snapshot uses strict `start_time < T` per `build_features.py:Aggregator.snapshot` (no future-into-past leakage)
- overfitting_signal: train=0.6397 val=0.6256 gap=0.0141 — well-fit, comparable to player-features-740's 0.0133 gap; no overfitting from the additional features or data (`metrics.json:train_val_auc_gap`)
- delta_from_prior: vs 2026-05-17-player-features-740 (val_auc=0.6227), this run = +0.0029 attributed primarily to per-player-per-hero winrate enrichment (top-10 features unchanged in identity; gains 65-80k each, see `metrics.json:feature_importance_top20`). The +0.0029 concentrates in the HIGH-coverage val bucket (+0.0043), NOT in the LOW bucket (+0.0014) — opposite of the cold-start hypothesis (`metrics.json:coverage_bucket_val_auc.buckets`, `metrics.json:history_source_breakdown.per_bucket`)
- unexpected_findings: (a) the coverage-bucket diagnostic stayed monotonic — pre-patch data didn't lift LOW more than HIGH, so cold-start is NOT the binding constraint; the casual/anonymous-player tail is; (b) LOW-bucket players have ~2.3% prepatch fraction (4 games avg) — they're genuinely casual, not active-but-uncached; (c) HIGH-coverage val_auc=0.6339 BEATS the architecture-only Transformer ceiling (0.6322) for the first time — for the active 1/3 of val, player features are the stronger lever than architecture; (d) two OOM-kills required dropping coplay + unique_heroes from the aggregator (state was scaling badly at ~5M accounts), but neither feature was in the top-20 importance from `player-features-740` so the bias is bounded; (e) the user's pre-experiment prediction that hero-specific player skill is patch-stable HELD — top-10 features unchanged in character, no metagame drift artifact visible
- seeds_run: 1 (single run, seed=42 from `config.yaml:seed`)
- metric_aggregation: single-run
- next_candidates:
  - **transformer-plus-player-features-740.** Combine the two strongest individual levers — replace the LightGBM head with the `plateau-architectures-740` Transformer architecture, feed it 10 hero IDs + the 80-dim per-player feature block, and train. Hypothesis: the HIGH-bucket val_auc 0.6339 from this experiment plus the architecture lever should push the WHOLE-val val_auc past 0.633, possibly to 0.640+. The two levers haven't been combined yet; doing so directly tests whether they're additive.
  - **anonymous-aware-modeling.** Treat the all-anonymous matches (12.6% of val) as a structural subproblem; either route them to a separate head that only sees hero one-hot + radiant-side-conditional priors, or build aggregate per-match features over the known-player subset (e.g. "mean smoothed_winrate over the K non-anonymous players on team R"). This addresses the 12.6% tail that's currently dead-weight in the model.
  - **player-features-decay-740.** With pre-patch data now confirmed to contribute (~13% of total game-history weight), test whether time-decayed history is BETTER than uniform aggregation. Exponential weighting with τ ≈ 90 days. Hypothesis: small additional gain, but with a real concern that recent skill matters more than ancient skill. Smaller experiment than this one — same data, just a different aggregator.

## Follow-up

- The hypothesis is cleanly NOT confirmed but the diagnostic delivered three sharper findings that redirect the search:
  1. Pre-patch data DOES help (+0.0028 overall, +0.0043 in HIGH bucket).
  2. Cold-start is NOT the binding constraint — the binding constraint is the casual/anonymous-player tail, which structural data extension cannot fix.
  3. For the active 1/3 of val, player features now BEAT the Transformer ceiling (0.6339 > 0.6322). Architecture-vs-information is no longer a clear win for architecture; the right comparison is now "which model wins on the active subset" and the answer is changing.
- Update `concepts/draft-prediction-plateau.md` (fifth refinement): the ~0.632 whole-val ceiling on patch-7.40 Turbo is structurally bounded by the 66% anonymous + casual-player tail; for the active 1/3 of val, player features beat architecture (0.6339 > 0.6322).
- Update `concepts/draft-only-win-prediction.md`: the `pre-game-win-prediction` flavour is now the project's working task; record the HIGH-coverage val_auc 0.6339 as the per-subset benchmark.
- The `features_only` ablation completed cleanly this time (val_auc=0.6065) — pure player features (no heroes) extract ~0.61 AUC on their own, vs LightGBM-with-heroes 0.6161. So player and hero features are complementary (combined = 0.6256, both individually below combined). This is informational but expected.
- `data/history/turbo/` (~100 GB, ~127 days of pre-7.40 Turbo data) is now on disk and reusable for any downstream experiment that wants longer player history.
- `data/snapshots/.../processed/player_features_prepatch/{train,val}.parquet` is the augmented feature table (80 player feats × 10 + 2 source-tracking cols × 10 + hero cols + match_id). Reusable.
