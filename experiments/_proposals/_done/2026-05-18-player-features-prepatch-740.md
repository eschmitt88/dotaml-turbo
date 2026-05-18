---
kind: proposal
slug: player-features-prepatch-740
date: 2026-05-18
status: implemented
experiment: experiments/2026-05-18-player-features-prepatch-740/
hypothesis: "Extending the per-player history aggregator from player-features-740 with ~127 days of pre-7.40 Turbo data (Aug 2025 → Dec 15 2025, ~118 GB additional raw) raises val_auc by ≥ 0.005 over the patch-only result (target val_auc ≥ 0.6277, vs player-features-740's 0.6227), and meaningfully flattens the coverage-bucket diagnostic, confirming that the cold-start signal observed in player-features-740 was the binding constraint rather than a feature-engineering ceiling."
rationale: >
  player-features-740 added per-player history to the LightGBM baseline
  and lifted val_auc by +0.0067 (0.6161 → 0.6227), with a clear
  coverage-bucket monotonic; low/med/high val_auc =
  0.6159/0.6230/0.6296. The bottom tercile saw zero lift because
  early-train matches have no in-patch lookback; pre-patch data
  directly addresses this. Per the analysis preceding this proposal,
  hero-specific winrate (top-10 features by importance in
  player-features-740) is more patch-transferable than the
  metagame-drift framing suggested; the Bayesian shrinkage formula
  shrinks toward each hero's base rate, so what survives the shrinkage
  is the player's deviation from the meta — and per-player deviations
  are largely patch-stable for the 145+ heroes that get only balance
  tweaks. NOTE; this experiment also implicitly tests Hodge 2017's
  metagame-drift warning by using cross-patch hero-specific data
  without any decay weighting.
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/draft-only-win-prediction]]"
  - "[[concepts/hero-embedding-vs-onehot]]"
  - "[[literature/papers/hodge2017win]]"
  - "[[literature/repos/eschmitt88-DotaML]]"
  - "[[experiments/2026-05-17-player-features-740]]"
  - "[[experiments/2026-05-15-plateau-baseline-740]]"
  - "[[experiments/2026-05-15-plateau-architectures-740]]"
  - "[[experiments/2026-05-16-transformer-hp-sweep-740]]"
expected_metric:
  name: val_auc
  target: 0.6277
  direction: higher-is-better
design_sketch:
  - Pull missing Azure data for date range [2025-08-01, 2025-12-15] (127 days, ~118 GB at observed 70 MB/s ≈ 30 min wall). Land under a NEW directory `data/history/turbo/year=YYYY/month=MM/day=DD/*.parquet`, semantically distinct from `data/snapshots/7.40-2025-12-16/raw/`. The history corpus is for feature derivation ONLY; the snapshot remains the authoritative evaluation universe (per splits.yaml). Don't pull Apr-May 2026 (post-snapshot, future-projects scope) or the test window [2026-03-10, 2026-03-23] (HCE-sealed).
  - Extend `build_features.py` to walk `data/history/turbo/` chronologically first (Aug 1 → Dec 15), then continue into the existing `data/snapshots/7.40-2025-12-16/raw/turbo/` (Dec 16 → Mar 9 train+val). Aggregator state carries seamlessly across the patch boundary; no state reset, no decay. ALL features (including hero-specific) get the extended history.
  - Same Bayesian smoothing (alpha=5), same recent-form window (10), same anonymous-account handling. NO time-decay this iteration — keep the "more data" lever isolated from the "weighting scheme" lever.
  - Model; LightGBM identical to player-features-740 + plateau-baseline-740 (500 rounds, lr 0.1, 31 leaves), same 5M-row stratified subset (seed=42) of train.
  - Same 3 ablations (heroes_plus_features, heroes_only sanity rebuild, features_only) and same coverage-bucket val_auc diagnostic; expect coverage buckets to flatten if pre-patch worked (the low bucket should rise toward the medium/high range; high bucket should rise modestly).
  - NEW diagnostic; "history-source breakdown" per player-row at snapshot time — what fraction of `n_games` came from pre-patch vs patch? Reported as a mean per coverage bucket. If pre-patch contributes substantially but val_auc doesn't lift, skill-drift is the next suspect (motivates time-decay aggregator).
  - HCE-strict; aggregator uses ONLY matches with start_time < current. Test window never read at any stage. Assert at build time AND train time as before.
risks:
  - Diminishing returns; adding 4 months of pre-patch history may give meager gains if most active patch-7.40 players already had enough in-patch history by the val window. Mitigation; the coverage-bucket + history-source-breakdown diagnostics directly distinguish "no signal" from "signal but small".
  - Player skill drift; a player active in Aug 2025 may have different skill in Mar 2026 (~7 months later). Without time-decay, the aggregator weighs old and recent games equally. Could regress val_auc rather than improve it. Mitigation; if val_auc < player-features-740's 0.6227 OR history-source breakdown shows pre-patch contributing substantially without lift, schedule a follow-up `player-features-decay` with exponential time-weighting (τ ≈ 90 days).
  - Sparse early-period collection; Aug-Oct 2025 has known gaps (Aug only 16/31 days, Oct 21/31 days per the Azure listing). Players with intermittent presence have noisier signal. Mitigation; just means features default to current values for those players — no model damage, just no additional lift from those periods.
  - Anonymous ceiling unchanged; 66% of player-slots remain anonymous regardless of how much history we have. The fundamental upper bound on what this whole approach can reach is bounded by the high-coverage bucket val_auc from player-features-740 (0.6296) plus whatever pre-patch history adds to the medium bucket. We will not break 0.640 with this experiment alone.
  - Disk and wall budget; 118 GB additional pull + 187 min feature rebuild (extrapolated from 69 min for 98 days × 266/98 ratio) + 15 min training ≈ 3.5 h total wall. Within budget.yaml max_wall_hours=24 and max_disk_gb=500. Total disk after pull ≈ 200 GB raw + ~3 GB processed.
related_prior:
  - 2026-05-17-player-features-740
  - 2026-05-15-plateau-baseline-740
estimated_runtime: "≈3.5 h wall (30 min Azure pull, 3 h feature rebuild including pre-patch chronological scan over ~265 days of raw, 15 min for 3 LightGBM ablations). Disk; ~118 GB additional raw under data/history/turbo/ (totals ~200 GB raw); ~3 GB processed parquet. Well under budget.yaml max_wall_hours=24 and max_disk_gb=500."
---

# Pre-patch history extension — testing the cold-start binding

`player-features-740` confirmed that per-player history contains signal (+0.0067 AUC), confirmed that the dominant lever is per-player-per-hero winrate (top-10 features by importance), and confirmed that cold-start is binding via a monotonic coverage-bucket diagnostic (low/med/high val_auc = 0.6159/0.6230/0.6296). The natural follow-up is to extend the history corpus and see whether the bottom-tercile val matches lift toward the high-tercile values once their players have meaningful lookback.

Hodge 2017 (`[[literature/papers/hodge2017win]]`) flagged metagame drift as a concern for cross-patch transfer, but the concern is specific to hero base rates, which our Bayesian-shrunk per-player-per-hero winrate already separates from the per-player deviation. A player who wins 62% on Pudge when the global is 50% is a +12pp deviation — and that deviation is dominated by "this player knows Pudge mechanics," not by patch-specific balance tweaks. So pre-patch hero-specific data is probably 80-90% as informative as in-patch data, and we get the full benefit by including everything we have.

Three result forks:

- **val_auc ≥ 0.6277 (confirmed).** Cold-start was the binding constraint. Coverage-bucket flattens. Pre-patch data is now baked into the feature derivation; downstream experiments (deeper per-player-per-hero, anonymous handling redesign, eventual player embeddings) start from this richer base. Next-natural step is time-decay aggregator to gracefully handle the older portion of the history.
- **val_auc in [0.6227, 0.6277) (partial / no clear lift).** Diminishing returns hit harder than expected, OR skill drift cancels the data-volume gain. The new history-source breakdown diagnostic distinguishes these cases. If pre-patch IS contributing substantially to feature values but val_auc doesn't move → drift is binding → `player-features-decay` (exponential time-weighting) is the next experiment. If pre-patch isn't contributing → diminishing returns is binding → pivot to deeper features (player-on-hero-pair, anonymous redesign) or to the Transformer + player features combination.
- **val_auc < 0.6227 (regression).** Skill drift catastrophically outweighs the data-volume gain. Schedule `player-features-decay` immediately and rerun with τ ≈ 90 days exponential decay.

This is the cheapest experiment that directly answers "is cold-start the binding constraint?" — a question the previous experiment's coverage-bucket diagnostic raised but couldn't answer alone.
