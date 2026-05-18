---
kind: proposal
slug: player-features-740
date: 2026-05-17
status: implemented
experiment: experiments/2026-05-17-player-features-740/
hypothesis: "Adding per-player historical features (smoothed overall and hero-specific win rates, recent form, premade-detection co-play counts) computed from the patch-7.40 snapshot's leading-window history to the existing LightGBM baseline raises val_auc by ≥ 0.020 over plateau-baseline-740 (target val_auc ≥ 0.6361), clearly exceeding both the architectural ceiling at 0.6322 (plateau-architectures-740) and the HP-tuned ceiling at 0.6318 (transformer-hp-sweep-740), and confirming the plateau is an information bottleneck (missing player identity) rather than a representation bottleneck."
rationale: >
  Three converging in-project signals say the ~0.632 ceiling is
  information-bound, not model-bound; 6 prior-art architectures
  spanning 4x capacity (DotaML), 3 mirrored architectures
  (plateau-architectures-740), and a 60-trial Optuna sweep finding
  only a 0.0008 AUC envelope around the control HP point
  (transformer-hp-sweep-740) all confirm any further gain requires a
  new INPUT axis. Hodge et al. 2017 independently attest the hero-only
  ceiling at 55-59% accuracy on Dota 2 (matching our 58.66 % within
  0.01) and show that admitting richer features (in-game telemetry)
  raises accuracy to 75-76 % — a 17 pp gap that demonstrates the
  broader prediction task has substantial headroom once feature sets
  beyond hero IDs are admitted. Hodge does NOT directly isolate the
  "player identity / MMR" contribution (the authors note this gap and
  defer to Yang/Qin/Lei 2016, not yet ingested); our `raw_json`
  already contains account_id × 10 per match, so this experiment
  tests the simplest pre-game-knowable feature axis that is currently
  unused. NOTE; this proposal deliberately broadens the task scope
  beyond `concepts/draft-only-win-prediction` (which excludes player
  identity by definition). On success, that concept will be refined
  or superseded by a `pre-game-win-prediction` concept.
reads:
  - "[[concepts/draft-only-win-prediction]]"
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/hero-embedding-vs-onehot]]"
  - "[[literature/repos/eschmitt88-DotaML]]"
  - "[[literature/papers/hodge2017win]]"
  - "[[experiments/2026-05-15-plateau-baseline-740]]"
  - "[[experiments/2026-05-16-transformer-hp-sweep-740]]"
expected_metric:
  name: val_auc
  target: 0.6361
  direction: higher-is-better
design_sketch:
  - Build a leading-window per-account history index from all 19.6M raw matches under data/snapshots/7.40-2025-12-16/raw/. Process matches in chronological order (sort by start_time); maintain per-account_id running aggregates; snapshot each player's features at match time T using only matches with start_time strictly < T, then update aggregates with the current match.
  - Per-player features (~10 cols × 10 players ≈ 100 cols), Bayesian-smoothed where appropriate; (a) n_games_in_window (log1p), (b) smoothed_winrate = (alpha + n_wins) / (2*alpha + n_games) with alpha chosen for ~10-game shrinkage to global prior, (c) smoothed_winrate_on_current_hero (same shrinkage with hero base rate), (d) last_10_winrate (recent form), (e) days_since_last_match (log1p), (f) n_games_on_current_hero (log1p), (g) hero_diversity = unique heroes played, (h) co_play_mean = mean co-play count with the 4 teammates (premade signal), (i) is_anonymous indicator.
  - Cold-start handling; anonymous accounts (account_id in {0, 4294967295} per Steam convention) get global priors; accounts with zero history use base rates. Include a per-match n_anonymous_in_match feature so the model can downweight low-info matches.
  - HCE-strict; aggregates use ONLY matches with start_time strictly less than the current match's start_time. Test window [2026-03-10, 2026-03-23] never read during search. Assert this at feature-build time.
  - Model; LightGBM with 300-dim hero one-hot + 1-bit Radiant side + ~100 player features. Same hyperparameters as plateau-baseline-740 (500 rounds, lr 0.1, 31 leaves) to isolate the feature effect.
  - Same 5M-row stratified train subset (seed=42) and the same val parquet as plateau-baseline-740, so direct A/B with baseline numbers is fair.
  - Diagnostic ablations; (1) heroes-only rebuild should match plateau-baseline-740 within 0.001 AUC (sanity check on the data pipeline); (2) features-only (no hero one-hot) measures the pure player-signal lift.
  - **Coverage-bucketed val_auc diagnostic** (tells us whether cold-start is binding without committing to a pre-patch ingest); for each val match, compute coverage = mean(n_games_in_window across 10 players), bucket into terciles (low / medium / high), report val_auc per bucket. Outcomes; flat across buckets → cold-start NOT binding (skip pre-patch ingest). Monotonic increase → cold-start IS binding → schedule `player-features-prepatch-740` follow-up to ingest patch-7.39 data for overall/recent/co-play features (skipping cross-patch hero-specific to avoid metagame-drift confound per Hodge 2017).
  - Metrics; val_auc, val_acc, val_log_loss, val_brier, calibration curve, LightGBM feature_importance ranking, and the coverage-bucketed val_auc table.
risks:
  - Cold-start dominance; the patch-7.40 snapshot is only ~98 days, so early-train matches have minimal player lookback. Mitigation; include n_games as a feature so the model can learn to weight low-history players' contributions appropriately, AND the coverage-bucketed val_auc diagnostic above directly measures whether this matters.
  - Anonymous-account prevalence; many Steam profiles hide match history (account_id reports as 4294967295 = UINT32_MAX, or 0). If >50% of player-rows are anonymous, the feature lever shrinks substantially. We will measure n_anonymous up front and report it.
  - Turbo skill signal is hidden; Turbo has no public MMR and is unranked, so the skill proxies derived here are necessarily noisier than ranked-AP literature. Expected effect could be smaller than ranked-MOBA work would suggest.
  - Quantitative target is loosely grounded. The +0.020 AUC target is based on (a) "any improvement beyond the architectural ceiling counts" reasoning, plus (b) Hodge's 17pp in-game-features gap as an order-of-magnitude proof-of-concept that information beats architecture, NOT on a paper that isolates the player-history contribution. If results disappoint, ingesting Yang/Qin/Lei 2016 (or similar) for a tighter prior is the natural next step.
  - Feature pipeline is non-trivial (~500-1000 LOC of sequential aggregation). Engineering risk of bugs. Mitigation; the heroes-only ablation must reproduce plateau-baseline-740 within 0.001 AUC; if it doesn't, the pipeline has a bug and we stop.
  - Scope expansion vs the original `draft-only-win-prediction` task definition. The user has explicitly broadened scope to "what's knowable before the game starts" — this proposal acts on that. The concept will be refined post-experiment.
related_prior:
  - 2026-05-15-plateau-baseline-740
  - 2026-05-16-transformer-hp-sweep-740
estimated_runtime: "≈2.5-3.5 h on CPU (feature build ~2-3 h sequential scan of 100 GB raw parquet with JSON parse + per-account aggregation across 19.6M matches; LightGBM training ~5 min). Disk; ~2-5 GB for the augmented processed parquet. Well under budget.yaml max_wall_hours=24."
---

# Player-history features — testing the information-bottleneck hypothesis

Three prior in-project experiments triangulate on the same conclusion: the ~0.632 AUC ceiling on patch-7.40 draft-only win prediction is **information-bound, not model-bound**. DotaML's 6-architecture grid, our 3-architecture mirror, and a 60-trial Optuna sweep all land within 0.02 AUC of each other despite spanning 4× model capacity and a 9-dimensional hyperparameter search. That pattern is the canonical signature of a model that has fully extracted the information in its input.

Hodge et al. 2017 (`[[literature/papers/hodge2017win]]`) independently confirms the hero-only ceiling: 55-59% accuracy on Dota 2 across LR and RF with feature selection, matching our `plateau-baseline-740` val_acc=0.5866 within 0.01. Crucially, the same paper shows that adding in-game telemetry (kills, gold, net worth, tower damage at 20 min) raises accuracy to 75-76% — a 17 pp gap that demonstrates the prediction task as a whole has substantial headroom once feature sets richer than hero IDs are admitted. Hodge does not directly isolate the player-identity contribution (they note this gap explicitly and defer to Yang/Qin/Lei 2016 for the claim that player-on-hero matters; we have not yet ingested that paper).

Our `raw_json` already contains `account_id × 10` per match — the simplest pre-game-knowable feature axis that we are currently throwing away. This experiment derives leading-window per-account aggregates from the snapshot itself (HCE-strict, no future leakage) and tests whether that lever moves val_auc beyond the architectural ceiling we've established.

A built-in **coverage-bucket diagnostic** decides whether a follow-up experiment fetching pre-patch (7.39 era) history would be useful. The snapshot is only 98 days, so early-train matches have minimal player lookback — but if val_auc is roughly flat across low/medium/high coverage terciles in val, cold-start is not binding and pre-patch ingestion is needless complexity. If val_auc rises monotonically with coverage, the follow-up `player-features-prepatch-740` (Sept-Dec 2025, ~75 days of 7.39 Turbo data) becomes high-priority for the next round.

Three result forks:

- **val_auc ≥ 0.6361 (confirmed).** Information bottleneck hypothesis validated for the pre-game feature axis. Next experiments target richer player features (lane/role preference, per-hero matchup history, time-of-day effects) and the player × hero interaction; structural model improvements become second-order. Likely drives a new `pre-game-win-prediction` concept that supersedes `draft-only-win-prediction`. The coverage-bucket diagnostic then informs whether pre-patch ingest is needed to push further.
- **val_auc in [0.6322, 0.6361) (partial).** Player features help but only marginally exceed the Transformer ceiling. Most likely cause: Turbo's hidden MMR and anonymous-account prevalence diluting the proxy. The coverage-bucket diagnostic is critical here: if low-coverage val matches drag the average down while high-coverage matches are healthy, the fix is pre-patch ingest; if all buckets are flat, the limit is genuinely the noisiness of Turbo skill proxies and the next move is to ingest Yang/Qin/Lei 2016 for a tighter prior.
- **val_auc < 0.6322 (not confirmed).** Either the feature engineering is broken (sanity check via the heroes-only ablation), or Turbo player skill is genuinely uninformative — which would be surprising given the broader MOBA evidence and would itself be a publishable finding.

This is the highest-leverage next experiment by a wide margin: it's grounded in independently-attested literature, addresses the specific failure mode three prior experiments converge on, requires no new data ingest, falsifies cleanly either way, and produces a coverage-bucket diagnostic that directly answers the "should we fetch pre-patch data?" question instead of guessing.
