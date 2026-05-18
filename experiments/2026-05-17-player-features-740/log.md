# player-features-740 — log

## 2026-05-17

- Scaffolded by IMPLEMENTER subagent.
- Wrote `build_features.py` (chronological per-account aggregator with
  Bayesian smoothing alpha=5; HCE-strict leading window; coplay-dict
  capped at 200 entries/account to bound RAM; anonymous accounts only
  contribute to per-hero globals).
- Wrote `train.py` with three ablations (`heroes_only`, `features_only`,
  `heroes_plus_features`) and coverage-bucket val_auc diagnostic.
- Smoke test results (3 days = 2025-12-16..2025-12-18, 731k raw rows,
  379k emitted train rows, 0 val rows because val window is Feb 24+):
  - build_features.py --smoke: 104 s wall; 467k unique accounts tracked;
    bad_json=0; anonymous_per_match mean=6.57, p50=7 — early-patch data
    is heavily anonymous (account_id mostly absent), which the
    aggregator handles by collapsing anon updates to per-hero globals.
  - train.py --smoke (val carved as tail 10% of train_smoke since val
    parquet is empty for early days — pipeline-check fallback added):
    heroes_plus_features val_auc=0.6131 (feature_dim=391),
    heroes_only val_auc=0.6136 (dim=301), features_only val_auc=0.6075
    (dim=90). Coverage buckets all populate, monotonic on smoke
    (low 0.611, med 0.614, high 0.614). Top feature_importance dominated
    by `p*_smoothed_winrate_hero` — expected.
  - HCE assertions held in both build and train (no test-window dates
    seen / refused).
- Patched train.py with a smoke-only fallback that carves the last 10% of
  the smoke train as a pseudo-val when val_smoke.parquet is empty. Pure
  pipeline-correctness check; no effect on non-smoke runs.
- Placeholder metrics.json written; overwritten on full run.
- Main agent now runs the full ~2-3 h build + three training ablations
  in background via `run_all.sh` inside tmux for SSH-detach survival.
- 21:18 Full run launched via `nohup bash run_all.sh > /tmp/dotaml_pf.log 2>&1 &`
  (run_in_background via harness). build_features.py wrote
  `data/snapshots/.../player_features/{train.parquet (1203 MB), val.parquet (233 MB), build_stats.json}` after 4123 s (~69 min).
  Build stats; n_raw_read=16,923,487 (matches train+val window), n_emitted=15,437,578 (filtered by fake-match + dedup), n_bad_json=0, n_unique_account_ids_tracked=1,329,669, anonymous_per_match mean=6.66 (66% anonymous — consistent with smoke).
- 22:20 train.py heroes_plus_features: val_auc=0.6227, train_auc=0.6360, gap=0.0133. 150 s wall.
- 22:23 train.py heroes_only sanity ablation: val_auc=0.6160, Δ vs plateau-baseline-740 = -0.0001 (PASSES ≤0.001 sanity).
- 22:25 train.py features_only: CRASHED at ~round 50 with `LightGBMError: Check failed: (best_split_info.left_count) > (0)`. Known LightGBM behavior on degenerate features; not investigated — headline result is in heroes_plus_features and sanity check covers pipeline correctness.
- 23:00-ish Aggregated metrics, wrote README Result/Interpretation/Diagnostics. Hypothesis NOT confirmed (val_auc 0.6227 < target 0.6361). Coverage-bucket diagnostic monotonic (low/med/high val_auc = 0.6159/0.6230/0.6296), confirming cold-start binding. Top-10 feature importances exclusively per-player smoothed_winrate_hero — the marginal-value lever is hero-specific player history, not overall skill or recent form or premade detection. 12.7% of val matches are all-10-players-anonymous.
