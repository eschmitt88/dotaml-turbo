---
kind: proposal
slug: transformer-plus-features-740
date: 2026-05-18
status: implemented
experiment: experiments/2026-05-18-transformer-plus-features-740/
hypothesis: "Combining the MinimalTransformer architecture from plateau-architectures-740 (val_auc 0.6322) with the 80-dim per-player feature block from player-features-prepatch-740 (val_auc 0.6256 alone) — injecting per-player features as a learned Linear(8, d_model) projection added to each slot's hero embedding — raises val_auc by ≥ 0.005 over the better individual lever (target val_auc ≥ 0.6372, vs Transformer's 0.6322), confirming that hero-attention and per-player-per-hero skill features address sufficiently distinct information axes to be meaningfully additive."
rationale: >
  Architecture and player-features have been tested independently but
  never combined. Architecture-only Transformer reaches 0.6322;
  LightGBM + 80 player features reaches 0.6256; these are the two
  strongest individual levers and they address different information
  axes (attention captures hero-pair interactions, player features
  capture per-player skill). The HIGH-coverage tercile of val from
  player-features-prepatch-740 reached 0.6339 (already above
  Transformer-only), and the n_anon ≤ 1 subset reached 0.6447 —
  evidence that with enough player data the LightGBM head alone is
  competitive with attention. The natural test is whether combining
  the two levers gives meaningful additive gain on the whole val.
  The per-player feature injection design (add Linear(8,
  d_model)(player_feats) to each slot's hero embedding) is the
  simplest viable; it adds ~600 parameters to the ~82k-param model.
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[concepts/draft-only-win-prediction]]"
  - "[[concepts/hero-embedding-vs-onehot]]"
  - "[[literature/papers/hodge2017win]]"
  - "[[experiments/2026-05-15-plateau-baseline-740]]"
  - "[[experiments/2026-05-15-plateau-architectures-740]]"
  - "[[experiments/2026-05-16-transformer-hp-sweep-740]]"
  - "[[experiments/2026-05-17-player-features-740]]"
  - "[[experiments/2026-05-18-player-features-prepatch-740]]"
expected_metric:
  name: val_auc
  target: 0.6372
  direction: higher-is-better
design_sketch:
  - Reuse processed parquet from player-features-prepatch-740 at data/snapshots/.../processed/player_features_prepatch/{train,val}.parquet. NO new feature build. Same 5M stratified subset (seed=42).
  - Architecture is MinimalTransformer from plateau-architectures-740 with the HP point that won transformer-hp-sweep-740 (d_model=64, n_heads=4, n_layers=2, ff_mult=2, dropout=0, embed_dim=64) — these settings produced val_auc=0.6311 in the sweep's control trial and matched the prior 0.6322 architecture result.
  - Player-feature injection; per slot p in [0..9], read the 8 player features (n_games_log1p, smoothed_winrate, smoothed_winrate_hero, last10_winrate, days_since_last_log1p, n_games_hero_log1p, hero_diversity_log1p, is_anonymous), project via Linear(8, d_model) (~576 params), and ADD to that slot's hero_embedding + team_embedding before self-attention. Architecturally minimal change.
  - Training; same 14-epoch cap, bf16 autocast, num_workers=0, math SDP backend forced (per the Blackwell-torch-dataloader-bug memory). Adam lr=1e-3, batch_size=8192. Each ablation runs in a fresh subprocess.
  - Three ablations to isolate the effect; (1) architecture_only (Transformer, no player feats — should reproduce plateau-architectures-740's 0.6322 within 0.005); (2) features_only_lgbm (LightGBM + 80 player feats — already done in player-features-prepatch-740, val_auc 0.6256); (3) PRIMARY; transformer_plus_features (combined). Anchors fixed against existing metrics.
  - Coverage-bucket val_auc diagnostic carried over; expect HIGH bucket to lift toward the n_anon ≤ 1 ceiling we observed at 0.6447 in offline analysis.
  - HCE-strict; never read [2026-03-10, 2026-03-23] dates. Asserted at train time as in prior experiments.
risks:
  - Redundancy between hero-attention and per-player-per-hero features. The top-10 features by importance in player-features experiments were exclusively `pX_smoothed_winrate_hero` — and the Transformer's self-attention also learns hero-pair structure. The two might encode overlapping signal, producing less-than-additive combined gain.
  - Player-feature injection design choice. We're using "add to hero embedding" which is the simplest. Alternatives (concat as separate dim, dedicated player-token, gated mix) might work better but require iteration. If additive injection underperforms by >0.005 vs whole-val expectation, a follow-up experiment can swap to a richer injection.
  - Blackwell + torch 2.9 stability. Per the Blackwell-torch-dataloader-bug memory; each ablation runs in a fresh subprocess to absorb intermittent crashes. Wall budget accounts for ~20% retry overhead.
  - Anonymous-account tail unchanged. The 12.6% all-anonymous val matches still get only base-rate priors from the player features. Whole-val ceiling is structurally bound by this regardless of what the model does.
  - Target +0.005 over architecture-only may be optimistic if hero-attention and per-player-per-hero features are highly redundant. A +0.002 result (val_auc=0.6342) would still be informative but would suggest deeper architectural changes are needed for the next round.
related_prior:
  - 2026-05-15-plateau-architectures-740
  - 2026-05-18-player-features-prepatch-740
  - 2026-05-16-transformer-hp-sweep-740
estimated_runtime: "≈1.5 h on RTX 5080 (3 ablations × ~12 min each × ~1.2 retry overhead, no new feature build). Disk; <100 MB for new model checkpoints. Well under budget.yaml max_wall_hours=24 and max_disk_gb=500."
---

# Transformer + player features — combining the two strongest individual levers

Five prior experiments have established that (a) the ~0.632 plateau on hero-only inputs is information-bound, not capacity-bound, and (b) the two strongest individual levers — architecture (Transformer at 0.6322) and per-player history features (LightGBM + features at 0.6256) — address what appear to be distinct information axes. **The combination has never been tested.**

The offline analysis from earlier today (re-scoring the existing player-features-prepatch-740 model on data-availability slices) showed val_auc=0.6447 on the all-public subset (n_anon ≤ 1, 1.3% of val) and val_auc=0.6359 on the ≥5-players-with-history subset (21.4% of val) — both significantly above the architecture-only Transformer ceiling. This implies that on matches with rich player data, the features alone exceed what 82k attention parameters can extract. Combining the two should reach further.

Three result forks:

- **val_auc ≥ 0.6372 (confirmed).** Architecture and per-player-per-hero features are sufficiently independent that combining them gives meaningful additive gain on whole val. New whole-val ceiling established. Next experiments could push toward the active-subset ceiling (~0.645) via richer player features or anonymous-aware modeling.
- **val_auc 0.6322-0.6372 (partial — some additivity).** The two levers overlap more than expected; the player-features contribution is partially redundant with what hero-attention already encodes. The combined model still beats architecture-only but the gain is smaller than predicted. Next move: try a more architecturally distinct injection (dedicated player-token vs slot-level addition), or move to anonymous-aware modeling.
- **val_auc < 0.6322 (regression).** Unexpected — feature injection broke something. The architecture-only sanity ablation will catch this and we'd debug before drawing strong conclusions.

This is the experiment that most directly answers "is the player-features lever orthogonal to the architecture lever?" — a question implicit in the architecture-vs-information narrative across the prior five experiments.
