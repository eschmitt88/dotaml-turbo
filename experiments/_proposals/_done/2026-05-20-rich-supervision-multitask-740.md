---
kind: proposal
slug: rich-supervision-multitask-740
date: 2026-05-20
status: implemented
experiment: experiments/2026-05-20-rich-supervision-multitask-740/
hypothesis: "A multi-task Transformer (shared encoder over draft + player aggregates; separate heads for win / duration / items / KDA-GPM-damage) trained jointly with weighted losses lifts whole-val win val_auc by ≥ 0.001 over the cleanup-confirmed 0.6477 reference (target val_auc ≥ 0.6487). Even if win val_auc doesn't move, the auxiliary outputs (duration-bucket curve, per-slot item recommendation) are useful in their own right, and the multi-task diagnostics tell us whether the encoder bottleneck is capacity, training-signal density, or true pre-game information."
rationale: >
  Three independent runs (`extended-740`, `cleanup-740`,
  `player-embedding-prelim-740` sanity replication) now pin the
  whole-val win ceiling at 0.6477 ± 2e-5 for the Transformer-over-
  draft-plus-aggregated-features architecture family. The
  embedding-prelim NULL result showed that adding 16M params of
  identity lookup yielded zero lift AND zero train-loss
  improvement, suggesting the gradient signal is the binding
  constraint, not the parameter count. Multi-task supervision on
  rich in-game telemetry (duration, items, KDA, GPM, hero_damage)
  gives every match ~10-50× more bits of supervision than the
  single radiant_win label, while keeping the encoder inputs
  strictly pre-game (HCE preserved). This tests "richer supervision
  helps the encoder learn" through a different mechanism than the
  embedding lookup.

  Audit confirms the raw parquets carry the full Steam Web API
  match-details payload per match — duration, picks_bans with order,
  per-player items (item_0..5 + neutrals), KDA, GPM, XPM, hero_damage,
  tower_damage, ability_upgrades with timestamps — all 175 GB already
  on disk
  (81 GB patch-7.40 + 94 GB pre-patch). A one-time JSON-parse pass
  materializes the rich columns into a sidecar parquet that the
  multi-task trainer joins to the existing clean per-player-features
  parquet at load time. No new ingest required.

  Beyond the win-prediction ceiling test, the two auxiliary outputs
  are useful products in their own right:
    - **Duration head** answers "given this draft, what's the win
      curve P(radiant_win | duration)?" → tactical guidance
      ("end early vs scale").
    - **Item head** answers "given this matchup and my slot, what
      items do players typically end up with?" → matchup-aware item
      recommendation. Descriptive in v1 (can layer skill-weighting
      later for prescriptive).
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[literature/papers/hodge2017win]]"
  - "[[experiments/2026-05-19-upstream-data-cleanup-740]]"
  - "[[experiments/2026-05-19-transformer-plus-features-extended-740]]"
  - "[[experiments/2026-05-19-player-embedding-prelim-740]]"
expected_metric:
  name: val_auc
  target: 0.6487
  direction: higher-is-better
design_sketch:
  - **Build rich-columns sidecar.** New script `build_rich_cols.py` walks `data/snapshots/7.40-2025-12-16/raw/turbo/` (filtered to match_ids in the clean parquet), parses the `raw_json` column once, emits `data/snapshots/7.40-2025-12-16/processed/rich_cols/{train,val}.parquet` with columns per match (1.7 GB est.):
    - `match_id` (key)
    - `duration` (int, seconds)
    - `p{0..9}_items` (list<int>, includes item_0..item_5 + item_neutral + item_neutral2, deduplicated; "0" / empty slots dropped)
    - `p{0..9}_kills`, `p{0..9}_deaths`, `p{0..9}_assists` (int)
    - `p{0..9}_gpm`, `p{0..9}_xpm`, `p{0..9}_hero_damage`, `p{0..9}_net_worth` (int)
    - Per-feature physical-bounds clamp at extract time + post-write column-stat check (avoid the fp32-sentinel pattern). Sidecar restricted to train+val match_ids only (HCE; test window not parsed).
  - **Build item vocab.** Stream `build_item_vocab.py` over train sidecar. Count item-ID occurrences across all 10 slot-lists. Keep all items with ≥ 100 train occurrences (expect ~150 items). Save as `results/item_vocab.json`. Anything sub-cutoff maps to a single "rare-item" bucket.
  - **Duration bucketing.** Quantile-based, computed from train.parquet at vocab-build time. Target 8 buckets, each with ~12.5% of train mass. Save edges in vocab JSON for reproducibility. (Turbo skews short; expected edges roughly at 15/18/21/24/27/30/35 min.)
  - **Architecture (shared encoder + multi-head).** Reuse `experiments/2026-05-19-upstream-data-cleanup-740/models.py`'s MinimalTransformerWithFeatures verbatim for the encoder + CLS token. Heads:
    - `win_head`: Linear(d_model=64, 1) over CLS → BCEWithLogits target=radiant_win
    - `duration_head`: Linear(d_model, 8) over CLS → CrossEntropy target=duration_bucket
    - `item_head`: per-slot Linear(d_model, item_vocab_size) → BCEWithLogits multi-label target=p{slot}_items_one_hot
    - `aux_head` (representation-shaping): per-slot Linear(d_model, 3) → SmoothL1 target=(kda_norm, gpm_norm, hero_damage_norm), where each target is standardized over the train split
  - **Joint loss.** `L = α_w·L_win + α_d·L_dur + α_i·L_item + α_a·L_aux` with α_w=1.0, α_d=0.5, α_i=0.3, α_a=0.1 (initial; subagent may tune on smoke). Aim: win remains the primary task with reasonable contribution from the rest. Track each component loss separately in `history`.
  - **Training recipe identical to cleanup-740.** Transformer 64-dim, 4-head, 2-layer. Adam lr=1e-3, batch_size=8192, bf16 autocast, max_epochs=30, early-stop patience=5 on val WIN log-loss (not total log-loss — we want to early-stop on the primary task to avoid the aux dominating early-stop decisions).
  - **Two ablations:**
    1. `win_only_sanity` — only the win head, all other heads/losses disabled (α_d = α_i = α_a = 0). Sanity: should reproduce cleanup-740's 0.6477054 to ~1e-4 (the new rich-cols pipeline doesn't perturb the data the encoder sees).
    2. `multitask_all` — PRIMARY: all four heads active.
  - **Per-trial subprocess isolation** via run_all.sh (MAX_RETRIES=3 per ablation), per the Blackwell torch DataLoader memory.
  - **HCE strict.** Train ≤ 2026-02-23, val ≤ 2026-03-09, test [2026-03-10, 2026-03-23] sealed (no `build_rich_cols.py` walk into the test window). Encoder inputs are pre-game only at both train and inference time; rich-cols are training TARGETS only — they never enter the encoder.
  - **Diagnostics (NON-NEGOTIABLE):**
    - Per-head val metrics: win (auc, log_loss, brier), duration (top-1 acc, brier across buckets), items (per-slot mAP @ 10 + qualitative top-5 items for 3 hand-picked matchups), aux (per-target val MSE).
    - Coverage-bucket win val_auc (low/med/high terciles by mean n_games_log1p) for both ablations, comparable across the project.
    - Train-vs-val gap on the win component AND on the aux components (overfit signal — does the aux start overfitting before win?).
    - `delta_vs_cleanup_anchor`: vs 0.6477054.
    - `multitask_helped_win`: bool, true iff `multitask_all.val_auc - win_only_sanity.val_auc ≥ 0.001`.
risks:
  - **Win head may regress.** Multi-task can hurt the primary head if the aux losses pull the encoder toward representations that are useful for items/duration but not for win. Mitigation: the sanity ablation reproduces 0.6477; if multitask drops below the cleanup band [0.6467, 0.6487], the loss weights are wrong. First retry: drop α_a to 0.05 and α_i to 0.15. Second retry: only keep the duration head.
  - **Item vocab + multi-label loss is computationally heavy.** ~150-item vocab × 10 slots = 1500 BCE outputs per match. Tractable but may slow per-epoch time ~2×. Acceptable.
  - **Rich-cols build is ~3-4 h CPU.** Add post-write column-stats check; do NOT re-read the full sidecar (see `[[aiserver2026-postwrite-parquet-reread-oom]]`). Sidecar is sized similarly to the clean parquet (~1.7-3 GB est.), so OOM risk is lower than the cleanup-740 case, but the lesson applies.
  - **Duration head conditional bias.** Long Turbo games are selection-biased toward close matches; reading the curve as a counterfactual ("what if we forced 40 min") would be wrong. Document this in the experiment README and in the duration head's notes; the model is correctly learning a CONDITIONAL distribution.
  - **Item head is descriptive not prescriptive.** Predicts what players typically buy, not what they should. Adequate for v1; layer skill-weighting in a follow-up if v1 looks promising.
  - **Joint α-weighting may need retuning.** Initial 1.0/0.5/0.3/0.1 is a guess; smoke results may suggest a different ratio. The subagent should report whether the per-head losses look balanced (all decreasing, none dominating).
related_prior:
  - 2026-05-19-upstream-data-cleanup-740
  - 2026-05-19-transformer-plus-features-extended-740
  - 2026-05-19-player-embedding-prelim-740
estimated_runtime: "≈5-6 h total on RTX 5080: ~3-4 h rich-cols JSON parse (CPU-bound, single-pass over raw parquets) + ~2 min item vocab + ~30-45 min win_only_sanity (similar to cleanup-740's Transformer step) + ~60-90 min multitask_all (~2× per-epoch due to per-slot multi-label item head). Disk delta: ~2-3 GB rich-cols sidecar + ~70 MB vocab + checkpoints. Well under budget.yaml's 24-h and 500-GB ceilings."
---

# Rich-supervision multi-task — close (or characterize) the 0.6477 ceiling

The 0.6477 ceiling is now anchored across three independent runs within 2e-5. The embedding-prelim's NULL result was diagnostically informative: 16M added params yielded ZERO train-loss improvement, meaning the embedding gradient signal got washed out by the dominant hero+feature gradients rather than overfit. This points at **training-signal density** rather than encoder capacity as the binding constraint.

Hodge 2017 reports a 17 pp accuracy gap between hero-only (~58%) and in-game-telemetry-included (~75-76%) Dota 2 win prediction. To the extent that in-game outcomes (duration, items, KDA, damage) are STABLE attributes of the players involved given the draft, predicting them as auxiliary targets should give the encoder richer per-match gradient signal — without changing the encoder's pre-game-only input contract.

This is a different test of the same intuition as `player-embedding-prelim-740` (richer per-match learning signal helps the encoder), but via a different mechanism: instead of adding embedding capacity that competes with the well-fit feature signal, we add training-signal density across multiple heads sharing one encoder.

**Three result forks:**

- **win val_auc ≥ 0.6487 (CONFIRMED).** Multi-task supervision lifts the primary task. The new whole-val reference is the multitask number. Follow-ups: (a) layer player embeddings back on top with the multitask encoder as initialization, (b) iterate loss weights for further lift, (c) add the talent-pick sequence as another aux head.
- **win val_auc within [0.6467, 0.6487] (FLAT).** Multi-task didn't hurt, didn't help. The encoder's representation was already capacity-bound by pre-game information. Still useful: duration + item heads as standalone tools. Next ceiling-breakers must come from genuinely NEW pre-game information (draft-order via picks_bans, anonymous-aware modeling).
- **win val_auc < 0.6467 (REGRESSION).** Loss weights pulled encoder toward aux representations at the expense of win. Tighten α's and retry; if persistent, multi-task is the wrong tool here.

**The auxiliary outputs are useful regardless.** The duration model and per-slot item recommender are immediately actionable products for the user's personal-use framing — "end early vs scale" intuition for any draft, and "what to buy on Pudge vs this Lifestealer comp" recommendations. Even a flat or regressing win head still gives the user two new working tools.

**Engineering shape.** The rich-cols extraction is a one-time pass; the resulting sidecar becomes shared infrastructure for any future multi-task or post-game-conditioned experiment (e.g., a future "given the first 10 min, predict the rest" model). The per-slot multi-label item head is straightforward (BCEWithLogits over ~150-class output); the per-slot regression aux is even simpler. The largest engineering risk is the rich-cols build OOM-ing during validation, which is mitigated by adopting the pyarrow row-group-stats verification pattern from cleanup-740.
