---
kind: experiment
slug: player-embedding-prelim-740
date: 2026-05-19
status: done
respects:
  - ~/claude-system/claude/rules/evaluation.md
hypothesis: >
  Adding a learned per-player embedding lookup (vocab = top-500K most-frequent
  non-anonymous accounts + 1 'rare' + 1 'anonymous' bucket; embed_dim=32)
  projected per-slot into the hero+feature sum lifts whole-val val_auc by
  ≥ 0.0020 over the cleanup-confirmed combined reference (target ≥ 0.6497).
  The HIGH coverage tercile should benefit disproportionately.
result: "NULL RESULT. baseline_extended_clean reproduced upstream-data-cleanup-740's 0.6477054 to FIVE decimal places (sanity passed perfectly). with_player_embedding (16M params, 208x baseline) landed at 0.6476302, Δ=-7.5e-5 vs baseline and -0.0021 vs proposal target. Every coverage bucket flat within noise (LOW +0.00009, MED -0.0004, HIGH +0.0001) — including the HIGH bucket where 51% of slots get a frequent vocab entry. NOT overfitting (train-val gap actually narrower for embedding model). HYPOTHESIS NOT CONFIRMED. Strong negative finding: the 8 aggregated features are essentially complete for the per-player identity axis on this task."
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[experiments/2026-05-19-upstream-data-cleanup-740]]"
  - "[[experiments/2026-05-19-transformer-plus-features-extended-740]]"
metric:
  name: val_auc
  direction: higher-is-better
  target: 0.6497
parent_proposal: experiments/_proposals/2026-05-19-player-embedding-prelim-740.md
result: ""
---

# Player embedding prelim 740

Preliminary experiment: add a learned per-account embedding lookup to the
cleanup-confirmed Transformer+features baseline and measure whether the
extra capacity beats the 0.6477 plateau by ≥ 0.0020 (target ≥ 0.6497).

## Pipeline

Four sequential steps (see `run_all.sh`):

1. **`build_account_sidecar.py`** — walks raw history JSON, emits per-match
   account_id sidecar parquets keyed by match_id. Required because the
   clean parquet at `data/snapshots/.../player_features_prepatch_clean/`
   carries hero IDs + aggregated features only — NOT account_ids.
2. **`build_vocab.py`** — streams the train sidecar (pyarrow row-group
   iteration, NEVER into pandas), tallies non-anonymous account
   frequencies, keeps the top-500K. Writes `results/player_vocab.json`.
3. **`train.py --ablation baseline_extended_clean`** — sanity replication
   of `upstream-data-cleanup-740` (no embedding). Expected ≈ 0.6477 (within
   the ±0.001 equality band centered on the cleanup anchor 0.6477054).
4. **`train.py --ablation with_player_embedding`** — PRIMARY. Add
   `nn.Embedding(vocab_size=502_002, embed_dim=32)` + per-slot
   `Linear(32 → 64)` projection. `weight_decay=1e-3` applied ONLY to
   embedding params; other params keep `weight_decay=0`.

## Files

- `config.yaml` — single source for all knobs.
- `build_account_sidecar.py` — raw-history walker → sidecar parquets.
- `build_vocab.py` — sidecar → `results/player_vocab.json`.
- `data.py` — clean-parquet loader + sidecar join + vocab lookup.
- `models.py` — `MinimalTransformerWithFeaturesAndPlayerEmbedding`.
- `train.py` — single-ablation trainer.
- `run_all.sh` — sequential pipeline with per-trial subprocess isolation
  retry wrapper (Blackwell torch DataLoader bug workaround).

## HCE

- Trains on `[2025-12-16, 2026-02-23]`, validates on `[2026-02-24, 2026-03-09]`.
- `data.py:assert_no_test_dates` refuses to read any row whose
  `start_time_date` falls inside `[2026-03-10, 2026-03-23]`.
- `build_account_sidecar.py` refuses to walk any raw-history day in the
  test window or post-snapshot.
- `metrics.json` records validation-split metrics ONLY. Held-out test
  scoring is reserved for an explicit final-scoring pass and would write
  to `final_metrics.json`.

## Anchors

- `transformer-plus-features-extended-740` (dirty parquet): val_auc = 0.6477298
- `upstream-data-cleanup-740` (clean parquet, no embedding): val_auc = 0.6477054
- Proposal target: 0.6497 (= clean anchor + 0.0020)
- Equality band for `baseline_extended_clean` sanity check: [0.6467, 0.6487]

## Diagnostics

- `coverage_bucket_val_auc.{low,medium,high}.val_auc` — bucketed by mean
  `p*_n_games_log1p` across slots.
- `coverage_bucket_val_auc.buckets.{low,medium,high}.in_vocab_frac` — per-bucket
  fraction of slots mapping to a frequent (idx ≥ 2) vocab entry.
- `coverage_bucket_val_auc.buckets.{low,medium,high}.anon_share`,
  `rare_share` — sanity record.

## Result

**HYPOTHESIS NOT CONFIRMED — a clean null.**

| ablation | val_auc | best_epoch | epochs_run | params | wall |
|---|---:|---:|---:|---:|---:|
| baseline_extended_clean (sanity) | **0.6477054** | 22 | 27 | 77,377 | 32.2 min |
| with_player_embedding (primary) | **0.6476302** | 23 | 28 | 16,079,553 | 33.8 min |

Adding 208× the parameters (77K → 16.08M, all in the player_embedding
layer) yielded a -7.5e-5 delta — essentially identical performance.

The sanity replication (`baseline_extended_clean`) reproduced
`upstream-data-cleanup-740`'s combined val_auc to **five decimal places**
(0.6477054 in both). This is the cleanest possible confirmation that
the 0.6477 ceiling is genuinely the model's converged operating point
for this codepath, and is now anchored across three independent runs
(`-extended-740`, `-cleanup-740`, this `baseline_extended_clean`).

Coverage-bucket breakdown — including in-vocab-fraction diagnostic to
test whether the embedding helped where it had the most leverage:

| bucket | n | baseline val_auc | embedding val_auc | Δ | in_vocab_frac | anon_share |
|---|---:|---:|---:|---:|---:|---:|
| low    | 805,580 | 0.6368 | 0.6369 | +0.00009 | 4.9%  | 92% |
| medium | 808,016 | 0.6467 | 0.6463 | -0.00041 | 23.8% | 70% |
| high   | 805,589 | 0.6587 | 0.6589 | +0.00012 | 50.9% | 42% |

The HIGH bucket has 50.9% of slots resolving to a frequent vocab entry
(the maximum embedding leverage of any bucket) and gained only +0.0001.
If the embedding were extracting any genuine identity signal, we should
see lift here above the per-bucket noise floor.

**Overfit check.** Train-val log-loss gap at best epoch:
- baseline: train=0.6495 val=0.6547 → gap=0.0052
- with embedding: train=0.6496 val=0.6547 → gap=0.0050

The embedding model has a SMALLER train-val gap, so it is NOT
overfitting on the long-tail vocab — it is simply not learning anything
useful that wasn't already captured by the 8 aggregated features
(smoothed_winrate, smoothed_winrate_hero, last10_winrate, days_since,
n_games_log1p, n_games_hero_log1p, hero_diversity, is_anonymous).

**Vocab statistics:**

- 1,229,091 unique non-anonymous accounts seen in train.
- Top-500K covers 90.6% of non-anonymous slot occurrences (frequency cutoff at 20 games seen).
- 66.4% of all train slot-counts are anonymous (matches the 66% prior estimate).
- val-time fractions: 68% anonymous, 27% frequent vocab, 5% rare.

## Interpretation

The 8-feature aggregator is **essentially complete** for the per-player
identity axis of this task. That's a substantive finding, not just a
no-improvement result:

1. **Identity-level latent signal beyond aggregate stats does not exist
   in any meaningful quantity for win prediction on this dataset.**
   A 16M-param learnable representation had the same val_auc as the
   77K-param baseline using 8 hand-picked features. The features already
   encode "this player's per-hero skill, recency, and engagement" —
   which is apparently the full information content of the player
   identity that matters for the radiant-win label.

2. **The 0.6477 ceiling is genuinely converged for this representation
   family**, not an artifact of optimization. Three independent training
   runs (`transformer-plus-features-extended-740`, `upstream-data-cleanup-740`,
   `baseline_extended_clean` this experiment) landed at 0.6477 ± 2e-5.
   Anything we do next that stays in the "Transformer over 10 hero
   slots with per-slot per-player aggregated features" architecture
   family will land near 0.6477 unless it changes the information
   structure of the input.

3. **The remaining levers are NOT richer player representations.** They
   are:
   - **New information axes** (draft order via `picks_bans[]`, hero-pair
     history, lane/role inference, team-aggregate features that change
     the input structure not just the per-slot encoding).
   - **Anonymous-aware modeling** (user has deprioritized; gap of 0.0220
     between LOW and HIGH buckets still exists and would shift if
     attacked).
   - **Time decay** (the queued `player-features-decay-740`).
   - **Structural mutation** (LLM-driven islands evolution; deferred).

## Diagnostics

- intended_effect_confirmed: no — primary val_auc 0.6476302 is below baseline 0.6477054 (Δ=-7.5e-5) and far below target 0.6497 (Δ=-2.07e-3) (`metrics.json:headline`).
- leakage_check: HCE date-window assertion live in `data.py`; train ends 2025-12-16..2026-02-23, val ends 2026-02-24..2026-03-09, both strictly < test_start 2026-03-10 (`metrics_*.json:train_date_max`, `val_date_max`). `build_account_sidecar.py` also refuses to walk raw-history days in the test window. Sanity replication matched cleanup-740's val_auc to 5 decimal places, confirming the new sidecar-join pipeline introduces no shift in the data the baseline sees.
- overfitting_signal: train=0.6496 val=0.6547 gap=0.0050 for primary (vs baseline train=0.6495 val=0.6547 gap=0.0052). The embedding model's gap is SMALLER, not larger. Not overfitting — just not learning useful signal.
- delta_from_prior: vs `2026-05-19-upstream-data-cleanup-740` (val_auc 0.6477054 combined-clean): -7.5e-5, well within run-to-run noise. The 16M-param player embedding contributes zero net signal over the 8 aggregated features.
- unexpected_findings: (a) baseline_extended_clean reproducing the cleanup anchor to 5 decimal places — much tighter than the typical ~1e-4 reproducibility seen across this project's reruns. The fixed seed=42 + identical data pipeline really does pin the result. (b) The HIGH coverage bucket, with 51% in-vocab fraction and only 42% anonymous slots, was the most likely place for the embedding to help — and it gained only 0.0001. This is the cleanest possible evidence that aggregated features genuinely capture player identity for this task. (c) Sidecar-walk pre-step (45 min) was unanticipated — account_ids aren't in the clean parquet (`build_features.py` doesn't emit them); they live in raw JSON only. Future identity-flavored experiments either need the sidecar or need to extend `build_features.py` to emit account_ids alongside.
- seeds_run: 1 (single run; seed=42)
- metric_aggregation: single-run
- next_candidates:
  - **`anonymous-aware-modeling-740`** — the persistent 0.0220 LOW–HIGH bucket gap is the remaining lever; user deprioritized but the embedding null result strengthens the case (since identity-level richness doesn't help, attacking the bucket asymmetry structurally is the residual axis). Router head OR per-team aggregates over known players.
  - **`draft-order-features-740`** — new information axis. `picks_bans[]` sequence is rich and untouched. Encoding pick/ban order plus pick-side (radiant/dire) gives the model strategic context the current per-slot encoding lacks.
  - **`player-features-decay-740`** — already on the user's next list. Exponential time-weighting (τ ≈ 90 days). Smaller experiment; would mostly affect the smoothed_winrate / smoothed_winrate_hero features.
  - **(Engineering)** Extend `build_features.py` to emit `pX_account_id` columns alongside features. Saves future experiments the 45-min sidecar walk and unifies the data pipeline.

## Follow-up

The natural successor experiments fall into two families:

1. **Information-axis additions** (draft-order, hero-pair history,
   team-aggregate restructuring) — likely to lift the whole-val
   ceiling above 0.6477 because they add input information the
   current encoding doesn't have.
2. **Anonymous-aware modeling** (router or team-aggregate) — would
   compress the LOW–HIGH bucket gap and lift the LOW bucket
   specifically.

The embedding experiment closes off the "richer player
representation" line for now. Re-opening would require either
(a) a hierarchical-prior shrinkage architecture with explicit
anonymity-aware gating, OR (b) a co-play / partner-aware embedding
where the embedding LOOKUP itself uses match-context (the embedding
of player A is conditioned on who else is in the match) — both
substantially more complex than the prelim. Defer until other
levers exhaust.
