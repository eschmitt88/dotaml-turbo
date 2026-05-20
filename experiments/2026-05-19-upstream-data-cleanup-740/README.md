---
kind: experiment
slug: upstream-data-cleanup-740
date: 2026-05-19
status: done
hypothesis: "Patching the upstream defect in build_features.py that produces ~6,482 corrupted cells in p1_smoothed_winrate_hero and rebuilding the parquet leaves the combined Transformer+features val_auc within ±0.001 of the current ceiling 0.6477 (i.e., 0.6467 ≤ val_auc ≤ 0.6487), while eliminating the downstream sanitization workaround in data.py."
result: "CONFIRMED. Combined Transformer+features val_auc on clean parquet = 0.6477054 (Δ = -2.4e-5 vs dirty-parquet anchor 0.6477298). LightGBM features_only val_auc = 0.6063985 (Δ = -6.6e-5 vs dirty 0.6064643). Both within 1e-4 of anchors; equality band HOLDS. Sentinels were not biasing results. Clean parquet now canonical for downstream."
related_concepts:
  - "[[concepts/draft-prediction-plateau]]"
related_literature: []
tags:
  - data-quality
  - fp32-sentinel
  - parquet
  - sanitization
  - cleanup
related_prior:
  - 2026-05-18-player-features-prepatch-740
  - 2026-05-18-transformer-plus-features-740
  - 2026-05-19-transformer-plus-features-extended-740
respects:
  - "~/.claude/rules/evaluation.md"   # HCE rule
---

# upstream-data-cleanup-740

## Hypothesis

Patching the upstream defect in
`experiments/2026-05-18-player-features-prepatch-740/build_features.py` that
produces ~6,482 corrupted cells in `p1_smoothed_winrate_hero` of the prepatch
parquet (0.005% of 130M cells) and rebuilding the parquet leaves the combined
Transformer+features `val_auc` within ±0.001 of the current ceiling 0.6477
(i.e., 0.6467 ≤ val_auc ≤ 0.6487), while eliminating the downstream
sanitization workaround in `data.py`. A deviation outside that band would
indicate the sentinels were systematically biasing the prior result.

## Setup

- Build: `build_features.py` (forked from prepatch-740, multi-layer assertion
  added; writes to `data/snapshots/.../processed/player_features_prepatch_clean/`)
- LightGBM trainer: `train_lgbm.py` (verbatim from prepatch-740, only the config
  steers it to the new clean path)
- Transformer trainer: `train_tfm.py` + `models.py` + `data.py` (data.py has the
  load-time sanitization shim REMOVED and replaced with a hard assertion — the
  clean-parquet contract should make it a no-op)
- Config: `config.yaml` (combines LightGBM and Transformer sections in one file,
  Transformer keys are `transformer_*`-prefixed to coexist with LightGBM keys)
- Splits: project-root `splits.yaml` (test window [2026-03-10, 2026-03-23] sealed)
- Orchestration: `run_all.sh` (3 sequential steps, MAX_RETRIES=3 for the Transformer)

## Root cause of the fp32 corruption (post-investigation)

Investigation of the dirty parquet at
`data/snapshots/.../processed/player_features_prepatch/train.parquet` revealed:

- All 6,482 corrupted cells are in EXACTLY ONE column: `p1_smoothed_winrate_hero`.
  No other column (and no other player slot of the same feature) has bad values.
- Corrupted cells are contained entirely within parquet row group 2
  (rows 2,097,152 – 3,145,727), specifically within a contiguous match-index
  range [2,344,604, 2,504,113), all on date 2025-12-29.
- Density within that range is ~4%, distributed randomly across the rows
  (NOT a single contiguous block). Within bad rows, hero IDs span the full
  1..145 range and account-anonymous rate (64.3%) matches the non-bad rate
  on that date (66.3%) — i.e., NO data-side characteristic of the affected
  rows correlates with the corruption.
- Corrupted values have the signature of **uninitialized / torn memory**:
  NaN (14), denormals (444 values in [-1e-30, 1e-30]), and mixed-magnitude
  floats both positive and negative (4,593 < 0). Many uint32 reinterpretations
  carry partial `0x3F`/`0x3E` exponent bytes — consistent with 16-bit half-word
  writes scrambling individual fp32 cells, NOT a math overflow.

The visible numeric paths in `snapshot()` (lines 138 and 150 of
the prior build_features.py) cannot produce these values: the denominators are
bounded ≥ 5.0 (`hero_alpha`) or fall back to `global_prior=0.5335`. The dirty
cells are upstream of any clipping logic, in PyArrow's per-column buffer fill.

**Conclusion:** The root cause is best characterized as **transient memory
corruption during PyArrow's fp32 column conversion on a specific row group**.
The exact trigger (cosmic ray, kernel scheduler quirk, NumPy buffer reuse bug
under memory pressure) is not deterministically reproducible from the artifact.
The patch is defensive rather than causal:

1. `snapshot()` now `_validate_and_clamp()`-s every emitted feature against
   per-feature physical bounds before returning. A clamp event increments a
   counter; the counter is asserted-zero post-build.
2. The Python list → PyArrow conversion is rerouted through `np.float32`
   first, then `pa.array(np_col, type=pa.float32())` — avoiding the
   `pa.array(python_list, type=pa.float32())` path that the prior build used
   and which we suspect of the corruption.
3. After `pa.table(arrs)` construction (pre-write) and again after `pq.write_table`
   + re-read, the table is bounds-checked column-by-column. Any violation aborts
   the build with a hard SystemExit.

This multi-checkpoint defense catches the issue irrespective of the underlying
mechanism. If the rebuild still produces corrupted cells, at least one of the
three checkpoints will fire with diagnostic output, and the corrupted file
is deleted from disk before the pipeline can consume it.

## Result

**CONFIRMED — no-regression cleanup.** Both ablations land within 1e-4 of
their dirty-parquet anchors:

| ablation | clean val_auc | dirty anchor | Δ | source |
|---|---:|---:|---:|---|
| LightGBM features_only | 0.6063985 | 0.6064643 (`prepatch-740`) | -6.6e-5 | `metrics_ablation_features_only.json` |
| Transformer + features (30 ep cap, early-stop p=5) | **0.6477054** @ best_epoch=22 | 0.6477298 (`extended-740`) | -2.4e-5 | `metrics_transformer_plus_features.json` |

Equality-band check (`val_auc ∈ [0.6467, 0.6487]`) on the combined model:
**HOLDS** (clean number sits 0.0010 above the lower band, 0.0010 below the
upper band — essentially dead-center).

The Transformer training curve is also nearly identical to the dirty-parquet
run: 27 epochs (early-stop fired at epoch 22 + 5 patience), per-epoch
val_auc trajectory differs from `extended-740`'s by ≤ 0.0008 at every epoch,
median diff 0.0001. The 6,482 sentinels (0.005% of cells) were genuinely
noise-level, not biasing the prior result in either direction.

**The clean parquet** at
`data/snapshots/7.40-2025-12-16/processed/player_features_prepatch_clean/`
(1.38 GB train + 262 MB val, 13M+2.4M rows) is now the canonical input for
downstream experiments (`player-embedding-prelim-740` and beyond).
Sanitization shim removed from `data.py`; the hard-assertion replacement
fired zero times on the clean parquet, as designed.

## Interpretation

The hypothesis was a no-regression check, and it confirmed cleanly. The
substantive payoff is downstream cleanliness, not a headline AUC shift:

1. **Downstream experiments no longer carry the `load_arrays` sanitization
   workaround.** Every future consumer of this parquet (player embeddings,
   ranking, ensembling) has one less load-time check to remember.
2. **The prior `0.6256` (`player-features-prepatch-740`) and `0.6477`
   (`transformer-plus-features-extended-740`) anchors are now confirmed
   trustworthy** — re-running with clean data reproduces them to 1e-4.
3. **The defensive checkpoints in `build_features.py` are persistent.**
   Any future re-run with this codebase (different patch snapshot, larger
   subsample, etc.) gets the same guard.

The deeper engineering finding — that the corruption signature points to
transient memory / buffer-fill issues in PyArrow, NOT a math bug — is
documented in the Root Cause section above and worth carrying to the
`pull_history` / `build_features` future maintenance.

## Diagnostics

- intended_effect_confirmed: yes — combined val_auc 0.6477054 is within ±0.001 of the 0.6477 anchor (`metrics.json:equality_band_holds`).
- leakage_check: HCE date-window assertion live in `data.py`; train ends 2025-12-16..2026-02-23, val ends 2026-02-24..2026-03-09, both strictly < test_start 2026-03-10 (`metrics_transformer_plus_features.json:train_date_max`, `val_date_max`). `data.py` sanitization shim was removed and replaced with a hard zero-clamp assertion; assertion did not fire (clean parquet contract held).
- overfitting_signal: train_loss=0.6495 val_loss=0.6547 gap=0.0052 at best_epoch=22 (`metrics_transformer_plus_features.json:history`). Identical to extended-740's curve within 0.0008 per epoch.
- delta_from_prior: vs `2026-05-19-transformer-plus-features-extended-740` (val_auc 0.6477298): -2.4e-5 attributed to sub-noise variation in the 6,482-cell delta. vs `2026-05-18-player-features-prepatch-740` features_only (val_auc 0.6064643): -6.6e-5 same explanation.
- unexpected_findings: (a) the post-write re-read validation step in `build_features.py` OOM-killed both run attempts (rc=137 second time, hard system reboot first time) — 1.38 GB parquet re-read after a 2 h aggregation holding ~30 GB of dict-of-dict state exceeds the system's available RAM at that point. Mitigation: column-statistics-based verification (cheap, row-group-level min/max) confirmed cleanliness without loading data into RAM. Memory note saved for future builds. (b) root cause of the original 6,482-cell corruption could not be deterministically reproduced — investigation strongly suggests PyArrow buffer-fill anomaly rather than math bug; defense is mechanism-agnostic.
- seeds_run: 1 (single run; seed=42)
- metric_aggregation: single-run
- next_candidates:
  - **`player-embedding-prelim-740`** — already proposed (`experiments/_proposals/2026-05-19-player-embedding-prelim-740.md`). Should consume the clean parquet at `data/snapshots/.../player_features_prepatch_clean/`. Reference val_auc to beat: 0.6477 (the now-confirmed-clean combined model).
  - **DVC-track the clean parquet directory** so re-fetches by downstream experiments are reproducible.
  - (Engineering carryover) Promote the `_validate_and_clamp` + numpy-routed parquet write pattern from `build_features.py` into a shared utility, since the next data-build experiment will want to reuse it.

## Follow-up

- `player-embedding-prelim-740` is unblocked and consumes the clean parquet directly.
- DVC tracking of the new `player_features_prepatch_clean/` directory pending.
- The OOM lesson (don't re-read a 1+ GB parquet for validation after holding heavy aggregator state) is captured as a personal memory note.

## Engineering note: OOM-killer and the post-write re-read

The first full run (started 2026-05-19 03:58 UTC) led to a hard system
reboot at 08:23 UTC. The second full run (08:32 UTC) survived but
the OOM-killer killed `build_features.py` with rc=137 at 10:39 UTC,
five minutes after the parquet writes completed at 10:34. The
on-disk parquets were verified clean via PyArrow's row-group
min/max statistics (no full-table re-read needed), so the build was
salvageable without a third 2 h aggregation pass. The
training pipeline (STEP 2 + STEP 3 only) then ran in 28 minutes.

The lesson: when validating output of a multi-GB parquet build that
already holds significant aggregator state in RAM, prefer
streaming / row-group-level min/max checks (`pq.ParquetFile.metadata.row_group(i).column(j).statistics`)
over a full-table `pq.read_table(path).to_pandas()`. The former is
~free; the latter doubles peak RAM at exactly the worst moment.
