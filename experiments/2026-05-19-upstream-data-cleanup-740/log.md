# log — upstream-data-cleanup-740

## 2026-05-19 (scaffold + smoke)

- Investigated the prior parquet (`player_features_prepatch/train.parquet`).
  All 6,482 bad cells live in column `p1_smoothed_winrate_hero`, contained
  within row group 2, contiguous match range [2,344,604, 2,504,113), all on
  date 2025-12-29, ~4% density within that range, values are uninitialized-
  memory-shaped (NaN, denormals, mixed-magnitude floats). No data-side
  characteristic of affected rows correlates with corruption — strongly
  suggesting transient memory corruption during PyArrow's fp32 buffer fill
  rather than a math bug. See README.md `Root cause` for full analysis.
- Wrote patched `build_features.py` with three defensive checkpoints:
  (1) `_validate_and_clamp` inside `snapshot()`, (2) pre-arrow
  numpy.float32 conversion + bounds-check, (3) post-write re-read
  bounds-check.
- Wrote `config.yaml` combining LightGBM and Transformer config blocks
  (Transformer keys prefixed `transformer_*` to coexist).
- Wrote `train_lgbm.py` (verbatim from prepatch-740 with anchor delta
  updates), `train_tfm.py` (forked from extended-740, config key paths
  updated), `data.py` (forked from extended-740, sanitization shim
  REMOVED and replaced with hard assertion).
- Wrote `run_all.sh` — 3 sequential steps, MAX_RETRIES=3 on Transformer.
- Smoke build PASSED in 175s — zero clamp events, all three validation
  checkpoints clean, 379,585 smoke train rows.
- Smoke LightGBM features_only PASSED in ~30s — val_auc=0.6064 on 100k
  rows pseudo-val (pipeline test only, not meaningful).
- Smoke Transformer + features PASSED in ~5s — val_auc=0.4990 (1 epoch
  on 50k rows, pipeline test only). GPU bf16-autocast active. data.py
  assertion silent (no bad cells in the smoke parquet).

Pending: main agent runs `run_all.sh` for the full ~4 h rebuild + ablations.
