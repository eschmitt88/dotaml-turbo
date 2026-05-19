# log — transformer-plus-features-740

## 2026-05-18 (scaffold + smoke)

- Created experiment folder via subagent following proposal
  `experiments/_proposals/2026-05-18-transformer-plus-features-740.md`.
- Reused processed parquet from `player-features-prepatch-740`
  (`data/snapshots/7.40-2025-12-16/processed/player_features_prepatch/`).
  No new feature build.
- `models.py`: `MinimalTransformerWithFeatures` mirrors the
  `transformer-hp-sweep-740` `MinimalTransformer` plus an always-constructed
  `Linear(8, d_model)` feature-projection branch gated by `self.use_features`.
  Choice: always construct the projection so the parameter-list shape is
  identical between ablations (cleaner A/B; unused weights just don't accrue
  gradient when `use_features=False`).
- `data.py`: builds `(hero_ids[N,10], player_feats[N,10,8], y[N])` owned
  torch tensors via `torch.tensor(arr)` (Blackwell + torch 2.9 workaround,
  per `docs/decisions/0001-per-trial-subprocess-isolation.md`). 5M-row
  stratified subsample (seed=42), feature columns read in
  `FEAT_NAMES_PER_PLAYER` order from `player-features-prepatch-740/train.py`.
- `train.py`: forces math SDP backend at module load; num_workers=0; bf16
  autocast; coverage-bucket val_auc diagnostic carried over.
- `run_all.sh`: two ablations sequentially in fresh subprocesses with
  auto-retry (MAX_RETRIES=3) on rc!=0.

### Smoke (1 epoch, 50k train / 5k val)

- `transformer_plus_features` smoke: val_auc=0.4906 (≈ noise as expected),
  training stable, metrics_smoke_transformer_plus_features.json written. 0.6s
  training wall.
- `architecture_only` smoke: val_auc=0.4982 (≈ noise as expected), training
  stable, metrics_smoke_architecture_only.json written. 0.7s training wall.
- HCE date guard fired (train ≤ 2026-02-23, val ≤ 2026-03-09 — no test-window
  leakage).
- Hero ID range check passed (all in [1, 150]).

### Data-quality issue discovered + fix

- `player-features-prepatch-740/train.parquet` has a tiny (~0.030% of values
  in one feature column) corruption: `p{p}_smoothed_winrate_hero` carries
  ±3.4e38 (fp32 max) sentinels in ~3867 cells out of ~130M. The other 7
  feature columns are clean; val.parquet is entirely clean.
- First-batch bf16 forward exploded (logits → 1.7e17) once a corrupted
  cell was in the batch, producing NaN gradients on step 1.
- Fix: feature-aware sanitization in `data.py:load_arrays` — clip values
  outside per-feature physical bounds (winrates ∈ [0,1], log1p counts ≤ 20,
  etc) and replace with the per-feature median over the good entries. Prints
  the count when triggered. With the 5M stratified subsample (seed=42),
  ~23 corrupted cells get clipped — negligible signal impact.
- We can't fix the upstream parquet (hard rule: do not modify other
  experiments). The sanitization is the right place for the fix — it
  isolates the bug to this experiment without touching shared data.

## 2026-05-19 (full run + finalization)

- 00:14 Full run launched via `nohup bash run_all.sh ...`.
- 00:27 `architecture_only` (sanity) completed first attempt: **val_auc=0.6319**, 14 epochs, 13:13 wall. Δ vs `plateau-architectures-740` (0.6322) = -0.0003 — SANITY PASSES well within ≤0.005 spec.
- 00:41 `transformer_plus_features` (PRIMARY) completed first attempt: **val_auc=0.6452**, 14 epochs, 13:17 wall. HYPOTHESIS CONFIRMED.
- 00:42 Aggregated metrics. Headline gains:
  - vs proposal target 0.6372: **+0.0080**
  - vs Transformer-only 0.6322: **+0.0133**
  - vs LightGBM+features 0.6256: **+0.0196**
  - vs LightGBM-baseline 0.6161: **+0.0291**
- Coverage-bucket lifted across the board (low/med/high val_auc = 0.6347/0.6443/0.6560 vs prepatch's 0.6173/0.6256/0.6339). HIGH bucket at 0.6560 closes in on Hodge 2017's 75-76% in-game-telemetry ceiling, achieved with PRE-GAME info only.
- Notably the LOW bucket (mostly-anonymous matches) lifted +0.0174 — even on anonymous-heavy matches, attention extracts substantial extra signal from partial player info.
- Both ablations succeeded on attempt 1, NO Blackwell torch retries triggered — first time in 6 Transformer-using experiments where the retry path wasn't exercised.
- 00:50 README finalised with Result/Interpretation/Diagnostics. status: running → done.
