# log — player-embedding-prelim-740

## 2026-05-19

### Scaffold
- Forked from `experiments/2026-05-19-upstream-data-cleanup-740/`.
- Copied config.yaml, data.py, models.py, train_tfm.py → train.py.
- Added new modules: `build_account_sidecar.py`, `build_vocab.py`.
- Extended models.py with `MinimalTransformerWithFeaturesAndPlayerEmbedding`.
- Extended data.py with sidecar join + vocab lookup.
- Extended config.yaml with `[player_embedding]`, `[account_sidecar]`.

### Key non-obvious decision
- Clean parquet has NO `pX_account_id` columns (verified via pyarrow schema
  inspection). Account_ids live only in the raw history JSON. The proposal
  assumed they were present. Implemented a sidecar-parquet workaround:
  walk raw history once, emit per-match account_id parquets keyed by
  match_id, joined at load time. Walk is filtered to the set of match_ids
  present in the clean parquet (so it only touches train+val dates), and
  has the same HCE date guard as `build_features.py`.

### Smoke results
(to be filled by smoke runs)

### Full-run results
(to be filled post run_all.sh)
