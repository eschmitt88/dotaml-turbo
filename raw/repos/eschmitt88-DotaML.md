---
source_url: https://github.com/eschmitt88/DotaML
fetched_at: 2026-05-15
fetched_by: fetch-paper
repo_default_branch: main
repo_pushed_at: 2026-05-15T01:57:50Z
repo_private: true
capture_kind: repo-bundle
contents:
  - README.md
  - CLAUDE.md (design notes)
  - DATA_ACCESS.md
  - MATCH_DATA_REFERENCE.md
  - FEATURE_DESIGN.md
  - FAKE_MATCH_CRITERIA.md
  - DUPLICATION_REPORT.md
  - experiments/v1_baseline/metrics.json
  - experiments/v2_full_dataset/{metrics.json, RESULTS.md}
  - experiments/v3_hero_fix/{metrics.json, RESULTS.md}
  - experiments/v4_tensorflow_simple/{metrics.json, RESULTS.md}
  - experiments/v5_tensorflow_residual/{metrics.json, config.json, README.md}
  - experiments/v6_tensorflow_transformer/{README.md, TRAINING_SUMMARY.md}
---

# eschmitt88/DotaML — prior-art capture

This is a self-contained snapshot of the prior-art DotaML repo as of 2026-05-15.
It is a personal exploratory repo by the same author (eschmitt88), preceding the
current `dotaml-turbo` project. The current project does **not** reuse its code;
it treats the repo as a literature-style reference. Key signals to extract:

- Six model generations (v1 LightGBM → v6 Transformer) trained on Turbo matches
  from a pre-patch-7.40 window (Aug 15 – Oct 20, 2025).
- A clear performance plateau at ~59.9% test accuracy / ~0.635 test AUC across
  architectures of widely different size. The v5 README explicitly says
  "we may be approaching fundamental limits of hero draft prediction."
- Data-quality findings worth replaying: fake-match filtering (forfeit via T4
  towers + empty inventories), Azure file overlap producing ~4.9% match_id
  duplication, a `max_hero_id=130` bug silently dropping 20 heroes.
- Feature representations tried: ordered hero positions → one-hot 260
  (130 heroes × 2 teams) → one-hot 300 (150 × 2) → learned 64-dim embeddings.
- A 4-phase UCB combo-search procedure for ranking 3-hero cores by predicted
  win rate.
- A confirmed Radiant-side advantage of roughly +5 – +7 percentage points.

The new `dotaml-turbo` project freezes its snapshot at patch 7.40
(2025-12-16 → 2026-03-23, ~19.6M Turbo matches). The prior-art models are
therefore on a different patch and not directly comparable, but their
experiment grid is the most relevant baseline available.

---

## TOP-LEVEL README

```
# DotaML - Dota 2 Turbo Match Analysis & ML Pipeline

A scalable Python pipeline for analyzing Dota 2 Turbo matches using Azure Data
Lake Storage. Designed for machine learning workloads including hero
combination recommendations, item analysis, and win prediction.

## Features
- Azure Data Lake Integration: Direct access to 200k+ matches per day stored
  in Azure
- Scalable Architecture: Partitioned Parquet storage optimized for analytical
  queries
- Modern Stack: Polars, DuckDB, PyArrow for high-performance data processing
- ML-Ready: Extract features for hero drafting, item recommendations, and win
  prediction
- Efficient Querying: Date-based partitioning with columnar storage

## Quick Start
Install: `uv sync`; auth: `az login`.

## Data Lake Structure
Both Azure and local data use the same partition structure:

turbo/year=YYYY/month=MM/day=DD/matches_{min_seq}_{max_seq}.parquet
~200k matches/day, ~20 files/day (~50MB each), each ~10k matches.

Schema (top-level columns): match_id, game_mode (=23 Turbo),
start_time, start_time_date, collected_at, match_seq_num, raw_json.

## ML Use Cases (as listed)
1. Hero Combination Win Prediction
2. Item Recommendation
3. Ability Draft Analysis

## Authentication
DefaultAzureCredential (env → managed identity → az login → VS Code).
For local dev: `az login`. For prod, set AZURE_CLIENT_ID/TENANT_ID/SECRET.

## Data Sources
Storage Account: dota2datalake, Container: matches.
Collection System: DotaDB (sister repo).
Game Mode: Turbo (23) only. Date Range: 2025-08-15 onwards. ~200k/day.

## Performance
- Load single day: ~200k matches in <5s.
- JSON parse: ~50k matches/s.
- Storage: ~1GB/day (compressed Parquet).
```

## CLAUDE.md — repo agent notes (key excerpts)

```
Project structure:
  src/ml_models/             # Model abstraction layer (BaseHeroPredictor)
    lightgbm_model.py
    tensorflow_model.py
    loader.py                # auto-load by file extension
  src/tf_architectures/      # TF model architectures registry
    feedforward.py           # simple, deep, wide, residual
  src/features.py            # feature engineering (one-hot hero encoding)
  experiments/               # production models ONLY (promoted from mlruns/)
    v1_baseline / v2_full_dataset / v3_hero_fix / v4_tensorflow_simple ...
  mlruns/                    # MLflow experiment tracking (scratch)
  archive/                   # old models and results

Workflow: all training goes to mlruns/ first; manually promote to experiments/
  after evaluation.

Supported model formats: .txt (LightGBM), .keras/.h5/SavedModel (TF/Keras),
  planned: scikit-learn (.pkl/.joblib), ONNX (.onnx).

TF architectures registry: list_architectures() returns
  ['simple', 'deep', 'wide', 'residual'] (v6 added 'transformer' separately).

Data flow:
  Azure Data Lake (Parquet + JSON) → MatchLoader → Local Processing (opt) →
  Analytics/ML

Storage strategy:
  - Date-partitioned Parquet for partition pruning.
  - JSON parsed on demand (lazy).
  - Streaming ops for large date ranges.
  - Optional local cache as columnar Parquet.

Required permissions on dota2datalake: Storage Blob Data Reader.
```

## DATA_ACCESS.md — Azure access guide (key excerpts)

```
Storage account: dota2datalake
URL:             https://dota2datalake.dfs.core.windows.net
Container:       matches
Base path:       turbo/

Auth: DefaultAzureCredential (managed identity / service principal /
  az login / SAS). Permission: Storage Blob Data Reader.

Folder hierarchy (partitioned by match start_time, not collection time):
  turbo/year=YYYY/month=MM/day=DD/
    matches_{min_seq}_{max_seq}.parquet     # ~10k matches each, ~50MB
    daily_summary.parquet                   # {date, match_count, finalized_at}

Per-file schema:
  match_id        int64
  game_mode       int32 (always 23)
  start_time      string (ISO 8601)
  start_time_date string (YYYY-MM-DD)
  collected_at    string (ISO 8601)
  match_seq_num   int64
  raw_json        string (full match JSON)

Code recipe to enumerate files for a date and read into pandas via
DataLakeServiceClient + DefaultAzureCredential is included in the doc.

Data quality notes:
  - Files organized by match start_time, not collection time.
  - Turbo only (game_mode=23).
  - "Complete records: only includes matches where game_mode was successfully
    determined."
```

## MATCH_DATA_REFERENCE.md — field catalogue (key excerpts)

The doc is a long field-by-field catalogue. The fields most relevant to
draft-only win prediction are reproduced here verbatim.

```
Match-level (from raw_json):
  match_id, match_seq_num, start_time (Unix), start_time_date,
  duration (s), radiant_win (bool, primary target),
  game_mode (23), lobby_type (0=public, 7=ranked),
  pre_game_duration, picks_bans (array of {hero_id, team, order, is_pick};
    Turbo has no bans),
  radiant_score, dire_score, first_blood_time,
  tower_status_radiant, tower_status_dire (bitmask; bits 9, 10 = T4 towers),
  barracks_status_radiant, barracks_status_dire,
  cluster, human_players (always 10), leagueid (0 for public), engine, flags.

Per-player (10 per match) — draft-relevant subset:
  account_id (4294967295 if anonymous), player_slot (0-4=Radiant, 128-132=Dire),
  team_number (0=Radiant, 1=Dire), team_slot (0-4 within team),
  hero_id (1..150, not all IDs valid),
  hero_variant (facet), leaver_status (0=stayed),
  kills, deaths, assists, last_hits, denies, level (usually 30 in Turbo),
  gold_per_min, xp_per_min, net_worth, gold, gold_spent,
  hero_damage, scaled_hero_damage, tower_damage, scaled_tower_damage,
  hero_healing, scaled_hero_healing,
  item_0..item_5, backpack_0..2, item_neutral, item_neutral2,
  aghanims_scepter (0/1), aghanims_shard (0/1), moonshard (0/1),
  ability_upgrades: [{ability, time(s), level}].

Notes on signal availability:
  - No per-event purchase timeline, no ward/positioning data, no chat,
    no Roshan timing.
  - hero_id range 1..150 (not all IDs are valid; the v3 fix expanded the
    feature space from 130 → 150).

Draft feature extraction snippet from the doc:
  radiant_heroes = sorted([p['hero_id'] for p in match_data['players'][:5]])
  dire_heroes    = sorted([p['hero_id'] for p in match_data['players'][5:]])
  pick_order     = [p['hero_id'] for p in match_data['picks_bans']]
  y              = match_data['radiant_win']
```

## FEATURE_DESIGN.md — order-invariant draft features

The author considered four representations:

1. **One-hot encoding (chosen).** 2 × 150 = 300 binary features. Order-
   invariant by construction. Plays well with both LightGBM (v1-v3) and FFNs
   (v4-v5). Used in v1 through v5 (with the v3 bug fix lifting max_hero_id
   from 130 to 150).
2. **Hero-pair interactions.** Explicit synergy/counter features per pair.
   Rejected as too sparse (130 choose 2 = 8,385 combinations).
3. **Aggregated statistics.** Mean/sum of hero win rate or pick rate per
   team. Rejected as lossy (cannot recover hero identity).
4. **Learned embeddings + set pooling.** What v6 Transformer ultimately did
   (64-dim hero embeddings, shared across teams, separate position embeddings
   for Radiant vs Dire, multi-head self-attention over the 10-hero sequence,
   team-wise pooling before classifier).

Implementation function (reproduced):

```python
def create_onehot_hero_features(radiant_heroes, dire_heroes, max_hero_id=150):
    features = {}
    for hero_id in range(1, max_hero_id + 1):
        features[f'radiant_hero_{hero_id}'] = 0
        features[f'dire_hero_{hero_id}']    = 0
    for h in radiant_heroes: features[f'radiant_hero_{h}'] = 1
    for h in dire_heroes:    features[f'dire_hero_{h}']    = 1
    return features
```

## FAKE_MATCH_CRITERIA.md — boosting-service filter

About ~10,000 matches in the early dataset had identical hero compositions
where one team always wins; suspected boosting service / behavior-score
recovery / quest farming. Two filters were applied:

**Primary filter — forfeit detection (T4 towers).**

```python
def is_forfeit_match(m):
    losing = m['tower_status_dire'] if m['radiant_win'] else m['tower_status_radiant']
    t4a = bool(losing & (1 << 9))
    t4b = bool(losing & (1 << 10))
    return t4a and t4b   # both T4 of the losing team still standing
```

Rationale: real Dota matches almost never end with both T4s of the loser
intact; a "gg"-after-30-min surrender does.

**Secondary filter — empty inventories.**

```python
def has_too_many_empty_inventories(m):
    empty = 0
    for p in m['players']:
        if all((p.get(f'item_{i}', 0) or 0) == 0 for i in range(6)):
            empty += 1
    return empty > 2
```

Rationale: real losers still buy items; multiple empty inventories signal
non-genuine play.

## DUPLICATION_REPORT.md — Azure file overlap

Processed dataset `data/processed/hero_draft_onehot_v3/features.parquet`:

- 7,674,142 rows total, 7,297,966 unique match_ids.
- 376,176 duplicate rows (4.902%).
- Every duplicate appears exactly twice — a perfect 2× pattern, indicating
  systematic overlap, not random collection noise.

Root cause: overlapping `match_seq_num` ranges in adjacent Parquet filenames
in the Azure container — e.g. file A ending at seq 7072684405 and file B
starting at seq 7072662568, with both files containing matches in the
overlap region. 4 days had this problem in Azure itself (Aug 17, 19, 20, 22).

Fix applied (already done in the upstream repo): identified 40 overlapping
files in Azure, deleted them, kept 16 forming clean sequential chains.
Recommendation in the report: add automated duplicate detection with an
alert if dup rate > 0.1%, and add a match_id uniqueness constraint in the
DotaDB pipeline.

**Implication for dotaml-turbo:** before assuming patch 7.40 data is clean,
verify no overlapping sequence ranges in the 7.40 window.

---

## EXPERIMENTS — six-generation grid

All numbers below are from the upstream repo's own `metrics.json` /
RESULTS.md files. Splits are time-based chronological 80/20 train/test.

### v1_baseline — LightGBM, small dataset

`experiments/v1_baseline/metrics.json`:

```json
{
  "accuracy": 0.5854983518390007,
  "roc_auc": 0.61405137202002,
  "log_loss": 0.6706646506857886,
  "confusion_matrix": [[38592, 47515], [28939, 69402]],
  "best_iteration": 500
}
```

Approx 922k matches (per v2 RESULTS comparison: "v2 is 8.3× larger than v1").

### v2_full_dataset — LightGBM, 7.6M matches, max_hero_id=130 bug

```json
{
  "accuracy": 0.5876001885569667,
  "roc_auc": 0.6178308033266192,
  "log_loss": 0.6688988121958905,
  "best_iteration": 500
}
```

Training window Aug 15 – Sep 30, 2025 (46 days). 260 features
(130 heroes × 2 teams). Heroes 131-145 silently dropped (bug). Top combo:
Queen of Pain + Anti-Mage + Ancient Apparition, 62.2% predicted win rate.
Anti-Mage appears in 100% of the top 10 combos in v2 — later partially
attributed to the bug.

### v3_hero_fix — LightGBM, max_hero_id=150 fix, 80% subset

```json
{
  "accuracy": 0.5881832451445521,
  "roc_auc": 0.6188935987347888,
  "log_loss": 0.6685356697020771,
  "best_iteration": 500,
  "train_fraction": 0.8
}
```

300 features (150 × 2). Trained on 5.0M / 6.3M samples due to memory.
Same Aug 15 – Sep 30 window. Top combo: Lion + Weaver + Lich, 58.2%.
Anti-Mage's dominance drops from 10/10 (v2) → 2/10 (v3); Weaver rises
1/10 → 9/10. Authors conclude the v2 max_hero_id=130 bug had biased
results toward Anti-Mage.

### v4_tensorflow_simple — first neural net

```json
{
  "train_accuracy": 0.6004,
  "test_accuracy":  0.5983,
  "train_auc":      0.6361,
  "test_auc":       0.6341,
  "train_loss":     0.6600,
  "test_loss":      0.6608,
  "training_time_seconds": 28928.5,
  "training_samples": 6259185,
  "test_samples":     1414957,
  "epochs": 15,
  "batch_size": 16384,
  "architecture": "simple",
  "model_type": "tensorflow"
}
```

Simple FFN: 300 → 128 → 64 → 1 with Dropout(0.3) between hidden layers,
Adam(lr=1e-3), 46,849 parameters. Trained on full 6.26M training set.
Test accuracy +1.01% over v3 LightGBM; test AUC +0.0152. Notes the meta
again shifts (Lion 30% → 80% of top combos, Weaver 90% → 0%) — i.e. the
discovered "optimal combos" are unstable across architectures.

### v5_tensorflow_residual — deeper residual FFN

```json
{
  "test_accuracy":  0.5995,
  "test_auc":       0.6354,
  "train_accuracy": 0.6046,
  "train_auc":      0.6426,
  "test_loss":      0.6606,
  "train_loss":     0.6572,
  "training_time_seconds": 70192,
  "num_parameters": 228353,
  "train_samples": 7562549,
  "test_samples":  1890638
}
```

`config.json`:
```json
{
  "architecture": "residual",
  "hidden_units": [256, 256, 128, 128],
  "dropout_rate": 0.3,
  "batch_size": 16384,
  "epochs": 30,
  "learning_rate": 0.001,
  "optimizer": "adam"
}
```

Test acc +0.12% over v4; test AUC +0.0013. Train-test gap 0.51%. README
explicitly states: "modest improvement suggests we may be approaching
fundamental limits of hero draft prediction."

### v6_tensorflow_transformer — embeddings + masked-input training

Test set 1.89M matches:
- Accuracy: **59.87%**
- AUC: **0.6354**
- Loss: 0.6600

Architecture: 64-dim hero embedding (shared across teams) + position
embedding (Radiant vs Dire), 4 transformer layers, 4 attention heads/layer,
FFN dim 128, classifier dropout 0.3, 152,001 parameters total.
Training: 50 epochs, batch 1024, Adam(1e-3) with ReduceLROnPlateau, 29 hrs.
Trained with 30% random masking on input hero IDs (hero_id=0 = [MASK])
so the model can score incomplete drafts. Total dataset: 9.45M matches
across 67 days (Aug 15 – Oct 20, 2025), 80/20 chronological split.

### Plateau summary

| Model         | Test acc | Test AUC | Params | Notes                               |
|---------------|----------|----------|--------|-------------------------------------|
| v1 LightGBM   | 58.55%   | 0.6141   | n/a    | 922k matches                        |
| v2 LightGBM   | 58.76%   | 0.6178   | n/a    | 7.6M, max_hero_id=130 bug           |
| v3 LightGBM   | 58.82%   | 0.6189   | n/a    | bug fix, 80% subset                 |
| v4 SimpleFFN  | 59.83%   | 0.6341   | 47k    | first NN, +1.0% acc                 |
| v5 ResidualFFN| 59.95%   | 0.6354   | 228k   | deeper, +0.12% acc                  |
| v6 Transformer| 59.87%   | 0.6354   | 152k   | embeddings, masking support         |

Bottom line: across orders-of-magnitude changes in capacity and a switch
from tabular trees to attention, the ceiling sits at ~60% accuracy /
~0.635 AUC on a pre-7.40 patch. This is the single strongest signal in
the prior art for the new project.

### UCB combo-search procedure (used in v2-v4)

Four-phase escalating-N sampling over candidate 3-hero cores:

- Phase 1: 19,600 combos @ N=20  → 392,000 predictions
- Phase 2:    500 combos @ N=100 → 50,000
- Phase 3:    100 combos @ N=500 → 50,000
- Phase 4:     20 combos @ N=990 → 19,800

For each phase the model predicts Radiant win % across N random opposing
drafts; UCB(beta=1.0) advances the most promising combos to the next phase.

### Side asymmetry (Radiant advantage)

Reported consistently across model generations:
- v2: ~+5pp Radiant advantage.
- v3: ~+7pp.
- v4: ~+6.3pp.

This is a real feature of Turbo data, not an artifact of any one model.

---

## Notes for downstream ingest

Concepts worth seeding (non-exhaustive):
- `draft-only-win-prediction` — the prediction task itself.
- `radiant-side-advantage` — confirmed empirically across architectures.
- `fake-match-filtering` — T4-tower forfeit + empty-inventory heuristics.
- `azure-file-overlap-duplication` — data hygiene risk to verify.
- `max-hero-id-bug` — feature-space coverage failure mode.
- `one-hot-vs-embeddings` — feature representation tradeoff.
- `ucb-combo-search` — search procedure for top-K combinations.
- `draft-prediction-plateau` — the ~60% / 0.635 ceiling observed across
  architectures of varying capacity.
- `time-based-chronological-split` — 80/20 by start_time used throughout.
- `masked-draft-training` — v6's 30% mask rate enables incomplete-draft
  scoring.

Patch-context caveat: every metric above is on a pre-7.40 window
(Aug-Oct 2025). The current project's snapshot is patch 7.40
(2025-12-16 → 2026-03-23, ~19.6M Turbo matches), so these numbers serve as
a prior, not a baseline.
