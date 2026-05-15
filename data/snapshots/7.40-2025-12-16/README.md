---
kind: snapshot
snapshot_id: "7.40-2025-12-16"
patch: "7.40"
game_mode: 23
window_start: "2025-12-16"
window_end:   "2026-03-23"
source: "azure-data-lake://dota2datalake/matches/turbo"
status: structure-only      # structure-only | downloading | populated | frozen
created: "2026-05-15"
splits_spec: "../../../splits.yaml"
---

# Snapshot 7.40-2025-12-16

The single frozen dataset for `dotaml-turbo`. All experiments in this
project train and evaluate against this snapshot. Patch boundary
(2025-12-16 → 2026-03-23) is fixed for HCE comparability — newer-patch
or live-data work belongs in the sister repo `dotaml-serve`.

Status as of creation: **structure-only.** Directories exist; no data
has been pulled.

## Layout

```
data/snapshots/7.40-2025-12-16/
├── README.md          # this file
├── raw/               # Parquet mirrored from Azure (DVC-tracked, gitignored)
│   └── turbo/year=YYYY/month=MM/day=DD/matches_*.parquet
└── processed/         # feature-extracted Parquet for training (DVC-tracked)
    └── (created by the first experiment's data stage)
```

`*.parquet` is gitignored project-wide; DVC owns this content.

## Source

- **Account:** `dota2datalake`
- **Container:** `matches`
- **Base path:** `turbo/`
- **Partition:** `year=YYYY/month=MM/day=DD/matches_{min_seq}_{max_seq}.parquet`
- **Auth:** `DefaultAzureCredential` (uses `az login` locally).
- **Read recipe:** see `literature/repos/eschmitt88-DotaML.md` (DATA_ACCESS section)
  and `literature/repos/eschmitt88-DotaDB.md` (collector + schema).

Approximate volume: 200k matches/day × 98 days = ~19.6M matches.
At ~50 MB per file and ~20 files/day, raw mirror ≈ 100 GB. Lives on
the SN850X (`~/projects/` symlink), not in `~/`.

## Splits

Authoritative spec lives at `../../../splits.yaml` (project root) per
the HCE rule for single-task projects. 70-14-14 chronological:

| split | dates (inclusive)         | days |
| ----- | ------------------------- | ---- |
| train | 2025-12-16 → 2026-02-23   |  70  |
| val   | 2026-02-24 → 2026-03-09   |  14  |
| test  | 2026-03-10 → 2026-03-23   |  14  |

Partition on `start_time_date`, **not** `match_seq_num` — see
`concepts/match-id-vs-seq-num-ordering.md`.

Test is sealed during search. Only a final-scoring pass at chain end
may read it; it writes to `final_metrics.json`, not `metrics.json`.

## Filters

Applied uniformly across all three splits before any model sees the data:

1. **Turbo only** — `game_mode == 23`. Already guaranteed upstream by DotaDB
   but re-verified at read.
2. **Forfeit filter** — drop matches where both Tier-4 towers of the
   losing team are still standing (bits 9-10 of `tower_status_*`). See
   `concepts/fake-match-filtering.md`.
3. **Empty-inventory filter** — drop matches with > 2 players showing
   zero items across all six inventory slots. Same concept.

A separate dedup-by-`match_id` step covers any residual seq-num-range
overlap from the upstream collector.

## What lives where

- Raw Parquet (Azure mirror): `raw/turbo/year=…/`. DVC-tracked, ~100 GB.
- Processed feature parquet: `processed/`. DVC-tracked, much smaller —
  ≤10 GB for a 300-dim one-hot draft feature table over 19.6M rows.
- Run metrics, models, plots: NOT here. They live with the experiment
  that produced them, under `experiments/YYYY-MM-DD-<slug>/`.

## Lifecycle

Once populated and a first experiment has consumed it, this snapshot is
**frozen**. New ingest of newer patches goes to a new snapshot directory
(`data/snapshots/<patch>-<date>/`), not in place. A re-snapshot of the
same patch (e.g. to pick up a corrected upstream file) is allowed, but
must (a) record an ADR under `docs/decisions/`, and (b) be cut as a new
dated directory rather than mutating this one.
