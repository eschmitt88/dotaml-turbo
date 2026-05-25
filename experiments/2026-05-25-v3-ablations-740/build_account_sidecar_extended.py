"""Build per-match account_id sidecar parquet for the EXTENDED train corpus.

The extended player_features parquet (Aug 2025 -> Feb 2026 cross-patch)
does NOT carry per-player account_id columns. The prior 7.40-only
sidecar at experiments/2026-05-19-player-embedding-prelim-740/sidecar/
covers ~13M of the 32M extended train rows AND the entire val set (all
match_ids match), so this script only walks raw for the ~19M pre-patch
match_ids that are missing from the prior sidecar.

Reads only the missing days (pre-patch) per the date filter. Emits
sidecar/account_ids_train_extended.parquet (the supplementary chunk;
the prior sidecar's _train file is reused verbatim by data.py).

HCE: refuses to read test-window or post-snapshot dates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import sys
import time
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"

N_PLAYERS = 10
ANON_IDS = {0, 4294967295}


def enumerate_raw_files(raw_roots: list[Path]) -> dict[str, list[Path]]:
    by_day: dict[str, list[Path]] = {}
    for root in raw_roots:
        if not root.exists():
            print(f"  (warn) raw root missing: {root}")
            continue
        files = sorted(root.rglob("matches_*.parquet"))
        print(f"  {root}: {len(files)} files")
        for fp in files:
            parts = fp.parts
            year = month = day = None
            for p in parts:
                if p.startswith("year="):
                    year = p.split("=")[1]
                elif p.startswith("month="):
                    month = p.split("=")[1]
                elif p.startswith("day="):
                    day = p.split("=")[1]
            if not (year and month and day):
                continue
            d = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            by_day.setdefault(d, []).append(fp)
    return by_day


def load_mids_from_extended_train(extended_train_path: Path) -> set[int]:
    pf = pq.ParquetFile(extended_train_path)
    print(f"  streaming match_id from {extended_train_path.name} "
          f"({pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups)")
    mids: set[int] = set()
    for rg in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(rg, columns=["match_id"]).column("match_id").to_numpy()
        mids.update(int(m) for m in col)
    print(f"    -> {len(mids):,} unique match_ids in extended train")
    return mids


def load_mids_from_prior_sidecar(prior_path: Path) -> set[int]:
    if not prior_path.exists():
        print(f"  prior sidecar missing: {prior_path}")
        return set()
    tbl = pq.read_table(prior_path, columns=["match_id"])
    mids = set(int(m) for m in tbl.column("match_id").to_numpy())
    print(f"  prior sidecar: {len(mids):,} match_ids already covered")
    return mids


def write_sidecar(out_path: Path, rows: dict[str, list]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema_fields = [pa.field("match_id", pa.int64())]
    for p in range(N_PLAYERS):
        schema_fields.append(pa.field(f"p{p}_account_id", pa.int64()))
    schema = pa.schema(schema_fields)
    arrays = [pa.array(rows["match_id"], type=pa.int64())]
    for p in range(N_PLAYERS):
        arrays.append(pa.array(rows[f"p{p}_account_id"], type=pa.int64()))
    tbl = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(tbl, out_path, compression="zstd")
    print(f"  wrote {out_path} ({tbl.num_rows:,} rows, "
          f"{out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="Walks <=5 days only for plumbing verification.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    # Inputs.
    extended_pf_train = (PROJECT_ROOT
                          / cfg["player_features_transformer"]["source_dir"]
                          / ("train_smoke.parquet" if args.smoke else "train.parquet"))
    prior_sidecar = (PROJECT_ROOT
                      / cfg["account_sidecar"]["prior_sidecar_train_path"])

    side_cfg = cfg["account_sidecar"]
    out_path = (PROJECT_ROOT
                / (side_cfg["smoke_extended_train_out"] if args.smoke
                    else side_cfg["extended_train_out"]))

    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    snapshot_end = dt.date.fromisoformat(splits["snapshot_end_date"])

    print(f"Loading filter set from extended train + diffing prior sidecar...")
    ext_train_mids = load_mids_from_extended_train(extended_pf_train)
    prior_mids = load_mids_from_prior_sidecar(prior_sidecar)
    missing_mids = ext_train_mids - prior_mids
    print(f"  extended train: {len(ext_train_mids):,}")
    print(f"  prior covers : {len(prior_mids & ext_train_mids):,}")
    print(f"  to walk      : {len(missing_mids):,}")

    if not missing_mids:
        print("  No missing match_ids; sidecar build not needed.")
        return 0

    print("Enumerating raw files...")
    raw_roots = [PROJECT_ROOT / r for r in side_cfg["raw_roots"]]
    by_day = enumerate_raw_files(raw_roots)
    if not by_day:
        sys.exit("No raw files; cannot build sidecar.")
    days = sorted(by_day.keys())
    print(f"  {len(days)} days enumerated (first={days[0]}, last={days[-1]})")

    days_ok = []
    for d in days:
        d_obj = dt.date.fromisoformat(d)
        if test_lo <= d_obj <= test_hi:
            continue
        if d_obj > snapshot_end:
            continue
        days_ok.append(d)
    days = days_ok
    print(f"  {len(days)} days after HCE date filter")

    if args.smoke:
        days = days[:5]
        print(f"  SMOKE: limiting walk to {days[:3]}... ({len(days)} days)")

    rows: dict[str, list] = {"match_id": []}
    for p in range(N_PLAYERS):
        rows[f"p{p}_account_id"] = []

    n_raw = 0
    n_bad_json = 0
    n_emit = 0
    t0 = time.time()

    for d_idx, day in enumerate(days):
        d_obj = dt.date.fromisoformat(day)
        if test_lo <= d_obj <= test_hi:
            sys.exit(f"REFUSED: refusing to read test day {day}")
        if d_obj > snapshot_end:
            sys.exit(f"REFUSED: refusing to read post-snapshot day {day}")
        for fp in by_day[day]:
            try:
                tbl = pq.read_table(fp, columns=["match_id", "raw_json", "game_mode"])
            except Exception as e:  # noqa: BLE001
                print(f"  read fail {fp}: {e}")
                continue
            mids = tbl.column("match_id").to_numpy(zero_copy_only=False)
            gms = tbl.column("game_mode").to_numpy(zero_copy_only=False)
            jsons = tbl.column("raw_json").to_pylist()
            del tbl
            for i in range(len(mids)):
                n_raw += 1
                if int(gms[i]) != 23:
                    continue
                mid = int(mids[i])
                if mid not in missing_mids:
                    continue
                try:
                    m = orjson.loads(jsons[i])
                except Exception:
                    n_bad_json += 1
                    continue
                players = m.get("players")
                if not players or len(players) != 10:
                    continue
                accts = [int(p.get("account_id") or 0) for p in players]
                rows["match_id"].append(mid)
                for p in range(N_PLAYERS):
                    rows[f"p{p}_account_id"].append(accts[p])
                n_emit += 1
            del jsons
        gc.collect()
        if d_idx % 10 == 0 and d_idx > 0:
            elapsed = time.time() - t0
            est_remain = elapsed * (len(days) - d_idx) / max(d_idx, 1)
            print(f"  [{d_idx}/{len(days)}] day={day} n_raw={n_raw:,} n_emit={n_emit:,} "
                  f"elapsed={elapsed:.0f}s est_remain={est_remain:.0f}s")

    elapsed = time.time() - t0
    print(f"Walk done in {elapsed:.0f}s. raw_read={n_raw:,} bad_json={n_bad_json:,} "
          f"n_emit={n_emit:,}")

    if n_emit == 0 and not args.smoke:
        sys.exit("No rows emitted; sidecar build broken.")

    print("Writing sidecar parquet...")
    write_sidecar(out_path, rows)
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
