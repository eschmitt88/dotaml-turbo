"""Build per-match account_id sidecar parquets for player-embedding-prelim-740.

The clean parquet at data/snapshots/.../player_features_prepatch_clean/ does
NOT carry per-player account_id columns — only hero IDs, the 8 aggregated
features, source counters, and n_anonymous_in_match. To add a learned
per-player embedding we need account_ids per slot. This script walks the
raw history JSON once, filtered to the set of match_ids present in the
clean train/val parquets, and emits two compact sidecar parquets:

  sidecar/account_ids_train.parquet (match_id + p0..p9_account_id)
  sidecar/account_ids_val.parquet   (match_id + p0..p9_account_id)

The same player-slot ordering as the clean parquet is preserved: index 0..4
= Radiant in match['players'] order, 5..9 = Dire — identical to
build_features.py's `accts = [int(p.get('account_id') or 0) for p in players]`.

Anonymous account_ids (0 and 4294967295) are written through as-is; the
vocab/lookup step in data.py routes them to the 'anon' bucket.

HCE: refuses to walk any day in [test_start_date, test_end_date] or
post-snapshot. Test parquet match_ids are never in the filter set anyway,
but the date guard is a belt-and-braces second layer.

Smoke mode: walks only `account_sidecar_smoke.n_days` post-patch days,
filters to match_ids present in the {train,val}_smoke.parquet if those
exist (else falls back to whatever subset of clean parquet rows live on
the walked days).
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
from tqdm import tqdm

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


def load_mids_from_clean(clean_path: Path, name: str) -> set[int]:
    """Read just the match_id column via pyarrow — never the full table."""
    pf = pq.ParquetFile(clean_path)
    print(f"  {name}: streaming match_id from {clean_path.name} "
          f"({pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups)")
    mids: set[int] = set()
    for rg in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(rg, columns=["match_id"]).column("match_id").to_numpy()
        mids.update(int(m) for m in col)
    print(f"    -> {len(mids):,} unique match_ids in {name}")
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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    src_dir = PROJECT_ROOT / cfg["player_features_transformer"]["source_dir"]
    side = cfg["account_sidecar"]
    out_dir = PROJECT_ROOT / side["out_dir"]
    train_out = out_dir / (side["smoke_train_filename"] if args.smoke else side["train_filename"])
    val_out = out_dir / (side["smoke_val_filename"] if args.smoke else side["val_filename"])

    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    snapshot_end = dt.date.fromisoformat(splits["snapshot_end_date"])

    # Load match_id filter sets from clean parquet.
    print("Loading match_id filter sets from clean parquet...")
    if args.smoke and (src_dir / "train_smoke.parquet").exists():
        train_mids = load_mids_from_clean(src_dir / "train_smoke.parquet", "train_smoke")
        val_mids = load_mids_from_clean(src_dir / "val_smoke.parquet", "val_smoke")
    else:
        train_mids = load_mids_from_clean(src_dir / "train.parquet", "train")
        val_mids = load_mids_from_clean(src_dir / "val.parquet", "val")

    emit_mids = train_mids | val_mids
    print(f"  total filter set: {len(emit_mids):,} match_ids")

    # Enumerate raw files.
    print("Enumerating raw files...")
    raw_roots = [PROJECT_ROOT / r for r in side["raw_roots"]]
    by_day = enumerate_raw_files(raw_roots)
    if not by_day:
        sys.exit("No raw files; cannot build sidecar.")
    days = sorted(by_day.keys())
    print(f"  {len(days)} days enumerated (first={days[0]}, last={days[-1]})")

    # HCE date guard. Filter days to non-test, non-post-snapshot.
    days_ok = []
    for d in days:
        d_obj = dt.date.fromisoformat(d)
        if test_lo <= d_obj <= test_hi:
            continue  # never read test
        if d_obj > snapshot_end:
            continue
        days_ok.append(d)
    days = days_ok
    print(f"  {len(days)} days after HCE date filter")

    if args.smoke:
        smk = cfg.get("account_sidecar_smoke", {})
        # Walk first n_days post-patch + any prepatch history days that
        # appear (similar shape to build_features.py smoke).
        patch_start = dt.date.fromisoformat(splits["train_start_date"])
        prepatch = [d for d in days if dt.date.fromisoformat(d) < patch_start]
        postpatch = [d for d in days if dt.date.fromisoformat(d) >= patch_start]
        n_smk = int(smk.get("n_days", 3))
        days = prepatch[:2] + postpatch[:n_smk]
        print(f"SMOKE: walking {len(days)} days: {days}")

    # Walk raw, emit per-split rows.
    train_rows: dict[str, list] = {"match_id": []}
    val_rows: dict[str, list] = {"match_id": []}
    for p in range(N_PLAYERS):
        train_rows[f"p{p}_account_id"] = []
        val_rows[f"p{p}_account_id"] = []

    n_raw = 0
    n_bad_json = 0
    n_emit_train = 0
    n_emit_val = 0
    t0 = time.time()

    for day in tqdm(days, desc="days"):
        d_obj = dt.date.fromisoformat(day)
        # Double-check HCE.
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
                if mid not in emit_mids:
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
                # Route into train or val bucket.
                if mid in train_mids:
                    train_rows["match_id"].append(mid)
                    for p in range(N_PLAYERS):
                        train_rows[f"p{p}_account_id"].append(accts[p])
                    n_emit_train += 1
                else:  # mid in val_mids (guaranteed by emit_mids membership)
                    val_rows["match_id"].append(mid)
                    for p in range(N_PLAYERS):
                        val_rows[f"p{p}_account_id"].append(accts[p])
                    n_emit_val += 1
            del jsons
        gc.collect()

    elapsed = time.time() - t0
    print(f"Walk done in {elapsed:.0f}s. raw_read={n_raw:,} bad_json={n_bad_json:,} "
          f"emit_train={n_emit_train:,} emit_val={n_emit_val:,}")

    if n_emit_train == 0 and not args.smoke:
        sys.exit("No train rows emitted — sidecar build broken.")

    print("Writing sidecar parquets...")
    write_sidecar(train_out, train_rows)
    write_sidecar(val_out, val_rows)

    # Sanity: counts should match clean parquet (full-mode only).
    if not args.smoke:
        if n_emit_train != len(train_mids):
            print(f"  WARN: emit_train {n_emit_train:,} != train_mids {len(train_mids):,}; "
                  f"missing {len(train_mids) - n_emit_train:,} match_ids in raw.")
        if n_emit_val != len(val_mids):
            print(f"  WARN: emit_val {n_emit_val:,} != val_mids {len(val_mids):,}; "
                  f"missing {len(val_mids) - n_emit_val:,} match_ids in raw.")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
