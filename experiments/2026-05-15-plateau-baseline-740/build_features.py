"""Build the feature parquet for plateau-baseline-740 from raw.

Reads raw parquet files day-by-day, parses raw_json, extracts:
  - radiant_win (bool target)
  - 10 hero_ids
  - tower_status_radiant / tower_status_dire (for forfeit filter)
  - per-player item_0..item_5 (for empty-inventory filter)

Applies fake-match filter (forfeit + >2 empty inventories), dedups by
match_id, splits by start_time_date per splits.yaml, and writes
processed/{train,val}.parquet.

HCE: this script must NOT touch the test window. We assert it.

Output schema:
  match_id (int64)
  start_time_date (string)
  radiant_win (uint8)
  r0..r4 (uint16) — radiant hero_ids (1..150)
  d0..d4 (uint16) — dire hero_ids (1..150)
  split (string) "train" | "val"
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
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
SNAPSHOT_DIR = PROJECT_ROOT / "data/snapshots/7.40-2025-12-16"
RAW_ROOT = SNAPSHOT_DIR / "raw" / "turbo"
PROCESSED_DIR = SNAPSHOT_DIR / "processed"
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"


def is_forfeit(radiant_win: bool, ts_radiant: int, ts_dire: int) -> bool:
    """Both T4 towers (bits 9, 10) of the LOSING team still standing."""
    losing_ts = ts_dire if radiant_win else ts_radiant
    t4a = bool(losing_ts & (1 << 9))
    t4b = bool(losing_ts & (1 << 10))
    return t4a and t4b


def too_many_empty_inv(players: list) -> bool:
    """> 2 players with all 6 inventory slots == 0."""
    empty = 0
    for p in players:
        # item_0..item_5; treat None / missing / 0 as empty.
        if all((p.get(f"item_{i}") or 0) == 0 for i in range(6)):
            empty += 1
            if empty > 2:
                return True
    return False


def extract_one(raw_json_str: str) -> tuple | None:
    """Returns (radiant_win, r_heroes(5), d_heroes(5)) or None to drop."""
    try:
        m = orjson.loads(raw_json_str)
    except Exception:
        return None
    players = m.get("players")
    if not players or len(players) != 10:
        return None
    rw = m.get("radiant_win")
    if rw is None:
        return None
    ts_r = m.get("tower_status_radiant", 0) or 0
    ts_d = m.get("tower_status_dire", 0) or 0
    if is_forfeit(bool(rw), int(ts_r), int(ts_d)):
        return None
    if too_many_empty_inv(players):
        return None
    # Heroes: first 5 = radiant (slot 0-4), last 5 = dire (slot 128-132).
    # Documented as such in MATCH_DATA_REFERENCE.md.
    r_heroes = []
    d_heroes = []
    for p in players[:5]:
        h = p.get("hero_id")
        if h is None or h < 1 or h > 150:
            return None
        r_heroes.append(int(h))
    for p in players[5:]:
        h = p.get("hero_id")
        if h is None or h < 1 or h > 150:
            return None
        d_heroes.append(int(h))
    return (bool(rw), r_heroes, d_heroes)


def assert_no_test_dates(dates: list[str], test_lo: dt.date, test_hi: dt.date) -> None:
    bad = []
    for s in dates:
        try:
            d = dt.date.fromisoformat(s)
        except Exception:
            continue
        if test_lo <= d <= test_hi:
            bad.append(s)
    if bad:
        sys.exit(f"REFUSED: feature build saw test-window dates: {bad[:5]}... — HCE rule violated.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-rows-per-file", type=int, default=0,
                    help="Optional cap (debug only); 0 = no cap.")
    args = ap.parse_args()

    splits = yaml.safe_load(SPLITS_PATH.read_text())
    train_lo = dt.date.fromisoformat(splits["train_start_date"])
    train_hi = dt.date.fromisoformat(splits["train_end_date"])
    val_lo = dt.date.fromisoformat(splits["val_start_date"])
    val_hi = dt.date.fromisoformat(splits["val_end_date"])
    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Enumerate raw files. Format: turbo/year=YYYY/month=MM/day=DD/matches_*.parquet
    files = sorted(RAW_ROOT.rglob("matches_*.parquet"))
    print(f"Found {len(files)} raw parquet files under {RAW_ROOT}")
    if not files:
        sys.exit("No raw files; run pull_raw.py first.")

    # Walk by day; assign to split; refuse to read any test-window day.
    by_day: dict[str, list[Path]] = {}
    for fp in files:
        # path like .../year=YYYY/month=MM/day=DD/matches_*.parquet
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

    days = sorted(by_day.keys())
    print(f"Days available: {len(days)} (first={days[0]}, last={days[-1]})")
    assert_no_test_dates(days, test_lo, test_hi)

    # Per-day → split assignment (date strings).
    def split_for(d_str: str) -> str | None:
        d = dt.date.fromisoformat(d_str)
        if train_lo <= d <= train_hi:
            return "train"
        if val_lo <= d <= val_hi:
            return "val"
        if test_lo <= d <= test_hi:
            sys.exit(f"REFUSED: encountered test-window day {d_str} — HCE rule.")
        return None  # outside snapshot range; ignore.

    # Aggregate per-split row buffers — flush to parquet at end.
    # Memory: ~19M rows * (8 + small + 10*2 + 1) bytes ≈ <600 MB; fits in 64 GB.
    cols = {
        "match_id": [],
        "start_time_date": [],
        "radiant_win": [],
    }
    for k in ("r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4"):
        cols[k] = []
    splits_col: list[str] = []

    seen_match_ids: set[int] = set()
    n_read = 0
    n_kept = 0
    n_dup = 0
    n_filt = 0
    n_bad = 0

    t0 = time.time()
    for day in tqdm(days, desc="days"):
        sp = split_for(day)
        if sp is None:
            continue
        for fp in by_day[day]:
            try:
                tbl = pq.read_table(
                    fp,
                    columns=["match_id", "start_time_date", "raw_json", "game_mode"],
                )
            except Exception as e:  # noqa: BLE001
                print(f"  read fail {fp}: {e}")
                continue
            mids = tbl.column("match_id").to_numpy(zero_copy_only=False)
            sds = tbl.column("start_time_date").to_pylist()
            gms = tbl.column("game_mode").to_numpy(zero_copy_only=False)
            jsons = tbl.column("raw_json").to_pylist()
            limit = len(jsons) if not args.limit_rows_per_file else min(len(jsons), args.limit_rows_per_file)
            for i in range(limit):
                n_read += 1
                if int(gms[i]) != 23:
                    n_bad += 1
                    continue
                mid = int(mids[i])
                if mid in seen_match_ids:
                    n_dup += 1
                    continue
                ext = extract_one(jsons[i])
                if ext is None:
                    # Either parse error or filtered. Distinguish below.
                    # Cheap re-check to count filter vs bad: skip; counted as filt.
                    n_filt += 1
                    continue
                rw, rh, dh = ext
                seen_match_ids.add(mid)
                cols["match_id"].append(mid)
                cols["start_time_date"].append(sds[i])
                cols["radiant_win"].append(1 if rw else 0)
                for j, h in enumerate(rh):
                    cols[f"r{j}"].append(h)
                for j, h in enumerate(dh):
                    cols[f"d{j}"].append(h)
                splits_col.append(sp)
                n_kept += 1
            del tbl, jsons
        # GC at day boundary to keep working set small.
        gc.collect()

    elapsed = time.time() - t0
    print(
        f"Done in {elapsed:.0f}s. read={n_read} kept={n_kept} "
        f"filtered={n_filt} dup_match_id={n_dup} bad_game_mode={n_bad}"
    )

    if n_kept == 0:
        sys.exit("No rows kept — pipeline broken.")

    # Build arrow table once, write per split.
    arrays = {k: pa.array(v) for k, v in cols.items()}
    arrays["split"] = pa.array(splits_col)
    tbl = pa.table(arrays)

    print(f"Built table: {tbl.num_rows} rows, {tbl.num_columns} columns")
    train_tbl = tbl.filter(pa.compute.equal(tbl.column("split"), "train"))
    val_tbl = tbl.filter(pa.compute.equal(tbl.column("split"), "val"))
    print(f"  train rows: {train_tbl.num_rows}")
    print(f"  val   rows: {val_tbl.num_rows}")

    out_train = PROCESSED_DIR / "train.parquet"
    out_val = PROCESSED_DIR / "val.parquet"
    pq.write_table(train_tbl, out_train, compression="zstd")
    pq.write_table(val_tbl, out_val, compression="zstd")
    print(f"Wrote {out_train} ({out_train.stat().st_size/1e6:.1f} MB)")
    print(f"Wrote {out_val} ({out_val.stat().st_size/1e6:.1f} MB)")

    # Sanity stats to a JSON file for the train script to anchor on.
    stats = {
        "n_read": n_read,
        "n_kept": n_kept,
        "n_filtered_or_bad_extract": n_filt,
        "n_dup_match_id": n_dup,
        "n_bad_game_mode": n_bad,
        "n_train": int(train_tbl.num_rows),
        "n_val": int(val_tbl.num_rows),
        "build_seconds": elapsed,
        "train_date_min": str(min(by_day.keys() & {d for d in by_day if split_for(d) == "train"})),
        "train_date_max": str(max(d for d in by_day if split_for(d) == "train")),
        "val_date_min": str(min(d for d in by_day if split_for(d) == "val")),
        "val_date_max": str(max(d for d in by_day if split_for(d) == "val")),
    }
    (PROCESSED_DIR / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print("Stats:", json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
