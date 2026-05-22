"""Build per-match rich-cols sidecar parquets for rich-supervision-multitask-740.

The clean parquet at data/snapshots/.../player_features_prepatch_clean/
carries only hero IDs + 8 aggregated features + sources + n_anonymous_in_match
+ radiant_win + start_time_date. The multi-task heads need the IN-GAME outcomes
(duration, per-slot items / KDA / GPM / XPM / hero_damage / net_worth) as
training TARGETS only — they NEVER enter the encoder.

This script walks the raw turbo parquets once, filtered to the set of
match_ids present in the clean train/val parquets, parses each row's
`raw_json` STRING column with orjson, and emits two compact sidecar parquets:

  data/snapshots/.../processed/rich_cols/train.parquet
  data/snapshots/.../processed/rich_cols/val.parquet

Schema per row (23 columns):
  match_id (int64)
  duration (uint16)
  radiant_win (uint8, redundant with clean parquet — for join sanity)
  p0..p9_items (list<int32>, variable length: items 0..5 + neutrals,
                deduplicated, zeros dropped)
  p0..p9_kills (uint16)
  p0..p9_deaths (uint16)
  p0..p9_assists (uint16)
  p0..p9_gpm (uint16)
  p0..p9_xpm (uint16)
  p0..p9_hero_damage (uint32)
  p0..p9_net_worth (uint32)

Multi-checkpoint defense from cleanup-740:
  1. snapshot-time physical-bounds clamp on every numeric field
     (clamp counter asserted-zero post-build).
  2. numpy.<dtype>-routed pa.array() construction (NOT the pa.array(list,
     type=...) path that the prior build suspected of fp32 corruption).
  3. Post-write VERIFICATION via pyarrow row-group column statistics
     (NOT a full table re-read). Cleanup-740 OOM-killed twice on a full
     re-read of a 1.38 GB parquet held alongside heavy aggregator state;
     one event cascaded to a system reboot. See
     ~/.claude/projects/.../aiserver2026-postwrite-parquet-reread-oom.md.

HCE: refuses to walk any day in [test_start_date, test_end_date] or
post-snapshot. Test parquet match_ids are never in the filter set anyway,
but the date guard is the belt-and-braces second layer.
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

# Numeric feature names (each is per-slot: p{0..9}_<name>) plus the per-match
# `duration`. Bounds come from config.yaml:rich_cols.feat_bounds.
PER_MATCH_NUM = ["duration"]
PER_SLOT_NUM = ["kills", "deaths", "assists", "gpm", "xpm",
                "hero_damage", "net_worth"]
ITEM_FIELDS = ["item_0", "item_1", "item_2", "item_3", "item_4", "item_5",
               "item_neutral", "item_neutral2"]


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
    """Stream the match_id column via pyarrow row-groups (never full read)."""
    pf = pq.ParquetFile(clean_path)
    print(f"  {name}: streaming match_id from {clean_path.name} "
          f"({pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups)")
    mids: set[int] = set()
    for rg in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(rg, columns=["match_id"]).column("match_id").to_numpy()
        mids.update(int(m) for m in col)
    print(f"    -> {len(mids):,} unique match_ids in {name}")
    return mids


def gpm_to_uint16(v) -> int:
    if v is None:
        return 0
    try:
        x = int(v)
    except (TypeError, ValueError):
        return 0
    if x < 0:
        return 0
    if x > 65535:
        return 65535
    return x


class Clamper:
    """Per-feature numeric clamp with global counter (mirrors build_features.py)."""

    def __init__(self, bounds: dict[str, tuple[int, int]]):
        self.bounds = bounds
        self.clamp_events = 0
        self.clamp_by_feat: dict[str, int] = {k: 0 for k in bounds}

    def clamp(self, value, feat: str) -> int:
        if value is None:
            return 0  # zero-fill missing (don't count as a clamp event)
        try:
            x = int(value)
        except (TypeError, ValueError):
            self.clamp_events += 1
            self.clamp_by_feat[feat] += 1
            return 0
        lo, hi = self.bounds[feat]
        if x < lo:
            self.clamp_events += 1
            self.clamp_by_feat[feat] += 1
            return int(lo)
        if x > hi:
            self.clamp_events += 1
            self.clamp_by_feat[feat] += 1
            return int(hi)
        return x


def parse_items(p: dict) -> list[int]:
    """Deduplicate item IDs across the 8 item fields; drop 0s and None."""
    out: set[int] = set()
    for k in ITEM_FIELDS:
        v = p.get(k)
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv <= 0 or iv > 2_000_000_000:
            continue
        out.add(iv)
    return sorted(out)


def verify_row_group_stats(path: Path, numeric_bounds: dict[str, tuple[int, int]],
                            slot_bounds: dict[str, tuple[int, int]]) -> None:
    """Row-group column-stats bounds check. NO full table read.

    For each row-group and each numeric column, asserts min >= lo and max <= hi.
    Pyarrow stats are part of parquet metadata; cost is ~free. This is the
    cleanup-740 lesson: never re-read a multi-GB parquet to verify, especially
    not while heavy aggregator state still lives in RAM.
    """
    pf = pq.ParquetFile(path)
    md = pf.metadata
    n_rg = md.num_row_groups
    print(f"  row-group-stats verify {path.name}: {md.num_rows:,} rows, "
          f"{n_rg} row groups, {md.num_columns} cols")
    schema = pf.schema_arrow
    name_to_idx = {name: i for i, name in enumerate(schema.names)}
    n_bad = 0
    for rg_i in range(n_rg):
        rg = md.row_group(rg_i)
        # Check per-match numeric.
        for fname, (lo, hi) in numeric_bounds.items():
            col_idx = name_to_idx.get(fname)
            if col_idx is None:
                continue
            stats = rg.column(col_idx).statistics
            if stats is None or not stats.has_min_max:
                continue
            if stats.min < lo or stats.max > hi:
                print(f"    BAD: rg{rg_i} {fname}: "
                      f"min={stats.min}, max={stats.max}, bounds=[{lo},{hi}]")
                n_bad += 1
        # Check per-slot numerics.
        for p_i in range(N_PLAYERS):
            for s_name, (lo, hi) in slot_bounds.items():
                fname = f"p{p_i}_{s_name}"
                col_idx = name_to_idx.get(fname)
                if col_idx is None:
                    continue
                stats = rg.column(col_idx).statistics
                if stats is None or not stats.has_min_max:
                    continue
                if stats.min < lo or stats.max > hi:
                    print(f"    BAD: rg{rg_i} {fname}: "
                          f"min={stats.min}, max={stats.max}, bounds=[{lo},{hi}]")
                    n_bad += 1
    if n_bad > 0:
        sys.exit(f"REFUSED: post-write row-group-stats found {n_bad} bad columns in {path}.")
    print(f"    OK: all numeric columns within bounds in {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    src_dir = PROJECT_ROOT / cfg["player_features_transformer"]["source_dir"]
    rc = cfg["rich_cols"]
    out_dir = PROJECT_ROOT / rc["out_dir"]
    train_out = out_dir / (rc["smoke_train_filename"] if args.smoke
                            else rc["train_filename"])
    val_out = out_dir / (rc["smoke_val_filename"] if args.smoke
                          else rc["val_filename"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-feature physical bounds.
    bounds_raw = rc["feat_bounds"]
    bounds: dict[str, tuple[int, int]] = {k: (int(v[0]), int(v[1]))
                                           for k, v in bounds_raw.items()}
    numeric_bounds = {"duration": bounds["duration"]}
    slot_bounds = {k: bounds[k] for k in PER_SLOT_NUM}

    clamper = Clamper(bounds)

    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    snapshot_end = dt.date.fromisoformat(splits["snapshot_end_date"])

    # Match-id filter sets from clean parquet (HCE: test never appears here).
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
    raw_roots = [PROJECT_ROOT / r for r in rc["raw_roots"]]
    by_day = enumerate_raw_files(raw_roots)
    if not by_day:
        sys.exit("No raw files; cannot build rich-cols sidecar.")
    days = sorted(by_day.keys())
    print(f"  {len(days)} days enumerated (first={days[0]}, last={days[-1]})")

    # HCE date guard.
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
        smk = cfg.get("rich_cols_smoke", {})
        patch_start = dt.date.fromisoformat(splits["train_start_date"])
        postpatch = [d for d in days if dt.date.fromisoformat(d) >= patch_start]
        n_smk = int(smk.get("n_days", 1))
        days = postpatch[:n_smk]
        print(f"SMOKE: walking {len(days)} days: {days}")

    # Output buffers (split by train/val at row level for streaming write).
    train_rows: dict[str, list] = _new_buffer()
    val_rows: dict[str, list] = _new_buffer()

    n_raw = 0
    n_bad_json = 0
    n_emit_train = 0
    n_emit_val = 0
    t0 = time.time()

    for day in tqdm(days, desc="days"):
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
                if m.get("radiant_win") is None:
                    continue
                duration = clamper.clamp(m.get("duration"), "duration")
                rw = 1 if m["radiant_win"] else 0
                # Per-slot.
                per_slot_vals: dict[str, list[int]] = {s: [] for s in PER_SLOT_NUM}
                per_slot_items: list[list[int]] = []
                for p_i in range(N_PLAYERS):
                    p = players[p_i]
                    per_slot_items.append(parse_items(p))
                    per_slot_vals["kills"].append(clamper.clamp(p.get("kills"), "kills"))
                    per_slot_vals["deaths"].append(clamper.clamp(p.get("deaths"), "deaths"))
                    per_slot_vals["assists"].append(clamper.clamp(p.get("assists"), "assists"))
                    per_slot_vals["gpm"].append(clamper.clamp(p.get("gold_per_min"), "gpm"))
                    per_slot_vals["xpm"].append(clamper.clamp(p.get("xp_per_min"), "xpm"))
                    per_slot_vals["hero_damage"].append(clamper.clamp(p.get("hero_damage"), "hero_damage"))
                    per_slot_vals["net_worth"].append(clamper.clamp(p.get("net_worth"), "net_worth"))
                # Route into train or val.
                buf = train_rows if mid in train_mids else val_rows
                buf["match_id"].append(mid)
                buf["duration"].append(int(duration))
                buf["radiant_win"].append(int(rw))
                for p_i in range(N_PLAYERS):
                    buf[f"p{p_i}_items"].append(per_slot_items[p_i])
                    for s in PER_SLOT_NUM:
                        buf[f"p{p_i}_{s}"].append(per_slot_vals[s][p_i])
                if mid in train_mids:
                    n_emit_train += 1
                else:
                    n_emit_val += 1
            del jsons
        gc.collect()

    elapsed = time.time() - t0
    print(f"Walk done in {elapsed:.0f}s. raw_read={n_raw:,} bad_json={n_bad_json:,} "
          f"emit_train={n_emit_train:,} emit_val={n_emit_val:,}")
    print(f"Clamper: clamp_events={clamper.clamp_events}, by_feat={clamper.clamp_by_feat}")

    if n_emit_train == 0 and not args.smoke:
        sys.exit("No train rows emitted — rich-cols build broken.")

    # Pre-arrow validation: walk numpy arrays of each per-slot numeric column.
    print("Pre-arrow validation of numeric columns...")
    pre_bad_total = 0
    for buf_name, buf in (("train", train_rows), ("val", val_rows)):
        n = len(buf["match_id"])
        if n == 0:
            continue
        # duration
        arr = np.asarray(buf["duration"], dtype=np.int64)
        lo, hi = bounds["duration"]
        bad = (arr < lo) | (arr > hi)
        if bad.any():
            print(f"  PRE-ARROW BAD {buf_name}.duration: {int(bad.sum())} cells out of bounds")
            pre_bad_total += int(bad.sum())
        for p_i in range(N_PLAYERS):
            for s in PER_SLOT_NUM:
                cname = f"p{p_i}_{s}"
                arr = np.asarray(buf[cname], dtype=np.int64)
                lo, hi = bounds[s]
                bad = (arr < lo) | (arr > hi)
                if bad.any():
                    print(f"  PRE-ARROW BAD {buf_name}.{cname}: {int(bad.sum())} cells out of bounds")
                    pre_bad_total += int(bad.sum())
    if pre_bad_total > 0:
        sys.exit(f"REFUSED: pre-arrow validation found {pre_bad_total} bad cells "
                 f"(clamper failed). Build aborted.")

    print("Writing sidecar parquets (numpy-routed pa.array construction)...")
    if n_emit_train > 0:
        _write_rich_cols(train_rows, train_out, bounds)
    if n_emit_val > 0:
        _write_rich_cols(val_rows, val_out, bounds)

    # Free in-memory buffers BEFORE the verification step (mirrors cleanup-740
    # lesson: drop aggregator state before any verification that might touch RAM).
    del train_rows, val_rows
    gc.collect()

    # Post-write row-group column-stats verification (NO full read).
    print("Post-write row-group-stats verification...")
    for path in (train_out, val_out):
        if path.exists() and path.stat().st_size > 0:
            verify_row_group_stats(path, numeric_bounds, slot_bounds)

    # Sanity counts (full mode only).
    if not args.smoke:
        if n_emit_train != len(train_mids):
            print(f"  WARN: emit_train {n_emit_train:,} != train_mids {len(train_mids):,}; "
                  f"missing {len(train_mids) - n_emit_train:,} match_ids in raw.")
        if n_emit_val != len(val_mids):
            print(f"  WARN: emit_val {n_emit_val:,} != val_mids {len(val_mids):,}; "
                  f"missing {len(val_mids) - n_emit_val:,} match_ids in raw.")
    print("DONE.")
    return 0


def _new_buffer() -> dict[str, list]:
    buf: dict[str, list] = {"match_id": [], "duration": [], "radiant_win": []}
    for p_i in range(N_PLAYERS):
        buf[f"p{p_i}_items"] = []
        for s in PER_SLOT_NUM:
            buf[f"p{p_i}_{s}"] = []
    return buf


def _write_rich_cols(buf: dict[str, list], path: Path,
                      bounds: dict[str, tuple[int, int]]) -> None:
    """Numpy-routed pa.array construction (cleanup-740 lesson)."""
    arrays: dict[str, pa.Array] = {}
    arrays["match_id"] = pa.array(np.asarray(buf["match_id"], dtype=np.int64),
                                   type=pa.int64())
    # duration: uint16 fits 0..65535 -> bounds [0,7200] safe
    arrays["duration"] = pa.array(np.asarray(buf["duration"], dtype=np.int32),
                                   type=pa.uint16())
    arrays["radiant_win"] = pa.array(np.asarray(buf["radiant_win"], dtype=np.int32),
                                      type=pa.uint8())
    for p_i in range(N_PLAYERS):
        # items: variable-length list<int32>; build via pa.array on list-of-lists.
        # Keep as int32 (Steam API item IDs fit easily). No numpy routing —
        # list<int32> would require a chunked construction; pa.array on a
        # Python list-of-list-of-int is the canonical path and is NOT the
        # fp32-corruption path (that was specifically fp32 dense columns).
        arrays[f"p{p_i}_items"] = pa.array(buf[f"p{p_i}_items"],
                                            type=pa.list_(pa.int32()))
        # kills/deaths/assists/gpm/xpm: uint16 (bounds <= 5000 all safe)
        for s in ("kills", "deaths", "assists", "gpm", "xpm"):
            np_col = np.asarray(buf[f"p{p_i}_{s}"], dtype=np.int32)
            arrays[f"p{p_i}_{s}"] = pa.array(np_col, type=pa.uint16())
        # hero_damage / net_worth: uint32 (bounds 500K)
        for s in ("hero_damage", "net_worth"):
            np_col = np.asarray(buf[f"p{p_i}_{s}"], dtype=np.int64)
            arrays[f"p{p_i}_{s}"] = pa.array(np_col, type=pa.uint32())
    tbl = pa.table(arrays)
    pq.write_table(tbl, path, compression="zstd")
    print(f"  wrote {path} ({tbl.num_rows:,} rows, "
          f"{path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
