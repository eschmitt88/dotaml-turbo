"""Build EXTENDED rich_cols sidecar parquet for foundation-v3-740.

Forked from experiments/2026-05-20-rich-supervision-multitask-740/build_rich_cols.py.
Differences:

1. Source match_id set comes from the EXTENDED player_features parquet
   (data/snapshots/.../player_features_extended/{train,val}.parquet),
   built by build_features_extended.py.

2. Output to data/snapshots/.../processed/rich_cols_extended/.

3. Walks the same extended raw window: 2025-08-15 -> 2026-02-23 (train)
   + 2026-02-24 -> 2026-03-09 (val).

4. Same multi-checkpoint defense as multitask-740: pyarrow row-group
   column statistics post-write verification (NO full re-read).

5. **Chunked disk-persistent output (2026-05-24 fix).** Mirrors the same fix
   applied to build_features_extended.py: accumulating ~133 Python lists of
   per-emitted-row data across all 196 days OOM-killed the sibling features
   build at day 123/196. rich_cols has the same shape (train_rows + val_rows
   dicts grow with every emitted row across the full walk), so we apply the
   same chunked flush + stream-concat pattern preemptively rather than
   discover the OOM after another ~3-4h burn. Per CHUNK_DAYS days the
   in-flight buffers are written to per-split chunk parquets under
   `_chunks_train/` and `_chunks_val/`, then stream-concatenated at end via
   pq.ParquetWriter over per-row-group reads.

HCE: walks no test-window or post-snapshot days.
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

TRAIN_EXTENDED_START = dt.date.fromisoformat("2025-08-15")

N_PLAYERS = 10
ANON_IDS = {0, 4294967295}

PER_MATCH_NUM = ["duration"]
PER_SLOT_NUM = ["kills", "deaths", "assists", "gpm", "xpm",
                "hero_damage", "net_worth"]
ITEM_FIELDS = ["item_0", "item_1", "item_2", "item_3", "item_4", "item_5",
               "item_neutral", "item_neutral2"]

# Mirrors build_features_extended.py: flush buffers every CHUNK_DAYS days.
CHUNK_DAYS = 30


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
    pf = pq.ParquetFile(clean_path)
    print(f"  {name}: streaming match_id from {clean_path.name} "
          f"({pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups)")
    mids: set[int] = set()
    for rg in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(rg, columns=["match_id"]).column("match_id").to_numpy()
        mids.update(int(m) for m in col)
    print(f"    -> {len(mids):,} unique match_ids in {name}")
    return mids


class Clamper:
    def __init__(self, bounds: dict[str, tuple[int, int]]):
        self.bounds = bounds
        self.clamp_events = 0
        self.clamp_by_feat: dict[str, int] = {k: 0 for k in bounds}

    def clamp(self, value, feat: str) -> int:
        if value is None:
            return 0
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

    bounds_raw = rc["feat_bounds"]
    bounds: dict[str, tuple[int, int]] = {k: (int(v[0]), int(v[1]))
                                           for k, v in bounds_raw.items()}
    numeric_bounds = {"duration": bounds["duration"]}
    slot_bounds = {k: bounds[k] for k in PER_SLOT_NUM}

    clamper = Clamper(bounds)

    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    snapshot_end = dt.date.fromisoformat(splits["snapshot_end_date"])
    train_hi = dt.date.fromisoformat(splits["train_end_date"])
    val_lo = dt.date.fromisoformat(splits["val_start_date"])
    val_hi = dt.date.fromisoformat(splits["val_end_date"])
    train_lo = TRAIN_EXTENDED_START

    # Match-id filter sets from the EXTENDED clean parquet.
    print("Loading extended player_features match_id filter sets...")
    if args.smoke and (src_dir / "train_smoke.parquet").exists():
        train_mids = load_mids_from_clean(src_dir / "train_smoke.parquet", "train_smoke")
        val_mids = load_mids_from_clean(src_dir / "val_smoke.parquet", "val_smoke")
    else:
        train_mids = load_mids_from_clean(src_dir / "train.parquet", "train")
        val_mids = load_mids_from_clean(src_dir / "val.parquet", "val")
    emit_mids = train_mids | val_mids
    print(f"  total filter set: {len(emit_mids):,} match_ids")

    print("Enumerating raw files...")
    raw_roots = [PROJECT_ROOT / r for r in rc["raw_roots"]]
    by_day = enumerate_raw_files(raw_roots)
    if not by_day:
        sys.exit("No raw files; cannot build rich-cols sidecar.")
    days = sorted(by_day.keys())
    print(f"  {len(days)} days enumerated (first={days[0]}, last={days[-1]})")

    # HCE date guard + window filter.
    days_ok = []
    for d in days:
        d_obj = dt.date.fromisoformat(d)
        if test_lo <= d_obj <= test_hi:
            continue
        if d_obj > snapshot_end:
            continue
        if not (train_lo <= d_obj <= val_hi):
            continue
        days_ok.append(d)
    days = days_ok
    print(f"  {len(days)} days after HCE + window filter [{train_lo}..{val_hi}]")

    if args.smoke:
        # Smoke: align with build_features_extended smoke days.
        smk_days = [d for d in days if d in ("2025-09-15", "2025-12-20", "2026-02-25")]
        if len(smk_days) < 2:
            smk_days = [days[0], days[len(days) // 2], days[-1]]
        days = smk_days
        print(f"SMOKE: walking {len(days)} days: {days}")

    train_rows: dict[str, list] = _new_buffer()
    val_rows: dict[str, list] = _new_buffer()

    # Chunked output dirs: same pattern as build_features_extended.py.
    suffix = "_smoke" if args.smoke else ""
    chunks_train_dir = out_dir / f"_chunks_train{suffix}"
    chunks_val_dir = out_dir / f"_chunks_val{suffix}"
    for cd in (chunks_train_dir, chunks_val_dir):
        if cd.exists():
            for p in cd.glob("chunk_*.parquet"):
                p.unlink()
        cd.mkdir(parents=True, exist_ok=True)
    chunk_paths_train: list[Path] = []
    chunk_paths_val: list[Path] = []
    next_chunk_idx = 0

    n_raw = 0
    n_bad_json = 0
    n_emit_train = 0
    n_emit_val = 0
    t0 = time.time()
    days_processed_in_chunk = 0

    for day_idx_outer, day in enumerate(tqdm(days, desc="days")):
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
        days_processed_in_chunk += 1

        # Chunked flush. Same pattern as build_features_extended.py: bounds
        # peak RSS to one chunk's worth of rows.
        is_final_day = (day_idx_outer == len(days) - 1)
        if (days_processed_in_chunk >= CHUNK_DAYS) or is_final_day:
            if len(train_rows["match_id"]) > 0:
                cp_t, _ = flush_rich_chunk(train_rows, chunks_train_dir,
                                            next_chunk_idx, bounds, "train")
                if cp_t is not None:
                    chunk_paths_train.append(cp_t)
            if len(val_rows["match_id"]) > 0:
                cp_v, _ = flush_rich_chunk(val_rows, chunks_val_dir,
                                            next_chunk_idx, bounds, "val")
                if cp_v is not None:
                    chunk_paths_val.append(cp_v)
            next_chunk_idx += 1
            train_rows = _new_buffer()
            val_rows = _new_buffer()
            days_processed_in_chunk = 0
            gc.collect()

    elapsed = time.time() - t0
    print(f"Walk done in {elapsed:.0f}s. raw_read={n_raw:,} bad_json={n_bad_json:,} "
          f"emit_train={n_emit_train:,} emit_val={n_emit_val:,}")
    print(f"Clamper: clamp_events={clamper.clamp_events}, by_feat={clamper.clamp_by_feat}")
    print(f"  flushed {len(chunk_paths_train)} train chunks + "
          f"{len(chunk_paths_val)} val chunks")

    if n_emit_train == 0 and not args.smoke:
        sys.exit("No train rows emitted -- rich-cols build broken.")

    del train_rows, val_rows
    gc.collect()

    print("Stream-concatenating chunks into final parquets...")
    n_train_concat = stream_concat_rich_chunks(chunk_paths_train, train_out)
    n_val_concat = stream_concat_rich_chunks(chunk_paths_val, val_out)
    print(f"  train: {n_train_concat:,} rows -> {train_out.name}")
    print(f"  val:   {n_val_concat:,} rows -> {val_out.name}")

    print("Post-write row-group-stats verification...")
    for path in (train_out, val_out):
        if path.exists() and path.stat().st_size > 0:
            verify_row_group_stats(path, numeric_bounds, slot_bounds)

    # Cleanup chunk shards.
    cleaned = 0
    for cp in chunk_paths_train + chunk_paths_val:
        if cp is None:
            continue
        try:
            if cp.exists():
                cp.unlink()
                cleaned += 1
        except OSError as e:
            print(f"  (warn) failed to remove {cp}: {e}")
    for cd in (chunks_train_dir, chunks_val_dir):
        try:
            if cd.exists() and not any(cd.iterdir()):
                cd.rmdir()
        except OSError:
            pass
    print(f"  cleaned up {cleaned} chunk shards")

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


def _validate_rich_buf_pre_arrow(buf: dict[str, list],
                                  bounds: dict[str, tuple[int, int]],
                                  where: str) -> int:
    """Pre-arrow validation of one rich-cols buffer. Returns bad-cell count."""
    n = len(buf["match_id"])
    if n == 0:
        return 0
    pre_bad = 0
    arr = np.asarray(buf["duration"], dtype=np.int64)
    lo, hi = bounds["duration"]
    bad = (arr < lo) | (arr > hi)
    if bad.any():
        print(f"  PRE-ARROW BAD {where}.duration: {int(bad.sum())} out of bounds")
        pre_bad += int(bad.sum())
    for p_i in range(N_PLAYERS):
        for s in PER_SLOT_NUM:
            cname = f"p{p_i}_{s}"
            arr = np.asarray(buf[cname], dtype=np.int64)
            lo, hi = bounds[s]
            bad = (arr < lo) | (arr > hi)
            if bad.any():
                print(f"  PRE-ARROW BAD {where}.{cname}: {int(bad.sum())} out of bounds")
                pre_bad += int(bad.sum())
    return pre_bad


def flush_rich_chunk(buf: dict[str, list], chunks_dir: Path,
                      chunk_idx: int, bounds: dict[str, tuple[int, int]],
                      split_name: str) -> tuple[Path | None, int]:
    """Flush one rich-cols buffer to chunks_dir/chunk_{idx:03d}.parquet."""
    n_rows = len(buf["match_id"])
    if n_rows == 0:
        return None, 0
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pre_bad = _validate_rich_buf_pre_arrow(buf, bounds, where=f"{split_name}_chunk{chunk_idx}")
    if pre_bad > 0:
        sys.exit(f"REFUSED: rich {split_name} chunk {chunk_idx} pre-arrow validation "
                 f"found {pre_bad} bad cells.")
    chunk_path = chunks_dir / f"chunk_{chunk_idx:03d}.parquet"
    _write_rich_cols(buf, chunk_path, bounds)
    return chunk_path, n_rows


def stream_concat_rich_chunks(chunk_paths: list[Path], out_path: Path) -> int:
    """Stream rich-cols chunk parquets into one output parquet, one row-group
    at a time. Returns total row count."""
    non_empty = [cp for cp in chunk_paths if cp is not None
                 and cp.exists() and cp.stat().st_size > 1024]
    if not non_empty:
        return 0
    schema = pq.ParquetFile(non_empty[0]).schema_arrow
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    n = 0
    try:
        for cp in non_empty:
            pf = pq.ParquetFile(cp)
            for rg_i in range(pf.metadata.num_row_groups):
                rg_tbl = pf.read_row_group(rg_i)
                writer.write_table(rg_tbl)
                n += rg_tbl.num_rows
                del rg_tbl
            del pf
            gc.collect()
    finally:
        writer.close()
    return n


def _write_rich_cols(buf: dict[str, list], path: Path,
                      bounds: dict[str, tuple[int, int]]) -> None:
    arrays: dict[str, pa.Array] = {}
    arrays["match_id"] = pa.array(np.asarray(buf["match_id"], dtype=np.int64),
                                   type=pa.int64())
    arrays["duration"] = pa.array(np.asarray(buf["duration"], dtype=np.int32),
                                   type=pa.uint16())
    arrays["radiant_win"] = pa.array(np.asarray(buf["radiant_win"], dtype=np.int32),
                                      type=pa.uint8())
    for p_i in range(N_PLAYERS):
        arrays[f"p{p_i}_items"] = pa.array(buf[f"p{p_i}_items"],
                                            type=pa.list_(pa.int32()))
        for s in ("kills", "deaths", "assists", "gpm", "xpm"):
            np_col = np.asarray(buf[f"p{p_i}_{s}"], dtype=np.int32)
            arrays[f"p{p_i}_{s}"] = pa.array(np_col, type=pa.uint16())
        for s in ("hero_damage", "net_worth"):
            np_col = np.asarray(buf[f"p{p_i}_{s}"], dtype=np.int64)
            arrays[f"p{p_i}_{s}"] = pa.array(np_col, type=pa.uint32())
    tbl = pa.table(arrays)
    pq.write_table(tbl, path, compression="zstd")
    print(f"  wrote {path} ({tbl.num_rows:,} rows, "
          f"{path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
