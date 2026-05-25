"""Build EXTENDED player_features parquet for foundation-v3-740.

Forked from experiments/2026-05-19-upstream-data-cleanup-740/build_features.py.
Key differences:

1. **Walks raw inline + applies filters during walk.** The cleanup-740
   build_features.py relied on a pre-built processed/{train,val}.parquet
   for the row set to emit (`emit_mids`). That filter parquet only covers
   the 7.40 window (2025-12-16+). v3 needs to extend train to 2025-08-15,
   which means walking 2x the raw and applying the forfeit + empty-inv
   filters inline (as plateau-baseline did) rather than depending on
   the pre-built filter set. Val rows still come from raw days
   2026-02-24..2026-03-09; we re-derive val match_ids inline.

2. **EXTENDED train window**: 2025-08-15 -> 2026-02-23 (was 2025-12-16
   -> 2026-02-23). Val window unchanged (2026-02-24 -> 2026-03-09).

3. **New output dir**: data/snapshots/.../processed/player_features_extended/
   (legacy player_features_prepatch_clean/ stays in place).

4. **Same multi-checkpoint defense as cleanup-740**:
   - snapshot-time float32 clamp + bounds check
   - numpy.float32-routed pa.array() construction (avoid the suspect
     pa.array(python_list, type=pa.float32()) path)
   - row-group column statistics post-write verification (NOT a full
     re-read; cleanup-740 OOM-killed twice on a full re-read alongside
     heavy aggregator state). See
     ~/.claude/projects/.../aiserver2026-postwrite-parquet-reread-oom.md.

5. **Saves patch_id metadata** to results/patch_id_meta.json (which patches,
   how many train + val matches per patch_id). data.py's
   `_patch_id_from_dates` derives the IDs identically.

6. **Chunked disk-persistent output (2026-05-24 fix).** The cleanup-740 build
   accumulated every emitted row into a dict of ~130 Python lists across all
   days, then materialized the full pa.Table at end. With the v3 ~3x larger
   corpus that pattern OOM-killed at day 123/196 (anon-rss=91 GB of 91 GB).
   v3 now flushes those buffers to disk every CHUNK_DAYS days as a chunk
   parquet under `_chunks/`, then stream-concatenates the chunks into the
   final train+val parquets via a pq.ParquetWriter row-group-at-a-time pass
   (no full in-memory materialization). The PlayerAggregator dict itself
   stays in RAM across chunks (lookback continuity), but the per-row output
   buffers no longer grow unboundedly. Peak RSS is now bounded by
   {agg_dict (~5GB at end) + one chunk's pa.Table (~1GB) + pyarrow overhead}.

HCE: refuses to read any date in [test_start_date, test_end_date] or
post-snapshot. The walk drops them at the day-enumeration level.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import sys
import time
from collections import defaultdict, deque
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

# v3-specific: extended training window starts here.
TRAIN_EXTENDED_START = dt.date.fromisoformat("2025-08-15")
PATCH_START_DATE = dt.date.fromisoformat("2025-12-16")  # 7.40 boundary

# Hand-curated patch_id schedule (kept in sync with data.py:_patch_id_from_dates).
# (start_date_iso, patch_id) -- assigned by np.searchsorted right-side.
PATCH_EDGES = [("2025-08-01", 2), ("2025-09-10", 3), ("2025-12-16", 1)]

ANON_IDS = {0, 4294967295}

FEAT_NAMES_PER_PLAYER = [
    "n_games_log1p",
    "smoothed_winrate",
    "smoothed_winrate_hero",
    "last10_winrate",
    "days_since_last_log1p",
    "n_games_hero_log1p",
    "hero_diversity_log1p",
    "is_anonymous",
]
N_FEATS_PER_PLAYER = len(FEAT_NAMES_PER_PLAYER)
N_PLAYERS = 10

SOURCE_COL_NAMES = ["n_games_prepatch", "n_games_inpatch"]

FEAT_BOUNDS: dict[str, tuple[float, float]] = {
    "n_games_log1p":          (0.0, 25.0),
    "smoothed_winrate":       (0.0, 1.0),
    "smoothed_winrate_hero":  (0.0, 1.0),
    "last10_winrate":         (0.0, 1.0),
    "days_since_last_log1p":  (0.0, 25.0),
    "n_games_hero_log1p":     (0.0, 25.0),
    "hero_diversity_log1p":   (0.0, 10.0),
    "is_anonymous":           (0.0, 1.0),
}

# Number of days to buffer in RAM before flushing to a chunk parquet.
# Sized so one chunk's emitted rows (~30 days * ~30k turbo matches/day = ~900k
# rows, ~115 columns) stays well under ~3 GB of Python-list overhead.
CHUNK_DAYS = 30


def player_feat_cols() -> list[str]:
    cols = []
    for p in range(N_PLAYERS):
        for f in FEAT_NAMES_PER_PLAYER:
            cols.append(f"p{p}_{f}")
    return cols


def player_source_cols() -> list[str]:
    cols = []
    for p in range(N_PLAYERS):
        for s in SOURCE_COL_NAMES:
            cols.append(f"p{p}_{s}")
    return cols


def patch_id_for(date_str: str, default: int = 1) -> int:
    """Match data.py:_patch_id_from_dates exactly for a single date."""
    out = default
    for edge_date, edge_pid in PATCH_EDGES:
        if date_str >= edge_date:
            out = edge_pid
    return out


def is_forfeit(radiant_win: bool, ts_radiant: int, ts_dire: int) -> bool:
    losing_ts = ts_dire if radiant_win else ts_radiant
    return bool(losing_ts & (1 << 9)) and bool(losing_ts & (1 << 10))


def too_many_empty_inv(players: list) -> bool:
    empty = 0
    for p in players:
        if all((p.get(f"item_{i}") or 0) == 0 for i in range(6)):
            empty += 1
            if empty > 2:
                return True
    return False


class PlayerAggregator:
    """Per-account running aggregates updated chronologically.

    Same aggregator as cleanup-740. Snapshot emits float32 + validates
    bounds inline (clamp counter asserted-zero post-build).
    """

    def __init__(self, recent_window: int, alpha: float, hero_alpha: float,
                 global_radiant_prior: float = 0.5335):
        self.alpha = float(alpha)
        self.hero_alpha = float(hero_alpha)
        self.recent_window = int(recent_window)
        self.global_prior = float(global_radiant_prior)
        self.n_games: dict[int, int] = defaultdict(int)
        self.n_wins: dict[int, int] = defaultdict(int)
        self.last_time: dict[int, int] = {}
        self.recent_wins: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.recent_window)
        )
        self.hero_n: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.hero_w: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.hero_global_n: dict[int, int] = defaultdict(int)
        self.hero_global_w: dict[int, int] = defaultdict(int)
        self.n_pre: dict[int, int] = defaultdict(int)
        self.n_in: dict[int, int] = defaultdict(int)
        self.clamp_events: int = 0
        self.clamp_by_feat: dict[str, int] = {f: 0 for f in FEAT_NAMES_PER_PLAYER}

    def _hero_base(self, hero_id: int) -> float:
        n = self.hero_global_n.get(hero_id, 0)
        if n == 0:
            return self.global_prior
        w = self.hero_global_w.get(hero_id, 0)
        return w / n

    def _validate_and_clamp(self, value: float, feat_name: str) -> float:
        f32 = float(np.float32(value))
        lo, hi = FEAT_BOUNDS[feat_name]
        if not math.isfinite(f32) or f32 < lo or f32 > hi:
            self.clamp_events += 1
            self.clamp_by_feat[feat_name] += 1
            if feat_name in ("smoothed_winrate", "smoothed_winrate_hero",
                             "last10_winrate"):
                f32 = 0.5
            else:
                f32 = 0.0
        return f32

    def snapshot(self, acct: int, hero_id: int, now_ts: int) -> tuple[list[float], int, int]:
        is_anon = 1 if acct in ANON_IDS else 0
        n = self.n_games.get(acct, 0)
        w = self.n_wins.get(acct, 0)
        sw = (self.alpha + w) / (2 * self.alpha + n)
        hn = self.hero_n.get(acct, {}).get(hero_id, 0) if acct in self.hero_n else 0
        hw = self.hero_w.get(acct, {}).get(hero_id, 0) if acct in self.hero_w else 0
        hero_prior = self._hero_base(hero_id)
        sw_hero_denom = self.hero_alpha + hn
        sw_hero = (self.hero_alpha * hero_prior + hw) / sw_hero_denom
        rd = self.recent_wins.get(acct)
        if rd is not None and len(rd) > 0:
            last10 = sum(rd) / len(rd)
        else:
            last10 = 0.5
        last_t = self.last_time.get(acct)
        if last_t is None:
            days_since = 365.0
        else:
            days_since = max(0.0, (now_ts - last_t) / 86400.0)
        n_hero = hn
        n_diverse = len(self.hero_n.get(acct, {}))

        raw = [
            ("n_games_log1p",         math.log1p(n)),
            ("smoothed_winrate",      sw),
            ("smoothed_winrate_hero", sw_hero),
            ("last10_winrate",        last10),
            ("days_since_last_log1p", math.log1p(days_since)),
            ("n_games_hero_log1p",    math.log1p(n_hero)),
            ("hero_diversity_log1p",  math.log1p(n_diverse)),
            ("is_anonymous",          float(is_anon)),
        ]
        feats = [self._validate_and_clamp(v, name) for name, v in raw]
        return feats, int(self.n_pre.get(acct, 0)), int(self.n_in.get(acct, 0))

    def update(self, acct: int, hero_id: int, won: int, now_ts: int,
               is_prepatch: bool) -> None:
        if acct in ANON_IDS:
            self.hero_global_n[hero_id] += 1
            self.hero_global_w[hero_id] += won
            return
        self.n_games[acct] += 1
        self.n_wins[acct] += won
        self.last_time[acct] = now_ts
        self.recent_wins[acct].append(won)
        self.hero_n[acct][hero_id] += 1
        self.hero_w[acct][hero_id] += won
        self.hero_global_n[hero_id] += 1
        self.hero_global_w[hero_id] += won
        if is_prepatch:
            self.n_pre[acct] += 1
        else:
            self.n_in[acct] += 1


def assert_no_test_or_postsnapshot(dates: list[str], test_lo: dt.date,
                                    test_hi: dt.date,
                                    snapshot_end: dt.date) -> None:
    bad_test = []
    bad_post = []
    for s in dates:
        try:
            d = dt.date.fromisoformat(s)
        except Exception:
            continue
        if test_lo <= d <= test_hi:
            bad_test.append(s)
        elif d > snapshot_end:
            bad_post.append(s)
    if bad_test:
        sys.exit(f"REFUSED: feature build saw test-window dates: {bad_test[:5]}... -- HCE rule violated.")
    if bad_post:
        sys.exit(f"REFUSED: feature build saw post-snapshot dates: {bad_post[:5]}... -- out of scope.")


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


def validate_column_array(col: np.ndarray, feat_name: str) -> tuple[int, list[float]]:
    lo, hi = FEAT_BOUNDS[feat_name]
    bad = ~np.isfinite(col) | (col < lo) | (col > hi)
    n_bad = int(bad.sum())
    sample = col[bad][:10].tolist() if n_bad else []
    return n_bad, sample


def verify_row_group_stats_features(path: Path) -> None:
    """Per-row-group min/max bounds verification on every per-player float column.

    Pyarrow embeds min/max statistics in row-group metadata, so this is
    near-free and avoids any full table read of the multi-GB parquet
    (cleanup-740 OOM lesson).
    """
    pf = pq.ParquetFile(path)
    md = pf.metadata
    print(f"  row-group-stats verify {path.name}: {md.num_rows:,} rows, "
          f"{md.num_row_groups} row groups, {md.num_columns} cols")
    schema = pf.schema_arrow
    name_to_idx = {n: i for i, n in enumerate(schema.names)}
    n_bad = 0
    for rg_i in range(md.num_row_groups):
        rg = md.row_group(rg_i)
        for p in range(N_PLAYERS):
            for f in FEAT_NAMES_PER_PLAYER:
                cname = f"p{p}_{f}"
                col_idx = name_to_idx.get(cname)
                if col_idx is None:
                    continue
                stats = rg.column(col_idx).statistics
                if stats is None or not stats.has_min_max:
                    continue
                lo, hi = FEAT_BOUNDS[f]
                if stats.min < lo - 1e-6 or stats.max > hi + 1e-6:
                    print(f"    BAD: rg{rg_i} {cname}: "
                          f"min={stats.min}, max={stats.max}, bounds=[{lo},{hi}]")
                    n_bad += 1
    if n_bad > 0:
        sys.exit(f"REFUSED: post-write row-group-stats verify found {n_bad} bad columns in {path}.")
    print(f"    OK: all numeric columns within bounds in {path.name}")


def _new_out_cols() -> dict[str, list]:
    """Fresh empty per-chunk output buffer with the full v3 schema."""
    out_cols: dict[str, list] = {
        "match_id": [], "start_time_date": [], "radiant_win": [],
    }
    for k in ("r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4"):
        out_cols[k] = []
    for c in player_feat_cols():
        out_cols[c] = []
    for c in player_source_cols():
        out_cols[c] = []
    out_cols["n_anonymous_in_match"] = []
    out_cols["split"] = []
    return out_cols


def _out_cols_to_table(out_cols: dict[str, list]) -> pa.Table:
    """Convert one chunk's out_cols dict to a pa.Table using numpy-routed
    float construction (same belt-and-braces as cleanup-740). Caller is
    responsible for pre-arrow validation BEFORE invoking.
    """
    arrs = {}
    arrs["match_id"] = pa.array(out_cols["match_id"], type=pa.int64())
    arrs["start_time_date"] = pa.array(out_cols["start_time_date"], type=pa.string())
    arrs["radiant_win"] = pa.array(out_cols["radiant_win"], type=pa.uint8())
    for k in ("r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4"):
        arrs[k] = pa.array(out_cols[k], type=pa.uint16())
    for c in player_feat_cols():
        if c.endswith("_is_anonymous"):
            arrs[c] = pa.array(out_cols[c], type=pa.uint8())
        else:
            np_col = np.asarray(out_cols[c], dtype=np.float32)
            arrs[c] = pa.array(np_col, type=pa.float32())
    for c in player_source_cols():
        arrs[c] = pa.array(out_cols[c], type=pa.uint32())
    arrs["n_anonymous_in_match"] = pa.array(out_cols["n_anonymous_in_match"], type=pa.uint8())
    arrs["split"] = pa.array(out_cols["split"], type=pa.string())
    return pa.table(arrs)


def _validate_chunk_pre_arrow(out_cols: dict[str, list]) -> int:
    """Same per-cell bounds check as the legacy end-of-build pass, but on a
    single chunk's buffers. Returns count of bad cells; callers should sys.exit
    if non-zero.
    """
    pre_bad_total = 0
    for p in range(N_PLAYERS):
        for f in FEAT_NAMES_PER_PLAYER:
            cname = f"p{p}_{f}"
            if not out_cols[cname]:
                continue
            arr = np.asarray(out_cols[cname], dtype=np.float32)
            n_bad, sample = validate_column_array(arr, f)
            if n_bad > 0:
                print(f"  PRE-ARROW BAD: {cname}: {n_bad} cells; sample={sample[:5]}")
                pre_bad_total += n_bad
    return pre_bad_total


def flush_chunk(out_cols: dict[str, list], chunks_dir: Path,
                chunk_idx: int) -> tuple[Path, int]:
    """Write the current per-chunk buffer to `chunks_dir/chunk_{idx:03d}.parquet`
    and return (path, n_rows). Caller is expected to discard `out_cols` after
    this returns (creating a fresh one via _new_out_cols).

    Runs the per-cell bounds validation BEFORE conversion so corruption is
    caught at chunk granularity, not only at end-of-build.
    """
    n_rows = len(out_cols["match_id"])
    if n_rows == 0:
        return chunks_dir / f"chunk_{chunk_idx:03d}.parquet", 0
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pre_bad = _validate_chunk_pre_arrow(out_cols)
    if pre_bad > 0:
        sys.exit(f"REFUSED: chunk {chunk_idx} pre-arrow validation found "
                 f"{pre_bad} bad cells.")
    tbl = _out_cols_to_table(out_cols)
    chunk_path = chunks_dir / f"chunk_{chunk_idx:03d}.parquet"
    pq.write_table(tbl, chunk_path, compression="zstd")
    print(f"  flushed chunk {chunk_idx}: {n_rows:,} rows -> "
          f"{chunk_path.name} ({chunk_path.stat().st_size/1e6:.1f} MB)")
    del tbl
    return chunk_path, n_rows


def stream_concat_chunks_split(chunk_paths: list[Path], out_train: Path,
                                out_val: Path) -> tuple[int, int]:
    """Stream chunk parquets into separate train and val parquets, splitting
    per-row by the 'split' column. Never materializes more than one row group
    at a time -- safe against the cleanup-740 post-write full-re-read OOM.
    Returns (n_train_rows, n_val_rows).
    """
    # Probe schema from first non-empty chunk.
    schema = None
    for cp in chunk_paths:
        if cp.exists() and cp.stat().st_size > 1024:
            schema = pq.ParquetFile(cp).schema_arrow
            break
    if schema is None:
        sys.exit("stream_concat: no non-empty chunks to read.")

    train_writer = pq.ParquetWriter(out_train, schema, compression="zstd")
    val_writer = pq.ParquetWriter(out_val, schema, compression="zstd")
    n_train = 0
    n_val = 0
    try:
        for cp in chunk_paths:
            if not cp.exists() or cp.stat().st_size <= 1024:
                continue
            pf = pq.ParquetFile(cp)
            for rg_i in range(pf.metadata.num_row_groups):
                rg_tbl = pf.read_row_group(rg_i)
                split_col = rg_tbl.column("split")
                train_mask = pa.compute.equal(split_col, "train")
                val_mask = pa.compute.equal(split_col, "val")
                t_part = rg_tbl.filter(train_mask)
                v_part = rg_tbl.filter(val_mask)
                if t_part.num_rows > 0:
                    train_writer.write_table(t_part)
                    n_train += t_part.num_rows
                if v_part.num_rows > 0:
                    val_writer.write_table(v_part)
                    n_val += v_part.num_rows
                del rg_tbl, t_part, v_part
            del pf
            gc.collect()
    finally:
        train_writer.close()
        val_writer.close()
    return n_train, n_val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Walk a tiny subset (a few days each side of patch boundary).")
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--limit-rows-per-file", type=int, default=0)
    ap.add_argument("--max-matches", type=int, default=0,
                    help="Hard cap on total matches emitted (for instrumented small runs).")
    ap.add_argument("--max-days", type=int, default=0,
                    help="Hard cap on number of days walked (post-window-filter). "
                         "Intermediate-test path -- triggers the chunked-flush code over "
                         "more days than smoke without committing to a full ~3-4h run.")
    ap.add_argument("--chunk-days", type=int, default=CHUNK_DAYS,
                    help=f"Days per chunk flush (default {CHUNK_DAYS}). Smaller "
                         "values bound RSS more tightly at the cost of more "
                         "intermediate parquet writes.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    pf_cfg_src = cfg["player_features_transformer"]
    out_dir = PROJECT_ROOT / pf_cfg_src["source_dir"]   # player_features_extended/
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_roots = [PROJECT_ROOT / r for r in cfg["rich_cols"]["raw_roots"]]

    # v3: aggregator HPs match cleanup-740 defaults verbatim (not in the v3
    # config — just inlined here).
    alpha = 5.0
    hero_alpha = 5.0
    recent_window = 10

    # v3 train window OVERRIDES splits.yaml["train_start_date"] (which only
    # covers the 7.40 window). Val window unchanged.
    train_lo = TRAIN_EXTENDED_START
    train_hi = dt.date.fromisoformat(splits["train_end_date"])
    val_lo = dt.date.fromisoformat(splits["val_start_date"])
    val_hi = dt.date.fromisoformat(splits["val_end_date"])
    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    snapshot_end = dt.date.fromisoformat(splits["snapshot_end_date"])

    print(f"v3 EXTENDED build window:")
    print(f"  train: {train_lo} .. {train_hi}  (was 2025-12-16)")
    print(f"  val:   {val_lo} .. {val_hi}      (unchanged)")
    print(f"  out_dir: {out_dir}")

    print("Enumerating raw files across roots...")
    by_day = enumerate_raw_files(raw_roots)
    if not by_day:
        sys.exit("No raw files; cannot build extended features.")
    days = sorted(by_day.keys())
    print(f"Total {len(days)} days across roots (first={days[0]}, last={days[-1]})")
    assert_no_test_or_postsnapshot(days, test_lo, test_hi, snapshot_end)

    # Filter to days actually in our window.
    keep_days = []
    for d in days:
        d_obj = dt.date.fromisoformat(d)
        if train_lo <= d_obj <= val_hi:
            keep_days.append(d)
    days = keep_days
    print(f"  {len(days)} days in [train_lo, val_hi]")

    if args.smoke:
        # Smoke: 1 pre-patch + 1 post-patch + 1 val day, in order.
        smk_days = []
        for d in days:
            d_obj = dt.date.fromisoformat(d)
            if d_obj.isoformat() in ("2025-09-15", "2025-12-20", "2026-02-25"):
                smk_days.append(d)
        # Fallback: take first/middle/end of the range.
        if len(smk_days) < 2:
            smk_days = [days[0], days[len(days) // 2], days[-1]]
        days = smk_days
        print(f"SMOKE MODE: walking {len(days)} days: {days}")
    elif args.max_days > 0:
        days = days[: int(args.max_days)]
        print(f"MAX-DAYS MODE: walking first {len(days)} days "
              f"(first={days[0]}, last={days[-1]})")

    agg = PlayerAggregator(recent_window=recent_window, alpha=alpha,
                           hero_alpha=hero_alpha)

    # Chunked output buffers: out_cols holds only the in-flight chunk's rows
    # and is flushed to disk every CHUNK_DAYS days (see the chunk-flush block
    # at the bottom of the day loop). This bounds RSS regardless of how many
    # days are processed.
    out_cols = _new_out_cols()
    suffix = "_smoke" if args.smoke else ""
    chunks_dir = out_dir / (f"_chunks{suffix}")
    # Wipe any stale chunks from a prior failed run before starting (cheap;
    # the directory only ever holds intermediates that get cleaned at the end).
    if chunks_dir.exists():
        for p in chunks_dir.glob("chunk_*.parquet"):
            p.unlink()
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    next_chunk_idx = 0

    n_raw_read = 0
    n_bad_json = 0
    n_emitted = 0
    n_prepatch_days = 0
    n_inpatch_days = 0
    n_anon_hist: list[int] = []
    max_matches = int(args.max_matches) if args.max_matches > 0 else 0
    n_seen_mids: set[int] = set()
    n_filt_forfeit = 0
    n_filt_empty_inv = 0

    # Per-patch counts are accumulated incrementally so we can flush + drop
    # out_cols without losing the distribution metadata.
    train_patch_dist: dict[int, int] = {}
    val_patch_dist: dict[int, int] = {}

    chunk_days_threshold = int(args.chunk_days)
    print(f"  chunk_days_threshold = {chunk_days_threshold}")

    t0 = time.time()
    aborted_early = False
    days_processed_in_chunk = 0
    for day_idx_outer, day in enumerate(tqdm(days, desc="days")):
        if aborted_early:
            break
        d_obj = dt.date.fromisoformat(day)
        if test_lo <= d_obj <= test_hi:
            sys.exit(f"REFUSED: refusing to process test-window day {day}")
        if d_obj > snapshot_end:
            sys.exit(f"REFUSED: refusing to process post-snapshot day {day}")
        is_prepatch_day = (d_obj < PATCH_START_DATE)
        if is_prepatch_day:
            n_prepatch_days += 1
        else:
            n_inpatch_days += 1
        # Which split? Train if in [train_lo, train_hi]; val if [val_lo, val_hi].
        if train_lo <= d_obj <= train_hi:
            day_split = "train"
        elif val_lo <= d_obj <= val_hi:
            day_split = "val"
        else:
            continue
        day_matches: list[tuple] = []
        for fp in by_day[day]:
            try:
                tbl_in = pq.read_table(
                    fp, columns=["match_id", "raw_json", "game_mode"]
                )
            except Exception as e:  # noqa: BLE001
                print(f"  read fail {fp}: {e}")
                continue
            mids = tbl_in.column("match_id").to_numpy(zero_copy_only=False)
            gms = tbl_in.column("game_mode").to_numpy(zero_copy_only=False)
            jsons = tbl_in.column("raw_json").to_pylist()
            limit = (len(jsons) if not args.limit_rows_per_file
                     else min(len(jsons), args.limit_rows_per_file))
            for i in range(limit):
                n_raw_read += 1
                if int(gms[i]) != 23:
                    continue
                mid = int(mids[i])
                if mid in n_seen_mids:
                    continue
                try:
                    m = orjson.loads(jsons[i])
                except Exception:
                    n_bad_json += 1
                    continue
                players = m.get("players")
                if not players or len(players) != 10:
                    continue
                st = m.get("start_time")
                if st is None:
                    continue
                rw_raw = m.get("radiant_win")
                if rw_raw is None:
                    continue
                # Apply forfeit + empty-inv filters inline (replaces dep on
                # processed/train.parquet that only covers 7.40 window).
                ts_r = int(m.get("tower_status_radiant", 0) or 0)
                ts_d = int(m.get("tower_status_dire", 0) or 0)
                if is_forfeit(bool(rw_raw), ts_r, ts_d):
                    n_filt_forfeit += 1
                    continue
                if too_many_empty_inv(players):
                    n_filt_empty_inv += 1
                    continue
                day_matches.append((int(st), mid, m))
            del tbl_in, jsons
        day_matches.sort(key=lambda x: (x[0], x[1]))
        for start_ts, mid, m in day_matches:
            rw = 1 if m["radiant_win"] else 0
            players = m["players"]
            accts = [int(p.get("account_id") or 0) for p in players]
            heroes = [int(p.get("hero_id") or 0) for p in players]
            # Hero-id sanity (matches plateau-baseline rule).
            if any(h < 1 or h > 150 for h in heroes):
                continue

            # Emit the row (filter set is now inline — every passing match
            # gets emitted into train or val).
            feat_row: list[float] = []
            src_row: list[int] = []
            n_anon_match = 0
            for i in range(10):
                a = accts[i]
                h = heroes[i]
                if a in ANON_IDS:
                    n_anon_match += 1
                feats, n_pre, n_in = agg.snapshot(a, h, start_ts)
                feat_row.extend(feats)
                src_row.extend([n_pre, n_in])
            out_cols["match_id"].append(mid)
            out_cols["start_time_date"].append(day)
            out_cols["radiant_win"].append(rw)
            for j in range(5):
                out_cols[f"r{j}"].append(heroes[j])
                out_cols[f"d{j}"].append(heroes[5 + j])
            idx = 0
            for p in range(N_PLAYERS):
                for f in FEAT_NAMES_PER_PLAYER:
                    out_cols[f"p{p}_{f}"].append(feat_row[idx])
                    idx += 1
            sidx = 0
            for p in range(N_PLAYERS):
                for s in SOURCE_COL_NAMES:
                    out_cols[f"p{p}_{s}"].append(src_row[sidx])
                    sidx += 1
            out_cols["n_anonymous_in_match"].append(n_anon_match)
            out_cols["split"].append(day_split)
            n_seen_mids.add(mid)
            n_emitted += 1
            n_anon_hist.append(n_anon_match)
            if max_matches > 0 and n_emitted >= max_matches:
                print(f"  max_matches cap {max_matches:,} hit -- aborting walk early.")
                aborted_early = True
                break

            # Aggregator update (always, including post-emit).
            for i in range(10):
                won = rw if i < 5 else (1 - rw)
                agg.update(accts[i], heroes[i], won, start_ts, is_prepatch_day)

            # Patch-id distribution: accumulate per-emitted-row so we don't
            # need to re-scan out_cols later (which gets flushed periodically).
            pid = patch_id_for(day)
            if day_split == "train":
                train_patch_dist[pid] = train_patch_dist.get(pid, 0) + 1
            else:
                val_patch_dist[pid] = val_patch_dist.get(pid, 0) + 1

        gc.collect()
        days_processed_in_chunk += 1

        # Chunked flush. Triggers every CHUNK_DAYS days OR on early-abort OR
        # on the final day. This is the actual OOM fix: out_cols never holds
        # more than one chunk's rows.
        is_final_day = (day_idx_outer == len(days) - 1)
        if (days_processed_in_chunk >= chunk_days_threshold) or is_final_day or aborted_early:
            if len(out_cols["match_id"]) > 0:
                cp, _ = flush_chunk(out_cols, chunks_dir, next_chunk_idx)
                chunk_paths.append(cp)
                next_chunk_idx += 1
            out_cols = _new_out_cols()
            days_processed_in_chunk = 0
            gc.collect()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. raw_read={n_raw_read:,} bad_json={n_bad_json:,} "
          f"emitted={n_emitted:,} filt_forfeit={n_filt_forfeit:,} "
          f"filt_empty_inv={n_filt_empty_inv:,}")
    print(f"Aggregator clamp events: total={agg.clamp_events}, by_feat={agg.clamp_by_feat}")
    print(f"  flushed {len(chunk_paths)} chunk parquets under {chunks_dir}")

    if n_emitted == 0:
        sys.exit("No rows emitted -- pipeline broken.")

    # Drop the (now empty) out_cols buffer before the concat pass so all RAM
    # is freed for the stream-concat step.
    del out_cols
    gc.collect()

    out_train = out_dir / f"train{suffix}.parquet"
    out_val = out_dir / f"val{suffix}.parquet"

    # Stream-concatenate the per-chunk parquets into train + val splits.
    # Reads one row group at a time; never materializes the full table.
    print(f"Stream-concatenating {len(chunk_paths)} chunks into "
          f"{out_train.name} + {out_val.name}...")
    n_train_out, n_val_out = stream_concat_chunks_split(
        chunk_paths, out_train, out_val
    )
    print(f"  train rows: {n_train_out:,}")
    print(f"  val   rows: {n_val_out:,}")
    print(f"  train patch_id distribution: {train_patch_dist}")
    print(f"  val   patch_id distribution: {val_patch_dist}")
    print(f"Wrote {out_train} ({out_train.stat().st_size/1e6:.1f} MB)")
    print(f"Wrote {out_val} ({out_val.stat().st_size/1e6:.1f} MB)")

    # Post-write row-group-stats verification (NO full re-read).
    print("Post-write row-group-stats verification...")
    for path in (out_train, out_val):
        if path.exists() and path.stat().st_size > 1024:
            verify_row_group_stats_features(path)

    # Cleanup chunk shards now that the final concat passed verification.
    # Smoke mode skipped chunks_dir setup (writes are tiny), so guard the
    # cleanup on actual chunk paths existing.
    if chunk_paths:
        cleaned = 0
        for cp in chunk_paths:
            try:
                if cp.exists():
                    cp.unlink()
                    cleaned += 1
            except OSError as e:
                print(f"  (warn) failed to remove chunk {cp}: {e}")
        try:
            if chunks_dir.exists() and not any(chunks_dir.iterdir()):
                chunks_dir.rmdir()
        except OSError:
            pass
        print(f"  cleaned up {cleaned} chunk shards from {chunks_dir}")

    anon_arr = np.array(n_anon_hist, dtype=np.int32) if n_anon_hist else np.array([0])
    stats = {
        "smoke": bool(args.smoke),
        "build_seconds": float(elapsed),
        "n_raw_read": int(n_raw_read),
        "n_bad_json": int(n_bad_json),
        "n_emitted": int(n_emitted),
        "n_filt_forfeit": int(n_filt_forfeit),
        "n_filt_empty_inv": int(n_filt_empty_inv),
        "n_train_emitted": int(n_train_out),
        "n_val_emitted": int(n_val_out),
        "n_days_total": len(days),
        "days_processed": [days[0], days[-1]] if days else [],
        "n_prepatch_days_walked": n_prepatch_days,
        "n_inpatch_days_walked": n_inpatch_days,
        "n_unique_account_ids_tracked": len(agg.n_games),
        "raw_roots": [str(r) for r in raw_roots],
        "aggregator_clamp_events_total": int(agg.clamp_events),
        "aggregator_clamp_events_by_feat": dict(agg.clamp_by_feat),
        "anonymous_per_match_hist": {
            "min": int(anon_arr.min()), "max": int(anon_arr.max()),
            "mean": float(anon_arr.mean()), "p50": int(np.median(anon_arr)),
            "p95": int(np.quantile(anon_arr, 0.95)),
        },
        "train_window": [str(train_lo), str(train_hi)],
        "val_window": [str(val_lo), str(val_hi)],
        "patch_edges": PATCH_EDGES,
        "train_patch_id_distribution": train_patch_dist,
        "val_patch_id_distribution": val_patch_dist,
        "config": {"alpha": alpha, "hero_alpha": hero_alpha,
                   "recent_window": recent_window},
        "feat_bounds": {k: list(v) for k, v in FEAT_BOUNDS.items()},
    }
    stats_path = out_dir / f"build_stats{suffix}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    results_dir = EXP_DIR / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    (results_dir / f"patch_id_meta{suffix}.json").write_text(json.dumps({
        "patch_edges": PATCH_EDGES,
        "train_patch_id_distribution": train_patch_dist,
        "val_patch_id_distribution": val_patch_dist,
    }, indent=2))
    print(f"Stats: {stats_path}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
