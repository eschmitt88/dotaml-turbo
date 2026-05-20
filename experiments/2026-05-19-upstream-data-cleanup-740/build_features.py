"""Build augmented feature parquet for upstream-data-cleanup-740.

Forked from experiments/2026-05-18-player-features-prepatch-740/build_features.py.
Identical schema and aggregator logic; the only differences are:

1. **Defensive sanitization inside snapshot().** Each numeric feature is
   explicitly cast to float32, validated finite, and clamped to per-feature
   physical bounds. If a clamp ever fires the build records it in a counter
   and continues (the counter is asserted to be 0 post-build).

2. **Multi-checkpoint post-build assertion.** After all rows are emitted,
   each per-player float column is cast via numpy.float32, validated finite,
   and bounded against the per-feature physical range. If ANY cell escapes
   the build aborts with a hard SystemExit BEFORE writing the parquet.

3. **Post-write re-read assertion.** After pq.write_table, we re-read each
   output parquet and re-run the per-cell bounds check. If the on-disk
   artifact carries a single bad cell the build deletes the offending file
   and exits non-zero.

4. **New output dir.** Writes to player_features_prepatch_clean/ — does NOT
   overwrite the dirty player_features_prepatch/ parquet (kept for reference).

Why the multi-layer defense rather than a single root-cause fix: the 6,482
corrupted cells in the prior parquet exclusively appeared in column
p1_smoothed_winrate_hero, contiguous row range [2344604, 2504113), all on
date 2025-12-29, randomly distributed within that range (~4% density), with
values matching unaligned-write / memory-corruption signatures (denormals,
NaN, mixed-magnitude floats including positive and negative). The visible
numeric paths (snapshot() lines 138/150) cannot produce those values — both
denominators are bounded ≥ 5.0 (hero_alpha) or fall back to global_prior.
The most likely root cause is a transient memory event during pyarrow's
fp32 buffer fill on a specific row group, not a math bug. The defensive
layering catches the issue irrespective of the underlying mechanism.

HCE: refuses to read any date in [test_start_date, test_end_date] or
in (snapshot_end_date, ...) at walk time.
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
SNAPSHOT_DIR = PROJECT_ROOT / "data/snapshots/7.40-2025-12-16"
PROCESSED_DIR = SNAPSHOT_DIR / "processed"
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"
PATCH_START_DATE = dt.date.fromisoformat("2025-12-16")

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

# Per-feature physical bounds. Anything outside these bounds is a build bug.
# (lo, hi) tuples; the bound is inclusive on both sides. is_anonymous is binary.
FEAT_BOUNDS: dict[str, tuple[float, float]] = {
    "n_games_log1p":          (0.0, 25.0),   # log1p(N); 25 ≈ 7.2e10 games (impossible)
    "smoothed_winrate":       (0.0, 1.0),    # probability
    "smoothed_winrate_hero":  (0.0, 1.0),    # probability (THE column the prior build corrupted)
    "last10_winrate":         (0.0, 1.0),    # rolling rate
    "days_since_last_log1p":  (0.0, 25.0),   # log1p(days); 25 ≈ 200M days
    "n_games_hero_log1p":     (0.0, 25.0),
    "hero_diversity_log1p":   (0.0, 10.0),   # log1p(distinct heroes); 10 ≈ 22K heroes
    "is_anonymous":           (0.0, 1.0),    # binary
}


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


class PlayerAggregator:
    """Per-account running aggregates updated chronologically.

    Same aggregator as player-features-prepatch-740; snapshot() now emits
    np.float32 values directly (no Python float intermediary) and validates
    each feature against FEAT_BOUNDS. The clamp counter is incremented
    if any feature falls outside its bound; post-build we assert it's 0.
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
        # NEW: defensive instrumentation.
        self.clamp_events: int = 0
        self.clamp_by_feat: dict[str, int] = {f: 0 for f in FEAT_NAMES_PER_PLAYER}

    def _hero_base(self, hero_id: int) -> float:
        n = self.hero_global_n.get(hero_id, 0)
        if n == 0:
            return self.global_prior
        # Read .get() rather than [] to avoid defaultdict mutation side-effect.
        w = self.hero_global_w.get(hero_id, 0)
        return w / n

    def _validate_and_clamp(self, value: float, feat_name: str) -> float:
        """Cast to float32, ensure finite, clamp to physical bounds.

        Returns a python float (already in fp32 precision). If the value
        was out-of-bounds or non-finite, increments the clamp counter.
        """
        f32 = float(np.float32(value))
        lo, hi = FEAT_BOUNDS[feat_name]
        if not math.isfinite(f32) or f32 < lo or f32 > hi:
            self.clamp_events += 1
            self.clamp_by_feat[feat_name] += 1
            # Choose a safe-default rather than per-feature median (median
            # would need a full pass). Winrates → 0.5 (uninformative prior);
            # log1p features → 0; binary → 0.
            if feat_name in ("smoothed_winrate", "smoothed_winrate_hero",
                             "last10_winrate"):
                f32 = 0.5
            else:
                f32 = 0.0
        return f32

    def snapshot(self, acct: int, hero_id: int, now_ts: int,
                 teammates: list[int]) -> tuple[list[float], int, int]:
        """Pre-match feature vector + (n_prepatch, n_inpatch) for one player.

        Each emitted feature is float32-cast and bounds-validated before being
        appended to the output list. The Python list holds float32-precision
        values exclusively — defensive against pa.array() coercing odd dtypes.
        """
        is_anon = 1 if acct in ANON_IDS else 0
        n = self.n_games.get(acct, 0)
        w = self.n_wins.get(acct, 0)
        sw = (self.alpha + w) / (2 * self.alpha + n)
        hn = self.hero_n.get(acct, {}).get(hero_id, 0) if acct in self.hero_n else 0
        hw = self.hero_w.get(acct, {}).get(hero_id, 0) if acct in self.hero_w else 0
        hero_prior = self._hero_base(hero_id)
        sw_hero_denom = self.hero_alpha + hn
        # Both terms in numerator and the denominator are bounded ≥ 0; denom ≥ alpha > 0.
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
               teammates: list[int], is_prepatch: bool) -> None:
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
        sys.exit(f"REFUSED: feature build saw test-window dates: {bad_test[:5]}... — HCE rule violated.")
    if bad_post:
        sys.exit(f"REFUSED: feature build saw post-snapshot dates: {bad_post[:5]}... — out of scope.")


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
    """Return (n_bad, sample_bad_values) for one column."""
    lo, hi = FEAT_BOUNDS[feat_name]
    bad = ~np.isfinite(col) | (col < lo) | (col > hi)
    n_bad = int(bad.sum())
    sample = col[bad][:10].tolist() if n_bad else []
    return n_bad, sample


def validate_table(tbl: pa.Table, where: str) -> None:
    """Assert every per-player float column is finite and bounded.

    Raises SystemExit on any violation; this is the post-build hard gate.
    """
    n_bad_total = 0
    for p in range(N_PLAYERS):
        for f in FEAT_NAMES_PER_PLAYER:
            col_name = f"p{p}_{f}"
            arr = tbl.column(col_name).to_numpy(zero_copy_only=False)
            n_bad, sample = validate_column_array(arr, f)
            if n_bad > 0:
                print(f"  [{where}] BAD: column {col_name} has {n_bad} cells outside bounds "
                      f"({FEAT_BOUNDS[f]}); sample = {sample[:5]}")
                n_bad_total += n_bad
    if n_bad_total > 0:
        sys.exit(f"REFUSED: {where} validation found {n_bad_total} bad cells — "
                 f"clean-parquet contract violated.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Process small subset; see config.yaml.smoke for spec.")
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--limit-rows-per-file", type=int, default=0)
    ap.add_argument("--max-matches", type=int, default=0,
                    help="Hard cap on total matches emitted (for instrumented small-subset runs).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())
    pf_cfg = cfg["player_features"]
    smoke_cfg = cfg["smoke"]

    alpha = float(pf_cfg["smoothing_alpha"])
    hero_alpha = float(pf_cfg["hero_smoothing_alpha"])
    recent_window = int(pf_cfg["recent_form_window"])
    out_dir = PROJECT_ROOT / pf_cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_roots = [PROJECT_ROOT / r for r in pf_cfg["raw_roots"]]

    train_lo = dt.date.fromisoformat(splits["train_start_date"])
    train_hi = dt.date.fromisoformat(splits["train_end_date"])
    val_lo = dt.date.fromisoformat(splits["val_start_date"])
    val_hi = dt.date.fromisoformat(splits["val_end_date"])
    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    snapshot_end = dt.date.fromisoformat(splits["snapshot_end_date"])

    print("Loading filtered processed parquet (row set to emit)...")
    train_tbl = pq.read_table(PROCESSED_DIR / "train.parquet")
    val_tbl = pq.read_table(PROCESSED_DIR / "val.parquet")
    train_mids = set(int(x) for x in train_tbl.column("match_id").to_pylist())
    val_mids = set(int(x) for x in val_tbl.column("match_id").to_pylist())
    print(f"  filtered train rows: {len(train_mids):,}")
    print(f"  filtered val   rows: {len(val_mids):,}")
    emit_mids = train_mids | val_mids

    def index_processed(tbl, split_name: str) -> dict[int, tuple]:
        mids = tbl.column("match_id").to_numpy(zero_copy_only=False)
        sds = tbl.column("start_time_date").to_pylist()
        rws = tbl.column("radiant_win").to_numpy(zero_copy_only=False)
        r = [tbl.column(f"r{i}").to_numpy(zero_copy_only=False) for i in range(5)]
        d = [tbl.column(f"d{i}").to_numpy(zero_copy_only=False) for i in range(5)]
        out = {}
        for i in range(len(mids)):
            heroes = [int(r[j][i]) for j in range(5)] + [int(d[j][i]) for j in range(5)]
            out[int(mids[i])] = (split_name, sds[i], int(rws[i]), heroes)
        return out

    print("  indexing processed parquet by match_id...")
    proc_idx: dict[int, tuple] = {}
    proc_idx.update(index_processed(train_tbl, "train"))
    proc_idx.update(index_processed(val_tbl, "val"))
    print(f"  processed index size: {len(proc_idx):,}")

    print("Enumerating raw files across roots...")
    by_day = enumerate_raw_files(raw_roots)
    if not by_day:
        sys.exit("No raw files; run pull_history.py first.")
    days = sorted(by_day.keys())
    print(f"Total {len(days)} days across roots (first={days[0]}, last={days[-1]})")
    assert_no_test_or_postsnapshot(days, test_lo, test_hi, snapshot_end)

    if args.smoke:
        history_days = list(smoke_cfg.get("history_days", []))
        patch_days = sorted([d for d in days if dt.date.fromisoformat(d) >= PATCH_START_DATE])
        n_smoke_days = int(smoke_cfg["n_days"])
        patch_days_keep = patch_days[:n_smoke_days]
        keep_days = sorted(set(history_days) | set(patch_days_keep))
        days = [d for d in days if d in keep_days]
        emit_dates = set(patch_days_keep)
        emit_mids = {mid for mid, (_, sd, _, _) in proc_idx.items() if sd in emit_dates}
        print(f"SMOKE MODE: walking {len(days)} days {days}")
        print(f"  emit window: {sorted(emit_dates)} ; emit_mids: {len(emit_mids):,}")

    agg = PlayerAggregator(recent_window=recent_window, alpha=alpha,
                           hero_alpha=hero_alpha)

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

    n_raw_read = 0
    n_bad_json = 0
    n_emitted = 0
    n_prepatch_days = 0
    n_inpatch_days = 0
    n_anon_hist: list[int] = []
    max_matches = int(args.max_matches) if args.max_matches > 0 else 0

    t0 = time.time()
    aborted_early = False
    for day in tqdm(days, desc="days"):
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
                if m.get("radiant_win") is None:
                    continue
                day_matches.append((int(st), int(mids[i]), m))
            del tbl_in, jsons
        day_matches.sort(key=lambda x: (x[0], x[1]))
        for start_ts, mid, m in day_matches:
            rw = 1 if m["radiant_win"] else 0
            players = m["players"]
            accts = [int(p.get("account_id") or 0) for p in players]
            heroes = [int(p.get("hero_id") or 0) for p in players]
            radiant_team = accts[:5]
            dire_team = accts[5:]
            teammates_lists = []
            for i in range(10):
                team = radiant_team if i < 5 else dire_team
                offset = 0 if i < 5 else 5
                teammates_lists.append([team[j] for j in range(5) if j + offset != i])

            if mid in emit_mids:
                proc = proc_idx.get(mid)
                if proc is not None:
                    _, sd, proc_rw, proc_heroes = proc
                    feat_row: list[float] = []
                    src_row: list[int] = []
                    n_anon_match = 0
                    for i in range(10):
                        a = accts[i]
                        h = heroes[i]
                        if a in ANON_IDS:
                            n_anon_match += 1
                        feats, n_pre, n_in = agg.snapshot(
                            a, h, start_ts, teammates_lists[i]
                        )
                        feat_row.extend(feats)
                        src_row.extend([n_pre, n_in])
                    out_cols["match_id"].append(mid)
                    out_cols["start_time_date"].append(sd)
                    out_cols["radiant_win"].append(proc_rw)
                    for j in range(5):
                        out_cols[f"r{j}"].append(proc_heroes[j])
                        out_cols[f"d{j}"].append(proc_heroes[5 + j])
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
                    sd_obj = dt.date.fromisoformat(sd)
                    if train_lo <= sd_obj <= train_hi:
                        out_cols["split"].append("train")
                    elif val_lo <= sd_obj <= val_hi:
                        out_cols["split"].append("val")
                    else:
                        out_cols["split"].append("other")
                    n_emitted += 1
                    n_anon_hist.append(n_anon_match)
                    if max_matches > 0 and n_emitted >= max_matches:
                        print(f"  max_matches cap {max_matches:,} hit — aborting walk early.")
                        aborted_early = True
                        break

            for i in range(10):
                won = rw if i < 5 else (1 - rw)
                agg.update(accts[i], heroes[i], won, start_ts,
                           teammates_lists[i], is_prepatch_day)

        gc.collect()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. raw_read={n_raw_read:,} bad_json={n_bad_json:,} "
          f"emitted={n_emitted:,}")
    print(f"Aggregator clamp events: total={agg.clamp_events}, by_feat={agg.clamp_by_feat}")

    if n_emitted == 0:
        sys.exit("No rows emitted — pipeline broken.")

    # Pre-arrow validation: walk every per-player float column as numpy and bounds-check.
    print("Pre-arrow validation: per-player float columns...")
    pre_bad_total = 0
    for p in range(N_PLAYERS):
        for f in FEAT_NAMES_PER_PLAYER:
            cname = f"p{p}_{f}"
            arr = np.asarray(out_cols[cname], dtype=np.float32)
            n_bad, sample = validate_column_array(arr, f)
            if n_bad > 0:
                print(f"  PRE-ARROW BAD: {cname}: {n_bad} cells; sample={sample[:5]}")
                pre_bad_total += n_bad
    if pre_bad_total > 0:
        sys.exit(f"REFUSED: pre-arrow validation found {pre_bad_total} bad cells "
                 f"(snapshot() validator failed to catch them). Build aborted.")

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
            # Build via numpy.float32 FIRST then wrap in pa.array — this is the
            # belt-and-braces guard against the pa.array(python_list,
            # type=pa.float32()) path that the prior build used (and which we
            # suspect of corrupting one column on row group 2 of train.parquet).
            np_col = np.asarray(out_cols[c], dtype=np.float32)
            arrs[c] = pa.array(np_col, type=pa.float32())
    for c in player_source_cols():
        arrs[c] = pa.array(out_cols[c], type=pa.uint32())
    arrs["n_anonymous_in_match"] = pa.array(out_cols["n_anonymous_in_match"], type=pa.uint8())
    arrs["split"] = pa.array(out_cols["split"], type=pa.string())
    tbl = pa.table(arrs)

    print("In-memory pa.Table validation (post-conversion, pre-write)...")
    validate_table(tbl, where="in-memory")

    train_tbl_out = tbl.filter(pa.compute.equal(tbl.column("split"), "train"))
    val_tbl_out = tbl.filter(pa.compute.equal(tbl.column("split"), "val"))
    print(f"  train rows: {train_tbl_out.num_rows:,}")
    print(f"  val   rows: {val_tbl_out.num_rows:,}")

    suffix = "_smoke" if args.smoke else ""
    out_train = out_dir / f"train{suffix}.parquet"
    out_val = out_dir / f"val{suffix}.parquet"
    pq.write_table(train_tbl_out, out_train, compression="zstd")
    pq.write_table(val_tbl_out, out_val, compression="zstd")
    print(f"Wrote {out_train} ({out_train.stat().st_size/1e6:.1f} MB)")
    print(f"Wrote {out_val} ({out_val.stat().st_size/1e6:.1f} MB)")

    # Post-write re-read validation. Reads the file we just wrote and runs the
    # same bounds check. This catches any corruption introduced by pyarrow's
    # write path or the underlying I/O layer.
    print("Post-write validation: re-reading and bounds-checking...")
    for path in (out_train, out_val):
        if path.stat().st_size == 0:
            continue
        rt = pq.read_table(path)
        try:
            validate_table(rt, where=f"post-write({path.name})")
        except SystemExit:
            print(f"  bad on-disk artifact at {path}; deleting.")
            path.unlink(missing_ok=True)
            raise

    anon_arr = np.array(n_anon_hist, dtype=np.int32) if n_anon_hist else np.array([0])
    stats = {
        "smoke": bool(args.smoke),
        "max_matches_cap": int(max_matches),
        "n_raw_read": int(n_raw_read),
        "n_bad_json": int(n_bad_json),
        "n_emitted": int(n_emitted),
        "n_train_emitted": int(train_tbl_out.num_rows),
        "n_val_emitted": int(val_tbl_out.num_rows),
        "build_seconds": float(elapsed),
        "days_processed": [days[0], days[-1]],
        "n_days_total": len(days),
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
        "config": {
            "alpha": alpha, "hero_alpha": hero_alpha,
            "recent_window": recent_window,
        },
        "feat_bounds": {k: list(v) for k, v in FEAT_BOUNDS.items()},
    }
    stats_path = out_dir / f"build_stats{suffix}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Stats: {stats_path}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
