"""Build the augmented feature parquet for player-features-740.

Walks the raw parquet day-by-day in chronological order, parses raw_json,
extracts per-player (account_id, hero_id) plus match-level (radiant_win,
start_time), and maintains per-account running aggregates. For each match
in the filtered processed-parquet row set, snapshots a leading-window
feature vector for all 10 players using ONLY matches with strictly earlier
start_time (HCE-strict).

Output: data/snapshots/7.40-2025-12-16/processed/player_features/{train,val}.parquet
        (or {train,val}_smoke.parquet in --smoke mode)

Schema per player p in [0..9] (radiant 0..4, then dire 0..4):
  p{i}_n_games_log1p           float32  log1p(games played in window)
  p{i}_smoothed_winrate        float32  (alpha + wins) / (2*alpha + games)
  p{i}_smoothed_winrate_hero   float32  hero-specific smoothed winrate
  p{i}_last10_winrate          float32  recent-form winrate (10-game)
  p{i}_days_since_last_log1p   float32  log1p(days since last match)
  p{i}_n_games_hero_log1p      float32  log1p(games on current hero)
  p{i}_hero_diversity_log1p    float32  log1p(unique heroes played)
  p{i}_coplay_mean             float32  mean coplay count w/ the 4 teammates
  p{i}_is_anonymous            uint8    1 if account_id in {0, 4294967295}

Plus:
  match_id, start_time_date, radiant_win, r0..r4, d0..d4, split
  n_anonymous_in_match (uint8)

HCE: refuses to read any date in [test_start_date, test_end_date].

Aggregator scope: ALL raw matches (no filter) update the aggregator state
so that a player's history reflects every game they actually played,
including ones that were filter-dropped from the output row set.
The OUTPUT row set is the filtered processed-parquet match_id set.
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
RAW_ROOT = SNAPSHOT_DIR / "raw" / "turbo"
PROCESSED_DIR = SNAPSHOT_DIR / "processed"
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"

ANON_IDS = {0, 4294967295}

FEAT_NAMES_PER_PLAYER = [
    "n_games_log1p",
    "smoothed_winrate",
    "smoothed_winrate_hero",
    "last10_winrate",
    "days_since_last_log1p",
    "n_games_hero_log1p",
    "hero_diversity_log1p",
    "coplay_mean",
    "is_anonymous",
]
N_FEATS_PER_PLAYER = len(FEAT_NAMES_PER_PLAYER)
N_PLAYERS = 10


def player_feat_cols() -> list[str]:
    cols = []
    for p in range(N_PLAYERS):
        for f in FEAT_NAMES_PER_PLAYER:
            cols.append(f"p{p}_{f}")
    return cols


class PlayerAggregator:
    """Per-account running aggregates updated chronologically.

    State per account (held in dict-of-arrays/dicts for memory efficiency).
    Memory budget: ~10M unique account_ids × ~150 B core state ≈ 1.5 GB.
    Coplay dict adds variable cost; we cap entries-per-account to 200 most
    recent teammates to keep total RAM bounded.
    """

    COPLAY_CAP = 200  # per-account teammate dict size cap

    def __init__(self, recent_window: int, alpha: float, hero_alpha: float,
                 global_radiant_prior: float = 0.5335):
        self.alpha = float(alpha)
        self.hero_alpha = float(hero_alpha)
        self.recent_window = int(recent_window)
        self.global_prior = float(global_radiant_prior)
        # core scalar state per account_id
        self.n_games: dict[int, int] = defaultdict(int)
        self.n_wins: dict[int, int] = defaultdict(int)
        self.last_time: dict[int, int] = {}
        self.recent_wins: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.recent_window)
        )
        self.unique_heroes: dict[int, set] = defaultdict(set)
        # hero-specific
        self.hero_n: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.hero_w: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # per-hero global base rate (computed in-window from won-side accounting)
        self.hero_global_n: dict[int, int] = defaultdict(int)
        self.hero_global_w: dict[int, int] = defaultdict(int)
        # coplay: account -> {teammate_account: count}
        self.coplay: dict[int, dict[int, int]] = defaultdict(dict)

    def _hero_base(self, hero_id: int) -> float:
        n = self.hero_global_n.get(hero_id, 0)
        if n == 0:
            return self.global_prior
        return self.hero_global_w[hero_id] / n

    def snapshot(self, acct: int, hero_id: int, now_ts: int, teammates: list[int]) -> list[float]:
        """Pre-match feature vector for one player. Uses only state at time T."""
        is_anon = 1 if acct in ANON_IDS else 0
        n = self.n_games.get(acct, 0)
        w = self.n_wins.get(acct, 0)
        sw = (self.alpha + w) / (2 * self.alpha + n)
        hn = self.hero_n.get(acct, {}).get(hero_id, 0) if acct in self.hero_n else 0
        hw = self.hero_w.get(acct, {}).get(hero_id, 0) if acct in self.hero_w else 0
        hero_prior = self._hero_base(hero_id)
        # shrink hero-specific rate toward per-hero base rate
        sw_hero = (self.hero_alpha * hero_prior + hw) / (self.hero_alpha + hn)
        rd = self.recent_wins.get(acct)
        if rd is not None and len(rd) > 0:
            last10 = sum(rd) / len(rd)
        else:
            last10 = 0.5  # neutral
        last_t = self.last_time.get(acct)
        if last_t is None:
            days_since = 365.0  # large
        else:
            days_since = max(0.0, (now_ts - last_t) / 86400.0)
        n_hero = hn
        n_diverse = len(self.unique_heroes.get(acct, ()))
        # coplay mean
        if is_anon or not teammates:
            coplay_mean = 0.0
        else:
            cp = self.coplay.get(acct, {})
            if not cp:
                coplay_mean = 0.0
            else:
                vals = [cp.get(t, 0) for t in teammates if t not in ANON_IDS]
                coplay_mean = sum(vals) / max(1, len(vals)) if vals else 0.0
        return [
            float(math.log1p(n)),
            float(sw),
            float(sw_hero),
            float(last10),
            float(math.log1p(days_since)),
            float(math.log1p(n_hero)),
            float(math.log1p(n_diverse)),
            float(coplay_mean),
            float(is_anon),
        ]

    def update(self, acct: int, hero_id: int, won: int, now_ts: int,
               teammates: list[int]) -> None:
        """Apply this match's outcome to the running state."""
        if acct in ANON_IDS:
            # We still keep hero_global stats from anonymous accounts so the
            # hero base rate is informed by all games, but we DON'T bloat the
            # per-account dicts for anonymous IDs (they all collapse to a few
            # huge keys which would wreck stats).
            self.hero_global_n[hero_id] += 1
            self.hero_global_w[hero_id] += won
            return
        self.n_games[acct] += 1
        self.n_wins[acct] += won
        self.last_time[acct] = now_ts
        self.recent_wins[acct].append(won)
        self.unique_heroes[acct].add(hero_id)
        self.hero_n[acct][hero_id] += 1
        self.hero_w[acct][hero_id] += won
        self.hero_global_n[hero_id] += 1
        self.hero_global_w[hero_id] += won
        # coplay (only with non-anon teammates)
        cp = self.coplay[acct]
        for t in teammates:
            if t in ANON_IDS:
                continue
            cp[t] = cp.get(t, 0) + 1
        # cap coplay dict size to bound memory
        if len(cp) > self.COPLAY_CAP:
            # drop the 50 smallest entries
            items = sorted(cp.items(), key=lambda kv: kv[1])
            for k, _ in items[:50]:
                del cp[k]


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
    ap.add_argument("--smoke", action="store_true",
                    help="Process only first N days from config.yaml.smoke.n_days.")
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--limit-rows-per-file", type=int, default=0,
                    help="Debug cap on rows per raw file.")
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

    train_lo = dt.date.fromisoformat(splits["train_start_date"])
    train_hi = dt.date.fromisoformat(splits["train_end_date"])
    val_lo = dt.date.fromisoformat(splits["val_start_date"])
    val_hi = dt.date.fromisoformat(splits["val_end_date"])
    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])

    # Load the filtered processed parquet so we know exactly which match_ids
    # to EMIT features for. The aggregator processes ALL raw matches.
    print("Loading filtered processed parquet (row set to emit)...")
    train_tbl = pq.read_table(PROCESSED_DIR / "train.parquet")
    val_tbl = pq.read_table(PROCESSED_DIR / "val.parquet")
    train_mids = set(int(x) for x in train_tbl.column("match_id").to_pylist())
    val_mids = set(int(x) for x in val_tbl.column("match_id").to_pylist())
    print(f"  filtered train rows: {len(train_mids):,}")
    print(f"  filtered val   rows: {len(val_mids):,}")
    emit_mids = train_mids | val_mids
    # Map: match_id -> (split, radiant_win, [hero_ids_10])
    # Pull from processed parquet so the emitted output rows are exactly aligned.
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

    # Enumerate raw files, group by day.
    files = sorted(RAW_ROOT.rglob("matches_*.parquet"))
    if not files:
        sys.exit("No raw files; run pull_raw.py first.")
    by_day: dict[str, list[Path]] = {}
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
    days = sorted(by_day.keys())
    print(f"Found {len(files)} raw files, {len(days)} days "
          f"(first={days[0]}, last={days[-1]})")
    assert_no_test_dates(days, test_lo, test_hi)

    if args.smoke:
        n_smoke_days = int(smoke_cfg["n_days"])
        days = days[:n_smoke_days]
        print(f"SMOKE MODE: limited to first {len(days)} days ({days[0]}..{days[-1]})")
        # Restrict emit set to match_ids in those days
        keep_dates = set(days)
        emit_mids = {mid for mid, (_, sd, _, _) in proc_idx.items() if sd in keep_dates}
        print(f"  smoke emit_mids: {len(emit_mids):,}")

    agg = PlayerAggregator(recent_window=recent_window, alpha=alpha,
                           hero_alpha=hero_alpha)

    # Output buffers (column-major lists).
    out_cols: dict[str, list] = {
        "match_id": [],
        "start_time_date": [],
        "radiant_win": [],
    }
    for k in ("r0", "r1", "r2", "r3", "r4", "d0", "d1", "d2", "d3", "d4"):
        out_cols[k] = []
    for c in player_feat_cols():
        out_cols[c] = []
    out_cols["n_anonymous_in_match"] = []
    out_cols["split"] = []

    n_raw_read = 0
    n_bad_json = 0
    n_emitted = 0
    n_anon_hist: list[int] = []
    n_proc_misses = 0  # match_ids in processed but not seen in raw walk

    t0 = time.time()
    # Walk days in chronological order. Within a day, accumulate matches into
    # a list, then sort by start_time and process in that order — this ensures
    # within-day chronology is respected for the leading-window invariant.
    for day in tqdm(days, desc="days"):
        d_obj = dt.date.fromisoformat(day)
        # HCE assert per day
        if test_lo <= d_obj <= test_hi:
            sys.exit(f"REFUSED: refusing to process test-window day {day}")
        # Collect all matches for this day (across multiple files).
        day_matches: list[tuple] = []  # (start_time, match_id, raw_json_obj)
        for fp in by_day[day]:
            try:
                tbl = pq.read_table(
                    fp, columns=["match_id", "raw_json", "game_mode"]
                )
            except Exception as e:  # noqa: BLE001
                print(f"  read fail {fp}: {e}")
                continue
            mids = tbl.column("match_id").to_numpy(zero_copy_only=False)
            gms = tbl.column("game_mode").to_numpy(zero_copy_only=False)
            jsons = tbl.column("raw_json").to_pylist()
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
                # Quick sanity: must have radiant_win, hero_ids, account_ids
                if m.get("radiant_win") is None:
                    continue
                day_matches.append((int(st), int(mids[i]), m))
            del tbl, jsons
        # Sort by start_time (deterministic; ties broken by match_id).
        day_matches.sort(key=lambda x: (x[0], x[1]))
        # Process in order.
        for start_ts, mid, m in day_matches:
            rw = 1 if m["radiant_win"] else 0
            players = m["players"]
            # 10 (account_id, hero_id) pairs, in player_slot order (r0..r4, d0..d4)
            accts = [int(p.get("account_id") or 0) for p in players]
            heroes = [int(p.get("hero_id") or 0) for p in players]
            radiant_team = accts[:5]
            dire_team = accts[5:]
            # Per-player teammate lists (the other 4 on their team)
            teammates_lists = []
            for i in range(10):
                team = radiant_team if i < 5 else dire_team
                offset = 0 if i < 5 else 5
                teammates_lists.append([team[j] for j in range(5) if j + offset != i])

            # SNAPSHOT pre-match features (only if this match is in emit set)
            if mid in emit_mids:
                # Cross-check with processed index: heroes must match
                proc = proc_idx.get(mid)
                if proc is None:
                    # in emit_mids but not proc_idx — shouldn't happen
                    pass
                else:
                    _, sd, proc_rw, proc_heroes = proc
                    # Use processed heroes / rw as authoritative for emission
                    feat_row: list[float] = []
                    n_anon_match = 0
                    for i in range(10):
                        a = accts[i]
                        h = heroes[i]
                        if a in ANON_IDS:
                            n_anon_match += 1
                        feats = agg.snapshot(a, h, start_ts, teammates_lists[i])
                        feat_row.extend(feats)
                    # Emit
                    out_cols["match_id"].append(mid)
                    out_cols["start_time_date"].append(sd)
                    out_cols["radiant_win"].append(proc_rw)
                    for j in range(5):
                        out_cols[f"r{j}"].append(proc_heroes[j])
                        out_cols[f"d{j}"].append(proc_heroes[5 + j])
                    # Fill player feat columns
                    idx = 0
                    for p in range(N_PLAYERS):
                        for f in FEAT_NAMES_PER_PLAYER:
                            out_cols[f"p{p}_{f}"].append(feat_row[idx])
                            idx += 1
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

            # UPDATE aggregator with this match (always, regardless of emit)
            for i in range(10):
                won = rw if i < 5 else (1 - rw)
                agg.update(accts[i], heroes[i], won, start_ts, teammates_lists[i])

        gc.collect()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. raw_read={n_raw_read:,} bad_json={n_bad_json:,} "
          f"emitted={n_emitted:,}")

    if n_emitted == 0:
        sys.exit("No rows emitted — pipeline broken.")

    # Build arrow tables.
    # Type schedule: heroes uint16, feats float32, indicators uint8.
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
            arrs[c] = pa.array(out_cols[c], type=pa.float32())
    arrs["n_anonymous_in_match"] = pa.array(out_cols["n_anonymous_in_match"], type=pa.uint8())
    arrs["split"] = pa.array(out_cols["split"], type=pa.string())
    tbl = pa.table(arrs)

    train_tbl = tbl.filter(pa.compute.equal(tbl.column("split"), "train"))
    val_tbl = tbl.filter(pa.compute.equal(tbl.column("split"), "val"))
    print(f"  train rows: {train_tbl.num_rows:,}")
    print(f"  val   rows: {val_tbl.num_rows:,}")

    suffix = "_smoke" if args.smoke else ""
    out_train = out_dir / f"train{suffix}.parquet"
    out_val = out_dir / f"val{suffix}.parquet"
    pq.write_table(train_tbl, out_train, compression="zstd")
    pq.write_table(val_tbl, out_val, compression="zstd")
    print(f"Wrote {out_train} ({out_train.stat().st_size/1e6:.1f} MB)")
    print(f"Wrote {out_val} ({out_val.stat().st_size/1e6:.1f} MB)")

    # Stats
    anon_arr = np.array(n_anon_hist, dtype=np.int32) if n_anon_hist else np.array([0])
    stats = {
        "smoke": bool(args.smoke),
        "n_raw_read": int(n_raw_read),
        "n_bad_json": int(n_bad_json),
        "n_emitted": int(n_emitted),
        "n_train_emitted": int(train_tbl.num_rows),
        "n_val_emitted": int(val_tbl.num_rows),
        "build_seconds": float(elapsed),
        "days_processed": [days[0], days[-1]],
        "n_unique_account_ids_tracked": len(agg.n_games),
        "anonymous_per_match_hist": {
            "min": int(anon_arr.min()), "max": int(anon_arr.max()),
            "mean": float(anon_arr.mean()), "p50": int(np.median(anon_arr)),
            "p95": int(np.quantile(anon_arr, 0.95)),
        },
        "config": {
            "alpha": alpha, "hero_alpha": hero_alpha,
            "recent_window": recent_window,
        },
    }
    stats_path = out_dir / f"build_stats{suffix}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Stats: {stats_path}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
