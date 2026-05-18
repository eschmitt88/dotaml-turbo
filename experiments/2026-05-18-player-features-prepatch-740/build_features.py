"""Build augmented feature parquet for player-features-prepatch-740.

Differs from player-features-740/build_features.py in TWO ways:

1. Walks MULTIPLE raw roots in chronological order. By default the
   aggregator processes data/history/turbo/ (Aug 1 → Dec 15 2025) first
   and then continues into data/snapshots/7.40-2025-12-16/raw/turbo/
   (Dec 16 → Mar 9). The aggregator state carries seamlessly across the
   patch boundary; no reset, no decay.

2. Adds per-player history-source tracking: for each player at snapshot
   time we report n_games_prepatch and n_games_inpatch alongside the
   usual feature vector, so the train.py history-source-breakdown
   diagnostic can compute mean(n_games_prepatch / n_games_total) per
   coverage bucket.

Output: data/snapshots/7.40-2025-12-16/processed/player_features_prepatch/
        {train,val}.parquet (or {train,val}_smoke.parquet in --smoke mode).

Schema is the same as player-features-740 PLUS two new columns per player:
  p{i}_n_games_prepatch        uint32
  p{i}_n_games_inpatch         uint32

HCE: refuses to read any date in [test_start_date, test_end_date] or
in (snapshot_end_date, ...) at walk time.

Aggregator scope: ALL raw matches across BOTH roots update aggregator
state; the OUTPUT row set is exactly the filtered processed-parquet
match_id set from data/snapshots/.../processed/{train,val}.parquet
(produced by the original plateau-baseline build).
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
# Patch-7.40 begins 2025-12-16. Used to label history-source per player.
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
    # NOTE 2026-05-18: `coplay_mean` removed from this experiment vs
    # player-features-740. Two reasons: (1) coplay_mean wasn't in the
    # top-20 feature_importances of player-features-740 (contributed
    # essentially nothing); (2) the per-account coplay dict was the
    # dominant memory hog (~75 GB at the ~5M-account scale this
    # experiment hits) and caused two OOM-kills before this edit.
    # The heroes_only sanity check vs plateau-baseline-740 still
    # holds; the heroes_plus_features comparison vs player-features-740
    # is slightly less clean (8 player features instead of 9) but the
    # removed feature contributed nothing, so the bias is bounded.
    "is_anonymous",
]
N_FEATS_PER_PLAYER = len(FEAT_NAMES_PER_PLAYER)
N_PLAYERS = 10

# Extra per-player cols for history-source tracking.
SOURCE_COL_NAMES = ["n_games_prepatch", "n_games_inpatch"]


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

    Adds n_games_prepatch / n_games_inpatch counters per account so
    that snapshot() can also return the source-of-history breakdown.
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
        # `unique_heroes` set removed — derive via len(hero_n[acct]) instead
        # (saves ~8 GB at 5M accounts).
        self.hero_n: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.hero_w: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.hero_global_n: dict[int, int] = defaultdict(int)
        self.hero_global_w: dict[int, int] = defaultdict(int)
        # `coplay` removed — see FEAT_NAMES_PER_PLAYER note above.
        # NEW: history-source counters per account.
        self.n_pre: dict[int, int] = defaultdict(int)
        self.n_in: dict[int, int] = defaultdict(int)

    def _hero_base(self, hero_id: int) -> float:
        n = self.hero_global_n.get(hero_id, 0)
        if n == 0:
            return self.global_prior
        return self.hero_global_w[hero_id] / n

    def snapshot(self, acct: int, hero_id: int, now_ts: int,
                 teammates: list[int]) -> tuple[list[float], int, int]:
        """Pre-match feature vector + (n_prepatch, n_inpatch) for one player."""
        is_anon = 1 if acct in ANON_IDS else 0
        n = self.n_games.get(acct, 0)
        w = self.n_wins.get(acct, 0)
        sw = (self.alpha + w) / (2 * self.alpha + n)
        hn = self.hero_n.get(acct, {}).get(hero_id, 0) if acct in self.hero_n else 0
        hw = self.hero_w.get(acct, {}).get(hero_id, 0) if acct in self.hero_w else 0
        hero_prior = self._hero_base(hero_id)
        sw_hero = (self.hero_alpha * hero_prior + hw) / (self.hero_alpha + hn)
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
        feats = [
            float(math.log1p(n)),
            float(sw),
            float(sw_hero),
            float(last10),
            float(math.log1p(days_since)),
            float(math.log1p(n_hero)),
            float(math.log1p(n_diverse)),
            float(is_anon),
        ]
        return feats, int(self.n_pre.get(acct, 0)), int(self.n_in.get(acct, 0))

    def update(self, acct: int, hero_id: int, won: int, now_ts: int,
               teammates: list[int], is_prepatch: bool) -> None:
        """Apply this match's outcome to the running state."""
        if acct in ANON_IDS:
            self.hero_global_n[hero_id] += 1
            self.hero_global_w[hero_id] += won
            return
        self.n_games[acct] += 1
        self.n_wins[acct] += won
        self.last_time[acct] = now_ts
        self.recent_wins[acct].append(won)
        # unique_heroes set update removed — derive via len(hero_n[acct]).
        self.hero_n[acct][hero_id] += 1
        self.hero_w[acct][hero_id] += won
        self.hero_global_n[hero_id] += 1
        self.hero_global_w[hero_id] += won
        if is_prepatch:
            self.n_pre[acct] += 1
        else:
            self.n_in[acct] += 1
        # coplay update block removed — see FEAT_NAMES_PER_PLAYER note.


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
    """Walk one or more raw roots, group all matches_*.parquet by YYYY-MM-DD."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Process small subset; see config.yaml.smoke for spec.")
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
        # Smoke: keep the configured history_days + first n_days of patch-7.40
        # (matching the player-features-740 smoke window). Emit features for
        # those patch-7.40 days only.
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

    t0 = time.time()
    for day in tqdm(days, desc="days"):
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
                if m.get("radiant_win") is None:
                    continue
                day_matches.append((int(st), int(mids[i]), m))
            del tbl, jsons
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

            for i in range(10):
                won = rw if i < 5 else (1 - rw)
                agg.update(accts[i], heroes[i], won, start_ts,
                           teammates_lists[i], is_prepatch_day)

        gc.collect()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. raw_read={n_raw_read:,} bad_json={n_bad_json:,} "
          f"emitted={n_emitted:,}")

    if n_emitted == 0:
        sys.exit("No rows emitted — pipeline broken.")

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
    for c in player_source_cols():
        arrs[c] = pa.array(out_cols[c], type=pa.uint32())
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
        "n_days_total": len(days),
        "n_prepatch_days_walked": n_prepatch_days,
        "n_inpatch_days_walked": n_inpatch_days,
        "n_unique_account_ids_tracked": len(agg.n_games),
        "raw_roots": [str(r) for r in raw_roots],
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
