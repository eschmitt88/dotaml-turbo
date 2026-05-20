"""Data loading for transformer-plus-features-740.

Reads the augmented parquet built by player-features-prepatch-740 at
data/snapshots/.../processed/player_features_prepatch/{train,val}.parquet.
Produces three owned torch tensors per split:
  hero_ids     : [N, 10]  long
  player_feats : [N, 10, n_player_feats]  float32
  y            : [N]      float32

HCE rule: only train.parquet and val.parquet are read. The test parquet at
[2026-03-10, 2026-03-23] is sealed; we assert against it via splits.yaml.

Same 5M-row stratified subsample (seed=42) as plateau-baseline-740,
plateau-architectures-740, transformer-hp-sweep-740, and
player-features-prepatch-740 — so the train rows match across experiments.

Owned-tensor (torch.tensor → deep copy) construction is the
Blackwell-torch-dataloader-bug workaround; see
docs/decisions/0001-per-trial-subprocess-isolation.md.
"""
from __future__ import annotations

import datetime as dt
import gc
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def stratified_subsample(y: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """Mirrors plateau-baseline-740/train.py:stratified_subsample exactly."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n_target >= n:
        return rng.permutation(n)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    p_pos = len(pos_idx) / n
    n_pos = int(round(n_target * p_pos))
    n_neg = n_target - n_pos
    pos_pick = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_pick = rng.choice(neg_idx, size=n_neg, replace=False)
    out = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(out)
    return out


def assert_no_test_dates(tbl, name: str, splits: dict) -> None:
    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    sds = tbl.column("start_time_date").to_pylist()
    bad = [s for s in sds if test_lo <= dt.date.fromisoformat(s) <= test_hi]
    if bad:
        raise SystemExit(
            f"REFUSED: {name} split contains test-window dates {bad[:3]}... — HCE rule."
        )


def load_arrays(table, feat_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (hero_ids[N, 10] int64, player_feats[N, 10, F] float32, y[N] float32)."""
    r_cols = [table.column(f"r{i}").to_numpy(zero_copy_only=False).astype(np.int64) for i in range(5)]
    d_cols = [table.column(f"d{i}").to_numpy(zero_copy_only=False).astype(np.int64) for i in range(5)]
    hero_ids = np.stack(r_cols + d_cols, axis=1)             # [N, 10]
    if hero_ids.min() < 1 or hero_ids.max() > 150:
        raise ValueError(
            f"hero_ids out of expected [1, 150] range: [{hero_ids.min()}, {hero_ids.max()}]"
        )
    y = table.column("radiant_win").to_numpy(zero_copy_only=False).astype(np.float32)

    # Build [N, 10, F] in (player, feat) order matching FEAT_NAMES_PER_PLAYER from
    # player-features-prepatch-740/train.py. Column name: p{p}_{feat_name}.
    n = table.num_rows
    F = len(feat_names)
    player_feats = np.empty((n, 10, F), dtype=np.float32)
    for p in range(10):
        for f_idx, fname in enumerate(feat_names):
            col = f"p{p}_{fname}"
            player_feats[:, p, f_idx] = (
                table.column(col).to_numpy(zero_copy_only=False).astype(np.float32)
            )

    # Defensive sanitization. player-features-prepatch-740/train.parquet has a
    # tiny fraction (~0.030%) of rows where one of the p{p}_smoothed_winrate_hero
    # columns carries a corrupted ~±3.4e38 (fp32 max) sentinel. The other 7
    # feature columns are clean. We can't modify the upstream experiment (hard
    # rule), so sanitize at load time with feature-aware bounds.
    feat_bounds = {
        "n_games_log1p":          (0.0, 20.0),    # log1p of n_games; n_games rarely > 10^9
        "smoothed_winrate":       (0.0, 1.0),     # winrate is a probability
        "smoothed_winrate_hero":  (0.0, 1.0),     # winrate is a probability  <-- corrupted column
        "last10_winrate":         (0.0, 1.0),     # rolling winrate
        "days_since_last_log1p":  (0.0, 20.0),    # log1p of days
        "n_games_hero_log1p":     (0.0, 20.0),    # log1p of n_games
        "hero_diversity_log1p":   (0.0, 10.0),    # log1p of distinct heroes
        "is_anonymous":           (0.0, 1.0),     # binary
    }
    total_bad = 0
    bad_by_feat = {}
    for f_idx, fname in enumerate(feat_names):
        lo, hi = feat_bounds.get(fname, (-1e6, 1e6))
        col = player_feats[:, :, f_idx]
        bad_mask = ~np.isfinite(col) | (col < lo) | (col > hi)
        nb = int(bad_mask.sum())
        if nb == 0:
            continue
        good = col[~bad_mask]
        med = float(np.median(good)) if good.size else 0.5 * (lo + hi)
        col[bad_mask] = med
        total_bad += nb
        bad_by_feat[fname] = nb
    if total_bad > 0:
        print(f"  data.load_arrays: sanitized {total_bad} corrupted feature values "
              f"to per-feature median {bad_by_feat} (likely upstream sentinel "
              f"in player-features-prepatch-740 build)")
    if not np.all(np.isfinite(player_feats)):
        raise ValueError("player_feats still contains non-finite values after sanitization")
    if float(np.abs(player_feats).max()) > 1e3:
        raise ValueError(
            f"player_feats abs-max {np.abs(player_feats).max()} > 1e3 after sanitization"
        )
    return hero_ids, player_feats, y


class DraftPlusFeaturesDataset(Dataset):
    """In-memory tensor dataset.

    5M rows: hero_ids 5M*10 int64 ≈ 400 MB, player_feats 5M*10*8 float32 ≈ 1.6 GB,
    y 5M float32 ≈ 20 MB. Total ≈ 2 GB host RAM — well under the 64 GB budget.
    """

    def __init__(self, hero_ids: np.ndarray, player_feats: np.ndarray, y: np.ndarray):
        # Use torch.tensor (deep copy) — owns memory, avoids the from_numpy
        # shared-memory path that triggered "Overflow when unpacking long long"
        # mid-training on Blackwell + torch 2.9-2.12.
        self.hero_ids = torch.tensor(hero_ids, dtype=torch.long).contiguous()
        self.player_feats = torch.tensor(player_feats, dtype=torch.float32).contiguous()
        self.y = torch.tensor(y, dtype=torch.float32).contiguous()

    def __len__(self) -> int:
        return self.hero_ids.size(0)

    def __getitem__(self, idx: int):
        return self.hero_ids[idx], self.player_feats[idx], self.y[idx]


def load_train_val(seed: int, n_target: int, feat_names: list[str],
                   source_dir: Path, splits: dict, smoke: bool = False,
                   smoke_n_train: int = 50_000, smoke_n_val: int = 5_000):
    """Load + subsample. Returns (train_ds, val_ds, meta_dict)."""
    train_path = source_dir / "train.parquet"
    val_path = source_dir / "val.parquet"

    print(f"Reading augmented parquet: {train_path}")
    train_tbl = pq.read_table(train_path)
    print(f"  train rows: {train_tbl.num_rows:,}")
    print(f"Reading augmented parquet: {val_path}")
    val_tbl = pq.read_table(val_path)
    print(f"  val   rows: {val_tbl.num_rows:,}")

    assert_no_test_dates(train_tbl, "train", splits)
    assert_no_test_dates(val_tbl, "val", splits)

    train_dr = (
        str(np.min(train_tbl.column("start_time_date").to_numpy(zero_copy_only=False))),
        str(np.max(train_tbl.column("start_time_date").to_numpy(zero_copy_only=False))),
    )
    val_dr = (
        str(np.min(val_tbl.column("start_time_date").to_numpy(zero_copy_only=False))),
        str(np.max(val_tbl.column("start_time_date").to_numpy(zero_copy_only=False))),
    )

    y_train_full = train_tbl.column("radiant_win").to_numpy(zero_copy_only=False).astype(np.int8)
    sub_idx = stratified_subsample(y_train_full, n_target, seed)
    train_tbl_sub = train_tbl.take(sub_idx)

    if smoke:
        rng = np.random.default_rng(seed + 1)
        train_pick = rng.choice(train_tbl_sub.num_rows, size=smoke_n_train, replace=False)
        val_pick = rng.choice(val_tbl.num_rows, size=smoke_n_val, replace=False)
        train_tbl_sub = train_tbl_sub.take(train_pick)
        val_tbl = val_tbl.take(val_pick)

    h_tr, pf_tr, y_tr = load_arrays(train_tbl_sub, feat_names)
    h_va, pf_va, y_va = load_arrays(val_tbl, feat_names)

    train_ds = DraftPlusFeaturesDataset(h_tr, pf_tr, y_tr)
    val_ds = DraftPlusFeaturesDataset(h_va, pf_va, y_va)

    # Snapshot what meta needs BEFORE releasing pyarrow refs.
    n_train_pre = int(train_tbl.num_rows)
    n_train_post = int(train_tbl_sub.num_rows)
    n_val = int(val_tbl.num_rows)
    radiant_base_rate_train_full = float(y_train_full.mean())
    radiant_base_rate_train_subsampled = float(y_tr.mean())
    radiant_base_rate_val = float(y_va.mean())

    # Eager release; see plateau-architectures + transformer-hp-sweep data.py.
    del train_tbl, train_tbl_sub, val_tbl, h_tr, pf_tr, y_tr, h_va, pf_va, y_va
    del y_train_full, sub_idx
    gc.collect()

    meta = {
        "n_train_pre_subsample": n_train_pre,
        "n_train_post_subsample": n_train_post,
        "n_val": n_val,
        "train_subset_size_target": int(n_target),
        "train_subset_seed": int(seed),
        "train_date_min": train_dr[0],
        "train_date_max": train_dr[1],
        "val_date_min": val_dr[0],
        "val_date_max": val_dr[1],
        "radiant_base_rate_train_full": radiant_base_rate_train_full,
        "radiant_base_rate_train_subsampled": radiant_base_rate_train_subsampled,
        "radiant_base_rate_val": radiant_base_rate_val,
        "smoke": bool(smoke),
        "feat_names": list(feat_names),
        "n_player_feats": len(feat_names),
    }
    return train_ds, val_ds, meta


__all__ = ["DraftPlusFeaturesDataset", "load_train_val",
           "stratified_subsample", "assert_no_test_dates", "load_arrays"]
