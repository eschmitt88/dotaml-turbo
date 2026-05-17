"""Data loading for transformer-hp-sweep-740.

HCE rule: only data/snapshots/.../processed/{train,val}.parquet are read.
The test parquet [2026-03-10, 2026-03-23] is sealed and is asserted against.

Same 5M-row stratified subsample (seed=42) as plateau-baseline-740 and
plateau-architectures-740. Side bit is intentionally NOT loaded — the
MinimalTransformer uses a binary team embedding instead.
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
SNAPSHOT_DIR = PROJECT_ROOT / "data/snapshots/7.40-2025-12-16"
PROCESSED = SNAPSHOT_DIR / "processed"


def stratified_subsample(y: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """Return indices that stratify on y to roughly preserve class balance.

    Mirrors plateau-baseline-740/train.py:stratified_subsample exactly so
    the same 5M-row subset is selected (same seed=42).
    """
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
    """Hard guard for HCE rule. Reads start_time_date strings."""
    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    sds = tbl.column("start_time_date").to_pylist()
    bad = [s for s in sds if test_lo <= dt.date.fromisoformat(s) <= test_hi]
    if bad:
        raise SystemExit(
            f"REFUSED: {name} split contains test-window dates {bad[:3]}... — HCE rule."
        )


def load_arrays(table) -> tuple[np.ndarray, np.ndarray]:
    """Return (hero_ids[N, 10] int64, y[N] float32)."""
    r_cols = [table.column(f"r{i}").to_numpy(zero_copy_only=False).astype(np.int64) for i in range(5)]
    d_cols = [table.column(f"d{i}").to_numpy(zero_copy_only=False).astype(np.int64) for i in range(5)]
    hero_ids = np.stack(r_cols + d_cols, axis=1)             # [N, 10]
    y = table.column("radiant_win").to_numpy(zero_copy_only=False).astype(np.float32)
    if hero_ids.min() < 1 or hero_ids.max() > 150:
        raise ValueError(
            f"hero_ids out of expected [1, 150] range: [{hero_ids.min()}, {hero_ids.max()}]"
        )
    return hero_ids, y


class DraftDataset(Dataset):
    """In-memory tensor dataset (5M rows × 10 hero IDs is ~400 MB int64)."""

    def __init__(self, hero_ids: np.ndarray, y: np.ndarray):
        # Use torch.tensor (deep copy) instead of from_numpy (shared memory).
        # On torch 2.11 + Blackwell, the shared-memory path produced
        # intermittent "Overflow when unpacking long long" errors deep
        # into training when indexing. Owning the memory avoids it.
        self.hero_ids = torch.tensor(hero_ids, dtype=torch.long).contiguous()
        self.y = torch.tensor(y, dtype=torch.float32).contiguous()

    def __len__(self) -> int:
        return self.hero_ids.size(0)

    def __getitem__(self, idx: int):
        return self.hero_ids[idx], self.y[idx]


def load_train_val(seed: int, n_target: int, splits: dict, smoke: bool = False,
                   smoke_n_train: int = 100_000, smoke_n_val: int = 10_000):
    """Load and subsample. Returns (train_ds, val_ds, meta_dict)."""
    train_tbl = pq.read_table(PROCESSED / "train.parquet")
    val_tbl = pq.read_table(PROCESSED / "val.parquet")

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

    h_tr, y_tr = load_arrays(train_tbl_sub)
    h_va, y_va = load_arrays(val_tbl)

    train_ds = DraftDataset(h_tr, y_tr)
    val_ds = DraftDataset(h_va, y_va)

    # Snapshot the values meta needs BEFORE releasing pyarrow refs.
    n_train_pre = int(train_tbl.num_rows)
    n_train_post = int(train_tbl_sub.num_rows)
    n_val = int(val_tbl.num_rows)
    radiant_base_rate_train_full = float(y_train_full.mean())
    radiant_base_rate_train_subsampled = float(y_tr.mean())
    radiant_base_rate_val = float(y_va.mean())

    # Eagerly release pyarrow Tables / numpy bridges after the deep-copy
    # into owned torch tensors. Pyarrow refcounts were investigated as a
    # potential cause of the GC-time DataLoader segfault on Blackwell + torch
    # 2.9-2.12 (see docs/decisions/0001-per-trial-subprocess-isolation.md);
    # explicit release was confirmed NOT to be the fix — the bug is in
    # torch's own DataLoader + tensor GC interaction, not pyarrow. We keep
    # the eager release anyway because it is good memory hygiene with no
    # downside.
    del train_tbl, train_tbl_sub, val_tbl, h_tr, h_va, y_tr, y_va, y_train_full, sub_idx
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
    }
    return train_ds, val_ds, meta


__all__ = ["DraftDataset", "load_train_val", "stratified_subsample", "assert_no_test_dates"]
