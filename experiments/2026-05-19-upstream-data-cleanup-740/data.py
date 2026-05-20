"""Data loading for upstream-data-cleanup-740.

Forked from experiments/2026-05-19-transformer-plus-features-extended-740/data.py
with one change: the load-time sanitization shim in `load_arrays` is
**removed and replaced with a hard assertion**. The clean parquet's
build pipeline (build_features.py) guarantees finite-and-bounded cells;
if a single cell escapes, this loader fails loudly rather than silently
clipping. That's the whole point of this experiment — no more band-aids.

HCE rule: only train.parquet and val.parquet are read. The test parquet
at [2026-03-10, 2026-03-23] is sealed; we assert against it via splits.yaml.
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

# Per-feature physical bounds — must match build_features.py:FEAT_BOUNDS exactly.
FEAT_BOUNDS = {
    "n_games_log1p":          (0.0, 25.0),
    "smoothed_winrate":       (0.0, 1.0),
    "smoothed_winrate_hero":  (0.0, 1.0),
    "last10_winrate":         (0.0, 1.0),
    "days_since_last_log1p":  (0.0, 25.0),
    "n_games_hero_log1p":     (0.0, 25.0),
    "hero_diversity_log1p":   (0.0, 10.0),
    "is_anonymous":           (0.0, 1.0),
}


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
    """Return (hero_ids[N, 10] int64, player_feats[N, 10, F] float32, y[N] float32).

    No sanitization — instead, hard-asserts every per-feature column is finite
    and within FEAT_BOUNDS. The clean parquet contract says this MUST hold.
    If it doesn't, the rebuild is broken; fix the build, don't paper over it.
    """
    r_cols = [table.column(f"r{i}").to_numpy(zero_copy_only=False).astype(np.int64) for i in range(5)]
    d_cols = [table.column(f"d{i}").to_numpy(zero_copy_only=False).astype(np.int64) for i in range(5)]
    hero_ids = np.stack(r_cols + d_cols, axis=1)             # [N, 10]
    if hero_ids.min() < 1 or hero_ids.max() > 150:
        raise ValueError(
            f"hero_ids out of expected [1, 150] range: [{hero_ids.min()}, {hero_ids.max()}]"
        )
    y = table.column("radiant_win").to_numpy(zero_copy_only=False).astype(np.float32)

    n = table.num_rows
    F = len(feat_names)
    player_feats = np.empty((n, 10, F), dtype=np.float32)
    for p in range(10):
        for f_idx, fname in enumerate(feat_names):
            col = f"p{p}_{fname}"
            player_feats[:, p, f_idx] = (
                table.column(col).to_numpy(zero_copy_only=False).astype(np.float32)
            )

    # Hard assertion — clean parquet contract.
    total_bad = 0
    bad_by_feat: dict[str, int] = {}
    for f_idx, fname in enumerate(feat_names):
        lo, hi = FEAT_BOUNDS.get(fname, (-1e6, 1e6))
        col = player_feats[:, :, f_idx]
        bad_mask = ~np.isfinite(col) | (col < lo) | (col > hi)
        nb = int(bad_mask.sum())
        if nb > 0:
            bad_by_feat[fname] = nb
            total_bad += nb
    if total_bad > 0:
        raise SystemExit(
            f"REFUSED: clean parquet violation — {total_bad} cells outside bounds "
            f"({bad_by_feat}). Re-run build_features.py; do NOT band-aid here."
        )
    # Stronger overall safeguards (would have been the second-line guard in the
    # prior loader; preserved here as belt-and-braces).
    if not np.all(np.isfinite(player_feats)):
        raise SystemExit("REFUSED: player_feats contains non-finite values.")
    if float(np.abs(player_feats).max()) > 1e3:
        raise SystemExit(
            f"REFUSED: player_feats abs-max {np.abs(player_feats).max()} > 1e3."
        )
    return hero_ids, player_feats, y


class DraftPlusFeaturesDataset(Dataset):
    """In-memory tensor dataset.

    Owned-tensor (torch.tensor → deep copy) construction is the
    Blackwell-torch-dataloader-bug workaround; see
    docs/decisions/0001-per-trial-subprocess-isolation.md.
    """

    def __init__(self, hero_ids: np.ndarray, player_feats: np.ndarray, y: np.ndarray):
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
    """Load + subsample. Returns (train_ds, val_ds, meta_dict).

    When smoke=True, prefer `{train,val}_smoke.parquet` if present in
    source_dir (smoke-build outputs); otherwise fall back to the full
    parquet and subsample heavily.
    """
    if smoke and (source_dir / "train_smoke.parquet").exists():
        train_path = source_dir / "train_smoke.parquet"
        val_path = source_dir / "val_smoke.parquet"
    else:
        train_path = source_dir / "train.parquet"
        val_path = source_dir / "val.parquet"

    print(f"Reading clean parquet: {train_path}")
    train_tbl = pq.read_table(train_path)
    print(f"  train rows: {train_tbl.num_rows:,}")
    print(f"Reading clean parquet: {val_path}")
    val_tbl = pq.read_table(val_path) if val_path.exists() else None
    if val_tbl is None or val_tbl.num_rows == 0:
        if smoke:
            # Smoke fallback: carve tail of train as pseudo-val (pipeline test only).
            n_t = train_tbl.num_rows
            n_v = max(1000, n_t // 10)
            print(f"  SMOKE: val parquet missing/empty; carving tail {n_v:,} rows off train.")
            val_tbl = train_tbl.slice(n_t - n_v, n_v)
            train_tbl = train_tbl.slice(0, n_t - n_v)
        else:
            raise SystemExit(f"REFUSED: val parquet at {val_path} is missing/empty.")
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
        train_pick = rng.choice(train_tbl_sub.num_rows,
                                size=min(smoke_n_train, train_tbl_sub.num_rows),
                                replace=False)
        val_pick = rng.choice(val_tbl.num_rows,
                              size=min(smoke_n_val, val_tbl.num_rows),
                              replace=False)
        train_tbl_sub = train_tbl_sub.take(train_pick)
        val_tbl = val_tbl.take(val_pick)

    h_tr, pf_tr, y_tr = load_arrays(train_tbl_sub, feat_names)
    h_va, pf_va, y_va = load_arrays(val_tbl, feat_names)

    train_ds = DraftPlusFeaturesDataset(h_tr, pf_tr, y_tr)
    val_ds = DraftPlusFeaturesDataset(h_va, pf_va, y_va)

    n_train_pre = int(train_tbl.num_rows)
    n_train_post = int(train_tbl_sub.num_rows)
    n_val = int(val_tbl.num_rows)
    radiant_base_rate_train_full = float(y_train_full.mean())
    radiant_base_rate_train_subsampled = float(y_tr.mean())
    radiant_base_rate_val = float(y_va.mean())

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
