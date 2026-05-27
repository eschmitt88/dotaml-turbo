"""Data loading for v7-unified-masked-multitask-740.

Forked from experiments/2026-05-25-v4-iso-teambias-extended-740/data.py.

Key change vs v4: every per-slot quantity that the rich-cols sidecar
provides (items, kills, deaths, assists, gpm, hero_damage) plus the
per-match duration is now ALSO carried as an INPUT (not only as a
supervision target). The model's masking machinery (see mae.py and
models.py) decides which inputs are visible vs replaced with learned
mask tokens at each batch.

Compared to v4, the dataset yield is extended:

  yield order (12 tensors per row):
    hero_ids       [10]         long
    player_feats   [10, F]      f32
    patch_id       scalar       long
    account_idx    [10]         long  (kept for backward-compat; unused
                                       in v7 since player embeddings off)
    items          [10, V]      f32  multi-hot
    kills_raw      [10]         f32  raw count (NaN-safe)
    deaths_raw     [10]         f32
    assists_raw    [10]         f32
    gpm_raw        [10]         f32
    hd_raw         [10]         f32
    dur_log        scalar       f32  log(seconds + 1)
    y_win          scalar       f32

Aux standardization (kills/gpm/hd) used in v4 is dropped from the
yield: v7 carries the RAW counts as inputs (model handles scaling
internally via log1p projections) and as targets (SmoothL1 on raw
count, separate K/D/A heads). Duration is scalar log-seconds in both
input AND target slots.

Canonical hero sort is preserved end-to-end; ALL per-slot arrays are
reordered in lockstep.

HCE assertion preserved: refuses to walk test-window dates.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

# v7 always reads these from the sidecar; they are both inputs AND targets.
RICH_PER_SLOT_FIELDS = ["kills", "deaths", "assists", "gpm", "hero_damage"]
N_PLAYERS = 10

ANON_IDS = {0, 4294967295}
ANON_IDX = 0


def stratified_subsample(y: np.ndarray, n_target: int, seed: int) -> np.ndarray:
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
            f"REFUSED: {name} split contains test-window dates {bad[:3]}... -- HCE rule."
        )


def canonical_sort_by_hero(hero_ids: np.ndarray,
                            player_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hero_ids.ndim != 2 or hero_ids.shape[1] != 10:
        raise ValueError(f"hero_ids must be [N, 10], got {hero_ids.shape}")
    N = hero_ids.shape[0]
    r_perm = np.argsort(hero_ids[:, :5], axis=1, kind="stable")
    d_perm = np.argsort(hero_ids[:, 5:], axis=1, kind="stable") + 5
    perm = np.concatenate([r_perm, d_perm], axis=1)
    rows = np.arange(N)[:, None]
    hero_sorted = hero_ids[rows, perm]
    pf_sorted = player_feats[rows, perm]
    return hero_sorted, pf_sorted, perm


def reorder_per_slot(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    N = arr.shape[0]
    rows = np.arange(N)[:, None]
    return arr[rows, perm]


def reorder_items_per_slot(items_per_slot: list, perm: np.ndarray) -> list:
    out = []
    for i, slots in enumerate(items_per_slot):
        new_slots = [slots[int(perm[i, k])] for k in range(10)]
        out.append(new_slots)
    return out


def load_arrays(table, feat_names: list[str]) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    mids = table.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
    r_cols = [table.column(f"r{i}").to_numpy(zero_copy_only=False).astype(np.int64) for i in range(5)]
    d_cols = [table.column(f"d{i}").to_numpy(zero_copy_only=False).astype(np.int64) for i in range(5)]
    hero_ids = np.stack(r_cols + d_cols, axis=1)
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
            f"REFUSED: clean parquet violation -- {total_bad} cells outside bounds "
            f"({bad_by_feat}). Re-run build_features.py."
        )
    if not np.all(np.isfinite(player_feats)):
        raise SystemExit("REFUSED: player_feats contains non-finite values.")
    if float(np.abs(player_feats).max()) > 1e3:
        raise SystemExit(
            f"REFUSED: player_feats abs-max {np.abs(player_feats).max()} > 1e3."
        )
    return mids, hero_ids, player_feats, y


class FoundationDatasetV7(Dataset):
    """v7 dataset. Yields the 12-tensor tuple documented at module top."""

    def __init__(self, hero_ids: np.ndarray, player_feats: np.ndarray,
                 patch_ids: np.ndarray, account_idx: np.ndarray,
                 items_per_slot: list,
                 kills_raw: np.ndarray, deaths_raw: np.ndarray,
                 assists_raw: np.ndarray, gpm_raw: np.ndarray, hd_raw: np.ndarray,
                 dur_log: np.ndarray, y_win: np.ndarray,
                 item_vocab_size: int):
        self.hero_ids = torch.tensor(hero_ids, dtype=torch.long).contiguous()
        self.player_feats = torch.tensor(player_feats, dtype=torch.float32).contiguous()
        self.patch_ids = torch.tensor(patch_ids, dtype=torch.long).contiguous()
        self.account_idx = torch.tensor(account_idx, dtype=torch.long).contiguous()
        self.items_per_slot = items_per_slot
        self.kills_raw = torch.tensor(kills_raw, dtype=torch.float32).contiguous()
        self.deaths_raw = torch.tensor(deaths_raw, dtype=torch.float32).contiguous()
        self.assists_raw = torch.tensor(assists_raw, dtype=torch.float32).contiguous()
        self.gpm_raw = torch.tensor(gpm_raw, dtype=torch.float32).contiguous()
        self.hd_raw = torch.tensor(hd_raw, dtype=torch.float32).contiguous()
        self.dur_log = torch.tensor(dur_log, dtype=torch.float32).contiguous()
        self.y_win = torch.tensor(y_win, dtype=torch.float32).contiguous()
        self.item_vocab_size = int(item_vocab_size)

    def __len__(self) -> int:
        return self.hero_ids.size(0)

    def __getitem__(self, idx: int):
        items = torch.zeros(10, self.item_vocab_size, dtype=torch.float32)
        slots = self.items_per_slot[idx]
        for p_i in range(10):
            row = slots[p_i]
            if row:
                items[p_i, row] = 1.0
        return (self.hero_ids[idx], self.player_feats[idx], self.patch_ids[idx],
                self.account_idx[idx], items,
                self.kills_raw[idx], self.deaths_raw[idx], self.assists_raw[idx],
                self.gpm_raw[idx], self.hd_raw[idx],
                self.dur_log[idx], self.y_win[idx])


def _read_clean_parquet_defensive(path: Path):
    try:
        tbl = pq.read_table(path)
        return tbl
    except OSError as e:
        print(f"  WARN: pq.read_table({path}) failed: {e}; switching to defensive read.")
    pf = pq.ParquetFile(path)
    all_cols = list(pf.schema_arrow.names)
    bad_cols = []
    for c in all_cols:
        try:
            pq.read_table(path, columns=[c])
        except OSError:
            bad_cols.append(c)
    if not bad_cols:
        tables = [pq.read_table(path, columns=[c]) for c in all_cols]
        out = tables[0]
        for t in tables[1:]:
            out = out.append_column(t.column_names[0], t.column(0))
        return out
    raise OSError(f"clean parquet {path} corrupted columns: {bad_cols}")


def _read_sidecar_tbl(path: Path, mid_set: set[int]):
    cols = ["match_id", "duration"]
    for p in range(10):
        cols.append(f"p{p}_items")
    for p in range(10):
        for name in RICH_PER_SLOT_FIELDS:
            cols.append(f"p{p}_{name}")
    pf = pq.ParquetFile(path)
    expect = pf.metadata.num_rows
    tbl = pf.read(columns=cols)
    bad = [(c, len(tbl.column(c))) for c in cols if len(tbl.column(c)) != expect]
    if bad:
        print(f"    WARN: pf.read short on {path.name}; retrying once.")
        del tbl
        gc.collect()
        tbl = pf.read(columns=cols)
        bad = [(c, len(tbl.column(c))) for c in cols if len(tbl.column(c)) != expect]
        if bad:
            raise SystemExit(f"REFUSED: sidecar {path.name} short columns after retry: {bad[:3]}")
    mids = tbl.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
    keep_mask = np.fromiter((int(m) in mid_set for m in mids), dtype=bool, count=len(mids))
    if keep_mask.sum() < len(mids):
        keep_idx = np.where(keep_mask)[0]
        tbl = tbl.take(keep_idx)
    return tbl


def _build_foundation_targets_v7(
    clean_mids: np.ndarray,
    hero_ids: np.ndarray,
    player_feats: np.ndarray,
    y_win: np.ndarray,
    sidecar_path: Path,
    vocab: dict[str, int],
    canonical_sort: bool,
):
    """Returns aligned arrays:
      (hero_ids_kept, player_feats_kept, y_win_kept,
       items_per_slot, kills_raw, deaths_raw, assists_raw, gpm_raw, hd_raw,
       dur_log, account_idx_kept, mids_kept)
    """
    clean_set = set(int(m) for m in clean_mids)
    side_tbl = _read_sidecar_tbl(sidecar_path, clean_set)
    side_mids = side_tbl.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
    print(f"    sidecar read: {side_tbl.num_rows:,} rows (filtered to clean set)")
    side_idx: dict[int, int] = {int(m): i for i, m in enumerate(side_mids)}
    clean_to_side = np.array([side_idx.get(int(m), -1) for m in clean_mids], dtype=np.int64)
    keep_mask = clean_to_side >= 0
    n_missing = int((~keep_mask).sum())
    if n_missing > 0:
        print(f"    NOTE: dropping {n_missing:,} clean-parquet rows with no sidecar match.")
    keep_idx_clean = np.where(keep_mask)[0]
    keep_idx_side = clean_to_side[keep_mask]
    hero_ids_kept = hero_ids[keep_idx_clean]
    player_feats_kept = player_feats[keep_idx_clean]
    y_win_kept = y_win[keep_idx_clean]
    mids_kept = clean_mids[keep_idx_clean]

    duration = side_tbl.column("duration").to_numpy(zero_copy_only=False).astype(np.int32)
    duration_kept = duration[keep_idx_side]
    dur_log = np.log1p(np.clip(duration_kept, 0, None).astype(np.float64)).astype(np.float32)

    # Items vocab mapping (vectorized).
    _t0 = time.time()
    vocab_int_pairs: list[tuple[int, int]] = []
    for k, v in vocab.items():
        try:
            kid = int(k)
        except (ValueError, TypeError):
            continue
        vocab_int_pairs.append((kid, int(v)))
    if not vocab_int_pairs:
        raise RuntimeError("vocab has no integer-keyed entries")
    max_item_id = max(k for k, _ in vocab_int_pairs)
    vocab_arr = np.full(max_item_id + 2, -1, dtype=np.int32)
    for kid, vid in vocab_int_pairs:
        vocab_arr[kid] = vid
    n_side = side_tbl.num_rows
    items_by_slot_full: list = [None] * 10
    for p in range(10):
        col_chunked = side_tbl.column(f"p{p}_items")
        col = col_chunked.combine_chunks()
        offsets = col.offsets.to_numpy()
        values = np.asarray(col.values.to_numpy())
        safe = (values >= 0) & (values <= max_item_id)
        mapped_flat = np.full(values.shape, -1, dtype=np.int32)
        if safe.any():
            mapped_flat[safe] = vocab_arr[values[safe]]
        per_row: list = [None] * n_side
        for i in range(n_side):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if s == e:
                per_row[i] = []
                continue
            row_vals = mapped_flat[s:e]
            row_vals = row_vals[row_vals >= 0]
            per_row[i] = row_vals.tolist()
        items_by_slot_full[p] = per_row
    items_per_slot: list = []
    for side_i in keep_idx_side:
        si = int(side_i)
        items_per_slot.append([items_by_slot_full[p][si] for p in range(10)])
    print(f"    items vectorized in {time.time() - _t0:.1f}s")

    # Per-slot raw counts (NaN-safe; clip to feat_bounds-style sane range).
    n_kept = int(keep_mask.sum())
    raw_arrs: dict[str, np.ndarray] = {}
    for fname in RICH_PER_SLOT_FIELDS:
        arr = np.empty((n_kept, 10), dtype=np.float32)
        for p in range(10):
            col = side_tbl.column(f"p{p}_{fname}").to_numpy(zero_copy_only=False)
            arr[:, p] = col[keep_idx_side].astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip absurd values (parquet outliers).
        if fname in ("kills", "deaths", "assists"):
            arr = np.clip(arr, 0.0, 100.0)
        elif fname in ("gpm",):
            arr = np.clip(arr, 0.0, 5000.0)
        elif fname == "hero_damage":
            arr = np.clip(arr, 0.0, 500_000.0)
        raw_arrs[fname] = arr

    kills_raw = raw_arrs["kills"]
    deaths_raw = raw_arrs["deaths"]
    assists_raw = raw_arrs["assists"]
    gpm_raw = raw_arrs["gpm"]
    hd_raw = raw_arrs["hero_damage"]

    # account_idx: keep shape so downstream callers don't break; all anon.
    account_idx = np.zeros((n_kept, 10), dtype=np.int64)

    if canonical_sort:
        hero_sorted, pf_sorted, perm = canonical_sort_by_hero(hero_ids_kept, player_feats_kept)
        hero_ids_kept = hero_sorted
        player_feats_kept = pf_sorted
        items_per_slot = reorder_items_per_slot(items_per_slot, perm)
        rows = np.arange(perm.shape[0])[:, None]
        kills_raw = kills_raw[rows, perm]
        deaths_raw = deaths_raw[rows, perm]
        assists_raw = assists_raw[rows, perm]
        gpm_raw = gpm_raw[rows, perm]
        hd_raw = hd_raw[rows, perm]
        account_idx = account_idx[rows, perm]
        assert np.all(hero_ids_kept[:, :5][:, :-1] <= hero_ids_kept[:, :5][:, 1:])
        assert np.all(hero_ids_kept[:, 5:][:, :-1] <= hero_ids_kept[:, 5:][:, 1:])

    return (hero_ids_kept, player_feats_kept, y_win_kept,
            items_per_slot, kills_raw, deaths_raw, assists_raw, gpm_raw, hd_raw,
            dur_log, account_idx, mids_kept)


def _patch_id_from_dates(dates: np.ndarray, default_patch_id: int = 1) -> np.ndarray:
    n = len(dates)
    patch_edges = [("2025-08-01", 2), ("2025-09-10", 3), ("2025-12-16", 1)]
    out = np.full(n, default_patch_id, dtype=np.int64)
    edge_dates = np.array([e[0] for e in patch_edges], dtype="U10")
    edge_pids = np.array([e[1] for e in patch_edges], dtype=np.int64)
    str_dates = np.asarray(dates, dtype="U10")
    idx = np.searchsorted(edge_dates, str_dates, side="right") - 1
    valid = idx >= 0
    out[valid] = edge_pids[idx[valid]]
    return out


def load_train_val(seed: int, n_target: int, feat_names: list[str],
                   source_dir: Path, splits: dict, smoke: bool = False,
                   smoke_n_train: int = 50_000, smoke_n_val: int = 5_000,
                   sidecar_dir: Path | None = None,
                   vocab_path: Path | None = None,
                   canonical_sort: bool = True,
                   default_patch_id: int = 1):
    """Load train+val for v7. Returns (train_ds, val_ds, meta)."""
    if smoke and (source_dir / "train_smoke.parquet").exists():
        train_path = source_dir / "train_smoke.parquet"
        val_path = source_dir / "val_smoke.parquet"
    else:
        train_path = source_dir / "train.parquet"
        val_path = source_dir / "val.parquet"

    print(f"Reading clean parquet: {train_path}")
    train_tbl = _read_clean_parquet_defensive(train_path)
    print(f"  train rows: {train_tbl.num_rows:,}")
    print(f"Reading clean parquet: {val_path}")
    val_tbl = _read_clean_parquet_defensive(val_path) if val_path.exists() else None
    if val_tbl is None or val_tbl.num_rows == 0:
        if smoke:
            n_t = train_tbl.num_rows
            n_v = max(1000, n_t // 10)
            print(f"  SMOKE: val parquet missing/empty; carving tail {n_v:,} off train.")
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

    train_dates = train_tbl_sub.column("start_time_date").to_numpy(zero_copy_only=False)
    val_dates = val_tbl.column("start_time_date").to_numpy(zero_copy_only=False)
    train_patch_ids = _patch_id_from_dates(train_dates, default_patch_id=default_patch_id)
    val_patch_ids = _patch_id_from_dates(val_dates, default_patch_id=default_patch_id)

    mids_tr, h_tr, pf_tr, y_tr = load_arrays(train_tbl_sub, feat_names)
    mids_va, h_va, pf_va, y_va = load_arrays(val_tbl, feat_names)

    n_train_pre = int(train_tbl.num_rows)
    n_train_post = int(train_tbl_sub.num_rows)
    n_val = int(val_tbl.num_rows)
    radiant_base_rate_train_full = float(y_train_full.mean())
    radiant_base_rate_train_subsampled = float(y_tr.mean())
    radiant_base_rate_val = float(y_va.mean())

    del train_tbl, train_tbl_sub, val_tbl, y_train_full, sub_idx
    gc.collect()

    if sidecar_dir is None or vocab_path is None:
        raise SystemExit("v7 requires sidecar_dir and vocab_path.")

    vocab_blob = json.loads(Path(vocab_path).read_text())
    vocab = vocab_blob["vocab"]
    item_vocab_size = int(vocab_blob["meta"]["vocab_size"])
    print(f"  vocab: item_vocab_size={item_vocab_size}")

    if smoke and (sidecar_dir / "train_smoke.parquet").exists():
        side_train_path = sidecar_dir / "train_smoke.parquet"
        side_val_path = sidecar_dir / "val_smoke.parquet"
        if not side_val_path.exists() or side_val_path.stat().st_size < 1024:
            side_val_path = side_train_path
    else:
        side_train_path = sidecar_dir / "train.parquet"
        side_val_path = sidecar_dir / "val.parquet"
    print(f"  rich-cols sidecar: train={side_train_path.name}, val={side_val_path.name}")

    print("  joining train sidecar...")
    (h_tr_k, pf_tr_k, y_win_tr, items_tr,
     k_tr, d_tr, a_tr, g_tr, hd_tr, dur_tr, acct_tr, mids_tr_kept) = \
        _build_foundation_targets_v7(
            mids_tr, h_tr, pf_tr, y_tr, side_train_path, vocab, canonical_sort)
    print("  joining val sidecar...")
    (h_va_k, pf_va_k, y_win_va, items_va,
     k_va, d_va, a_va, g_va, hd_va, dur_va, acct_va, mids_va_kept) = \
        _build_foundation_targets_v7(
            mids_va, h_va, pf_va, y_va, side_val_path, vocab, canonical_sort)

    # Trim patch_ids to survivors.
    def _trim_patch(mids_in, patch_in, side_path):
        clean_set = set(int(m) for m in mids_in)
        side_tbl_local = _read_sidecar_tbl(side_path, clean_set)
        side_mids_local = side_tbl_local.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
        side_idx_local = {int(m): i for i, m in enumerate(side_mids_local)}
        clean_to_side = np.array([side_idx_local.get(int(m), -1) for m in mids_in], dtype=np.int64)
        keep = clean_to_side >= 0
        return patch_in[keep]

    train_patch_ids_kept = _trim_patch(mids_tr, train_patch_ids, side_train_path)
    val_patch_ids_kept = _trim_patch(mids_va, val_patch_ids, side_val_path)
    assert len(train_patch_ids_kept) == len(h_tr_k)
    assert len(val_patch_ids_kept) == len(h_va_k)

    if smoke and len(h_va_k) == 0 and len(h_tr_k) > 100:
        n_carve = min(smoke_n_val, max(50, len(h_tr_k) // 10))
        print(f"  SMOKE: val join empty; carving tail {n_carve:,} off joined train.")
        h_va_k = h_tr_k[-n_carve:]; pf_va_k = pf_tr_k[-n_carve:]
        y_win_va = y_win_tr[-n_carve:]; items_va = items_tr[-n_carve:]
        k_va = k_tr[-n_carve:]; d_va = d_tr[-n_carve:]; a_va = a_tr[-n_carve:]
        g_va = g_tr[-n_carve:]; hd_va = hd_tr[-n_carve:]
        dur_va = dur_tr[-n_carve:]; acct_va = acct_tr[-n_carve:]
        val_patch_ids_kept = train_patch_ids_kept[-n_carve:]
        h_tr_k = h_tr_k[:-n_carve]; pf_tr_k = pf_tr_k[:-n_carve]
        y_win_tr = y_win_tr[:-n_carve]; items_tr = items_tr[:-n_carve]
        k_tr = k_tr[:-n_carve]; d_tr = d_tr[:-n_carve]; a_tr = a_tr[:-n_carve]
        g_tr = g_tr[:-n_carve]; hd_tr = hd_tr[:-n_carve]
        dur_tr = dur_tr[:-n_carve]; acct_tr = acct_tr[:-n_carve]
        train_patch_ids_kept = train_patch_ids_kept[:-n_carve]

    train_ds = FoundationDatasetV7(h_tr_k, pf_tr_k, train_patch_ids_kept, acct_tr,
                                    items_tr, k_tr, d_tr, a_tr, g_tr, hd_tr,
                                    dur_tr, y_win_tr, item_vocab_size)
    val_ds = FoundationDatasetV7(h_va_k, pf_va_k, val_patch_ids_kept, acct_va,
                                  items_va, k_va, d_va, a_va, g_va, hd_va,
                                  dur_va, y_win_va, item_vocab_size)
    n_train_post = int(len(train_ds))
    n_val = int(len(val_ds))

    del mids_tr, mids_va, h_tr, pf_tr, y_tr, h_va, pf_va, y_va
    gc.collect()

    train_pid_unique, train_pid_counts = np.unique(train_patch_ids_kept, return_counts=True)
    val_pid_unique, val_pid_counts = np.unique(val_patch_ids_kept, return_counts=True)
    meta = {
        "n_train_pre_subsample": int(n_train_pre),
        "n_train_post_subsample": int(n_train_post),
        "n_val": int(n_val),
        "train_subset_size_target": int(n_target),
        "train_subset_seed": int(seed),
        "train_date_min": train_dr[0], "train_date_max": train_dr[1],
        "val_date_min": val_dr[0], "val_date_max": val_dr[1],
        "radiant_base_rate_train_full": radiant_base_rate_train_full,
        "radiant_base_rate_train_subsampled": radiant_base_rate_train_subsampled,
        "radiant_base_rate_val": radiant_base_rate_val,
        "smoke": bool(smoke),
        "feat_names": list(feat_names),
        "n_player_feats": len(feat_names),
        "item_vocab_size": int(item_vocab_size),
        "canonical_sort": bool(canonical_sort),
        "default_patch_id": int(default_patch_id),
        "train_patch_id_distribution": {int(k): int(v) for k, v in zip(train_pid_unique, train_pid_counts)},
        "val_patch_id_distribution": {int(k): int(v) for k, v in zip(val_pid_unique, val_pid_counts)},
        "rich_per_slot_fields": list(RICH_PER_SLOT_FIELDS),
    }
    return train_ds, val_ds, meta


__all__ = ["FoundationDatasetV7", "load_train_val", "assert_no_test_dates",
           "canonical_sort_by_hero", "reorder_per_slot", "reorder_items_per_slot",
           "load_arrays", "stratified_subsample", "RICH_PER_SLOT_FIELDS"]
