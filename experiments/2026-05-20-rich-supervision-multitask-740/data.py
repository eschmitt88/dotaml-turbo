"""Data loading for rich-supervision-multitask-740.

Forked from experiments/2026-05-19-upstream-data-cleanup-740/data.py.

Differences:

1. Joins the clean parquet (hero_ids + 8 player features + radiant_win) with
   the rich-cols sidecar at data/snapshots/.../processed/rich_cols/ on match_id
   to produce per-match training targets for the multi-task heads:
       - duration -> duration_bucket (int via np.digitize against vocab edges)
       - per-slot items list<int32> -> stored as variable-length int arrays
         in the dataset; multi-hot tensor materialized lazily per __getitem__
         (item-vocab can be ~150-200 ids; per-row tensor ~10*200 bool = 2 KB)
       - per-slot aux (kills, gpm, hero_damage) -> standardized to train
         mean/std (per-slot mean is over all 10 slots; the model is symmetric
         over slots within a team only via the team_embed, so a global aux
         mean/std is the right reference scale)

2. Item-vocab lookup at load time: drop items not in vocab (route to RARE
   bucket = idx 1 -- but we don't include RARE in the multi-hot target, since
   it's not a meaningful learning signal; we simply drop unknown items). The
   PAD slot (idx 0) is also dropped from targets.

3. win_only_sanity callers ask for `multitask=False`, which skips the sidecar
   join and matches the cleanup-740 contract exactly (modulo the match-id
   intersection with sidecar in case any are missing -- expected
   `<= 0` missing in normal builds; smoke skips this).

4. HCE date assertion stays.
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

# Aux target field names in the rich-cols sidecar (per slot: p{i}_<name>).
DEFAULT_AUX_TARGETS = ["kills", "gpm", "hero_damage"]
N_PLAYERS = 10


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


def load_arrays(table, feat_names: list[str]) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """Return (match_ids[N] int64, hero_ids[N,10] int64, player_feats[N,10,F] f32, y[N] f32)."""
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

    # Hard assertion -- clean-parquet contract.
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


class DraftPlusFeaturesDataset(Dataset):
    """Win-only dataset (matches cleanup-740 contract). Yields (hero_ids, player_feats, y)."""

    def __init__(self, hero_ids: np.ndarray, player_feats: np.ndarray, y: np.ndarray):
        self.hero_ids = torch.tensor(hero_ids, dtype=torch.long).contiguous()
        self.player_feats = torch.tensor(player_feats, dtype=torch.float32).contiguous()
        self.y = torch.tensor(y, dtype=torch.float32).contiguous()

    def __len__(self) -> int:
        return self.hero_ids.size(0)

    def __getitem__(self, idx: int):
        return self.hero_ids[idx], self.player_feats[idx], self.y[idx]


class MultitaskDataset(Dataset):
    """Multi-task dataset. Yields:

      hero_ids:    [10] long
      player_feats:[10, F] f32
      y_win:       scalar f32
      y_dur:       scalar long (bucket idx)
      y_item:      [10, item_vocab_size] f32 multi-hot (built lazily here)
      y_aux:       [10, n_aux] f32 standardized

    Item targets are stored on the dataset as variable-length int arrays
    (one [10][?] per match) and materialized to multi-hot per __getitem__.
    Aux targets are pre-standardized at construction time.
    """

    def __init__(self, hero_ids: np.ndarray, player_feats: np.ndarray,
                 y_win: np.ndarray, y_dur: np.ndarray,
                 items_per_slot: list[list[list[int]]],  # [N][10][?]
                 aux_std: np.ndarray,                     # [N, 10, n_aux] f32
                 item_vocab_size: int):
        self.hero_ids = torch.tensor(hero_ids, dtype=torch.long).contiguous()
        self.player_feats = torch.tensor(player_feats, dtype=torch.float32).contiguous()
        self.y_win = torch.tensor(y_win, dtype=torch.float32).contiguous()
        self.y_dur = torch.tensor(y_dur, dtype=torch.long).contiguous()
        self.items_per_slot = items_per_slot
        self.aux_std = torch.tensor(aux_std, dtype=torch.float32).contiguous()
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
        return (self.hero_ids[idx], self.player_feats[idx], self.y_win[idx],
                self.y_dur[idx], items, self.aux_std[idx])


def _read_clean_parquet_defensive(path: Path):
    """Read the clean per-player-features parquet defensively.

    Defense against silent on-disk byte-flip corruption observed on this
    hardware: clean train.parquet had row-group 6 of column
    `p9_days_since_last_log1p` become unreadable (`OSError: Unexpected end
    of stream`) despite md5 + mtime + file size matching the original build.
    All other 114 columns read fine; all other row groups of the affected
    column also read fine.

    Strategy:
      1. Try the fast path: `pq.read_table(path)`. If it works, return.
      2. On failure, read all columns EXCEPT the corrupted one in one shot
         (fast). For the corrupted column, read row-group-by-row-group;
         for any failed row group, fill with the per-column median computed
         from successfully-read row groups.
      3. If MULTIPLE columns are unreadable (severe corruption), raise.

    The defensive path adds ~1-2 s for a clean file; for a corrupted file
    it recovers gracefully. Logged so the substitution is visible.
    """
    try:
        tbl = pq.read_table(path)
        return tbl
    except OSError as e:
        print(f"  WARN: pq.read_table({path}) failed: {e}")
        print(f"  WARN: switching to defensive per-column-per-row-group read.")

    pf = pq.ParquetFile(path)
    all_cols = list(pf.schema_arrow.names)

    # Find which columns fail. Read each column standalone (one-shot).
    bad_cols = []
    for c in all_cols:
        try:
            pq.read_table(path, columns=[c])
        except OSError:
            bad_cols.append(c)
    if not bad_cols:
        # Sometimes the full read fails but per-col reads all succeed.
        # Try concatenating per-column results.
        print(f"  defensive: no single-column failures; reading {len(all_cols)} cols individually.")
        tables = [pq.read_table(path, columns=[c]) for c in all_cols]
        out = tables[0]
        for t in tables[1:]:
            out = out.append_column(t.column_names[0], t.column(0))
        return out
    if len(bad_cols) > 1:
        raise OSError(f"clean parquet at {path} has {len(bad_cols)} corrupted columns "
                      f"({bad_cols}); too many to recover. Rebuild required.")

    bad_col = bad_cols[0]
    print(f"  defensive: ONE corrupted column '{bad_col}' — recovering via per-row-group + median fill.")

    # Read all good cols in one shot (fast).
    good_cols = [c for c in all_cols if c != bad_col]
    out = pq.read_table(path, columns=good_cols)

    # Read the bad col per row group, accumulate good chunks; mark bad RGs.
    good_chunks = []
    bad_rg_indices = []
    bad_rg_lengths = []
    for rg_idx in range(pf.metadata.num_row_groups):
        try:
            t = pf.read_row_group(rg_idx, columns=[bad_col])
            good_chunks.append(t.column(0).combine_chunks())
            bad_rg_lengths.append(t.num_rows)
        except OSError:
            # Use the metadata row count for this RG so we know the fill length.
            n = pf.metadata.row_group(rg_idx).num_rows
            bad_rg_indices.append((rg_idx, n))
            bad_rg_lengths.append(n)

    # Compute median from good chunks.
    if not good_chunks:
        raise OSError(f"clean parquet at {path}: ALL row groups of '{bad_col}' "
                      f"are corrupted; cannot derive fill value.")
    import numpy as _np
    good_concat = pa.concat_arrays(good_chunks)
    good_arr = good_concat.to_numpy(zero_copy_only=False)
    finite = good_arr[_np.isfinite(good_arr)]
    fill_val = float(_np.median(finite)) if finite.size else 0.0
    print(f"  defensive: '{bad_col}' rg {[i for i,_ in bad_rg_indices]} unreadable; "
          f"filling {sum(n for _,n in bad_rg_indices):,} rows with median={fill_val:.4f}")

    # Reconstruct the full column in original row-group order.
    # Walk RGs in order; for each, either take the next good_chunk or generate a fill chunk.
    out_chunks = []
    good_iter = iter(good_chunks)
    bad_set = {i for i, _ in bad_rg_indices}
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_len = pf.metadata.row_group(rg_idx).num_rows
        if rg_idx in bad_set:
            arr = pa.array(_np.full(rg_len, fill_val, dtype=_np.float32), type=pa.float32())
            out_chunks.append(arr)
        else:
            out_chunks.append(next(good_iter))
    full_col = pa.chunked_array(out_chunks)

    # Find original position of bad_col and insert it back.
    bad_col_idx = all_cols.index(bad_col)
    out = out.add_column(bad_col_idx, bad_col, full_col)
    assert out.num_rows == pf.metadata.num_rows, (
        f"defensive read length mismatch: got {out.num_rows}, expected {pf.metadata.num_rows}"
    )
    return out


def _read_sidecar_tbl(path: Path, mid_set: set[int]):
    """Read entire rich-cols sidecar, filter rows to mid_set.

    Defense against the transient PyArrow buffer-fill anomaly observed in
    cleanup-740 (and reproduced as the multitask_all attempt-1 ArrowInvalid:
    "Column 11 named p9_items expected length 13018393 but got length
    13018391"). The sidecar parquet itself is internally consistent
    (metadata.num_rows == per-row-group sum == column lengths when read via
    row-groups); the bug is in pf.read(columns=cols) returning short columns
    intermittently. We:

      1. Try pf.read(columns=cols). If all column lengths equal
         metadata.num_rows, accept.
      2. Otherwise retry once (transient anomaly often clears on a fresh
         reader handle).
      3. If still bad, fall back to row-group-by-row-group reads and
         pa.concat_tables() — slower but guaranteed-consistent because
         row-group reads use the parquet file's own row counts as the
         source of truth.
    """
    cols = ["match_id", "duration"]
    for p in range(10):
        cols.append(f"p{p}_items")
    for p in range(10):
        for name in DEFAULT_AUX_TARGETS:
            cols.append(f"p{p}_{name}")

    def _try_pf_read():
        pf = pq.ParquetFile(path)
        expect = pf.metadata.num_rows
        try:
            tbl = pf.read(columns=cols)
        except OSError as e:
            return None, expect, [("__oserror__", str(e))]
        # Validate every column's length matches metadata.num_rows.
        bad = []
        for c in cols:
            n = len(tbl.column(c))
            if n != expect:
                bad.append((c, n))
        return tbl, expect, bad

    tbl, expect, bad = _try_pf_read()
    if bad:
        print(f"    WARN: pf.read failed/short on {path.name}; expected {expect}, got {bad[:3]}... retrying once.")
        if tbl is not None:
            del tbl
        gc.collect()
        tbl, expect, bad = _try_pf_read()
    if bad:
        print(f"    WARN: retry still bad; falling back to per-row-group + per-column defensive read.")
        if tbl is not None:
            del tbl
        gc.collect()
        pf = pq.ParquetFile(path)

        # Read each requested column independently, identify bad ones.
        bad_cols = []
        for c in cols:
            try:
                pq.read_table(path, columns=[c])
            except OSError:
                bad_cols.append(c)
        if bad_cols:
            print(f"    defensive sidecar: bad columns = {bad_cols}")

        # Read good cols in one shot (fast).
        good_cols = [c for c in cols if c not in bad_cols]
        if good_cols:
            try:
                good_tbl = pq.read_table(path, columns=good_cols)
            except OSError:
                # If even good cols can't be read together, do per-col reads + assemble.
                good_tbls = [pq.read_table(path, columns=[c]) for c in good_cols]
                good_tbl = good_tbls[0]
                for t in good_tbls[1:]:
                    good_tbl = good_tbl.append_column(t.column_names[0], t.column(0))
        else:
            raise SystemExit(f"REFUSED: sidecar {path.name} has no readable columns at all.")

        # For each bad column, read RG by RG; fill bad RGs with sensible defaults.
        for bc in bad_cols:
            sch_field = pf.schema_arrow.field(bc)
            is_list = pa.types.is_list(sch_field.type) or pa.types.is_large_list(sch_field.type)
            good_chunks = []
            bad_rg_indices = []
            for rg_i in range(pf.metadata.num_row_groups):
                try:
                    rg_tbl = pf.read_row_group(rg_i, columns=[bc])
                    good_chunks.append(rg_tbl.column(0).combine_chunks())
                except OSError:
                    bad_rg_indices.append(rg_i)
            # Compute fill value from good chunks.
            if is_list:
                fill_val = []  # empty list per row
                if good_chunks:
                    fill_type = good_chunks[0].type
                else:
                    fill_type = sch_field.type
            else:
                # Numeric: median over good chunks.
                if not good_chunks:
                    raise SystemExit(f"REFUSED: bad col {bc} of {path.name} has no readable row groups.")
                concat = pa.concat_arrays(good_chunks)
                arr_np = concat.to_numpy(zero_copy_only=False)
                finite = arr_np[np.isfinite(arr_np)] if arr_np.dtype.kind == 'f' else arr_np
                fill_val = float(np.median(finite)) if len(finite) else 0.0
                fill_type = sch_field.type

            # Reassemble in RG order.
            out_chunks = []
            good_iter = iter(good_chunks)
            for rg_i in range(pf.metadata.num_row_groups):
                rg_len = pf.metadata.row_group(rg_i).num_rows
                if rg_i in bad_rg_indices:
                    if is_list:
                        # Build a ListArray of empty lists.
                        offsets = np.zeros(rg_len + 1, dtype=np.int32)
                        # ListArray.from_arrays(offsets, values)
                        values = pa.array([], type=fill_type.value_type if hasattr(fill_type, 'value_type') else pa.int64())
                        rg_arr = pa.ListArray.from_arrays(pa.array(offsets), values)
                    else:
                        rg_arr = pa.array(np.full(rg_len, fill_val, dtype=arr_np.dtype), type=fill_type)
                    out_chunks.append(rg_arr)
                else:
                    out_chunks.append(next(good_iter))
            full_col = pa.chunked_array(out_chunks)
            print(f"    defensive sidecar: '{bc}' rg {bad_rg_indices} unreadable; "
                  f"filled {sum(pf.metadata.row_group(i).num_rows for i in bad_rg_indices):,} rows with "
                  f"{'empty list' if is_list else f'median={fill_val:.3f}'}")
            # Insert column at its original position.
            bc_idx = cols.index(bc)
            good_tbl = good_tbl.add_column(bc_idx, bc, full_col)

        tbl = good_tbl
        # Final length check.
        for c in cols:
            n = len(tbl.column(c))
            if n != expect:
                raise SystemExit(
                    f"REFUSED: sidecar {path.name} column {c} length {n} != metadata {expect} "
                    f"even after defensive read."
                )

    mids = tbl.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
    keep_mask = np.fromiter((int(m) in mid_set for m in mids),
                            dtype=bool, count=len(mids))
    if keep_mask.sum() < len(mids):
        keep_idx = np.where(keep_mask)[0]
        tbl = tbl.take(keep_idx)
    return tbl


def _build_multitask_targets(
    clean_mids: np.ndarray,
    hero_ids: np.ndarray,
    player_feats: np.ndarray,
    y_win: np.ndarray,
    sidecar_path: Path,
    vocab: dict[str, int],
    duration_edges: list[float],
    aux_targets: list[str],
    aux_mean_std: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            list, np.ndarray, np.ndarray, np.ndarray]:
    """Returns aligned arrays:
      (hero_ids_kept, player_feats_kept, y_win_kept, y_dur_kept,
       items_per_slot, aux_std, aux_mean, aux_std_scale)
    Drops clean-parquet rows missing from the sidecar.
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

    duration = side_tbl.column("duration").to_numpy(zero_copy_only=False).astype(np.int32)
    duration_kept = duration[keep_idx_side]
    edges_arr = np.asarray(duration_edges, dtype=np.float64)
    y_dur = np.digitize(duration_kept.astype(np.float64), edges_arr).astype(np.int64)

    # Vectorized vocab mapping. Build a flat lookup array indexed by raw item ID.
    # vocab JSON keys are stringified ints (per build_item_vocab.py); skip non-int keys
    # defensively (PAD/RARE sentinel keys, if any).
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
    vocab_arr = np.full(max_item_id + 2, -1, dtype=np.int32)  # -1 = OOV (will be dropped)
    for kid, vid in vocab_int_pairs:
        vocab_arr[kid] = vid

    n_side = side_tbl.num_rows
    items_by_slot_full: list[list[list[int]]] = [None] * 10  # type: ignore[list-item]
    for p in range(10):
        col_chunked = side_tbl.column(f"p{p}_items")
        # Combine chunks to a single ListArray for clean offsets/values access.
        col = col_chunked.combine_chunks()
        offsets = col.offsets.to_numpy()           # shape (n_side+1,)
        values = np.asarray(col.values.to_numpy())  # shape (total_items,), raw int item IDs
        # Clip lookups to in-range IDs; OOV (incl. negative or > max) → -1 sentinel.
        safe = (values >= 0) & (values <= max_item_id)
        mapped_flat = np.full(values.shape, -1, dtype=np.int32)
        if safe.any():
            mapped_flat[safe] = vocab_arr[values[safe]]
        per_row: list[list[int]] = [None] * n_side  # type: ignore[list-item]
        for i in range(n_side):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if s == e:
                per_row[i] = []
                continue
            row_vals = mapped_flat[s:e]
            row_vals = row_vals[row_vals >= 0]
            per_row[i] = row_vals.tolist()
        items_by_slot_full[p] = per_row

    # Reindex to keep_idx_side (subset of side rows that survived the clean-set join).
    items_per_slot: list[list[list[int]]] = []
    for side_i in keep_idx_side:
        si = int(side_i)
        items_per_slot.append([items_by_slot_full[p][si] for p in range(10)])
    print(f"    items vectorized in {time.time() - _t0:.1f}s")

    n_kept = int(keep_mask.sum())
    n_aux = len(aux_targets)
    aux_raw = np.empty((n_kept, 10, n_aux), dtype=np.float32)
    for p in range(10):
        for a_i, a_name in enumerate(aux_targets):
            col = side_tbl.column(f"p{p}_{a_name}").to_numpy(zero_copy_only=False)
            aux_raw[:, p, a_i] = col[keep_idx_side].astype(np.float32)

    if aux_mean_std is None:
        flat = aux_raw.reshape(-1, n_aux)
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
    else:
        mu, sd = aux_mean_std
    aux_std = ((aux_raw - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)

    return (hero_ids_kept, player_feats_kept, y_win_kept, y_dur,
            items_per_slot, aux_std, mu.astype(np.float64), sd.astype(np.float64))


def load_train_val(seed: int, n_target: int, feat_names: list[str],
                   source_dir: Path, splits: dict, smoke: bool = False,
                   smoke_n_train: int = 50_000, smoke_n_val: int = 5_000,
                   multitask: bool = False,
                   sidecar_dir: Path | None = None,
                   vocab_path: Path | None = None,
                   aux_targets: list[str] | None = None):
    """Load + subsample.

    If multitask=False, returns (DraftPlusFeaturesDataset_train, _val, meta)
    -- identical contract to cleanup-740's load_train_val.

    If multitask=True, returns (MultitaskDataset_train, _val, meta) with
    rich-cols sidecar joined on match_id; aux standardized to train-mean/std.
    """
    aux_targets = aux_targets or DEFAULT_AUX_TARGETS
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

    if not multitask:
        train_ds = DraftPlusFeaturesDataset(h_tr, pf_tr, y_tr)
        val_ds = DraftPlusFeaturesDataset(h_va, pf_va, y_va)
        del mids_tr, mids_va, h_tr, pf_tr, y_tr, h_va, pf_va, y_va
        gc.collect()
        meta = _meta_dict(n_train_pre, n_train_post, n_val,
                          n_target, seed, train_dr, val_dr,
                          radiant_base_rate_train_full,
                          radiant_base_rate_train_subsampled,
                          radiant_base_rate_val, smoke, feat_names,
                          multitask=False, n_aux=0, item_vocab_size=0)
        return train_ds, val_ds, meta

    # Multi-task path.
    if sidecar_dir is None or vocab_path is None:
        raise SystemExit("multitask=True requires sidecar_dir and vocab_path.")
    vocab_blob = json.loads(Path(vocab_path).read_text())
    vocab = vocab_blob["vocab"]
    duration_edges = vocab_blob["duration_bucket_edges"]
    item_vocab_size = int(vocab_blob["meta"]["vocab_size"])
    print(f"  vocab: item_vocab_size={item_vocab_size}, "
          f"duration_buckets={len(duration_edges) + 1}")

    if smoke and (sidecar_dir / "train_smoke.parquet").exists():
        side_train_path = sidecar_dir / "train_smoke.parquet"
        side_val_path = sidecar_dir / "val_smoke.parquet"
        # SMOKE: val sidecar may be empty/missing because smoke clean parquet
        # carved val from train tail; in that case route val join to the train
        # sidecar (mid set is a subset, so the join is well-defined).
        if not side_val_path.exists() or side_val_path.stat().st_size < 1024:
            print(f"  SMOKE: val sidecar {side_val_path.name} missing/empty; "
                  f"routing val join through {side_train_path.name}.")
            side_val_path = side_train_path
    else:
        side_train_path = sidecar_dir / "train.parquet"
        side_val_path = sidecar_dir / "val.parquet"
    print(f"  rich-cols sidecar: train={side_train_path.name}, val={side_val_path.name}")

    print("  joining train sidecar...")
    h_tr_k, pf_tr_k, y_win_tr, y_dur_tr, items_tr, aux_tr_std, mu, sd = \
        _build_multitask_targets(
            mids_tr, h_tr, pf_tr, y_tr, side_train_path, vocab, duration_edges,
            aux_targets, aux_mean_std=None)
    print("  joining val sidecar...")
    h_va_k, pf_va_k, y_win_va, y_dur_va, items_va, aux_va_std, _, _ = \
        _build_multitask_targets(
            mids_va, h_va, pf_va, y_va, side_val_path, vocab, duration_edges,
            aux_targets, aux_mean_std=(mu, sd))

    # SMOKE fallback: if val join yielded 0 rows (smoke val sidecar empty AND
    # train-sidecar routing failed because carved val mids weren't in the train
    # sidecar's mid set), carve a small val from the END of the joined train.
    if smoke and len(h_va_k) == 0 and len(h_tr_k) > 100:
        n_carve = min(smoke_n_val, max(50, len(h_tr_k) // 10))
        print(f"  SMOKE: val join was empty; carving tail {n_carve:,} rows off joined train.")
        h_va_k = h_tr_k[-n_carve:]
        pf_va_k = pf_tr_k[-n_carve:]
        y_win_va = y_win_tr[-n_carve:]
        y_dur_va = y_dur_tr[-n_carve:]
        items_va = items_tr[-n_carve:]
        aux_va_std = aux_tr_std[-n_carve:]
        h_tr_k = h_tr_k[:-n_carve]
        pf_tr_k = pf_tr_k[:-n_carve]
        y_win_tr = y_win_tr[:-n_carve]
        y_dur_tr = y_dur_tr[:-n_carve]
        items_tr = items_tr[:-n_carve]
        aux_tr_std = aux_tr_std[:-n_carve]

    train_ds = MultitaskDataset(h_tr_k, pf_tr_k, y_win_tr, y_dur_tr,
                                 items_tr, aux_tr_std, item_vocab_size)
    val_ds = MultitaskDataset(h_va_k, pf_va_k, y_win_va, y_dur_va,
                               items_va, aux_va_std, item_vocab_size)
    n_train_post = int(len(train_ds))
    n_val = int(len(val_ds))

    del mids_tr, mids_va, h_tr, pf_tr, y_tr, h_va, pf_va, y_va
    del h_tr_k, pf_tr_k, y_win_tr, y_dur_tr, items_tr, aux_tr_std
    del h_va_k, pf_va_k, y_win_va, y_dur_va, items_va, aux_va_std
    gc.collect()

    meta = _meta_dict(n_train_pre, n_train_post, n_val,
                      n_target, seed, train_dr, val_dr,
                      radiant_base_rate_train_full,
                      radiant_base_rate_train_subsampled,
                      radiant_base_rate_val, smoke, feat_names,
                      multitask=True, n_aux=len(aux_targets),
                      item_vocab_size=item_vocab_size)
    meta["aux_targets"] = list(aux_targets)
    meta["aux_train_mean"] = mu.tolist()
    meta["aux_train_std"] = sd.tolist()
    meta["duration_bucket_edges"] = list(duration_edges)
    meta["item_vocab_size"] = item_vocab_size
    return train_ds, val_ds, meta


def _meta_dict(n_train_pre, n_train_post, n_val, n_target, seed,
               train_dr, val_dr, br_tr_full, br_tr_sub, br_va,
               smoke, feat_names, *, multitask: bool, n_aux: int,
               item_vocab_size: int) -> dict:
    return {
        "n_train_pre_subsample": int(n_train_pre),
        "n_train_post_subsample": int(n_train_post),
        "n_val": int(n_val),
        "train_subset_size_target": int(n_target),
        "train_subset_seed": int(seed),
        "train_date_min": train_dr[0],
        "train_date_max": train_dr[1],
        "val_date_min": val_dr[0],
        "val_date_max": val_dr[1],
        "radiant_base_rate_train_full": br_tr_full,
        "radiant_base_rate_train_subsampled": br_tr_sub,
        "radiant_base_rate_val": br_va,
        "smoke": bool(smoke),
        "feat_names": list(feat_names),
        "n_player_feats": len(feat_names),
        "multitask": bool(multitask),
        "n_aux": int(n_aux),
        "item_vocab_size": int(item_vocab_size),
    }


__all__ = ["DraftPlusFeaturesDataset", "MultitaskDataset",
           "load_train_val", "stratified_subsample", "assert_no_test_dates",
           "load_arrays"]
