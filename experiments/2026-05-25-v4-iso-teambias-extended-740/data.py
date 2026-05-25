"""Data loading for v3-ablations-740.

Forked from experiments/2026-05-24-foundation-v3-740/data.py.

Changes vs v3:

1. Optional `account_sidecar_paths` argument plumbs per-match per-slot
   account_ids into the dataset for the A2 player-embedding ablation.
   When `account_sidecar_paths` is None (the A1 case), behavior is
   identical to v3. The returned dataset always carries an account_idx
   tensor, but when no sidecar is provided every position is set to
   anon_idx=0 (which routes to the shared anonymous embedding row in
   models.py). Sidecar join handles missing match_ids gracefully:
   unmatched mids also become all-anon.

2. Account_idx is reordered in lockstep with the canonical hero sort.

3. y_dur_bucket is now a primary loss target for the A1 ablation (CE
   8-bucket); v3's regression target y_dur_log is also returned. Train
   loop picks which to use based on ablation.

All other v3 invariants preserved: HCE date assertion, defensive parquet
readers, multi-patch _patch_id_from_dates, canonical hero sort.
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

DEFAULT_AUX_TARGETS = ["kills", "gpm", "hero_damage"]
N_PLAYERS = 10

ANON_IDS = {0, 4294967295}
ANON_IDX = 0
RARE_IDX_BASE = 1   # base for hash-bucket indices (kept distinct from per-id top-K rows).


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
    """Sort each team's 5 slots by hero_id ascending.

    Args:
      hero_ids: [N, 10] int -- positions 0..4 = radiant, 5..9 = dire.
      player_feats: [N, 10, F] float

    Returns:
      hero_sorted: [N, 10] int
      pf_sorted:   [N, 10, F] float
      perm:        [N, 10] int -- the permutation used (so callers can
                   reorder per-slot aux targets in lockstep).

    Permutation invariance check: the model only sees the canonical input.
    """
    if hero_ids.ndim != 2 or hero_ids.shape[1] != 10:
        raise ValueError(f"hero_ids must be [N, 10], got {hero_ids.shape}")
    N = hero_ids.shape[0]
    # argsort within each team independently.
    r_perm = np.argsort(hero_ids[:, :5], axis=1, kind="stable")               # [N, 5]
    d_perm = np.argsort(hero_ids[:, 5:], axis=1, kind="stable") + 5            # [N, 5] -- offset
    perm = np.concatenate([r_perm, d_perm], axis=1)                            # [N, 10]
    rows = np.arange(N)[:, None]
    hero_sorted = hero_ids[rows, perm]
    pf_sorted = player_feats[rows, perm]
    return hero_sorted, pf_sorted, perm


def reorder_per_slot(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder per-slot aux/item arrays in lockstep with the canonical sort.

    arr: [N, 10, ...] -- can be any trailing shape.
    perm: [N, 10] int permutation from canonical_sort_by_hero.
    """
    N = arr.shape[0]
    rows = np.arange(N)[:, None]
    return arr[rows, perm]


def reorder_items_per_slot(items_per_slot: list, perm: np.ndarray) -> list:
    """Reorder the items list-of-lists structure in lockstep with the sort.

    items_per_slot is a Python list of length N, each element a list of 10
    sub-lists (one per slot). perm: [N, 10].
    """
    out = []
    for i, slots in enumerate(items_per_slot):
        new_slots = [slots[int(perm[i, k])] for k in range(10)]
        out.append(new_slots)
    return out


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


class FoundationDataset(Dataset):
    """v3-ablations dataset. Yields:

      hero_ids:    [10] long  (canonical: sorted asc within each team)
      player_feats:[10, F] f32
      patch_id:    scalar long
      account_idx: [10] long  (vocab idx; 0=anon for A1 default)
      y_win:       scalar f32
      y_dur:       scalar f32   (log(duration_seconds + 1.0); A2 uses this)
      y_dur_bucket: scalar long (8-bucket idx; A1 uses this for CE)
      y_item:      [10, item_vocab_size] f32 multi-hot (materialized per __getitem__)
      y_kda:       [10] f32   (kills only, standardized)
      y_gpm:       [10] f32   (standardized)
      y_hd:        [10] f32   (standardized)
    """

    def __init__(self, hero_ids: np.ndarray, player_feats: np.ndarray,
                 patch_ids: np.ndarray, account_idx: np.ndarray,
                 y_win: np.ndarray, y_dur: np.ndarray, y_dur_bucket: np.ndarray,
                 items_per_slot: list,
                 y_kda: np.ndarray, y_gpm: np.ndarray, y_hd: np.ndarray,
                 item_vocab_size: int):
        self.hero_ids = torch.tensor(hero_ids, dtype=torch.long).contiguous()
        self.player_feats = torch.tensor(player_feats, dtype=torch.float32).contiguous()
        self.patch_ids = torch.tensor(patch_ids, dtype=torch.long).contiguous()
        self.account_idx = torch.tensor(account_idx, dtype=torch.long).contiguous()
        self.y_win = torch.tensor(y_win, dtype=torch.float32).contiguous()
        self.y_dur = torch.tensor(y_dur, dtype=torch.float32).contiguous()
        self.y_dur_bucket = torch.tensor(y_dur_bucket, dtype=torch.long).contiguous()
        self.items_per_slot = items_per_slot
        self.y_kda = torch.tensor(y_kda, dtype=torch.float32).contiguous()
        self.y_gpm = torch.tensor(y_gpm, dtype=torch.float32).contiguous()
        self.y_hd = torch.tensor(y_hd, dtype=torch.float32).contiguous()
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
                self.account_idx[idx],
                self.y_win[idx], self.y_dur[idx], self.y_dur_bucket[idx], items,
                self.y_kda[idx], self.y_gpm[idx], self.y_hd[idx])


def _load_account_sidecar(side_paths: list[Path], mid_set: set[int]) -> dict[int, list[int]]:
    """Load account_id sidecar(s) and return {match_id: [10 account_ids]}.

    Multi-path: caller may pass several sidecar parquets; rows are
    keyed by match_id, last-write-wins on overlap. Reads only rows whose
    match_id is in `mid_set` (filter before materialize).
    """
    out: dict[int, list[int]] = {}
    n_paths_read = 0
    for sp in side_paths:
        sp = Path(sp)
        if not sp.exists() or sp.stat().st_size == 0:
            print(f"    account sidecar missing/empty: {sp}, skipping")
            continue
        n_paths_read += 1
        try:
            tbl = pq.read_table(sp, columns=(["match_id"] + [f"p{p}_account_id"
                                                              for p in range(10)]))
        except OSError as e:
            print(f"    WARN: pq.read_table({sp.name}) failed: {e}; skipping")
            continue
        mids = tbl.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
        keep = np.fromiter((int(m) in mid_set for m in mids), dtype=bool, count=len(mids))
        if not keep.any():
            print(f"    sidecar {sp.name}: 0 rows match filter; skipping")
            continue
        keep_idx = np.where(keep)[0]
        cols = [tbl.column(f"p{p}_account_id").to_numpy(zero_copy_only=False).astype(np.int64)
                for p in range(10)]
        for ii in keep_idx:
            mid = int(mids[ii])
            out[mid] = [int(cols[p][ii]) for p in range(10)]
        print(f"    sidecar {sp.name}: kept {int(keep.sum()):,}/{len(mids):,}")
    print(f"    total sidecar mids loaded: {len(out):,} from {n_paths_read} path(s)")
    return out


def _build_account_idx(account_id_matrix: np.ndarray, vocab: dict[int, int],
                       n_hash_buckets: int = 0,
                       hash_base_idx: int = 0,
                       anon_idx: int = 0) -> tuple[np.ndarray, dict]:
    """Map account_id[N, 10] int64 -> account_idx[N, 10] long via vocab.

    - Anonymous account_ids ({0, 4294967295}) -> anon_idx.
    - Vocab hits -> vocab[id].
    - Vocab misses (non-anon, non-frequent) ->
        if n_hash_buckets > 0: hash_base_idx + (id % n_hash_buckets).
        else:                  anon_idx (degrade to anon).

    Returns (account_idx, stats_dict).
    """
    if account_id_matrix.ndim != 2 or account_id_matrix.shape[1] != 10:
        raise ValueError(f"account_id_matrix must be [N, 10], got {account_id_matrix.shape}")
    n = account_id_matrix.shape[0]
    out = np.full((n, 10), anon_idx, dtype=np.int64)
    anon_arr = np.array(list(ANON_IDS), dtype=np.int64)
    n_anon = n_hit = n_rare = 0
    for p in range(10):
        col = account_id_matrix[:, p]
        anon_mask = np.isin(col, anon_arr)
        # vocab lookup row-by-row (col is small per matrix; total N*10 iters).
        col_out = np.full(n, anon_idx, dtype=np.int64)
        for i in range(n):
            if anon_mask[i]:
                col_out[i] = anon_idx
                n_anon += 1
                continue
            v = vocab.get(int(col[i]))
            if v is not None:
                col_out[i] = int(v)
                n_hit += 1
            else:
                if n_hash_buckets > 0:
                    col_out[i] = int(hash_base_idx + (int(col[i]) % n_hash_buckets))
                    n_rare += 1
                else:
                    col_out[i] = anon_idx
                    n_anon += 1
        out[:, p] = col_out
    total = n * 10
    stats = {
        "n_total_slots":   int(total),
        "n_anon_slots":    int(n_anon),
        "n_topK_hits":     int(n_hit),
        "n_rare_hash_slots": int(n_rare),
        "frac_anon":       float(n_anon / max(total, 1)),
        "frac_topK_hits":  float(n_hit / max(total, 1)),
        "frac_rare_hash":  float(n_rare / max(total, 1)),
    }
    return out, stats


def _read_clean_parquet_defensive(path: Path):
    """Read clean parquet defensively (per-row-group + median fill on one bad col).
    See multitask-740/data.py for the corruption-defense rationale (now hardware-fixed
    on JEDEC but the defense is harmless and free on healthy files).
    """
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
    if len(bad_cols) > 1:
        raise OSError(f"clean parquet {path} has {len(bad_cols)} corrupted columns; rebuild required.")
    bad_col = bad_cols[0]
    good_cols = [c for c in all_cols if c != bad_col]
    out = pq.read_table(path, columns=good_cols)
    good_chunks = []
    bad_rg_indices = []
    for rg_idx in range(pf.metadata.num_row_groups):
        try:
            t = pf.read_row_group(rg_idx, columns=[bad_col])
            good_chunks.append(t.column(0).combine_chunks())
        except OSError:
            n_rg = pf.metadata.row_group(rg_idx).num_rows
            bad_rg_indices.append((rg_idx, n_rg))
    if not good_chunks:
        raise OSError(f"clean parquet {path}: '{bad_col}' fully unreadable.")
    good_concat = pa.concat_arrays(good_chunks)
    good_arr = good_concat.to_numpy(zero_copy_only=False)
    finite = good_arr[np.isfinite(good_arr)]
    fill_val = float(np.median(finite)) if finite.size else 0.0
    out_chunks = []
    good_iter = iter(good_chunks)
    bad_set = {i for i, _ in bad_rg_indices}
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_len = pf.metadata.row_group(rg_idx).num_rows
        if rg_idx in bad_set:
            arr = pa.array(np.full(rg_len, fill_val, dtype=np.float32), type=pa.float32())
            out_chunks.append(arr)
        else:
            out_chunks.append(next(good_iter))
    full_col = pa.chunked_array(out_chunks)
    bad_col_idx = all_cols.index(bad_col)
    out = out.add_column(bad_col_idx, bad_col, full_col)
    return out


def _read_sidecar_tbl(path: Path, mid_set: set[int]):
    """Read entire rich-cols sidecar; filter to mid_set. Defensive vs PyArrow buffer-fill anomaly."""
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
        bad = [(c, len(tbl.column(c))) for c in cols if len(tbl.column(c)) != expect]
        return tbl, expect, bad

    tbl, expect, bad = _try_pf_read()
    if bad:
        print(f"    WARN: pf.read short on {path.name}; retrying once.")
        if tbl is not None:
            del tbl
        gc.collect()
        tbl, expect, bad = _try_pf_read()
    if bad:
        raise SystemExit(f"REFUSED: sidecar {path.name} short columns after retry: {bad[:3]}")

    mids = tbl.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
    keep_mask = np.fromiter((int(m) in mid_set for m in mids), dtype=bool, count=len(mids))
    if keep_mask.sum() < len(mids):
        keep_idx = np.where(keep_mask)[0]
        tbl = tbl.take(keep_idx)
    return tbl


def _build_foundation_targets(
    clean_mids: np.ndarray,
    hero_ids: np.ndarray,
    player_feats: np.ndarray,
    y_win: np.ndarray,
    sidecar_path: Path,
    vocab: dict[str, int],
    duration_edges: list[float],
    aux_targets: list[str],
    aux_mean_std: tuple[np.ndarray, np.ndarray] | None,
    canonical_sort: bool,
    account_sidecar_paths: list[Path] | None = None,
    player_vocab: dict[int, int] | None = None,
    n_hash_buckets: int = 0,
    hash_base_idx: int = 0,
):
    """Returns aligned arrays:
      (hero_ids_kept, player_feats_kept, y_win_kept,
       y_dur_log_kept, y_dur_bucket_kept,
       items_per_slot, y_kda, y_gpm, y_hd, aux_mean, aux_std)

    v3 change: y_dur is log(duration_seconds + 1.0) float32; y_dur_bucket
    (np.digitize on the same edges as multitask-740) is retained for the
    POST-HOC top1-acc diagnostic comparison vs prior anchors.

    Order of slots in all per-slot outputs is the CANONICAL order (after the
    per-team hero-id sort) if canonical_sort=True.
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
    # v3: log(seconds + 1.0) scalar regression target.
    y_dur_log = np.log1p(duration_kept.astype(np.float64)).astype(np.float32)
    # Post-hoc bucket label for diagnostic top1-acc comparison only (NOT a loss target).
    edges_arr = np.asarray(duration_edges, dtype=np.float64)
    y_dur_bucket = np.digitize(duration_kept.astype(np.float64), edges_arr).astype(np.int64)

    # Item vocab mapping (vectorized).
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
    aux_std_arr = ((aux_raw - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)

    # Split aux into the three per-task tensors. Default order: kills, gpm, hero_damage.
    name_to_idx = {n: i for i, n in enumerate(aux_targets)}
    y_kda = aux_std_arr[:, :, name_to_idx.get("kills", 0)]
    y_gpm = aux_std_arr[:, :, name_to_idx.get("gpm", 1)]
    y_hd = aux_std_arr[:, :, name_to_idx.get("hero_damage", 2)]

    # Build account_idx PRE-canonical-sort, then reorder in lockstep.
    # Default: all-anonymous (A1 case OR A2 with no sidecar coverage).
    mids_kept = clean_mids[keep_idx_clean]
    account_idx = np.zeros((mids_kept.shape[0], 10), dtype=np.int64)
    account_stats = {"n_total_slots": int(mids_kept.shape[0] * 10),
                     "n_anon_slots": int(mids_kept.shape[0] * 10),
                     "n_topK_hits": 0, "n_rare_hash_slots": 0,
                     "frac_anon": 1.0, "frac_topK_hits": 0.0, "frac_rare_hash": 0.0,
                     "frac_mids_covered": 0.0}
    if account_sidecar_paths and player_vocab is not None:
        mid_set = set(int(m) for m in mids_kept)
        side_map = _load_account_sidecar(account_sidecar_paths, mid_set)
        # Build per-row [10] account_id matrix from the map.
        acct_mat = np.zeros((mids_kept.shape[0], 10), dtype=np.int64)
        n_cov = 0
        for i, m in enumerate(mids_kept):
            row = side_map.get(int(m))
            if row is not None:
                acct_mat[i] = row
                n_cov += 1
        # acct_mat default-0 (anonymous) for unmatched rows.
        account_idx, stats = _build_account_idx(
            acct_mat, player_vocab,
            n_hash_buckets=n_hash_buckets, hash_base_idx=hash_base_idx,
            anon_idx=ANON_IDX,
        )
        stats["frac_mids_covered"] = float(n_cov / max(mids_kept.shape[0], 1))
        account_stats = stats

    if canonical_sort:
        hero_sorted, pf_sorted, perm = canonical_sort_by_hero(hero_ids_kept, player_feats_kept)
        hero_ids_kept = hero_sorted
        player_feats_kept = pf_sorted
        items_per_slot = reorder_items_per_slot(items_per_slot, perm)
        # Reorder per-slot scalar targets in lockstep.
        rows = np.arange(perm.shape[0])[:, None]
        y_kda = y_kda[rows, perm]
        y_gpm = y_gpm[rows, perm]
        y_hd = y_hd[rows, perm]
        # Reorder account_idx per-slot too (slot order must match per-feature order).
        account_idx = account_idx[rows, perm]
        # Sanity assertion: post-sort radiant heroes are non-decreasing.
        assert np.all(hero_ids_kept[:, :5][:, :-1] <= hero_ids_kept[:, :5][:, 1:]), \
            "post-sort radiant heroes not ascending"
        assert np.all(hero_ids_kept[:, 5:][:, :-1] <= hero_ids_kept[:, 5:][:, 1:]), \
            "post-sort dire heroes not ascending"

    return (hero_ids_kept, player_feats_kept, y_win_kept, y_dur_log, y_dur_bucket,
            items_per_slot, y_kda, y_gpm, y_hd, account_idx, account_stats,
            mu.astype(np.float64), sd.astype(np.float64))


def _patch_id_from_dates(dates: np.ndarray, default_patch_id: int = 1,
                          patch_edges: list[tuple[str, int]] | None = None) -> np.ndarray:
    """Derive patch_id from start_time_date.

    v3 change: supports multi-patch corpora via `patch_edges`, a sorted
    list of (start_date_iso, patch_id) tuples. For any row date >=
    edge[k].date and < edge[k+1].date (or end-of-list), assigns patch_id
    = edge[k].patch_id. Dates earlier than the first edge get
    `default_patch_id`.

    Empty `patch_edges` (the legacy 7.40-only case) -> all rows get
    `default_patch_id`. Hard-codes the known 7.40 boundary at 2025-12-16
    when patch_edges is not supplied.
    """
    n = len(dates)
    if patch_edges is None:
        # Built-in inference for known Turbo windows (Aug 2025 - Mar 2026):
        # < 2025-09-10  -> patch_id = 2 (7.39c era)
        # < 2025-12-16  -> patch_id = 3 (7.39d / 7.39e era; merged for vocab simplicity)
        # >= 2025-12-16 -> patch_id = 1 (7.40 -- keep "current patch" at id=1 for
        #                                 back-compat with the legacy single-patch
        #                                 token init)
        patch_edges = [("2025-08-01", 2), ("2025-09-10", 3), ("2025-12-16", 1)]
    out = np.full(n, default_patch_id, dtype=np.int64)
    # Vectorize via sorted edges + np.searchsorted on iso date strings.
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
                   aux_targets: list[str] | None = None,
                   canonical_sort: bool = True,
                   default_patch_id: int = 1,
                   account_sidecar_train_paths: list[Path] | None = None,
                   account_sidecar_val_paths: list[Path] | None = None,
                   player_vocab_path: Path | None = None,
                   n_hash_buckets: int = 0,
                   hash_base_idx: int = 0):
    """Load + subsample. Returns (FoundationDataset_train, _val, meta).

    Optional account-id sidecar wiring (A2 ablation):
      account_sidecar_{train,val}_paths: list of parquet paths with
        (match_id, p0_account_id..p9_account_id) columns.
      player_vocab_path: JSON {"<account_id>": idx, ...} + meta. account_id
        keys are int-castable; idx is the row in nn.Embedding.
      n_hash_buckets/hash_base_idx: vocab-miss non-anon ids -> hash bucket.
    When account_sidecar_*_paths is None or player_vocab_path is None, the
    dataset still has account_idx columns (all-anon=0), preserving the
    yield-tuple shape across A1 and A2.
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

    # Patch IDs derived BEFORE we drop the table.
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
        raise SystemExit("foundation-mvp requires sidecar_dir and vocab_path.")

    vocab_blob = json.loads(Path(vocab_path).read_text())
    vocab = vocab_blob["vocab"]
    duration_edges = vocab_blob["duration_bucket_edges"]
    item_vocab_size = int(vocab_blob["meta"]["vocab_size"])
    print(f"  vocab: item_vocab_size={item_vocab_size}, "
          f"duration_buckets={len(duration_edges) + 1}")

    if smoke and (sidecar_dir / "train_smoke.parquet").exists():
        side_train_path = sidecar_dir / "train_smoke.parquet"
        side_val_path = sidecar_dir / "val_smoke.parquet"
        if not side_val_path.exists() or side_val_path.stat().st_size < 1024:
            print(f"  SMOKE: val sidecar empty; routing val join through train sidecar.")
            side_val_path = side_train_path
    else:
        side_train_path = sidecar_dir / "train.parquet"
        side_val_path = sidecar_dir / "val.parquet"
    print(f"  rich-cols sidecar: train={side_train_path.name}, val={side_val_path.name}")

    # Optional player vocab for A2.
    player_vocab: dict[int, int] | None = None
    player_vocab_meta: dict | None = None
    if player_vocab_path is not None:
        pv_blob = json.loads(Path(player_vocab_path).read_text())
        player_vocab = {int(k): int(v) for k, v in pv_blob["vocab"].items()}
        player_vocab_meta = pv_blob.get("meta", {})
        print(f"  player vocab: top-K size={len(player_vocab):,}, "
              f"hash_buckets={n_hash_buckets}, hash_base_idx={hash_base_idx}")

    print("  joining train sidecar...")
    (h_tr_k, pf_tr_k, y_win_tr, y_dur_tr, y_durb_tr,
     items_tr, y_kda_tr, y_gpm_tr, y_hd_tr, acct_tr, acct_stats_tr,
     mu, sd) = \
        _build_foundation_targets(
            mids_tr, h_tr, pf_tr, y_tr, side_train_path, vocab, duration_edges,
            aux_targets, aux_mean_std=None, canonical_sort=canonical_sort,
            account_sidecar_paths=account_sidecar_train_paths,
            player_vocab=player_vocab,
            n_hash_buckets=n_hash_buckets, hash_base_idx=hash_base_idx)
    print("  joining val sidecar...")
    (h_va_k, pf_va_k, y_win_va, y_dur_va, y_durb_va,
     items_va, y_kda_va, y_gpm_va, y_hd_va, acct_va, acct_stats_va,
     _, _) = \
        _build_foundation_targets(
            mids_va, h_va, pf_va, y_va, side_val_path, vocab, duration_edges,
            aux_targets, aux_mean_std=(mu, sd), canonical_sort=canonical_sort,
            account_sidecar_paths=account_sidecar_val_paths,
            player_vocab=player_vocab,
            n_hash_buckets=n_hash_buckets, hash_base_idx=hash_base_idx)

    # Trim patch_ids to the join survivors. We need the post-join keep_idx_clean,
    # so re-derive the survivor mask via mid intersection here.
    def _trim_patch(mids_in, patch_in, side_path):
        clean_set = set(int(m) for m in mids_in)
        # We re-read side_tbl mid IDs cheaply for the trim.
        side_tbl_local = _read_sidecar_tbl(side_path, clean_set)
        side_mids_local = side_tbl_local.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
        side_idx_local = {int(m): i for i, m in enumerate(side_mids_local)}
        clean_to_side = np.array([side_idx_local.get(int(m), -1) for m in mids_in], dtype=np.int64)
        keep = clean_to_side >= 0
        return patch_in[keep]

    train_patch_ids_kept = _trim_patch(mids_tr, train_patch_ids, side_train_path)
    val_patch_ids_kept = _trim_patch(mids_va, val_patch_ids, side_val_path)
    # Sanity: same length as join survivors.
    assert len(train_patch_ids_kept) == len(h_tr_k), \
        f"patch_id length mismatch: {len(train_patch_ids_kept)} vs {len(h_tr_k)}"
    assert len(val_patch_ids_kept) == len(h_va_k), \
        f"patch_id length mismatch: {len(val_patch_ids_kept)} vs {len(h_va_k)}"

    # SMOKE fallback for val join empty.
    if smoke and len(h_va_k) == 0 and len(h_tr_k) > 100:
        n_carve = min(smoke_n_val, max(50, len(h_tr_k) // 10))
        print(f"  SMOKE: val join empty; carving tail {n_carve:,} off joined train.")
        h_va_k = h_tr_k[-n_carve:]; pf_va_k = pf_tr_k[-n_carve:]
        y_win_va = y_win_tr[-n_carve:]; y_dur_va = y_dur_tr[-n_carve:]
        y_durb_va = y_durb_tr[-n_carve:]
        items_va = items_tr[-n_carve:]
        y_kda_va = y_kda_tr[-n_carve:]; y_gpm_va = y_gpm_tr[-n_carve:]; y_hd_va = y_hd_tr[-n_carve:]
        acct_va = acct_tr[-n_carve:]
        val_patch_ids_kept = train_patch_ids_kept[-n_carve:]
        h_tr_k = h_tr_k[:-n_carve]; pf_tr_k = pf_tr_k[:-n_carve]
        y_win_tr = y_win_tr[:-n_carve]; y_dur_tr = y_dur_tr[:-n_carve]
        y_durb_tr = y_durb_tr[:-n_carve]
        items_tr = items_tr[:-n_carve]
        y_kda_tr = y_kda_tr[:-n_carve]; y_gpm_tr = y_gpm_tr[:-n_carve]; y_hd_tr = y_hd_tr[:-n_carve]
        acct_tr = acct_tr[:-n_carve]
        train_patch_ids_kept = train_patch_ids_kept[:-n_carve]

    train_ds = FoundationDataset(h_tr_k, pf_tr_k, train_patch_ids_kept, acct_tr,
                                  y_win_tr, y_dur_tr, y_durb_tr, items_tr,
                                  y_kda_tr, y_gpm_tr, y_hd_tr, item_vocab_size)
    val_ds = FoundationDataset(h_va_k, pf_va_k, val_patch_ids_kept, acct_va,
                                y_win_va, y_dur_va, y_durb_va, items_va,
                                y_kda_va, y_gpm_va, y_hd_va, item_vocab_size)
    n_train_post = int(len(train_ds))
    n_val = int(len(val_ds))

    del mids_tr, mids_va, h_tr, pf_tr, y_tr, h_va, pf_va, y_va
    gc.collect()

    # v3 diagnostic: patch_id distribution -- did extension actually pick up
    # multiple patches?
    train_pid_unique, train_pid_counts = np.unique(train_patch_ids_kept, return_counts=True)
    val_pid_unique, val_pid_counts = np.unique(val_patch_ids_kept, return_counts=True)
    train_patch_dist = {int(k): int(v) for k, v in zip(train_pid_unique, train_pid_counts)}
    val_patch_dist = {int(k): int(v) for k, v in zip(val_pid_unique, val_pid_counts)}

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
        "n_aux": len(aux_targets),
        "item_vocab_size": int(item_vocab_size),
        "aux_targets": list(aux_targets),
        "aux_train_mean": mu.tolist(),
        "aux_train_std": sd.tolist(),
        "duration_bucket_edges": list(duration_edges),
        "canonical_sort": bool(canonical_sort),
        "default_patch_id": int(default_patch_id),
        "train_patch_id_distribution": train_patch_dist,
        "val_patch_id_distribution": val_patch_dist,
        "account_idx_train_stats": acct_stats_tr,
        "account_idx_val_stats": acct_stats_va,
        "player_vocab_meta": player_vocab_meta,
        "n_hash_buckets": int(n_hash_buckets),
        "hash_base_idx": int(hash_base_idx),
    }
    return train_ds, val_ds, meta


__all__ = ["FoundationDataset", "load_train_val", "assert_no_test_dates",
           "canonical_sort_by_hero", "reorder_per_slot", "reorder_items_per_slot",
           "load_arrays", "stratified_subsample"]
