"""Data loading for player-embedding-prelim-740.

Forked from experiments/2026-05-19-upstream-data-cleanup-740/data.py with
two additions:

1. **Sidecar account_id join.** The clean parquet has no pX_account_id
   columns; this module loads the per-experiment account-id sidecar
   parquet (built by build_account_sidecar.py), joins it on match_id,
   and emits an `account_idx: int32[N, 10]` tensor of vocab indices
   alongside the existing hero_ids + player_feats + y.

2. **Vocab lookup.** A vocab loaded once at process start from
   `config.player_embedding.vocab_path` (built by build_vocab.py).
   account_id → idx via the vocab dict; anonymous account_ids
   (0, 4294967295) → 0, unknown account_ids → 1 (rare bucket).

HCE rule: only train.parquet and val.parquet from the clean parquet, and
the train/val account-id sidecars, are read. Test window is sealed.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
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

ANON_IDS = {0, 4294967295}


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


def load_vocab(vocab_path: Path) -> tuple[dict[int, int], dict]:
    """Load the vocab JSON; convert keys back to int. Returns (vocab, meta)."""
    raw = json.loads(vocab_path.read_text())
    vocab = {int(k): int(v) for k, v in raw["vocab"].items()}
    meta = raw.get("meta", {})
    return vocab, meta


def build_account_idx(account_ids_n10: np.ndarray, vocab: dict[int, int],
                      anon_idx: int = 0, rare_idx: int = 1) -> np.ndarray:
    """Map int64[N, 10] account_id matrix to int32 vocab indices.

    Anonymous IDs (0, 4294967295) -> anon_idx.
    Vocab hits                     -> vocab[id]  (>= 2 by construction).
    Vocab misses                   -> rare_idx.
    """
    n, ten = account_ids_n10.shape
    assert ten == 10
    out = np.full((n, ten), rare_idx, dtype=np.int32)
    # Per-slot lookup. dict.get on python ints is the cleanest path.
    for p in range(ten):
        col = account_ids_n10[:, p]
        col_py = col.tolist()
        anon_mask = np.isin(col, np.array(list(ANON_IDS), dtype=col.dtype))
        # Walk once; branch on anon.
        col_out = np.full(n, rare_idx, dtype=np.int32)
        for i in range(n):
            if anon_mask[i]:
                col_out[i] = anon_idx
            else:
                v = vocab.get(col_py[i])
                if v is not None:
                    col_out[i] = v
        out[:, p] = col_out
    return out


def join_sidecar_to_main(main_tbl, sidecar_tbl) -> np.ndarray:
    """Return account_ids[N, 10] int64 aligned to main_tbl's row order.

    Missing match_ids in sidecar -> all-zero (treated as 10 anonymous slots).
    """
    n = main_tbl.num_rows
    mids_main = main_tbl.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
    mids_side = sidecar_tbl.column("match_id").to_numpy(zero_copy_only=False).astype(np.int64)
    side_cols = [sidecar_tbl.column(f"p{p}_account_id").to_numpy(zero_copy_only=False).astype(np.int64)
                 for p in range(10)]
    mid_to_idx = {int(m): i for i, m in enumerate(mids_side)}
    out = np.zeros((n, 10), dtype=np.int64)
    n_missing = 0
    for i in range(n):
        idx = mid_to_idx.get(int(mids_main[i]), -1)
        if idx < 0:
            n_missing += 1
            continue
        for p in range(10):
            out[i, p] = side_cols[p][idx]
    if n_missing > 0:
        print(f"  WARN: {n_missing}/{n} match_ids missing from sidecar (-> all-anon slots)")
    return out


def load_arrays_and_acct(table, sidecar_table, feat_names: list[str],
                         vocab: dict[int, int], anon_idx: int, rare_idx: int):
    """Return (hero_ids[N,10] int64, player_feats[N,10,F] float32, y[N] float32,
              account_idx[N,10] int32, in_vocab_mask[N,10] bool).

    The sidecar_table is joined to `table` on match_id.
    """
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

    # Clean-parquet bounds assertion — same as parent.
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
    if not np.all(np.isfinite(player_feats)):
        raise SystemExit("REFUSED: player_feats contains non-finite values.")
    if float(np.abs(player_feats).max()) > 1e3:
        raise SystemExit(
            f"REFUSED: player_feats abs-max {np.abs(player_feats).max()} > 1e3."
        )

    # Sidecar join + vocab lookup.
    acct_mat = join_sidecar_to_main(table, sidecar_table)
    account_idx = build_account_idx(acct_mat, vocab, anon_idx, rare_idx)
    in_vocab_mask = account_idx >= 2

    return hero_ids, player_feats, y, account_idx, in_vocab_mask


class DraftPlusFeaturesPlusEmbedDataset(Dataset):
    """In-memory tensor dataset with account_idx tensor.

    Owned-tensor (torch.tensor → deep copy) construction is the
    Blackwell-torch-dataloader-bug workaround; see
    docs/decisions/0001-per-trial-subprocess-isolation.md.
    """

    def __init__(self, hero_ids: np.ndarray, player_feats: np.ndarray,
                 y: np.ndarray, account_idx: np.ndarray):
        self.hero_ids = torch.tensor(hero_ids, dtype=torch.long).contiguous()
        self.player_feats = torch.tensor(player_feats, dtype=torch.float32).contiguous()
        self.y = torch.tensor(y, dtype=torch.float32).contiguous()
        # int32 stored as long for embedding lookup safety.
        self.account_idx = torch.tensor(account_idx, dtype=torch.long).contiguous()

    def __len__(self) -> int:
        return self.hero_ids.size(0)

    def __getitem__(self, idx: int):
        return self.hero_ids[idx], self.player_feats[idx], self.account_idx[idx], self.y[idx]


def load_train_val(seed: int, n_target: int, feat_names: list[str],
                   source_dir: Path, sidecar_dir: Path,
                   sidecar_train_name: str, sidecar_val_name: str,
                   vocab: dict[int, int], anon_idx: int, rare_idx: int,
                   splits: dict, smoke: bool = False,
                   smoke_n_train: int = 50_000, smoke_n_val: int = 5_000):
    """Load + subsample. Returns (train_ds, val_ds, meta_dict)."""
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

    # Sidecar parquets.
    print(f"Reading sidecar: {sidecar_dir / sidecar_train_name}")
    sidecar_train = pq.read_table(sidecar_dir / sidecar_train_name)
    print(f"  sidecar train rows: {sidecar_train.num_rows:,}")
    print(f"Reading sidecar: {sidecar_dir / sidecar_val_name}")
    sidecar_val = pq.read_table(sidecar_dir / sidecar_val_name)
    print(f"  sidecar val   rows: {sidecar_val.num_rows:,}")

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

    h_tr, pf_tr, y_tr, ai_tr, inv_tr = load_arrays_and_acct(
        train_tbl_sub, sidecar_train, feat_names, vocab, anon_idx, rare_idx
    )
    h_va, pf_va, y_va, ai_va, inv_va = load_arrays_and_acct(
        val_tbl, sidecar_val, feat_names, vocab, anon_idx, rare_idx
    )

    train_ds = DraftPlusFeaturesPlusEmbedDataset(h_tr, pf_tr, y_tr, ai_tr)
    val_ds = DraftPlusFeaturesPlusEmbedDataset(h_va, pf_va, y_va, ai_va)

    # Coverage stats.
    train_in_vocab_frac = float(inv_tr.mean())
    val_in_vocab_frac = float(inv_va.mean())
    train_anon_share = float((ai_tr == anon_idx).mean())
    val_anon_share = float((ai_va == anon_idx).mean())

    n_train_pre = int(train_tbl.num_rows)
    n_train_post = int(train_tbl_sub.num_rows)
    n_val = int(val_tbl.num_rows)
    radiant_base_rate_train_full = float(y_train_full.mean())
    radiant_base_rate_train_subsampled = float(y_tr.mean())
    radiant_base_rate_val = float(y_va.mean())

    del train_tbl, train_tbl_sub, val_tbl, sidecar_train, sidecar_val
    del h_tr, pf_tr, y_tr, ai_tr, inv_tr, h_va, pf_va, y_va, ai_va, inv_va
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
        "train_in_vocab_frac": train_in_vocab_frac,
        "val_in_vocab_frac": val_in_vocab_frac,
        "train_anon_share": train_anon_share,
        "val_anon_share": val_anon_share,
    }
    return train_ds, val_ds, meta


__all__ = ["DraftPlusFeaturesPlusEmbedDataset", "load_train_val",
           "stratified_subsample", "assert_no_test_dates",
           "load_arrays_and_acct", "load_vocab", "build_account_idx",
           "join_sidecar_to_main"]
