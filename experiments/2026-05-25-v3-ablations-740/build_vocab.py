"""Build the per-account embedding vocab for v3-ablations-740 A2.

Streams account_id sidecar parquet(s) and counts non-anonymous account_id
appearances across all 10 slots. Keeps the top-K most frequent. Writes
vocab/player_id_vocab.json:

    {
      "vocab": {"<account_id_str>": <idx>, ...},   # idx in [1, K]
      "meta":  {
        "vocab_top_n":      <int>,                 # target K
        "n_top_kept":       <int>,                 # actual K
        "anon_idx":         0,
        "hash_base_idx":    K+1,                   # start of hash region
        "n_hash_buckets":   <int>,                 # from config
        "vocab_size_total": K + 1 + n_hash_buckets,  # 0=anon, 1..K top-K, K+1..K+B hash
        "frequency_cutoff_at_topN": <int>,
        "anon_share":       <float>,
        "coverage_of_topK_over_non_anon": <float>,
        "n_unique_non_anonymous_accounts": <int>,
        "walk_seconds":     <float>,
        "source_paths":     [...],
        "smoke":            <bool>,
      }
    }

Layout invariants:
  - idx 0  = shared "anonymous" embedding row.
  - idx 1..K  = individual rows for the top-K most-frequent non-anon accounts.
  - idx K+1..K+B = shared hash-bucket rows for the long-tail non-anon accounts
                   (acct_id % B + (K+1)).

Streams via pq.ParquetFile.iter_batches; never materializes a full multi-GB
parquet (defense per aiserver2026-postwrite-parquet-reread-oom memory).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

ANON_IDS = {0, 4294967295}


def stream_account_ids(parquet_path: Path, batch_size: int = 100_000,
                        max_batches: int | None = None):
    """Yield int64 numpy arrays of account_ids, batch by batch per slot column."""
    pf = pq.ParquetFile(parquet_path)
    cols = [f"p{p}_account_id" for p in range(10)]
    n = 0
    for rb in pf.iter_batches(batch_size=batch_size, columns=cols):
        for c in cols:
            yield rb.column(c).to_numpy(zero_copy_only=False)
        n += 1
        if max_batches is not None and n >= max_batches:
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="Use smoke_train_paths from config (or first batch only).")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Cap batches per source file (debug).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    side = cfg["account_sidecar"]
    pe = cfg["player_embedding"]
    if args.smoke:
        src_paths = [PROJECT_ROOT / p
                     for p in side.get("smoke_train_paths", side["train_paths"])]
    else:
        src_paths = [PROJECT_ROOT / p for p in side["train_paths"]]

    top_n = int(pe["vocab_top_n"])
    n_hash_buckets = int(pe.get("n_hash_buckets", 0))

    src_paths = [p for p in src_paths if p.exists() and p.stat().st_size > 0]
    if not src_paths:
        sys.exit("REFUSED: no account_sidecar train sources found / built.")
    print(f"Streaming account_ids from {len(src_paths)} path(s):")
    for p in src_paths:
        print(f"  - {p} ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"  top_n={top_n}, n_hash_buckets={n_hash_buckets}")

    t0 = time.time()
    counter: Counter[int] = Counter()
    n_total_slot = 0
    n_non_anon_slot = 0

    for src in src_paths:
        n_paths_before = n_total_slot
        try:
            for col_np in stream_account_ids(src, batch_size=200_000,
                                                max_batches=args.max_batches):
                n_total_slot += col_np.size
                mask = ~np.isin(col_np, np.array(list(ANON_IDS), dtype=col_np.dtype))
                non_anon = col_np[mask]
                n_non_anon_slot += non_anon.size
                uniq, cnts = np.unique(non_anon, return_counts=True)
                counter.update(dict(zip(uniq.tolist(), cnts.tolist())))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: {src}: {e}")
            continue
        added = n_total_slot - n_paths_before
        print(f"  {src.name}: added {added:,} slot reads -> {n_total_slot:,} total")

    walk_sec = time.time() - t0
    print(f"  walked in {walk_sec:.0f}s; "
          f"n_total_slot={n_total_slot:,} n_non_anon_slot={n_non_anon_slot:,} "
          f"n_unique_non_anon_accounts={len(counter):,}")

    top = counter.most_common(top_n)
    n_top = len(top)
    cutoff = int(top[-1][1]) if n_top > 0 else 0
    print(f"  kept top {n_top:,} (target {top_n:,}); frequency cutoff at idx {n_top - 1}: count={cutoff}")

    # Indices: 0=anon, 1..n_top = top-K, n_top+1..n_top+n_hash_buckets = hash.
    hash_base_idx = 1 + n_top
    vocab_size_total = 1 + n_top + n_hash_buckets
    vocab: dict[str, int] = {}
    for i, (acct, _) in enumerate(top):
        vocab[str(int(acct))] = i + 1  # 1..n_top

    cov = (sum(c for _, c in top) / n_non_anon_slot) if n_non_anon_slot > 0 else 0.0
    anon_share = (n_total_slot - n_non_anon_slot) / max(n_total_slot, 1)

    meta = {
        "vocab_top_n": top_n,
        "n_top_kept": n_top,
        "anon_idx": 0,
        "hash_base_idx": int(hash_base_idx),
        "n_hash_buckets": int(n_hash_buckets),
        "vocab_size_total": int(vocab_size_total),
        "n_total_slot_counts": int(n_total_slot),
        "n_non_anonymous_slot_counts": int(n_non_anon_slot),
        "n_unique_non_anonymous_accounts": int(len(counter)),
        "frequency_cutoff_at_topN": int(cutoff),
        "coverage_of_topK_over_non_anon": float(cov),
        "anon_share": float(anon_share),
        "walk_seconds": float(walk_sec),
        "source_paths": [str(p.relative_to(PROJECT_ROOT)) for p in src_paths],
        "smoke": bool(args.smoke),
    }

    out_path = EXP_DIR / pe["vocab_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"vocab": vocab, "meta": meta}))
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  vocab_size_total={vocab_size_total:,} (anon=1 + top_k={n_top:,} + hash={n_hash_buckets})")
    print(f"  coverage(top-K over non-anon)={cov:.3f}  anon_share={anon_share:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
