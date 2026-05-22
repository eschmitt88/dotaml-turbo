"""Build item vocab + duration-bucket edges from rich-cols train sidecar.

Streams the train sidecar parquet row-group by row-group (NEVER full read —
see ~/.claude/projects/.../aiserver2026-postwrite-parquet-reread-oom.md).

For each row-group:
  - For each of p0..p9_items (list<int32>), flatten via pa.compute.list_flatten
    and accumulate item-ID counts into a numpy bincount-backed table.
  - For duration (uint16), accumulate into a per-bucket histogram (online
    quantile estimation would be overkill; just collect all values — turbo
    train is at most ~13M rows, so a uint16 column is ~26 MB).

Outputs results/item_vocab.json with structure:

  {
    "vocab": {"<item_id>": <idx>, ...},   # idx in [2, vocab_size-1] for
                                          # kept items; 0=PAD, 1=RARE
    "duration_bucket_edges": [<float>, ...],  # n_buckets-1 cut points
    "meta": {
      "vocab_size": int (n_kept + 2),
      "freq_cutoff": int (config),
      "n_kept": int,
      "pad_idx": 0,
      "rare_idx": 1,
      "n_total_item_slots": int (10 * train_rows),
      "n_unique_item_ids": int,
      "top_items_by_freq": [[item_id, count], ...],   # top 10
      "min_freq_kept": int,
      "duration": {
        "n_buckets": int,
        "edges": [<float>, ...],
        "bucket_counts": [<int>, ...],   # train-set count per bucket
        "min": int, "max": int, "p50": int,
      },
      "walk_seconds": float,
      "source": "<relative path>",
      "smoke": bool,
    }
  }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

# Reserved index slots for the embedding/output layers.
PAD_IDX = 0
RARE_IDX = 1


def stream_train_sidecar(parquet_path: Path, max_row_groups: int | None = None):
    """Yield (item_arr_int32, duration_arr_int32) per row-group.

    item_arr is the flattened concatenation of p0..p9_items across the row-group.
    duration_arr is the row-group's duration column.
    """
    pf = pq.ParquetFile(parquet_path)
    item_cols = [f"p{p}_items" for p in range(10)]
    n_rg = pf.metadata.num_row_groups
    if max_row_groups is not None:
        n_rg = min(n_rg, max_row_groups)
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg, columns=item_cols + ["duration"])
        # Flatten each list-column to a 1-D pa.ChunkedArray, then numpy.
        flat_pieces: list[np.ndarray] = []
        for c in item_cols:
            flat = pc.list_flatten(tbl.column(c))
            flat_pieces.append(np.asarray(flat, dtype=np.int32))
        all_items = (np.concatenate(flat_pieces) if flat_pieces
                     else np.zeros(0, dtype=np.int32))
        durations = tbl.column("duration").to_numpy(zero_copy_only=False).astype(np.int32)
        del tbl
        yield all_items, durations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rc = cfg["rich_cols"]
    iv = cfg["item_vocab"]
    db = cfg["duration_bucket"]

    out_dir = PROJECT_ROOT / rc["out_dir"]
    if args.smoke and (out_dir / rc["smoke_train_filename"]).exists():
        src = out_dir / rc["smoke_train_filename"]
        max_rg = None
    elif args.smoke:
        src = out_dir / rc["train_filename"]
        max_rg = 1
        print(f"  smoke: sidecar smoke file missing, using first row group of {src.name}")
    else:
        src = out_dir / rc["train_filename"]
        max_rg = None
    if not src.exists():
        sys.exit(f"REFUSED: rich-cols train sidecar not found at {src}. Run build_rich_cols.py first.")

    freq_cutoff = int(iv["freq_cutoff"])
    n_buckets = int(db["n_buckets"])
    vocab_path = EXP_DIR / iv["vocab_path"]
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming item + duration from {src} (max_row_groups={max_rg})")
    t0 = time.time()

    # Online accumulators.
    # Use a dict[int, int] counter; numpy bincount would need a max-ID bound,
    # and Steam API item IDs can be in the 10K range.
    counts: dict[int, int] = {}
    duration_chunks: list[np.ndarray] = []
    n_total_item_slots_rows = 0

    for items_arr, dur_arr in stream_train_sidecar(src, max_row_groups=max_rg):
        n_total_item_slots_rows += dur_arr.size  # row-count this RG
        if items_arr.size > 0:
            uniq, cnt = np.unique(items_arr, return_counts=True)
            for u, c in zip(uniq.tolist(), cnt.tolist()):
                counts[int(u)] = counts.get(int(u), 0) + int(c)
        duration_chunks.append(dur_arr)

    walk_sec = time.time() - t0
    n_train_rows = n_total_item_slots_rows
    n_total_item_slots = n_train_rows * 10  # 10 slots/match
    n_unique = len(counts)
    print(f"  walked in {walk_sec:.0f}s; train_rows={n_train_rows:,}, "
          f"n_unique_item_ids={n_unique:,}")

    # Build vocab.
    kept = [(i, c) for i, c in counts.items() if c >= freq_cutoff]
    kept.sort(key=lambda x: -x[1])
    n_kept = len(kept)
    vocab: dict[str, int] = {}
    for rank, (iid, _) in enumerate(kept):
        vocab[str(int(iid))] = rank + 2   # 0=PAD, 1=RARE
    vocab_size = n_kept + 2
    min_freq_kept = kept[-1][1] if kept else 0
    top_items = [[int(iid), int(c)] for iid, c in kept[:10]]
    print(f"  kept {n_kept:,} items with freq >= {freq_cutoff}; "
          f"vocab_size={vocab_size}; min_freq_kept={min_freq_kept}")
    print(f"  top-10 items by freq: {top_items}")

    # Duration buckets via quantiles on the concatenated duration column.
    if duration_chunks:
        durs = np.concatenate(duration_chunks)
    else:
        durs = np.zeros(0, dtype=np.int32)
    if durs.size == 0:
        sys.exit("No durations seen; cannot build duration buckets.")
    qs = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]  # n_buckets-1 internal cuts
    edges = np.quantile(durs.astype(np.float64), qs).tolist()
    bucket_ids = np.digitize(durs, edges)   # 0..n_buckets-1
    bucket_counts = np.bincount(bucket_ids, minlength=n_buckets).tolist()
    print(f"  duration bucket edges (n_buckets={n_buckets}): {edges}")
    print(f"  duration bucket counts: {bucket_counts}")
    print(f"  duration stats: min={int(durs.min())}, "
          f"max={int(durs.max())}, p50={int(np.median(durs))}")

    out = {
        "vocab": vocab,
        "duration_bucket_edges": edges,
        "meta": {
            "vocab_size": int(vocab_size),
            "freq_cutoff": int(freq_cutoff),
            "n_kept": int(n_kept),
            "pad_idx": int(PAD_IDX),
            "rare_idx": int(RARE_IDX),
            "n_total_item_slots": int(n_total_item_slots),
            "n_unique_item_ids": int(n_unique),
            "top_items_by_freq": top_items,
            "min_freq_kept": int(min_freq_kept),
            "duration": {
                "n_buckets": int(n_buckets),
                "edges": edges,
                "bucket_counts": bucket_counts,
                "min": int(durs.min()),
                "max": int(durs.max()),
                "p50": int(np.median(durs)),
            },
            "walk_seconds": float(walk_sec),
            "source": str(src.relative_to(PROJECT_ROOT)),
            "smoke": bool(args.smoke),
        },
    }
    vocab_path.write_text(json.dumps(out))
    print(f"  wrote {vocab_path} ({vocab_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
