"""Build the per-account embedding vocab from the train account_id sidecar.

Streams the train sidecar parquet row-group by row-group (NEVER into pandas;
never read the full clean parquet here — see
~/.claude/projects/-mnt-projects-research-dotaml-turbo/memory/aiserver2026-postwrite-parquet-reread-oom.md).
Tallies per-account appearances across all 10 slots, excludes anonymous
account_ids (0 and 4294967295), keeps the top-N most frequent. Writes
results/player_vocab.json:

    {
      "<account_id_str>": <idx>,        # idx in [2, N+1] for frequent accounts
      ...
      "meta": {
        "vocab_size": <int>,            # N + 2 (anon + rare + frequents)
        "vocab_top_n": <int>,
        "anon_idx": 0,
        "rare_idx": 1,
        "n_total_slot_counts": <int>,   # 10 * train_rows (anon+non-anon)
        "n_non_anonymous_slot_counts": <int>,
        "n_unique_non_anonymous_accounts": <int>,
        "frequency_cutoff_at_topN": <int>,   # min count among frequent accounts
        "coverage_of_topN_over_non_anon": <float>,
        "anonymous_share": <float>,
      }
    }

Smoke mode reads {smoke_train_filename} if present, else single row group of
the full sidecar.
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


def stream_account_ids(parquet_path: Path, max_row_groups: int | None = None):
    """Yield int64 numpy arrays of account_ids, one per row-group per slot column."""
    pf = pq.ParquetFile(parquet_path)
    cols = [f"p{p}_account_id" for p in range(10)]
    n_rg = pf.metadata.num_row_groups
    if max_row_groups is not None:
        n_rg = min(n_rg, max_row_groups)
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg, columns=cols)
        for c in cols:
            yield tbl.column(c).to_numpy(zero_copy_only=False)
        del tbl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="Reads smoke sidecar if present, else first row-group.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    side = cfg["account_sidecar"]
    pe = cfg["player_embedding"]
    out_dir = PROJECT_ROOT / side["out_dir"]
    vocab_path = EXP_DIR / pe["vocab_path"]
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    top_n = int(pe["vocab_top_n"])
    anon_idx = int(pe["anon_idx"])
    rare_idx = int(pe["rare_idx"])

    if args.smoke:
        smoke_path = out_dir / side["smoke_train_filename"]
        if smoke_path.exists():
            src = smoke_path
            max_rg = None
        else:
            src = out_dir / side["train_filename"]
            max_rg = 1
            print(f"  smoke: sidecar smoke file missing, using first row group of {src.name}")
    else:
        src = out_dir / side["train_filename"]
        max_rg = None
    if not src.exists():
        sys.exit(f"REFUSED: sidecar not found at {src}. Run build_account_sidecar.py first.")

    print(f"Streaming account_ids from {src} (max_row_groups={max_rg})")
    t0 = time.time()
    counter: Counter[int] = Counter()
    n_total_slot = 0
    n_non_anon_slot = 0

    for col_np in stream_account_ids(src, max_row_groups=max_rg):
        n_total_slot += col_np.size
        # Mask out anonymous before counting.
        mask = ~np.isin(col_np, np.array(list(ANON_IDS), dtype=col_np.dtype))
        non_anon = col_np[mask]
        n_non_anon_slot += non_anon.size
        # np.unique is faster than Python Counter for large arrays, but
        # for a streaming aggregate we still need a global Counter. Use
        # numpy unique-counts per chunk and merge into the Counter.
        uniq, cnts = np.unique(non_anon, return_counts=True)
        # Counter.update with a dict-like of {int: count} is fastest path.
        counter.update(dict(zip(uniq.tolist(), cnts.tolist())))

    walk_sec = time.time() - t0
    print(f"  walked in {walk_sec:.0f}s; "
          f"n_total_slot={n_total_slot:,} n_non_anon_slot={n_non_anon_slot:,} "
          f"n_unique_non_anon_accounts={len(counter):,}")

    # Top-N.
    top = counter.most_common(top_n)
    n_top = len(top)
    print(f"  kept top {n_top:,} (target {top_n:,})")
    if n_top > 0:
        cutoff = int(top[-1][1])
        print(f"  frequency cutoff at idx {n_top-1}: count={cutoff}")
    else:
        cutoff = 0

    # Build vocab dict. Indices 2..n_top+1 for frequent accounts.
    vocab: dict[str, int] = {}
    for i, (acct, _) in enumerate(top):
        vocab[str(int(acct))] = i + 2  # 0=anon, 1=rare

    # Coverage of top-N over non-anonymous slots.
    if n_non_anon_slot > 0:
        cov = sum(c for _, c in top) / n_non_anon_slot
    else:
        cov = 0.0
    anon_share = (n_total_slot - n_non_anon_slot) / max(n_total_slot, 1)

    meta = {
        "vocab_size": n_top + 2,
        "vocab_top_n": top_n,
        "n_top_kept": n_top,
        "anon_idx": anon_idx,
        "rare_idx": rare_idx,
        "n_total_slot_counts": int(n_total_slot),
        "n_non_anonymous_slot_counts": int(n_non_anon_slot),
        "n_unique_non_anonymous_accounts": int(len(counter)),
        "frequency_cutoff_at_topN": int(cutoff),
        "coverage_of_topN_over_non_anon": float(cov),
        "anonymous_share": float(anon_share),
        "walk_seconds": float(walk_sec),
        "source": str(src.relative_to(PROJECT_ROOT)),
        "smoke": bool(args.smoke),
    }
    out = {"vocab": vocab, "meta": meta}
    vocab_path.write_text(json.dumps(out))
    print(f"  vocab_size={meta['vocab_size']:,}  coverage(top-N over non-anon)={cov:.3f}  "
          f"anon_share={anon_share:.3f}")
    print(f"  wrote {vocab_path} ({vocab_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
