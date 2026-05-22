"""
PyArrow write→read round-trip stress test.

Hypothesis: silent RAM bit-flips during high-memory parallel workloads
cause PyArrow to occasionally return mismatched data on parquet reads
(or write corrupted data). This script tries to reproduce that
deterministically under sustained memory pressure.

Strategy:
  1. Generate a 1.4 GB parquet with known content (deterministic from seed).
  2. Compute md5 of the on-disk file.
  3. In a loop:
     a. Read the file with pyarrow (full read, all columns).
     b. Compute md5 of the reconstructed bytes (write back to a temp BytesIO).
     c. Also check a few specific column statistics (sum, min, max, hash) against expected.
     d. Allocate a large numpy array to apply memory pressure.
  4. Log every iteration. Flag any mismatch.

If any read returns different data than was written, we have a smoking
gun for memory corruption. If 100+ iterations pass cleanly, the failure
is rarer than this test catches.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def gen_parquet(path: Path, n_rows: int, n_cols: int, seed: int = 42):
    """Write a deterministic parquet of shape (n_rows, n_cols) with mixed dtypes."""
    rng = np.random.default_rng(seed)
    cols = {}
    for c in range(n_cols):
        # mix of float32, int32, int64, float64 to stress different paths
        dtype_choice = c % 4
        if dtype_choice == 0:
            cols[f"f32_{c}"] = rng.standard_normal(n_rows, dtype=np.float32)
        elif dtype_choice == 1:
            cols[f"i32_{c}"] = rng.integers(-1_000_000, 1_000_000, size=n_rows, dtype=np.int32)
        elif dtype_choice == 2:
            cols[f"i64_{c}"] = rng.integers(0, 1_000_000_000, size=n_rows, dtype=np.int64)
        else:
            cols[f"f64_{c}"] = rng.standard_normal(n_rows)
    table = pa.table(cols)
    pq.write_table(table, path, compression="snappy", row_group_size=1_000_000)
    return table


def column_signature(table) -> dict:
    """Hash + summary stats for each column. Returns dict for comparison."""
    sig = {}
    for col_name in table.column_names:
        arr = table.column(col_name).to_numpy(zero_copy_only=False)
        h = hashlib.md5(arr.tobytes()).hexdigest()
        sig[col_name] = {
            "md5": h,
            "len": int(arr.shape[0]),
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "sum_abs": float(np.abs(arr).sum()),
        }
    return sig


def signatures_match(a: dict, b: dict, tol: float = 0.0) -> tuple[bool, list]:
    """Compare two column signatures. Returns (matches, list of mismatch reasons)."""
    if set(a.keys()) != set(b.keys()):
        return False, [f"column sets differ: {set(a.keys()) ^ set(b.keys())}"]
    diffs = []
    for col in a:
        for k in ("md5", "len"):
            if a[col][k] != b[col][k]:
                diffs.append(f"{col}.{k}: {a[col][k]} vs {b[col][k]}")
        # numerical fields: allow small tol for float drift but for byte-identical data we expect exact
        for k in ("min", "max", "sum_abs"):
            av, bv = a[col][k], b[col][k]
            if av is None and bv is None:
                continue
            if av != bv:
                diffs.append(f"{col}.{k}: {av} vs {bv}")
    return len(diffs) == 0, diffs


def memory_pressure_alloc(gb: float) -> np.ndarray:
    """Allocate a large numpy array to put memory pressure on the system."""
    n_bytes = int(gb * (1024**3))
    n_elem = n_bytes // 8
    arr = np.random.default_rng().standard_normal(n_elem)
    arr_sum = float(arr.sum())  # touch every page so it actually allocates
    return arr, arr_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/projects/research/dotaml-turbo/_meta/hardware-investigation-2026-05-21/pq")
    ap.add_argument("--n-rows", type=int, default=8_000_000, help="rows per parquet (~1 GB at default cols)")
    ap.add_argument("--n-cols", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pressure-gb", type=float, default=20.0,
                    help="GB of memory to allocate during reads (simulates aggregator state)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / "stress.parquet"

    print(f"=== PyArrow round-trip stress test ===")
    print(f"out: {pq_path}")
    print(f"params: n_rows={args.n_rows:,}, n_cols={args.n_cols}, iters={args.iters}, pressure_gb={args.pressure_gb}")
    print()

    # Phase A: generate baseline parquet
    print(f"[{time.strftime('%H:%M:%S')}] generating baseline parquet...")
    t0 = time.time()
    expected_table = gen_parquet(pq_path, args.n_rows, args.n_cols, seed=args.seed)
    file_size = pq_path.stat().st_size
    print(f"  wrote {file_size/1e9:.2f} GB in {time.time()-t0:.1f}s")

    # md5 of the on-disk file (should never change unless overwritten)
    with open(pq_path, "rb") as f:
        disk_md5 = hashlib.md5(f.read()).hexdigest()
    print(f"  on-disk md5: {disk_md5}")

    # Baseline signature: read fresh, compute signature
    expected_sig = column_signature(expected_table)
    del expected_table
    print(f"  computed expected signature for {len(expected_sig)} columns")
    print()

    # Phase B: stress loop
    print(f"[{time.strftime('%H:%M:%S')}] starting {args.iters}-iter stress loop with {args.pressure_gb} GB pressure...")
    fails = []
    pressure_alloc = None
    pressure_sum = None
    for i in range(args.iters):
        t0 = time.time()

        # Allocate pressure (fresh each iter to force memory writes/reads)
        if pressure_alloc is not None:
            del pressure_alloc
        pressure_alloc, pressure_sum = memory_pressure_alloc(args.pressure_gb)

        # Verify the on-disk file hasn't changed
        with open(pq_path, "rb") as f:
            cur_disk_md5 = hashlib.md5(f.read()).hexdigest()
        disk_drift = cur_disk_md5 != disk_md5

        # Read the parquet via pq.read_table
        try:
            tbl = pq.read_table(pq_path)
        except Exception as e:
            print(f"  iter {i:3d}  FAIL read: {type(e).__name__}: {e}  ({time.time()-t0:.1f}s)")
            fails.append((i, "read_exception", str(e)))
            continue

        # Compute signature on the just-read table
        try:
            got_sig = column_signature(tbl)
        except Exception as e:
            print(f"  iter {i:3d}  FAIL sig: {type(e).__name__}: {e}  ({time.time()-t0:.1f}s)")
            fails.append((i, "sig_exception", str(e)))
            del tbl
            continue

        ok, diffs = signatures_match(expected_sig, got_sig)
        elapsed = time.time() - t0
        if not ok or disk_drift:
            status = "FAIL"
            extra = f"  disk_drift={disk_drift}  diffs={diffs[:5]}"
            fails.append((i, "sig_mismatch", diffs[:5]))
        else:
            status = "ok  "
            extra = ""
        if i % 10 == 0 or not ok or disk_drift:
            print(f"  iter {i:3d}  {status}  {elapsed:.1f}s  pressure_sum={pressure_sum:.3g}{extra}")

        del tbl

    print()
    print(f"=== summary ===")
    print(f"iters: {args.iters}")
    print(f"fails: {len(fails)}")
    if fails:
        print("first 5 fail entries:")
        for f in fails[:5]:
            print(f"  {f}")
        sys.exit(1)
    print("all iterations passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
