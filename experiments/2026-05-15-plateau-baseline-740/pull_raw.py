"""Download patch-7.40 Turbo parquet from Azure to the local snapshot.

HCE rule (~/.claude/rules/evaluation.md): we are in the search phase, so
this script must NOT touch the test window [2026-03-10, 2026-03-23].
The download is restricted to train + val dates only.

Usage:
    python pull_raw.py [--include-test]   # --include-test reserved for the
                                          # explicit final-scoring pass; this
                                          # implementer run never sets it.

Idempotent: skips files already present on disk with the right size.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import sys
import time
from pathlib import Path

import yaml
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = PROJECT_ROOT / "data/snapshots/7.40-2025-12-16"
RAW_ROOT = SNAPSHOT_DIR / "raw" / "turbo"
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"

ACCOUNT = "dota2datalake"
CONTAINER = "matches"
BASE = "turbo"


def date_range(start: str, end: str):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    cur = s
    while cur <= e:
        yield cur
        cur += dt.timedelta(days=1)


def list_blobs_for_date(cc: ContainerClient, d: dt.date) -> list[tuple[str, int]]:
    prefix = f"{BASE}/year={d.year}/month={d.month:02d}/day={d.day:02d}/"
    out: list[tuple[str, int]] = []
    for b in cc.list_blobs(name_starts_with=prefix):
        if not b.name.endswith(".parquet"):
            continue
        # Skip aggregate daily_summary; we only want match files.
        if Path(b.name).name.startswith("matches_"):
            out.append((b.name, b.size))
    return out


def local_path_for_blob(blob_name: str) -> Path:
    # turbo/year=YYYY/month=MM/day=DD/matches_X_Y.parquet -> RAW_ROOT/year=.../matches_X_Y.parquet
    rel = blob_name[len(f"{BASE}/"):]
    return RAW_ROOT / rel


def download_one(cc: ContainerClient, name: str, expected_size: int) -> tuple[str, str]:
    dest = local_path_for_blob(name)
    if dest.exists() and dest.stat().st_size == expected_size:
        return ("skip", name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        bc = cc.get_blob_client(name)
        with open(tmp, "wb") as f:
            stream = bc.download_blob(max_concurrency=4)
            stream.readinto(f)
        tmp.rename(dest)
        return ("ok", name)
    except Exception as e:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return (f"err:{type(e).__name__}:{e}", name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--include-test",
        action="store_true",
        help="ONLY for the final-scoring pass. Default is search-phase (excludes test window).",
    )
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument(
        "--days",
        type=str,
        default=None,
        help="Optional comma-separated YYYY-MM-DD list; otherwise pull all train+val dates.",
    )
    args = ap.parse_args()

    splits = yaml.safe_load(SPLITS_PATH.read_text())

    # Compose the dates we will read, respecting HCE.
    if args.days:
        dates = [dt.date.fromisoformat(s.strip()) for s in args.days.split(",")]
    else:
        train = list(date_range(splits["train_start_date"], splits["train_end_date"]))
        val = list(date_range(splits["val_start_date"], splits["val_end_date"]))
        if args.include_test:
            test = list(date_range(splits["test_start_date"], splits["test_end_date"]))
            dates = train + val + test
        else:
            dates = train + val

    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    if not args.include_test:
        bad = [d for d in dates if test_lo <= d <= test_hi]
        if bad:
            sys.exit(
                f"REFUSED: search-phase pull includes test-window dates {bad} — HCE rule violated."
            )

    print(f"Snapshot dir: {SNAPSHOT_DIR}")
    print(f"Dates to consider: {len(dates)} (first={dates[0]}, last={dates[-1]})")
    print(f"--include-test: {args.include_test}")

    cred = DefaultAzureCredential()
    account_url = f"https://{ACCOUNT}.blob.core.windows.net"
    cc = ContainerClient(account_url=account_url, container_name=CONTAINER, credential=cred)

    # Phase 1: enumerate all blobs.
    print("Listing blobs...")
    all_blobs: list[tuple[str, int]] = []
    for d in tqdm(dates):
        all_blobs.extend(list_blobs_for_date(cc, d))
    print(f"Found {len(all_blobs)} parquet files. Total size = {sum(s for _, s in all_blobs) / 1e9:.1f} GB")

    # Phase 2: filter to ones we still need.
    needed: list[tuple[str, int]] = []
    skipped = 0
    for name, sz in all_blobs:
        dest = local_path_for_blob(name)
        if dest.exists() and dest.stat().st_size == sz:
            skipped += 1
        else:
            needed.append((name, sz))
    print(f"{skipped} already on disk, {len(needed)} to download.")
    if not needed:
        return 0

    # Phase 3: download in parallel.
    t0 = time.time()
    bytes_downloaded = 0
    errs: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(download_one, cc, n, s): (n, s) for n, s in needed}
        with tqdm(total=len(futs), desc="download") as pbar:
            for fut in cf.as_completed(futs):
                status, name = fut.result()
                _, sz = futs[fut]
                if status == "ok":
                    bytes_downloaded += sz
                elif status.startswith("err"):
                    errs.append(f"{name}: {status}")
                pbar.update(1)
                if bytes_downloaded > 0:
                    rate = bytes_downloaded / max(1, time.time() - t0) / 1e6
                    pbar.set_postfix(MBps=f"{rate:.1f}", errs=len(errs))

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Downloaded {bytes_downloaded/1e9:.2f} GB. Errors: {len(errs)}")
    if errs:
        for e in errs[:20]:
            print("  ", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
