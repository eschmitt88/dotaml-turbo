"""Download pre-patch-7.40 Turbo parquet from Azure to data/history/turbo/.

HCE-strict: refuses ANY date in the sealed test window
[2026-03-10, 2026-03-23] or in post-snapshot [2026-03-24, ...]. Default
pull range is [2025-08-01, 2025-12-15] from config.yaml.

Lands under PROJECT_ROOT/data/history/turbo/year=YYYY/month=MM/day=DD/
matches_*.parquet — semantically distinct from
data/snapshots/.../raw/turbo/ which holds the patch-7.40 snapshot.

Idempotent: skips files already present with the right size.

Usage:
    python pull_history.py                     # full pull per config.yaml
    python pull_history.py --days 2025-12-14,2025-12-15   # smoke pull
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import sys
import time
from pathlib import Path

import yaml
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"

ACCOUNT = "dota2datalake"
CONTAINER = "matches"
BASE = "turbo"


def date_range(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def list_blobs_for_date(cc: ContainerClient, d: dt.date) -> list[tuple[str, int]]:
    prefix = f"{BASE}/year={d.year}/month={d.month:02d}/day={d.day:02d}/"
    out: list[tuple[str, int]] = []
    for b in cc.list_blobs(name_starts_with=prefix):
        if not b.name.endswith(".parquet"):
            continue
        if Path(b.name).name.startswith("matches_"):
            out.append((b.name, b.size))
    return out


def local_path_for_blob(blob_name: str, out_root: Path) -> Path:
    # turbo/year=YYYY/month=MM/day=DD/matches_X_Y.parquet
    #   -> out_root/year=.../matches_X_Y.parquet
    rel = blob_name[len(f"{BASE}/"):]
    return out_root / rel


def download_one(cc: ContainerClient, name: str, expected_size: int,
                 out_root: Path) -> tuple[str, str]:
    dest = local_path_for_blob(name, out_root)
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
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--days", type=str, default=None,
                    help="Comma-separated YYYY-MM-DD list; otherwise full pull from config.")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())
    pull_cfg = cfg["pull_history"]
    out_root = PROJECT_ROOT / pull_cfg["out_root"]
    threads = int(args.threads if args.threads is not None else pull_cfg["threads"])

    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    snapshot_end = dt.date.fromisoformat(splits["snapshot_end_date"])

    if args.days:
        dates = [dt.date.fromisoformat(s.strip()) for s in args.days.split(",")]
    else:
        s = dt.date.fromisoformat(pull_cfg["start_date"])
        e = dt.date.fromisoformat(pull_cfg["end_date"])
        dates = list(date_range(s, e))

    # HCE: refuse test window and post-snapshot dates.
    bad_test = [d for d in dates if test_lo <= d <= test_hi]
    if bad_test:
        sys.exit(f"REFUSED: pull includes test-window dates {bad_test[:3]} — HCE rule violated.")
    bad_post = [d for d in dates if d > snapshot_end]
    if bad_post:
        sys.exit(f"REFUSED: pull includes post-snapshot dates {bad_post[:3]} — out of scope.")

    print(f"Out root: {out_root}")
    print(f"Dates to consider: {len(dates)} (first={dates[0]}, last={dates[-1]})")
    print(f"HCE checks passed (no test-window, no post-snapshot dates).")

    cred = DefaultAzureCredential()
    account_url = f"https://{ACCOUNT}.blob.core.windows.net"
    cc = ContainerClient(account_url=account_url, container_name=CONTAINER, credential=cred)

    print("Listing blobs...")
    all_blobs: list[tuple[str, int]] = []
    for d in tqdm(dates):
        all_blobs.extend(list_blobs_for_date(cc, d))
    print(f"Found {len(all_blobs)} parquet files. "
          f"Total size = {sum(s for _, s in all_blobs) / 1e9:.1f} GB")

    needed: list[tuple[str, int]] = []
    skipped = 0
    for name, sz in all_blobs:
        dest = local_path_for_blob(name, out_root)
        if dest.exists() and dest.stat().st_size == sz:
            skipped += 1
        else:
            needed.append((name, sz))
    print(f"{skipped} already on disk, {len(needed)} to download.")
    if not needed:
        return 0

    t0 = time.time()
    bytes_downloaded = 0
    errs: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(download_one, cc, n, s, out_root): (n, s) for n, s in needed}
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
    print(f"Done in {elapsed:.0f}s. Downloaded {bytes_downloaded/1e9:.2f} GB. "
          f"Errors: {len(errs)}")
    if errs:
        for e in errs[:20]:
            print("  ", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
