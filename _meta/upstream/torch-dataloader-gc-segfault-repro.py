"""
Minimal reproducer for: PyTorch DataLoader segfault during garbage collection
after several training epochs on RTX 5080 (Blackwell sm_120).

Stack trace at crash (observed across torch 2.9.1+cu128, 2.11.0+cu128, 2.12.0+cu130):

    Fatal Python error: Segmentation fault
    Current thread (most recent call first):
      Garbage-collecting
      File ".../torch/utils/data/_utils/fetch.py", line 52 in fetch
      File ".../torch/utils/data/dataloader.py", line 788 in _next_data
      File ".../torch/utils/data/dataloader.py", line 732 in __next__
      File "<user training loop>"

Heap-corruption signal: in our real (Optuna) workload, setting
MALLOC_CHECK_=3 MALLOC_PERTURB_=42 appeared to mask the bug (3 clean trials
where the default allocator crashed in 2), but this *minimal repro* still
crashes under the same allocator settings — sometimes even faster — and
once surfaced an abort with a libtorch C++ trace through
c10::TensorImpl::~TensorImpl(). The real-workload "masking" was likely
small-sample noise within the ~21% per-trial crash-free fraction we
observed across a 60-trial sweep.

Run:
    PYTHONFAULTHANDLER=1 python torch-dataloader-gc-segfault-repro.py

To increase the chance of triggering, raise --epochs and/or --rows.

Reproducer status: CONFIRMED to trigger the crash on the affected hardware
(NVIDIA RTX 5080 Blackwell sm_120, torch 2.9.1+cu128, Python 3.12.3,
glibc 2.39) within 3-5 epochs (~3-4 minutes wall time) on two consecutive
attempts during this script's drafting. Both crashes were the
DataLoader-fetch-during-GC stack documented above; one variant landed at
fetch.py:52 (list comprehension) and one at collate.py:208 (inside
default_collate). The bug remains timing-sensitive — exact epoch and exact
frame within the fetch/collate path vary run-to-run.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

faulthandler.enable()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class DraftDataset(Dataset):
    """Mirrors the working Dataset from the real workload.

    Deep-copies numpy arrays into owned, contiguous torch tensors at __init__
    so pyarrow / numpy buffer lifetimes are NOT involved at __getitem__ time.
    """

    def __init__(self, hero_ids: np.ndarray, y: np.ndarray) -> None:
        self.hero_ids = torch.tensor(hero_ids, dtype=torch.long).contiguous()
        self.y = torch.tensor(y, dtype=torch.float32).contiguous()

    def __len__(self) -> int:
        return self.hero_ids.size(0)

    def __getitem__(self, idx: int):
        return self.hero_ids[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MinimalTransformer(nn.Module):
    """Small encoder: hero embed + team embed -> N TransformerEncoderLayers ->
    mean-pool -> Linear(1). Matches the structure of the real model
    (~80-280k params)."""

    def __init__(
        self,
        n_heroes: int = 151,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.hero_embed = nn.Embedding(n_heroes, d_model)
        self.team_embed = nn.Embedding(2, d_model)
        # team mask: first 5 picks are radiant (team 0), last 5 are dire (team 1)
        team_ids = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)
        self.register_buffer("team_ids", team_ids, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, hero_ids: torch.Tensor) -> torch.Tensor:
        x = self.hero_embed(hero_ids) + self.team_embed(self.team_ids)
        x = self.encoder(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--gc-every", type=int, default=50, help="manual gc.collect() every N batches")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"[setup] generating {args.rows:,} rows of synthetic draft data", flush=True)
    hero_ids = rng.integers(low=1, high=151, size=(args.rows, 10), dtype=np.int64)
    y = rng.integers(low=0, high=2, size=(args.rows,)).astype(np.float32)
    ds = DraftDataset(hero_ids, y)
    # Drop the numpy arrays — mirror the real workload where pyarrow tables
    # were explicitly del'd + gc.collect()'d post-tensor-build.
    del hero_ids, y
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}", flush=True)
    print(f"[setup] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = MinimalTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] model params={n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    use_amp = device.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp
        else torch.autocast(device_type="cpu", enabled=False)
    )

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t_ep = time.time()
        losses: list[float] = []
        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(xb)
                loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.detach().float().item())
            if args.gc_every > 0 and (i + 1) % args.gc_every == 0:
                gc.collect()
        dt = time.time() - t_ep
        total = time.time() - t_start
        print(
            f"[epoch {epoch:3d}] mean_loss={np.mean(losses):.4f} "
            f"wall={dt:6.2f}s total={total/60:6.2f}m batches={len(losses)}",
            flush=True,
        )
        # extra GC pressure between epochs to amplify the suspected refcount/GC
        # interaction inside DataLoader.fetch
        gc.collect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
