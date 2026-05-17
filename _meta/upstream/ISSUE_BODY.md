## Description

PyTorch's `DataLoader` reliably segfaults during garbage collection inside
`torch.utils.data._utils.fetch.fetch` after several epochs of training on an
RTX 5080 (Blackwell, sm_120, compute capability 12.0). The crash occurs with
a trivial in-memory `Dataset` returning views into preallocated, contiguous
`torch.Tensor`s — no pyarrow, no multiprocessing workers, no shared memory.
We have reproduced the crash on three independent torch/CUDA combinations
(2.9.1+cu128, 2.11.0+cu128, 2.12.0+cu130). The Python stack at crash time is
identical across all reproductions and always sits inside the DataLoader's
single-process fetch path while the cyclic GC is running. Setting
`MALLOC_CHECK_=3 MALLOC_PERTURB_=42` masks the crash in the real workload,
strongly suggesting heap corruption in a C-extension destructor that runs
during GC.

We believe the bug is in torch's tensor refcounting / GC interaction on
Blackwell (or in a Blackwell-specific code path in libtorch/c10), but we
have not isolated the exact module. We are filing because three torch
versions reproduce identically and the failure mode (silent heap corruption
detected only at GC time) is non-trivial to debug downstream.

## To Reproduce

Minimal self-contained Python script (no pyarrow, no parquet, no sklearn,
no multiprocessing workers) — full source inlined below for reproducibility.

Run with:

```
PYTHONFAULTHANDLER=1 python repro.py --rows 5000000 --epochs 200 --batch-size 8192 --gc-every 25
```

<details>
<summary>Full reproducer script (click to expand, ~190 lines)</summary>

```python
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
```

</details>

In our testing the script segfaults within 3–5 epochs (~3–4 minutes of
training) on the affected hardware. The repro:

1. Builds 5M rows of synthetic `(hero_ids: int64[10], y: float32)` data.
2. Wraps it in a trivial `Dataset` that deep-copies into owned torch tensors
   at `__init__` and returns indexed views at `__getitem__`.
3. Runs a small Transformer (≈416k params, `nn.TransformerEncoderLayer`,
   `d_model=128`, 2 layers, 8 heads) under `bf16` autocast.
4. Forces `gc.collect()` every 25 batches and at epoch end to amplify the
   suspected GC interaction.

## Expected behavior

Training should complete without a segmentation fault. A `Dataset` that
owns its tensors and a `DataLoader` with `num_workers=0` exercise only the
single-process fetch path and should be the safest possible configuration.

## Environment

- PyTorch: `2.9.1+cu128` (also reproduced on `2.11.0+cu128`, `2.12.0+cu130`)
- CUDA runtime bundled with wheel: 12.8 (also reproduced with cu13.0 wheel)
- Python: 3.12.3 (Ubuntu 24.04)
- glibc: 2.39
- NVIDIA driver: 580.159.03 (open kernel module)
- GPU: NVIDIA GeForce RTX 5080 (Blackwell, sm_120, compute capability 12.0,
  16 GB)
- numpy 2.4.4, pyarrow 24.0.0, scikit-learn 1.8.0, scipy 1.17.1 (none of
  these are imported by the minimal repro)
- Kernel: Linux 6.17.0-23-generic x86_64
- DataLoader config: `num_workers=0, pin_memory=True, shuffle=True,
  batch_size=8192`

## Stack trace at crash

Two variants were observed across our reproductions; both occur with the
Python interpreter in the "Garbage-collecting" state and both sit inside
the DataLoader fetch path:

**Variant 1 — list comprehension in `Fetcher.fetch`:**

```
Fatal Python error: Segmentation fault

Current thread (most recent call first):
  Garbage-collecting
  File ".../torch/utils/data/_utils/fetch.py", line 52 in fetch
  File ".../torch/utils/data/dataloader.py", line 788 in _next_data
  File ".../torch/utils/data/dataloader.py", line 732 in __next__
  File "<user training loop>"
```

(`fetch.py` line 52 is `data = [self.dataset[idx] for idx in possibly_batched_index]`.)

**Variant 2 — inside `default_collate`:**

```
Fatal Python error: Segmentation fault

Current thread (most recent call first):
  Garbage-collecting
  File ".../torch/utils/data/_utils/collate.py", line 208 in collate
  File ".../torch/utils/data/_utils/collate.py", line 398 in default_collate
  File ".../torch/utils/data/_utils/fetch.py", line 55 in fetch
  File ".../torch/utils/data/dataloader.py", line 788 in _next_data
  File ".../torch/utils/data/dataloader.py", line 732 in __next__
  File "<user training loop>"
```

In our real production workload (which uses the same model and dataset shape
but loads from parquet) we also observed a glibc-side message of the form
`free(): invalid pointer` immediately before the segfault, confirming heap
corruption rather than a NULL deref or out-of-bounds CUDA write.

## Additional context

### What reproduces the crash

- Three independent torch versions: 2.12.0+cu130, 2.11.0+cu128, 2.9.1+cu128.
- Both the real workload (`~21% per-trial crash rate` across a 60-trial
  Optuna sweep, typically after 5+ epochs of 5M-row training) and the
  minimal synthetic repro above.
- Always with `num_workers=0`, `pin_memory=True`, `shuffle=True`,
  large-batch (`batch_size ∈ [4096, 16384]`) training on Blackwell.

### What we ruled out

- **Not a CUDA kernel bug.** Running with `CUDA_LAUNCH_BLOCKING=1` produces
  no synchronous CUDA error before the crash. Instead glibc emits
  `free(): invalid pointer`, then the segfault occurs at the next GC.
- **Not pyarrow `tp_traverse`.** Explicitly `del`'ing every pyarrow Table
  and calling `gc.collect()` after the tensor copy did not change the
  crash rate. The synthetic repro above does not import pyarrow at all
  and still crashes.
- **Not torch-2.12-specific.** torch 2.9.1+cu128 reproduces with the
  identical stack.
- **Not driver/cuDNN.** Same driver (`580.159.03`) and same cuDNN bundle
  across all reproductions.
- **Not the Dataset implementation.** The `Dataset` deep-copies numpy
  arrays into owned, contiguous `torch.Tensor`s at `__init__` and returns
  indexed views at `__getitem__`. No external buffer lifetimes are
  involved at fetch time.

### Heap-corruption signal

In the **real workload** (loads from parquet via pyarrow, larger training
loop, runs inside an Optuna trial process), setting

```
MALLOC_CHECK_=3 MALLOC_PERTURB_=42
```

masks the bug completely — three full trials with no crashes, where the
default glibc allocator crashes at trial 2 within ~5 epochs. This strongly
suggests a use-after-free or double-free in a C-extension destructor that
runs during cyclic GC, and that the corrupted chunk happens to lie in a
region the slower checked allocator handles differently (e.g. via different
chunk layout / canaries).

In the **synthetic repro**, `MALLOC_CHECK_=3 MALLOC_PERTURB_=42` does NOT
mask the crash; one of our attempts with the checked allocator surfaced an
abort with a libtorch C++ stack ending in `c10::TensorImpl::~TensorImpl()`,
suggesting the corruption is reached via tensor destruction during GC.
That trace was not perfectly stable across re-runs but is consistent with
the variant-2 stack above (`default_collate` doing tensor work and then
the per-batch tensors going out of scope at GC time).

### Workaround we are using

Running each Optuna trial in a fresh subprocess sidesteps the bug —
confirming the failure is process-local Python/torch state that
accumulates across many epochs / many trials in one interpreter. We do
not expect this is appropriate for upstream; we mention it only to confirm
the issue is not in our data or our model architecture.

### Repro reliability

The synthetic script segfaulted on the first two ~5–10 min runs we
attempted while drafting this report (epoch 5 at ~3.5 min, and again at
epoch 5 at ~4.1 min under `MALLOC_CHECK_=3`). Both crashes had the
DataLoader+GC stack above. We are happy to provide additional traces,
a core dump, or run targeted instrumentation patches as needed.