# DataLoader segfault during GC after several epochs on RTX 5080 (Blackwell sm_120) — heap corruption masked by `MALLOC_CHECK_=3`

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
no multiprocessing workers) at:

- `_meta/upstream/torch-dataloader-gc-segfault-repro.py`

Run with:

```
PYTHONFAULTHANDLER=1 python torch-dataloader-gc-segfault-repro.py \
    --rows 5000000 --epochs 200 --batch-size 8192 --gc-every 25
```

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
