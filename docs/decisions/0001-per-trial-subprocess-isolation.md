---
kind: adr
number: "0001"
slug: per-trial-subprocess-isolation
title: "Per-trial subprocess isolation for PyTorch sweeps on Blackwell"
date: "2026-05-17"
status: superseded
superseded_by: hardware-investigation-2026-05-21 (RAM bit-flip root cause)
superseded_date: 2026-05-21
note: "The torch DataLoader segfaults this ADR was written for were caused by RAM bit-flips on unstable DDR5 EXPO 6000 MT/s, NOT a torch bug. With EXPO disabled the issue is gone. The per-trial subprocess isolation pattern is still useful for OTHER reasons (resumable Optuna sweeps, clean state per trial) but is no longer needed as a workaround for the segfault. See _meta/upstream/RETRACTED.md and _meta/hardware-investigation-2026-05-21/."
context_tags: [torch, blackwell, sm-120, dataloader, gc, infrastructure]
---

# ADR 0001 — Per-trial subprocess isolation for PyTorch sweeps on Blackwell

## Context

The `2026-05-16-transformer-hp-sweep-740` experiment ran a 60-trial
Optuna sweep on an RTX 5080 (Blackwell sm_120). The in-process Optuna
loop (default pattern: one Python process running all trials) crashed
intermittently — total of 15 hard crashes across 70 wrapper iterations
(~21% per-trial crash rate). Symptoms varied: `SIGSEGV` (rc 139),
`free(): invalid pointer` (rc 134), `Overflow when unpacking long long`
in `DraftDataset.__getitem__`, NVRM `Xid 43` entries in `dmesg`, and
once a corrupted torch `.pyc` cache (`bad marshal data`) caused by a
SIGSEGV mid-write of the bytecode file.

The 2026-05-17 diagnostic investigation isolated the root cause:

| Hypothesis | Verdict |
| --- | --- |
| CUDA kernel bug (Blackwell sm_120) | **Ruled out** — `CUDA_LAUNCH_BLOCKING=1` showed NO CUDA error before the crash; instead `free(): invalid pointer` from glibc fired |
| Specific to torch 2.12+cu130 | **Ruled out** — reproduces identically on torch 2.9.1+cu128 (7 months bake time) and torch 2.11.0+cu128 |
| NVIDIA driver / cuDNN | **Ruled out** — same driver across all reproductions |
| pyarrow Table `tp_traverse` | **Ruled out** — explicit `del` of all pyarrow Tables plus `gc.collect()` after data load did NOT fix it |
| Hardware fault | **Ruled out** — synthetic minimal repro triggers the same crash in <5 min |
| Heap corruption (timing-sensitive) | **Confirmed** — `MALLOC_CHECK_=3` masks the bug in our real workload (3 trials, no crash) but NOT in the synthetic repro (still crashes, sometimes with a libtorch C++ trace through `c10::TensorImpl::~TensorImpl()`); the real-workload masking was probably noise within the ~21 % crash-free fraction. The libtorch destructor trace pins the corruption to tensor teardown during cyclic GC |

`PYTHONFAULTHANDLER=1` gave a consistent stack trace across every
crash:

```
Fatal Python error: Segmentation fault
Current thread (most recent call first):
  Garbage-collecting
  File ".../torch/utils/data/_utils/fetch.py", line 52 in fetch
  File ".../torch/utils/data/dataloader.py", line 788 in _next_data
  File ".../train_one.py", line 131 (for hero_ids, y in train_loader:)
```

So the bug is in **torch's DataLoader + tensor GC interaction**, surfaces
under memory pressure (typically after 5+ epochs of training, or
shortly after a prior Optuna trial completed in the same process), and
is reproducible across the torch 2.9-2.12 family. The NVRM Xid 43
entries in `dmesg` were **secondary** — when the Python process aborts
mid-CUDA-op, NVRM resets the GPU channel and logs Xid 43 as a side
effect.

The bug has been documented with:

- A self-contained 193-line synthetic reproducer at
  `_meta/upstream/torch-dataloader-gc-segfault-repro.py` (no pyarrow,
  no Optuna, no parquet — confirmed to trigger the crash at epoch 5
  within ~3-4 min on this hardware)
- A draft upstream report at
  `_meta/upstream/torch-dataloader-gc-segfault-report.md` ready to
  file at https://github.com/pytorch/pytorch/issues. The synthetic
  repro confirms the bug is **not** pyarrow-, parquet-, Optuna-, or
  multi-trial-related; a single ~5 min training loop with owned torch
  tensors is sufficient.

## Decision

**Run each Optuna trial in its own fresh Python process.** Concretely:

- `experiments/2026-05-16-transformer-hp-sweep-740/run_sweep_loop.sh`
  invokes `.venv/bin/python run_sweep.py --n-trials 1 --skip-top-k`
  per iteration, looping until the Optuna SQLite study has `N`
  COMPLETE+PRUNED trials.
- `cleanup_failed_trials.py` between iterations removes
  `FAIL`/`RUNNING`/`WAITING` rows so n_trials budget isn't burned by
  ghost trials from prior crashes.
- The Optuna SQLite store at `results/optuna.db` is the coordination
  point and is resumable across process restarts.
- Post-sweep top-k retraining runs as a separate isolated invocation
  (`run_sweep.py --retrain-only`).

The same pattern is the default for any future GPU-bound HP sweep in
this project. Treat in-process Optuna loops on this hardware as a
known-broken pattern.

Additionally, `objective.py` now does explicit per-trial cleanup
(`del model`, `torch.cuda.empty_cache()`, `gc.collect()`) before
returning, as belt-and-suspenders for local debug or smoke runs that
intentionally invoke the objective multiple times in one process.

## Why not the alternatives

- **Downgrade torch to 2.9.1+cu128 (most-baked stack).** Tried 2026-05-16.
  Same crash, same trace, same frequency. Not a torch-version fix.
- **Upgrade NVIDIA driver 580→595-open.** NVIDIA's docs are explicit:
  "Xid 43 is logged when a user application hits a software-induced
  fault, with the GPU remaining in a healthy state" — i.e. the
  application is at fault, not the driver. Still worth doing eventually
  as system hygiene, but won't address this bug.
- **`MALLOC_CHECK_=3` in production.** This masks the bug entirely but
  imposes 10-30 % allocation overhead on every malloc/free for the
  process lifetime. The architectural choice (subprocess isolation) is
  free at idle and ~5 sec overhead per trial.
- **Disable Python GC during training.** Standard workaround for
  C-extension GC bugs but treats the symptom. Subprocess isolation
  achieves the same effect cleanly (each subprocess's GC state is
  fresh).
- **Replace DataLoader with a hand-written batch iterator.** Sidesteps
  the bug, but pays in code complexity for marginal benefit over
  subprocess isolation.

## Consequences

**Positive:**

- Production sweeps complete reliably under the documented pattern.
  The 2026-05-16 sweep ran 60 trials in ~5 hr with 15 absorbed crashes
  using this architecture.
- Future projects in this org can adopt the same pattern without
  re-investigating the bug — see [[memory: blackwell-torch-dataloader-bug]].
- Per-trial subprocess isolation is also how Ray Tune, Kubeflow,
  SageMaker, etc. structure HP sweeps, so the pattern is portable
  rather than project-specific.

**Negative:**

- ~5 sec per-trial overhead for Python startup + data load. Acceptable
  for 3-5 min training trials; potentially noticeable for sub-30-sec
  trials (none in current scope).
- Top-k retraining post-sweep cannot run multiple architectures
  back-to-back in one process; must invoke per-arch. Already structured
  this way via `run_sweep.py --retrain-only`.
- Slightly more shell scripting (`run_sweep_loop.sh`) than a pure
  Python entry point would have.

## Open follow-ups

- **Filed upstream: https://github.com/pytorch/pytorch/issues/184062**
  (2026-05-17). The artifacts that went into the issue body are at
  `_meta/upstream/{ISSUE_TITLE.txt, ISSUE_BODY.md,
  torch-dataloader-gc-segfault-{repro.py, report.md}}`.
- Watch the upstream issue for triage. If a fix is merged and a torch
  release lands that claims a DataLoader / tensor GC fix on Blackwell,
  re-test in-process Optuna and — if stable — retire the subprocess
  wrapper. Record the change as a new ADR.
- Consider promoting `run_sweep_loop.sh` + `cleanup_failed_trials.py`
  to a reusable template under `_meta/templates/` for the next
  sweep-style experiment.
- The `c10::TensorImpl::~TensorImpl()` lead from the synthetic repro
  is the natural starting point for upstream investigators — points
  at PyTorch's CPU tensor destructor reached via Python cyclic GC.
