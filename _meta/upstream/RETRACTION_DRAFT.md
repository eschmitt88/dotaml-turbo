## Update — retracting the report

I owe an apology and a correction. After further investigation, **I no longer believe this is a PyTorch bug.** The root cause was silent RAM bit-flips on my workstation, masquerading as a DataLoader/GC interaction.

### What I found

Continued reliability issues on the same machine across totally unrelated workloads (PyArrow parquet reads, gcc compiling NVIDIA's DKMS kernel module) prompted a dedicated hardware investigation. Results:

| Diagnostic | EXPO 6000 MT/s | JEDEC 4800 MT/s |
|---|---|---|
| `memtester 30G 1` | Hundreds of single-bit-flip failures, clustered at specific physical addresses (0xc3xxxxxx, 0xebxxxxxx, 0x142xxxxxx) | **0 failures** |
| `stress-ng --vm 4 --vm-bytes 5G --vm-method all --timeout 1800` | 2,110 bit errors across 4 workers (all failed) | **0 bit errors** |
| PyArrow write→read consistency, 100 iters with 25 GB memory pressure | 2 silent data corruptions on int64 columns (one read returned `max = 141,837,450,994,669` from a column whose true values are bounded `[0, 10⁹]`) | **0 mismatches** |
| Kernel events during stress | `BUG: Bad page map`, `scheduling while atomic`, 5+ hours of RCU stalls → system hang requiring force-restart | None |

### Hardware

- AMD Ryzen 9 9950X (Zen 5)
- Corsair CMK96GX5M2E6000C36 (2 × 48 GB DDR5-6000, EXPO-rated)
- ASUS ROG CROSSHAIR X870E HERO, BIOS 1401 (May 2025)
- **Non-ECC** — so every bit-flip was undetectable at the OS level.

96 GB DDR5 at 6000 MT/s on Ryzen 9000 exceeds AMD's officially supported 5600 MT/s for 2-DIMM configurations. The memory controller wasn't reliably retaining bits at that speed/voltage.

### Why the symptom looked like a PyTorch bug

In a Python+torch workload that allocates and frees many tensors per epoch, the chance of a flipped bit landing in heap metadata (a free-list pointer, a refcount, a small-buffer-allocator size class field) accumulates quickly. When the cyclic GC then walks that corrupted heap, the typical failure mode is exactly what we reported: `free(): invalid pointer` followed by a segfault during tensor teardown — which puts the crash stack inside `DataLoader.fetch` or `default_collate` because that's where most of the per-batch allocation/freeing happens.

All the "evidence for an upstream bug" was consistent with hardware-induced heap corruption:

- **Reproduces across torch 2.9 / 2.11 / 2.12** — yes, because the bit-flips don't care which user-space library is touching the memory.
- **`MALLOC_CHECK_=3` partial masking** — different allocator layout, so the flipped bits sometimes landed in benign positions instead of free-list metadata. The minimal-repro run that surfaced a `c10::TensorImpl::~TensorImpl()` trace under `MALLOC_CHECK_=3` was already pointing at hardware corruption; I didn't read it that way at the time.
- **Couldn't reproduce on your 4080 Laptop** — different RAM, different memory controller, different motherboard.
- **NVIDIA's DKMS kernel module compile** also intermittently segfaulted on this machine during a routine `apt install` (succeeded on the automatic retry). A totally different binary doing totally different work, same intermittent-segfault pattern — that was the first observation that finally made me suspect hardware.

### Resolution

Disabled EXPO in BIOS; RAM now runs at JEDEC default 4800 MT/s. The same workload that previously crashed ~21% of trials has run sustained Transformer training plus a full validation suite (memtester + stress-ng + PyArrow round-trip with 25 GB pressure) without a single segfault or bit error.

The "per-trial subprocess isolation" workaround we'd built was effective for an unrelated reason: each new subprocess starts with a clean heap, so accumulated bit-flip damage didn't persist across trials.

### Recommended action

Please close as "not a bug" / "external (hardware)". Sorry for the noise and the time. Three independent torch versions reproducing the same stack felt like very strong evidence at the time. Next time `memtester` goes in the diagnostic checklist alongside "rule out CUDA / driver / pyarrow / dataset."
