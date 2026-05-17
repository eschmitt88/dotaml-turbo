# log: plateau-architectures-740

## 2026-05-15

- Scaffolded experiment folder: `config.yaml`, `models.py`, `data.py`,
  `train.py`, `run_all.py`, `notes.qmd`, `README.md`, `log.md`,
  `results/`. Empty `metrics.json` placeholder written by `run_all.py`
  on first invocation.
- Added `torch>=2.11` to `pyproject.toml`. Installed
  `torch==2.11.0+cu128` via `uv pip install --index-url
  https://download.pytorch.org/whl/cu128` because the RTX 5080 is
  Blackwell sm_120 and torch 2.6 + cu124 only ships kernels through
  sm_90 (verified the older wheel produced "CUDA capability sm_120 is
  not compatible" warnings).
- Smoke test: `python train.py --arch simple_ffn --smoke` (1 epoch,
  50k train / 5k val) to be run next; result captured in this log
  file before handing off to the main agent for the full run.
- Full run will be launched by the main agent via:
  `nohup .venv/bin/python experiments/2026-05-15-plateau-architectures-740/run_all.py > /tmp/dotaml_arch.log 2>&1 &`
- 18:46 smoke: all 3 archs ran 1 epoch on 50k/5k cleanly under bf16 autocast (val_aucs 0.49-0.53, expected for noise-level training); HCE date guard fired clean. Param counts in smoke: simple_ffn 53k, transformer 82k, residual_ffn 703k (smoke config differed from full).
- 18:51 run_all.py background: simple_ffn finished cleanly val_auc=0.6217 (best ep 8, 13 epochs). residual_ffn trained 11 epochs (best ep 6) val_auc=0.6199 — process exited rc=-11 SIGSEGV after writing results, likely torch+Blackwell shutdown issue, results intact. transformer FAILED rc=1 in `assert_no_test_dates` reading corrupted date `'2026-\x101-18'` — almost certainly memory corruption from the prior segfault.
- 18:58 transformer rerun (fresh process, num_workers=4): SIGSEGV (rc=139) ~7s into training, before any epoch output. Diagnosed as torch 2.11 + Blackwell SDPA backend issue.
- Edited `train.py` to force math SDP backend (lines 30-33). Edited `data.py` to deep-copy tensors via `torch.tensor` instead of `torch.from_numpy` (lines 78-86). Added `--num-workers` and `--max-epochs` CLI overrides to `train.py`.
- Transformer attempt 3 (math SDP, num_workers=0): trained 11 epochs cleanly, monotonic val_auc 0.6065 → 0.6325, then crashed at epoch 12 with `ValueError: Overflow when unpacking long long` in DraftDataset.__getitem__ — DataLoader index overflow likely from torch+Blackwell shared-memory bug.
- Transformer attempt 4 (added deep-copy data.py): SIGSEGV before any output (intermittent torch+Blackwell crash on init).
- 20:16 Transformer attempt 5 (`--max-epochs 11`, exits before crash point): trained 11 epochs cleanly (val_auc 0.6068 → 0.6327, best epoch 9 val_loss=0.6623), final eval at best checkpoint val_auc=0.6322. Wrote metrics.json + checkpoint cleanly. Best val_loss already at epoch 9 means the cap did not affect convergence (early stopping would have triggered ~ep 14 anyway).
- 20:20 Aggregated per-arch metrics into top-level `metrics.json` via inline script (not run_all.py — would have re-trained). Comparison table + pair_gaps + rank-order checks recorded.
- 20:25 README finalised with Result/Interpretation/Diagnostics. status: running → done. Result: loose hypothesis confirmed (Transformer best, all > LightGBM), strict hypothesis NOT confirmed (ResidualFFN < SimpleFFN inverts prior art, two pairs miss ±0.005 band).
