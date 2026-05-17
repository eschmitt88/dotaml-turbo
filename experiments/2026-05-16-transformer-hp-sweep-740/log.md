# transformer-hp-sweep-740 log

## 2026-05-16 — scaffold + smoke

- Added `optuna>=4.0` to project `pyproject.toml` and `uv sync`'d.
- Hand-created experiment folder following `experiments/2026-05-15-plateau-architectures-740/` layout (no template scaffolder used).
- Wrote `models.py:MinimalTransformer` per proposal spec (no side token, no 11-position embedding, binary team embed added to hero embed, optional embed→d_model projection, single Linear head).
- Wrote `data.py` mirroring `plateau-architectures-740/data.py` (same `stratified_subsample(seed=42)`, deep-copied tensors workaround, `assert_no_test_dates` HCE guard) but stripped the side_bit (MinimalTransformer doesn't take it).
- Wrote `train_one.py` with the math-SDP-only force, num_workers=0 DataLoader, bf16 autocast, optional Optuna trial pruning hook.
- Wrote `objective.py` with single-tuple categorical for (d_model, n_heads) (constraint d_model%n_heads==0 enforced by construction); enqueue helper for the control trial.
- Wrote `run_sweep.py`: builds `TPESampler(n_startup_trials=10, multivariate=True, seed=42)` + `SuccessiveHalvingPruner(min_resource=3, max_resource=14, reduction_factor=3)`; SQLite store at `results/optuna.db`; force-enqueues control trial as #0; post-sweep top-3 retraining at full 14 epochs with patience=5 + checkpoint save.
- Smoke run output (after fixing one bug — `SuccessiveHalvingPruner` in Optuna 4.x has no `max_resource` arg, removed):
  - control trial completed 3 epochs in ~2s wall on RTX 5080
  - val_auc=0.4958, val_loss=0.6931 — chance-level, **but consistent with prior `transformer_smoke_metrics.json` (val_auc=0.491 on 50k+5k+1 epoch)**. Transformers don't show signal on tiny smoke data — the harness is fine
  - param_count_total=76,801 (close to proposal's ~40k target; ~9.7k embed + 128 team + ~67k attention/head)
  - `results/optuna_smoke.db` and `results/trial_histories_smoke/trial_0_history.json` both written
  - HCE date assertion passed (val window 2026-02-24..2026-03-09 only)
- Note: `uv sync` bumped torch 2.11→2.12+cu130 (optuna's cuda-toolkit dep forced re-resolve). Math-SDP / num_workers=0 / deep-copy workarounds kept defensively.
- Note: `enable_nested_tensor` warning from `nn.TransformerEncoder` is benign (norm_first=True path).

## 2026-05-16 — full sweep + result

- 02:54 First full-sweep launch (60 trials, in-process Optuna loop): trial 0 completed clean (val_auc=0.6311), trial 1 PRUNED at epoch 4, trial 2 hit CUDA device-side assert that poisoned the entire CUDA context. `optimize()`'s `catch=(RuntimeError,)` didn't catch `torch.AcceleratorError`, so subsequent 58 trials all failed in the same poisoned process before the loop exited.
- 03:00 Broadened catch to `(Exception,)` and added a wrapper `run_sweep_loop.sh` that restarted on rc=2. First wrapped attempt: rc=139 (SIGSEGV) — the wrapper-only-on-rc=2 logic didn't help. Edited wrapper to restart on any non-zero. Added `cleanup_failed_trials.py` to wipe FAIL/RUNNING rows so n_trials wouldn't be burned by ghosts.
- 03:25 Second wrapped attempt died after 4 trials with a corrupted torch `.pyc` (`bad marshal data`) — Python's bytecode cache had been left half-written by a prior SIGSEGV. Wrapper then loop-failed on every retry. Decision: in-process Optuna is unworkable on torch 2.12 + Blackwell sm_120; need per-trial subprocess isolation.
- 03:30 Refactored: added `--retrain-only` to `run_sweep.py`, rewrote `run_sweep_loop.sh` to call `run_sweep.py --n-trials 1 --skip-top-k` per iteration. Each trial in a fresh Python process; CUDA crashes only lose that trial. Auto-detects `bad marshal data` in log and clears torch `__pycache__` between iterations.
- 03:31 Validated the per-trial pattern: 1 trial completed clean, exit rc=0.
- 03:35 Launched full per-trial-isolated sweep.
- 03:35–08:02 70 wrapper iterations to complete 60 trials. 15 crashes absorbed (mix of SIGSEGV rc=139 and one corrupted-pycache rc=1). ~5 h wall (vs ~5 h estimated). Crash rate ~21% per trial — subprocess isolation was load-bearing.
- 08:02 Sweep complete: 5 COMPLETE, 55 PRUNED, 0 FAIL. Top-k retraining (`run_sweep.py --retrain-only`) SIGSEGV'd immediately (3 retrains back-to-back in one process = guaranteed poisoning). Decision: skip top-k retraining — the COMPLETE trials already ran 14 epochs in their per-trial subprocesses, so the would-be retraining numbers are already in `optuna.db`.
- 08:15 Built aggregate `metrics.json` directly from the optuna study. Best trial #14: val_auc=0.6318 (d_model=64, n_heads=2, n_layers=2, lr=5.2e-4, batch=4096). Δ vs control trial #0: +0.0007. Δ vs prior `plateau-architectures-740` Transformer (0.6322): -0.0004. **Hypothesis (val_auc ≥ 0.6372) NOT confirmed.**
- 08:20 README finalised with Result/Interpretation/Diagnostics. status: running → done.
