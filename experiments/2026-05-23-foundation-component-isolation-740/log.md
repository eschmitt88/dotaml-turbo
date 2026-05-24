# log: foundation-component-isolation-740

## 2026-05-23

- Scaffolded the experiment folder.
- Bug A (PMAE collapse) diagnosed in foundation-mvp-740's mae.py + train.py:
  teacher pass used shared model weights, admitting BYOL/JEPA-style
  representational collapse. Fix: EMA-teacher (momentum=0.996, stop-grad).
- Bug B (UW-SO loss-scale misapplication) confirmed from
  foundation-mvp-740/loss.py and history. Fix: per-task initial-loss
  normalization over first 100 batches before softmax.
- Three ablations configured: iso_uwso, iso_pmae, iso_teambias.
- Smoke pending.
## 2026-05-23 EARLY HALT — iso_uwso

At 18:37 UTC (epoch 2 of iso_uwso), halted iso_uwso ablation early per
the new "Monitoring long-running ML jobs" discipline. Reason:

- omega(win, dur, item, kda, gpm, hd) at epoch 1 = [0.002, 0.002, 0.988,
  0.002, 0.002, 0.002] — items already dominating 99% of the loss
- omega at epoch 2 = [0.000, 0.000, 1.000, 0.000, 0.000, 0.000] —
  complete collapse to single-task
- val_auc 0.5380 → 0.5276 (degrading, near random baseline)
- train_win 0.6951 → 0.7041 (INCREASING — model actively anti-learning
  the primary task)
- T auto-tuner 0.496 → 0.362 (sharpening softmax, accelerating collapse)
- Math-deterministic: omega=1.000 to items means win head's gradient
  is identically zero from epoch 3 onward; model cannot recover

The Bug B fix (per-task initial-loss normalization) DID NOT WORK.
Hypothesized cause: L_k_init captured after the ~100-batch init window,
by which point items per-class BCE has already converged toward
steady-state (~0.08, vs the random-init ~0.69 expectation). Normalization
then captures a too-low baseline, so subsequent L_k for items looks
"even smaller," driving omega_items → 1.

**iso_uwso ABLATION RESULT: UW-SO (as we've implemented it, with or
without our normalization fix) is broken on this multi-task setup.**
The fix attempt was insufficient. v3 should either: (a) revert to
hand-tuned alpha weights (multitask-740's 1.0/0.15/0.3/0.1 worked),
(b) try a different multi-task loss (e.g., GradNorm, PCGrad), or
(c) re-implement UW-SO with bounded omega and a higher minimum T.

Letting iso_pmae and iso_teambias continue (uw_so=False for both).
