---
kind: proposal
slug: foundation-component-isolation-740
date: 2026-05-23
status: implemented
experiment: experiments/2026-05-23-foundation-component-isolation-740/
result: "CLEAN ATTRIBUTION. iso_uwso HALTED early at epoch 2 per new monitoring discipline (omega→1.000 to items by epoch 2, val_auc=0.5276, train_win INCREASING — UW-SO is the saboteur; the per-task initial-loss-normalization fix attempt was insufficient because items per-class BCE converges to ~0.08 within ~100 batches, capturing a too-low L_k_init baseline). iso_pmae val_auc=0.6464 @ best=21 (-0.0006 vs anchor) — SAFE with the EMA-teacher fix (Bug A root cause was the original BYOL/JEPA student=teacher collapse). iso_teambias val_auc=0.6493 @ best=14 (+0.0023 vs anchor) — HELPFUL; the ~64-param (team_query, team_key) attention bias gives a real lift. v3 design: keep canonical hero sort + (team, team) bias + PMAE with EMA teacher; drop UW-SO, revert to multitask-740's hand-tuned α (1.0/0.15/0.3/0.1); test patch token on broader cross-patch data. Live monitoring per the new discipline saved ~4-5h by halting iso_uwso early."
hypothesis: "Of the four new components added in `foundation-mvp-740` (PMAE auxiliary objective, UW-SO loss weighting, (team_query, team_key) attention bias, patch_id token) on top of the working multitask baseline, AT LEAST ONE introduces the training instability that caused val_auc to collapse to 0.5058 (random). Component-isolation ablations — each new component added individually on top of the known-good 5M-param `baseline_multitask_repro` — will identify which component(s) are the saboteur(s) AND which are safe. After isolation, a `foundation-v3` experiment can re-introduce only the safe components plus targeted fixes for the broken ones."
rationale: >
  `foundation-mvp-740` (2026-05-22) returned a clean diagnostic
  result: the `baseline_multitask_repro` ablation hit val_auc=0.6470
  (within noise of the cleanup anchor 0.6477054 — scaling from 77K
  to 5M params is neutral), while both `foundation_mvp` (val_auc=0.5058)
  and `foundation_no_patch_token` (val_auc=0.4984) collapsed to
  random. The proposal's explicit anticipated fork said
  "the baseline ablation will tell us whether it's the scale or
  the design that broke things" — answer: design.

  Symptoms observed in foundation_mvp training logs:
    - train_win loss INCREASED across epochs 1-5 (0.6947 → 0.7020) —
      model actively anti-learning the primary task.
    - PMAE auxiliary mae_loss collapsed to 0.0000 from epoch 3 onward
      in foundation_no_patch_token — implementation suspected buggy
      (information leak or trivially solvable).
    - UW-SO temperature T converged to ~0.45; with raw per-task loss
      scales differing by 30× (items per-class BCE ~0.07 vs duration
      CE ~2.1), the softmax(1/L/T) form over-weights low-loss tasks
      by similar magnitude, drowning the win head's gradient.

  This experiment runs 3 ablations, each adding ONE new component
  on top of `baseline_multitask_repro`'s known-working configuration.
  Cleanly attributes the failure. Two of the three may pass and one
  may fail; the failing component(s) become the focus of v3's fixes.
reads:
  - "[[experiments/2026-05-22-foundation-mvp-740]]"
  - "[[experiments/2026-05-20-rich-supervision-multitask-740]]"
  - "[[literature/papers/kim2024predict]]"
  - "[[literature/papers/kirchdorfer2024analytical]]"
  - "[[literature/papers/bi2022pangu]]"
  - "[[concepts/masked-modeling-tabular]]"
  - "[[concepts/uncertainty-weighted-multitask]]"
  - "[[concepts/attention-bias-positional]]"
expected_metric:
  name: val_auc
  target: 0.6470
  direction: each-ablation-near-baseline (a component is "safe" iff its ablation lands within [0.6440, 0.6500])
design_sketch:
  - "**Three ablations**, each = `baseline_multitask_repro` config + ONE new component added:"
  - "  • `iso_uwso` — adds UW-SO loss weighting. All other knobs match baseline (no PMAE, no patch token, no team bias). Tests whether UW-SO alone destabilizes training."
  - "  • `iso_pmae` — adds PMAE auxiliary objective. All other knobs match baseline. Tests whether PMAE alone destabilizes."
  - "  • `iso_teambias` — adds (team_query, team_key) 2×2 attention bias. All other knobs match baseline. Tests whether the bias alone destabilizes."
  - "**Reuses ALL `foundation-mvp-740` infrastructure** (data loaders, sidecar, vocab, model classes, loss modules). Just flips three boolean config flags per ablation. No new code unless a specific bug needs patching mid-experiment."
  - "**Bug-fix BEFORE running PMAE ablation**: read `mae.py` from foundation-mvp-740 carefully. The mae_loss → 0 collapse strongly suggests an information leak: either the masked tokens are reachable via a residual / skip path the loss is reading from, OR the mask values default to legal token IDs the model can predict trivially. Diagnose + patch. If a fix isn't obvious in 30 min of inspection, run the PMAE ablation anyway and document the bug for further investigation."
  - "**Sanity check for UW-SO**: log per-task ω weights every epoch. Verify they're not collapsing to single-task-dominant. If they are, the loss-scale-normalization bug is real — apply per-task initial-loss normalization (divide each L_k by L_k at epoch 1) BEFORE the UW-SO softmax."
  - "**LIVE MONITORING** (new per `~/.claude/CLAUDE.md` 'Monitoring long-running ML jobs'). Poll log every 10 min. Halt early on a PATTERN of 3+ consecutive epochs of: train loss increasing, val_auc at random, NaN, or wildly imbalanced multi-task weights. Don't burn 5h+ on a clearly broken run."
  - "**Training**: identical recipe to `baseline_multitask_repro` (Adam lr=1e-3 warmup → cosine to 1e-5; bf16 autocast; batch_size from foundation-mvp config; max_epochs=30, patience=5 on val_win_log_loss). `python -u` mandatory."
  - "**Wall budget**: each ablation ~3-5h based on baseline's 17-epoch wall time (2.7h) plus expected longer convergence; total ~10-15h sequential. Within `budget.yaml` 24h ceiling."
  - "**Diagnostics**:"
  - "  • Per-ablation val_auc trajectory (every epoch, all 4 task heads)."
  - "  • Per-ablation final val_metrics_at_best for win head."
  - "  • For `iso_uwso`: log per-task ω weights every epoch — diagnostic on the loss-weighting hypothesis."
  - "  • For `iso_pmae`: log mae_loss and mask-event count every epoch — diagnostic on the collapse hypothesis."
  - "  • Coverage-bucket val_auc for comparison to prior experiments."
risks:
  - "**Two or more components fail simultaneously** — possible but informative. If all three ablations break, the issue is in the shared infrastructure (e.g., shared decoder or task-as-token prompting) NOT in any individual component. That'd be a v3 architectural rethink rather than a fix."
  - "**A component fails subtly** (val_auc dips to 0.640 instead of crashing to 0.50) — harder to attribute. Diagnostic: compare per-task loss trajectories side-by-side; the bad component will show task-specific anomalies."
  - "**Compute budget**: 3 ablations × ~5h = ~15h sequential, plus monitoring overhead. Fits within budget.yaml's 24h. If first 2 ablations both fail, abort the third and switch to bug-diagnosis mode."
  - "**Component dependencies**: if PMAE requires UW-SO to work (e.g., needs auto-balancing because the MAE loss is on a different scale from supervised losses), `iso_pmae` will fail not because PMAE is buggy but because of the missing UW-SO. Mitigation: also include the inverse — `iso_no_pmae_only_uwso` is what `iso_uwso` already tests; if PMAE+UW-SO together work but PMAE alone doesn't, we know the dependency."
related_prior:
  - 2026-05-22-foundation-mvp-740
  - 2026-05-20-rich-supervision-multitask-740
estimated_runtime: "≈10-15h on RTX 5080 for 3 ablations sequential. With live monitoring, can halt early-failing runs and skip to the next, saving wall in the failure cases."
---

# foundation-component-isolation-740 — attribute the foundation-mvp failure

## Where this fits

`foundation-mvp-740` failed cleanly: baseline ablation worked (val_auc=0.6470, within noise of anchor), full design collapsed (val_auc=0.5058, random). The proposal's anticipated diagnostic fork resolved to "the architectural design is broken, not the scale." This experiment finishes the diagnosis: which of the four added components is the saboteur?

Approach: each ablation adds exactly ONE new component on top of the known-good `baseline_multitask_repro` configuration. If a component is safe, its ablation lands near 0.6470. If it's broken, its ablation collapses. After this experiment, the v3 design will re-introduce only the safe components, with targeted fixes for the broken ones.

## Specific bugs to fix BEFORE running each ablation

### PMAE (run `iso_pmae` only after this is fixed or documented)

The mae_loss → 0 collapse observed in `foundation_no_patch_token` strongly suggests an implementation bug. Three candidates:

1. **Information leak via residual/skip path**: the masked tokens are reachable somewhere in the encoder forward pass (e.g., via an unmasked side channel) so the reconstruction loss can be solved trivially.
2. **Mask values default to legal token IDs**: if the mask token replacement is e.g. zero or a special token that the model can predict from context too easily.
3. **Loss only counts the masked positions but the mask is empty**: the masking module fails to mark any positions as masked, so the loss is computed over an empty set and defaults to 0.

Read `mae.py` from `foundation-mvp-740` carefully. If the bug isn't obvious in 30 min of inspection, run the ablation anyway with extra logging (mask-event count per epoch + sample masked-position values) and resolve in v3.

### UW-SO (loss-scale normalization sanity)

The temperature T converged to ~0.45 in foundation_mvp. With raw per-task loss scales spanning 30×, the softmax(1/L/T) form over-weights low-scale tasks. Standard fix: normalize each per-task loss by its initial-epoch value BEFORE feeding to UW-SO, so all tasks start at ~1.0. Apply this normalization in the `iso_uwso` ablation. Log per-task ω weights every epoch to verify they're not collapsing to single-task-dominant.

## Live monitoring

Per the new `~/.claude/CLAUDE.md` rule "Monitoring long-running ML jobs": poll the log every 10 min during each ablation. Halt early if a PATTERN of 3+ consecutive epochs shows: train loss increasing, val_auc at random baseline (0.50), NaN/Inf in any loss component, or ω weights collapsing.

This is the lesson from `foundation-mvp-740`: catching the failure at epoch 3 would have saved ~80 min on that ablation, and recognizing the design needed v2 before launching the trailing two ablations would have saved another ~8h.

## Result interpretation

After all three ablations run:

| iso_uwso | iso_pmae | iso_teambias | Interpretation |
|---|---|---|---|
| pass | pass | pass | All components individually safe; failure was an INTERACTION. v3: re-introduce all four but watch for emergent instability. |
| fail | pass | pass | UW-SO is the saboteur. v3: fix loss normalization or revert to hand-tuned α. |
| pass | fail | pass | PMAE is the saboteur. v3: fix mae.py implementation. |
| pass | pass | fail | (team,team) bias is the saboteur. v3: debug bias interaction with attention. |
| 2+ fails | | | Multiple saboteurs; v3 is a redesign, not a fix. |
| all fail | | | Issue is in shared infrastructure (e.g., shared decoder, task-as-token prompting). v3 rethinks the multi-head architecture. |

This is a pure diagnostic experiment. No new ceiling claim — just attribution. The actual ceiling-pushing v3 builds on whatever this experiment finds.
