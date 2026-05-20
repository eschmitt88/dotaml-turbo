---
kind: proposal
slug: upstream-data-cleanup-740
date: 2026-05-19
status: implemented
experiment: experiments/2026-05-19-upstream-data-cleanup-740/
hypothesis: "Patching the upstream defect in `experiments/2026-05-18-player-features-prepatch-740/build_features.py` that produces ~6.5K fp32-max sentinel cells in the `p{p}_smoothed_winrate_hero` columns of the prepatch parquet (0.005% of 130M cells) and rebuilding the parquet leaves the combined-model val_auc within ±0.001 of the current ceiling 0.6477 (i.e., 0.6467 ≤ val_auc ≤ 0.6487), while eliminating the downstream sanitization workaround in `data.py`. A deviation outside that band would indicate the sentinels were systematically biasing the prior result."
rationale: >
  The prior two experiments (`player-features-prepatch-740` and
  `transformer-plus-features-740`) both relied on a `data.py`-side
  sanitization step (clip per-feature values outside physical bounds,
  replace with per-feature median) to neutralize ~6.5K corrupted
  cells in the `smoothed_winrate_hero` columns. The sanitization
  was a workaround, not a fix; the upstream parquet still carries
  the corrupted values, and any future consumer (player embeddings,
  ensembling, ranking experiments) would need to repeat the
  workaround or risk silent NaN propagation. The corruption volume
  is too small (~0.005% of cells) to plausibly bias the headline
  AUC by >0.001, but the downstream cleanliness matters because
  (a) the player-embedding experiment is next in queue and will
  consume the same parquet, and (b) any production-shaped artifact
  derived from this pipeline must not depend on a load-time band-aid.
  Likely culprits in `build_features.py:138` (`hero_global_w[h]/n`,
  defaults n=0 to global_prior but a numeric edge in counter dtypes
  could escape) or `build_features.py:150` (`(hero_alpha*hero_prior
  + hw) / (hero_alpha + hn)`, where hero_alpha=5.0 means the
  denominator is bounded ≥ 5.0 — so the bug is elsewhere, possibly
  in dtype conversion or the prior-fallback path; root cause to be
  isolated by the implementer).
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[experiments/2026-05-18-player-features-prepatch-740]]"
  - "[[experiments/2026-05-18-transformer-plus-features-740]]"
  - "[[experiments/2026-05-19-transformer-plus-features-extended-740]]"
expected_metric:
  name: val_auc
  target: 0.6477  # within ±0.001
  direction: equality-band
design_sketch:
  - Copy `experiments/2026-05-18-player-features-prepatch-740/build_features.py` into the new experiment folder.
  - **Root-cause the fp32-max source.** Add diagnostic instrumentation to the aggregator's `snapshot()` path; rebuild on a TINY subset (first ~50K matches) until at least one sentinel cell is reproduced; identify the offending code path; patch with explicit dtype handling and/or a defensive clamp INSIDE `snapshot()` (not as a post-hoc workaround in `load_arrays`).
  - Add a post-build assertion to the rebuild script: every column of the output parquet must have min/max within configured physical bounds; abort the build if any cell escapes (no silent overflow ever again).
  - Rebuild prepatch parquet to a NEW canonical path: `data/snapshots/7.40-2025-12-16/processed/player_features_prepatch_clean/{train,val}.parquet`. Do NOT overwrite the prior path — the prior parquet stays as a reference for delta comparisons.
  - Ablation A (LightGBM-only): re-run `player-features-prepatch-740`'s `features_only` ablation against the clean parquet. Reuse its `train.py` verbatim with a single `--processed-dir` override. Compare val_auc to prior 0.6256.
  - Ablation B (Transformer + features, extended-training cap): re-run `transformer-plus-features-extended-740`'s `transformer_plus_features` ablation against the clean parquet. Reuse its `train.py` verbatim; REMOVE the `load_arrays` sanitization shim (or assert that it never fires — should be a no-op now). Compare val_auc to prior 0.6477.
  - HCE strict: train ≤ 2026-02-23, val ≤ 2026-03-09, test [2026-03-10, 2026-03-23] sealed.
  - Per-trial subprocess isolation via run_all.sh for both ablations (precedent: 5 of 6 prior Transformer experiments needed at least one retry; budget for it).
  - Save the cleaned parquet under DVC tracking and update the cleanup-relevant `dvc.yaml` stage so future experiments can `dvc pull` the clean path.
  - **Output for the next experiment in queue**: a clean, sentinel-free prepatch parquet at the new path that `player-embedding-prelim-740` can consume directly.
risks:
  - **Root-cause time uncertain.** The sentinel pattern (specific to `smoothed_winrate_hero`, ~6.5K cells, 0.005% rate) doesn't match an obvious divide-by-zero on the visible code paths (both denominators are bounded). May involve dtype edge cases, defaultdict insertion ordering, or an unprintable hero_id. If root-causing takes >1 h of compute, fall back to a defensive clamp in `snapshot()` plus a flag for downstream consumers to know.
  - **Parquet rebuild is the long pole.** ~3 h wall for the full prepatch+inpatch aggregation; if it OOMs again (precedent: `coplay`+`unique_heroes` were dropped from the prior run for this reason), the dropped features need to stay dropped or a different rebuild strategy is needed.
  - **No-change result.** If val_auc lands exactly at 0.6477 (or within noise), the experiment is "successful by being uneventful" — the hypothesis was just that sentinels weren't biasing results materially. Worth recording but not a headline. Downstream cleanliness is the real payoff.
  - **Two ablations + 3 h build = ~4 h total wall**, well under `budget.yaml`'s 24-h ceiling. Disk: clean parquet ~same size as prior (~10-15 GB), well under 500 GB ceiling. No external API quota.
related_prior:
  - 2026-05-18-player-features-prepatch-740
  - 2026-05-18-transformer-plus-features-740
  - 2026-05-19-transformer-plus-features-extended-740
estimated_runtime: "≈4 h total on RTX 5080: ~3 h prepatch+inpatch parquet rebuild (CPU-bound aggregator, single-pass over ~30-40M matches) + ~30 min LightGBM training + ~25 min Transformer training + diagnostic + overhead. Disk delta: ~10-15 GB for the clean parquet. Well under budget.yaml ceilings."
---

# Upstream data cleanup — root-cause the sentinel, rebuild cleanly

The prior two big-deal experiments (`player-features-prepatch-740` at val_auc=0.6256 and `transformer-plus-features-740`/`-extended` at 0.6452/0.6477) both depended on a load-time sanitization step in `data.py` to neutralize ~6.5K fp32-max cells (3.4e38) in the `p{p}_smoothed_winrate_hero` columns of the prepatch parquet. The sanitization is correct on its own terms — it clips out-of-bounds values to the per-feature median — but it is a workaround, not a fix, and the corrupted parquet remains the canonical artifact on disk.

This experiment exists because the next experiment in the queue (`player-embedding-prelim-740`) is going to consume the same prepatch parquet, and the experiment after that (whatever it turns out to be) probably will too. Carrying a dtype-overflow band-aid through every downstream consumer is the kind of compounding tech-debt that derails arcs three months in. Fix it once at the source.

The corruption volume is so small (0.005% of cells) that the AUC effect is expected to be statistical noise — well within the ±0.001 equality band. If the result lands outside that band, that's the genuinely interesting outcome (it would mean the load-time sanitization was actually masking material bias in opposite directions for the two corruption regimes). If it lands inside, we get a clean parquet for downstream and a no-regression confirmation that the prior 0.6256 and 0.6477 numbers are trustworthy.

Two result forks:

- **val_auc on combined model in [0.6467, 0.6487] (NO-REGRESSION confirmed).** Sentinels were not biasing the prior results; the clean parquet replaces the dirty one as canonical; downstream experiments drop the `data.py` sanitization shim. The 0.6477 reference stands. Move on to `player-embedding-prelim-740`.
- **val_auc outside that band.** The sentinels were biasing results. The direction (up or down) tells us whether the prior 0.6477 was understating or overstating actual model quality. Either way, the clean number is the new reference and the prior comparison table in `concepts/draft-prediction-plateau.md` needs an asterisk + footnote.

The root-causing step is the highest-variance part of this experiment. The visible numeric paths (`build_features.py:138` and `:150`) both have bounded denominators with the configured `hero_alpha=5.0` — so the bug is in something less obvious: dtype edge cases (counter int overflow under defaultdict?), uninitialized memory paths in pyarrow write, an unexpected hero_id sentinel value. The implementer should reproduce on a small subset first rather than rebuild the full 30-40M-match aggregation only to discover the patch didn't work.
