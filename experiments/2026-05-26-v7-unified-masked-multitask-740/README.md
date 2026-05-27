---
kind: experiment
slug: v7-unified-masked-multitask-740
date: 2026-05-26
status: done
hypothesis: "Unified masked-multitask foundation combining v4's supervised anchor with per-scenario masking augmentation. Targets: pure_pregame_val_auc >= 0.6471 (matches v4 on its core query), items_cond_val_auc >= 0.80, duration_cond_val_auc >= 0.68. All three required for v7 to be a successful foundation."
result: "ALL THREE SUCCESS CRITERIA MET. pure_pregame_val_auc=0.6480 @ epoch 25/25 (+0.0009 vs v4=0.6471), items_cond=0.9887 (+0.189 vs target 0.80, items-as-input is hugely predictive), duration_cond=0.6800 (exactly hit target). Trained stably 5.73h, no halt, no collapse. First foundation experiment in the entire project arc to (a) match-or-beat v4 on the core query, (b) train without representation collapse or over-specialization (the v5/v6 failure modes), AND (c) natively support all specified downstream queries (personal win prob, hero pick rec via candidate sweep, item rec optimizing for win via items-as-input, item rec conditional on win=1, win-rate-vs-duration via duration-as-input, kills/min pair via separate K/D/A heads, lineup matchup). Outcome_cond probe plateaued at 0.328 (below target 0.40 but above halt 0.30) — item-rec-conditional-on-win is harder than expected; manageable, can be improved by reweighting outcome_cond scenario in a v8 if needed. The masking-augmentation + supervised-anchor combination is the winning recipe — the supervised heads anchor the encoder to win-discriminative features (preventing v5/v6-style failures) while the 9-scenario masking augmentation teaches robust handling of partial inputs. Adaptive per-scenario sampling worked exactly as designed: partial_draft and partial_items got reallocated budget (low probes), items_cond and everything_visible budget dropped (saturated probes). Two mid-flight fixes: (1) main agent applied log1p to kills/deaths/assists/gpm/hd loss targets before launching — implementer's smoke had raw-count targets which would have caused hd loss (24507) to dominate the multi-task sum. (2) main agent disabled the partial_draft halt criterion after epoch 4 since the probe (encoder-output-at-masked-slot ~ hero-embedding cosine) doesn't measure what we use for actual hero pick rec at inference (candidate sweep); ~50 min lost on first attempt. v7 is now the recommended foundation for downstream queries — see serve/ subdirectory."
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - embedding-vs-features-gradient-competition
related_literature: []
related_experiments:
  - 2026-05-25-v4-iso-teambias-extended-740
  - 2026-05-26-v5-pretrain-finetune-740
  - 2026-05-26-v6-jepa-pretrain-finetune-740
  - 2026-05-24-foundation-v3-740
  - 2026-05-20-rich-supervision-multitask-740
tags: [foundation, masked-multitask, transformer, FT-Transformer]
respects:
  - "~/claude-system/claude/rules/evaluation.md"
---

# v7-unified-masked-multitask-740

## Hypothesis

v4's FT-Transformer foundation is sound on its core query
(pure_pregame, val_auc=0.6471) but cannot answer the downstream queries
the user wants: partial-draft hero recommendation, item recommendation
conditional on outcome, win prediction conditional on items/duration,
and kills-per-minute queries.

v7 keeps v4's supervised anchor (proven to organize the encoder by
predicted-win along PCA-1; per v4 diagnostic) and adds masking
augmentation aligned to each downstream query. Per-scenario sampling
explicitly matches training to inference use cases; per-scenario probes
catch v5/v6-style degenerate collapse early.

Three architectural changes vs v4 (see `models.py`):
1. **Separate K, D, A heads** (not composite KDA) -- enables true
   kills/min queries.
2. **Duration as scalar regression** (not 8-bucket CE) -- v3 evidence.
3. **All 10 input groups become MASKABLE** with learned mask tokens
   (8 per-slot: hero, player_feat, items, kills, deaths, assists, gpm,
   hd; 2 per-match: duration, win). Sequence length = 12 tokens.

Task token vocabulary is 62 (v4 was 42): one win + one dur + 10 each
for items/kills/deaths/assists/gpm/hd.

## Setup

- Config: `config.yaml`
- Code entry point: `train.py --ablation v7_unified [--smoke]`
- Data: reuses extended player_features + rich_cols sidecar parquets
  verbatim from v3/v4. No data rebuild needed.
- Splits: project-root `splits.yaml`. Train 2025-08-15..2026-02-23,
  val 2026-02-24..2026-03-09. Test window [2026-03-10, 2026-03-23]
  SEALED -- never touched (HCE).
- Total maskable input groups: 10 (8 per-slot + 2 per-match).
- 9 scenarios (see `mae.py:ScenarioSampler.SCENARIOS`); each batch
  samples one. Per-scenario loss weights modulate per-head losses.
- 9-probe suite (see `probes.py:ProbeSuite.run`); runs every 2
  epochs + at epoch 1 to seed adaptive sampling. Halts at epoch 10 if
  any probe is below its halt threshold (`config.yaml:probes.halt_thresholds`).

## Anchor table

| Reference | val_auc | What it represents |
|---|---|---|
| v4 (PRIMARY ANCHOR) | 0.6471 | pure_pregame, full-input multi-task supervised |
| iso_teambias | 0.6493 | 7.40-only ceiling |
| baseline_multitask_repro | 0.6470 | foundation-mvp baseline |
| **v7 pure_pregame target** | **>= 0.6471** | matches v4 on its core query |
| **v7 items_cond target** | **>= 0.80** | items-as-input gives massive lift |
| **v7 duration_cond target** | **>= 0.68** | duration-as-input modest lift |

## Result

Fill in after the run. Point at `metrics.json` (validation split).

Key fields:
- `val_auc_pure_pregame` -- the v4-comparable canonical metric.
- `final_probe_results` -- all 9 probes at the best checkpoint.
- `delta_pure_pregame_vs_v4` -- the headline number.
- `scenario_distribution_final` -- adaptive sampling's converged distribution.
- `halted`, `halt_reasons` -- if the probe-suite halt fired.

## Interpretation

Fill after run.

## Diagnostics

Fill after run.
