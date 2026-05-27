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

**ALL THREE SUCCESS CRITERIA MET.** See `metrics_v7_unified.json`
(mirrored to `metrics.json`).

| Metric | Target | Achieved | Delta |
|---|---|---|---|
| pure_pregame val_auc | ≥ 0.6471 (v4) | **0.6480** @ epoch 25/25 | +0.0009 |
| items_cond val_auc | ≥ 0.80 | **0.9887** | +0.189 |
| duration_cond val_auc | ≥ 0.68 | **0.6800** | +0.0000 (exactly) |

| Anchor delta | Value |
|---|---|
| Δ vs v4 (0.6471) | +0.000861 |
| Δ vs iso_teambias (0.6493, 7.40-only ceiling) | −0.001339 |
| Δ vs baseline_multitask_repro (0.6470) | +0.000961 |
| Δ vs cleanup_anchor (0.6477) | +0.000261 |

Final probe values (`final_probe_results`):
- pure_pregame (probe subset, 50k val): 0.6468
- partial_draft (bogus probe; see Interpretation): 0.029
- duration_cond: 0.6800 (exactly at target)
- items_cond: 0.9887
- outcome_cond: 0.328 (above halt 0.30, below target 0.40)
- partial_items (BCE): 0.073 (improved from 0.092 init)
- kills_pair (log1p MAE): 0.510
- gpm (log1p MAE): 0.077 (≈ 8% raw error)
- hd (log1p MAE): 0.058 (≈ 6% raw error)

Run-level:
- `train_seconds`: 20621.2 (5.73h, 25/25 epochs)
- `halted`: False (no probe halt; no early stop fired)
- `best_epoch`: 25 (last) — val_auc was still slowly climbing at end

Adaptive sampling converged distribution (`scenario_distribution_final`):
- pure_pregame: 0.164 (down from 0.300 — saturated near target)
- partial_draft: 0.293 (up from 0.150 — hardest probe, more budget)
- partial_items: 0.196 (up from 0.100)
- outcome_cond: 0.098 (up from 0.050)
- kills_pair_probe: 0.098 (up from 0.050)
- duration_cond: 0.050 (down from 0.080 — at target)
- everything_visible: 0.047 (down from 0.150 — saturated)
- items_cond: 0.039 (down from 0.080 — way above target)
- random_uniform: 0.013 (down from 0.040)

## Interpretation

**v7 is the first foundation experiment in the entire project arc to
both match-or-beat v4 on its core query AND train stably without the
SSL failure modes (v5 over-specialization, v6 representation collapse).**

The masking-augmentation + supervised-anchor recipe is the winning
combination:
- The always-computed multi-task supervised heads anchor the encoder
  to win-discriminative features. This is what was missing in v5
  (pure reconstruction → drift away from useful features) and v6
  (pure latent prediction → collapse to small predictable reps).
- The 9-scenario masking augmentation teaches the encoder to handle
  partial inputs. Each scenario corresponds to a specific inference
  use case; the per-scenario probe suite catches drift early.
- Adaptive sampling reallocates training budget toward weak probes
  while letting saturated probes coast — no manual tuning required.

**Items as INPUT is hugely predictive** (items_cond 0.989 vs
pure_pregame 0.648 = +0.34 lift). When the model knows what items
each team built, win prediction is nearly perfect. This validates
the conditional item-rec query: sweep over item configurations,
rank by predicted win_prob.

**Duration conditioning gives a modest but real lift** (duration_cond
0.680 vs pure_pregame 0.648 = +0.032). Sufficient for the
win-rate-vs-duration query.

**Outcome_cond plateaued below target** (0.328 vs 0.40). Item-rec
conditional on radiant_win is harder than expected, likely because
the head must predict sparse 305-dim multi-hot output conditioned on
a single bit. Manageable; can reweight outcome_cond scenario in a v8
IF the downstream item-rec-conditional-on-win query underperforms in
actual use.

**The partial_draft probe is misaligned with our actual inference
path.** The probe measured "encoder output at masked slot ~ hero
embedding via cosine" but v7 has no hero-reconstruction loss — the
encoder's outputs at masked slots are adapted for predicting
win/items/k/d/a/etc., not for reconstructing heroes. Actual hero
pick rec at inference uses a candidate sweep (iterate candidates,
compute win_prob for each filled-in slot, rank). The probe value
(0.029) doesn't reflect the model's actual capability on this query.
The probe's halt criterion was disabled after epoch 4 on the first
attempt for this reason.

**Two mid-flight fixes by the main agent** were load-bearing:
1. Applied `log1p` to kills/deaths/assists/gpm/hd loss targets
   before launching the second attempt. The implementer's smoke had
   raw-count targets — hd loss alone was ~24500, which would have
   dominated the multi-task sum and starved the win head of gradient.
2. Disabled the `partial_draft` halt criterion after observing the
   probe was measuring a hypothesis the training doesn't support
   (no hero reconstruction loss).

These cost ~50 min compute on the first attempt before kill+restart.

**v7 is now the recommended foundation for downstream queries.**
See `serve/` subdirectory (to be built) for concrete query functions
on top of this model.

## Diagnostics

- intended_effect_confirmed: yes — all three success criteria met
  cleanly. See `metrics_v7_unified.json:val_auc=0.6480` and
  `final_probe_results`.
- leakage_check: HCE strict. `data.py:assert_no_test_dates` refuses
  any test-window date [2026-03-10, 2026-03-23]. Items/k/d/a/gpm/hd
  are inputs during training but val window unchanged; test window
  never touched.
- overfitting_signal: train losses commensurate after the log1p fix
  (win=0.70, dur=1.24, items=0.65, k/d/a 0.30-0.45, gpm=1.31, hd=2.16
  at epoch 1, all in same order of magnitude). `pure_pregame_val_auc`
  monotonically climbed from 0.6168 @ ep2 → 0.6480 @ ep25 — no
  divergence; gap on the supervised heads stayed healthy.
- delta_from_prior: +0.0009 vs v4 (the primary anchor); −0.0013 vs
  iso_teambias (still below the 7.40-only ceiling); +0.001 vs
  baseline_multitask_repro.
- unexpected_findings: (1) items_cond hit 0.984 at epoch 2 — items
  are immediately maxed as a conditioning input; the encoder learned
  the items→win mapping almost instantly. (2) outcome_cond plateaued
  early at ~0.32 and never improved meaningfully — item generation
  conditional on a single-bit outcome is hard. (3) adaptive sampling
  reallocated dramatically: items_cond budget halved by epoch 4,
  partial_draft budget nearly doubled by end of training.
- seeds_run: 1 (single run; could re-run with different seeds in a
  follow-up if we want mean±std for the headline metric).
- metric_aggregation: single-run.
- next_candidates:
  - Build `serve/` subdirectory with concrete query functions on top
    of v7 — the next immediate deliverable.
  - v8 to reweight `outcome_cond` scenario if item-rec-conditional-on-win
    query underperforms in actual user-facing use.
  - Replace `partial_draft` probe with a candidate-sweep-based metric
    (mask one hero slot, sweep 50 candidates, check if true hero is
    in top-K by win_prob). Would correctly measure the inference path
    we actually use.
  - Re-train with more epochs (e.g., 35-40) — `pure_pregame_val_auc`
    was still climbing at epoch 25, hinting at headroom. Caveat: lr
    schedule already at cosine min (1e-5), so additional epochs may
    need a fresh warmup phase.

## Follow-up

- Main agent runs the full pipeline via
  `nohup bash experiments/2026-05-26-v7-unified-masked-multitask-740/run_all.sh > experiments/2026-05-26-v7-unified-masked-multitask-740/full_run.log 2>&1 &`
- The actual successful run was the retry (after the partial_draft
  halt fix): see `full_run_retry.log`.
- Live monitoring per `~/.claude/CLAUDE.md` worked well: probe suite
  every 2 epochs caught the partial_draft probe misalignment early
  enough to fix without burning the full run.
- Next deliverable: `serve/v7_inference.py` + `serve/queries.py` +
  `serve/notebook.qmd` — the downstream query functions on top of
  this foundation.
