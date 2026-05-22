---
kind: experiment
slug: rich-supervision-multitask-740
date: 2026-05-20
status: done
hypothesis: "A multi-task Transformer (shared encoder over draft + player aggregates; separate heads for win / duration / items / KDA-GPM-damage) trained jointly with weighted losses lifts whole-val win val_auc by >= 0.001 over the cleanup-confirmed 0.6477054 reference (target >= 0.6487). Auxiliary outputs (duration curve, per-slot item recommendation) are useful regardless. HCE preserved (encoder sees only pre-game; rich-cols are training TARGETS)."
result: "CONFIRMED. multitask_all val_auc=0.6495 @ best_epoch=30 (+0.0018 vs cleanup anchor 0.6477054, +0.0008 over target 0.6487, +0.0022 vs same-data baseline_extended_clean sanity at 0.6473). Aux heads produced standalone-useful outputs: duration top1_acc=0.181 over 8 buckets (vs 0.125 random), item mAP@10=0.301 (mean precision 0.333 / mean recall 0.440). 4h 1m wall on JEDEC 4800 MT/s, zero retries, zero kernel events. After a multi-week saga of failed runs traced to RAM bit-flips at EXPO 6000 MT/s and a hardware fix, this is the first multitask training to complete end-to-end."
related_concepts:
  - "[[concepts/draft-prediction-plateau]]"
related_literature:
  - "[[literature/papers/hodge2017win]]"
tags:
  - multi-task
  - transformer
  - rich-supervision
  - items
  - duration
related_prior:
  - 2026-05-19-upstream-data-cleanup-740
  - 2026-05-19-transformer-plus-features-extended-740
  - 2026-05-19-player-embedding-prelim-740
respects:
  - "~/.claude/rules/evaluation.md"   # HCE rule
---

# rich-supervision-multitask-740

## Hypothesis

Three independent runs (`extended-740`, `cleanup-740`,
`player-embedding-prelim-740` sanity replication) pin the whole-val win
ceiling at `val_auc = 0.6477 +/- 2e-5` for the Transformer-over-draft-
plus-aggregated-features architecture family. The embedding-prelim
NULL result (16M added params, ZERO train-loss improvement) suggests
the binding constraint is **gradient-signal density**, not parameter
count.

This experiment tests the gradient-density hypothesis through a
different mechanism than the embedding lookup: shared encoder with
multi-task heads. Rich in-game telemetry (duration, items, KDA, GPM,
hero_damage) gives every match ~10-50x more bits of supervision than
the single radiant_win label, while keeping the encoder inputs
strictly pre-game.

Target: win `val_auc >= 0.6487` (`+0.001` over `cleanup-740`'s
`0.6477054`). Three result forks documented in the proposal.

## Setup

- Rich-cols build: `build_rich_cols.py` — walks raw turbo parquets
  filtered to clean-parquet `match_id`s, parses each row's `raw_json`,
  emits per-match `duration`, per-slot `kills/deaths/assists/gpm/xpm/
  hero_damage/net_worth` ints and a `pX_items` list-of-int32. HCE
  date guard at walk time. Multi-checkpoint defense (snapshot-time
  bound clamp, numpy-routed pyarrow write, row-group-stats post-write
  verification — NO full re-read; cleanup-740 OOM-killed twice on a
  re-read pass).
- Item vocab + duration buckets: `build_item_vocab.py` — streams the
  train sidecar row-group by row-group, tallies item IDs across all 10
  slot lists, keeps items with `>= 100` train occurrences, and computes
  8 quantile-based `duration` bucket edges from the train sidecar's
  duration column. Emits `results/item_vocab.json`.
- Trainer: `train.py` (forked from `cleanup-740/train_tfm.py`),
  `--ablation {win_only_sanity, multitask_all}`. Joint loss
  `L = α_w·L_win + α_d·L_dur + α_i·L_item + α_a·L_aux`, initial
  weights `1.0 / 0.5 / 0.3 / 0.1`. Early-stop on val win log-loss.
- Model: `models.py:MultiHeadTransformer` — `cleanup-740`'s
  `MinimalTransformerWithFeatures` encoder verbatim plus 4 heads. CLS
  pooling for win/duration, per-slot encoder outputs for item/aux.
- Data: `data.py` joins clean parquet + rich-cols sidecar on match_id.
  Builds win/dur_bucket/item_one_hot/aux normalized targets. HCE date
  assertion stays.
- Config: `config.yaml` (mirrors cleanup-740 shape, adds
  `rich_cols_*`, `item_vocab`, `duration_bucket`, `multitask_loss`,
  `transformer_ablations` blocks).
- Splits: project-root `splits.yaml` (test window
  `[2026-03-10, 2026-03-23]` sealed).
- Orchestration: `run_all.sh` (4 sequential steps, `MAX_RETRIES=3` on
  Transformer steps).

## Result

**HYPOTHESIS CONFIRMED.** Multi-task supervision lifts the win head
above the long-anchored single-head ceiling.

| ablation | val_auc | best_epoch | epochs_run | wall |
|---|---:|---:|---:|---:|
| baseline_extended_clean (sanity, single head) | 0.6473 | 28 | 30 | 30 min |
| **multitask_all (PRIMARY, 4 heads)** | **0.6495** | 30 | 30 | 4h 1m |

Deltas:

- vs cleanup-740 anchor (0.6477054): **+0.00179**
- vs same-data sanity (0.6473): **+0.0022**
- vs proposal target 0.6487: **+0.0008** (cleared)

The win head was **still trending upward at epoch 30** (val_auc 0.6485
@ ep 20 → 0.6489 @ ep 24 → 0.6495 @ ep 30; best_epoch was the last
epoch, no early-stopping fired). A longer training cap could yield
additional gain.

**Per-head val metrics at best epoch:**

- **Win** (CLS-pooled, BCE): auc=**0.6495**, acc=0.6087, log_loss=0.6537, brier=0.2314
- **Duration** (CLS-pooled, CE over 8 quantile buckets): top1_acc=**0.181**, brier=0.860 (random=0.125 → 45% above chance; useful but modest)
- **Items** (per-slot, BCE-multi-label over 305-item vocab): **mAP@10=0.301**, mean_precision_at_10=0.333, mean_recall_at_10=0.440 (over ~164K val slots — model picks 33% of actual end-game items in its top-10 per matchup)
- **Aux** (per-slot SmoothL1 on normalized KDA+GPM+hero_damage): MSE per dim = [1.047, 0.443, 0.913]

**Loss composition at best epoch** (post-α weighting):

- win × 1.0 = 0.6537 (65% of weighted_total 1.012)
- dur × 0.15 = 0.3044 (30%)
- item × 0.3 = 0.0221 (2%)
- aux × 0.1 = 0.0320 (3%)

The α_dur=0.15 (down from proposal's 0.5 per the smoke flag) kept the
joint loss roughly win-dominated rather than duration-dominated; the
smoke prediction was correct.

## Interpretation

This result resolves the encoder bottleneck question opened by the
embedding-prelim NULL. There are two distinct hypotheses for why the
0.6477 ceiling held across so many runs:

1. **Pre-game information bottleneck** — the encoder has already
   extracted everything pre-game data has to offer; only NEW input
   axes can lift it.
2. **Gradient-signal density bottleneck** — pre-game data carries more
   signal than the single radiant_win label allows the encoder to
   learn; richer supervision unblocks it.

The embedding-prelim NULL was consistent with #1 (more capacity
doesn't help). This experiment is consistent with #2 (richer
supervision DOES help). Both are partially true: the gain is modest
(+0.0022), confirming the encoder was close to its information limit
but not all the way there. The next experiments (anonymous-aware
modeling, hero-pair history, longer training at the new α weights)
will tell us how far below the true ceiling we still are.

**The aux heads are useful products in their own right** beyond the
win-head lift:

- The duration head answers "what's my win curve P(win | duration)?"
  conditional on draft (the bucketing discussion the user had with
  another agent applies — a future v2 with regression head could
  smooth the readout but the modest top1_acc=0.181 suggests
  duration is not strongly draft-determined; mostly conditioned on
  player skill which is partially captured via aggregates).
- The item head is a real matchup-aware item recommender at 30% mAP@10
  — useful as a personal-tool readout for "what do players typically
  end up with given THIS draft and my slot."

## Diagnostics

- intended_effect_confirmed: yes — multitask_all val_auc 0.6495 clears proposal target 0.6487 by +0.0008 AND beats same-data sanity baseline 0.6473 by +0.0022 (`metrics_multitask_all.json:val_auc`, `metrics_win_only_sanity.json:val_auc`).
- leakage_check: HCE date-window assertion live in `data.py`; train ends 2025-12-16..2026-02-23, val ends 2026-02-24..2026-03-09, both strictly < test_start 2026-03-10 (`metrics_*.json:train_date_max`, `val_date_max`). Encoder inputs are pre-game ONLY at both training and inference time; rich-cols (duration, items, KDA, GPM, hero_damage) are training TARGETS only, never encoder inputs. Verified by inspecting model forward signature.
- overfitting_signal: train win-log-loss=0.6479 val win-log-loss=0.6537 gap=0.0058 at best_epoch=30. Comparable to extended-740 (gap 0.0052). Modest, not overfit. The win head was still improving at epoch cap (val_auc 0.6488 @ ep 25 → 0.6495 @ ep 30); a longer cap may help.
- delta_from_prior: vs `2026-05-19-upstream-data-cleanup-740` (anchor val_auc 0.6477054), +0.00179 attributed to multi-task gradient density. vs same-data single-head `baseline_extended_clean` (0.6473), +0.0022 — same architecture, same data, same training recipe, only multi-task vs single-task differs. The +0.0022 is the clean attribution of the multi-task effect.
- unexpected_findings: (a) Win head was still improving at the 30-epoch cap (best_epoch=30); proposal expected convergence well before that. A v2 with `max_epochs=50` would likely yield small additional gain. (b) Item head mAP@10=0.301 is higher than expected for a multi-label problem over 305 classes — players' end-game item choices ARE quite predictable from pre-game draft + player aggregates. (c) Duration head top1_acc=0.181 is lower than expected for an 8-bucket problem; pre-game info is weakly informative about duration. (d) The α_dur=0.15 (vs proposal's 0.5, dropped per the smoke flag) was clearly correct — at α_dur=0.5 the win head had been lagging the single-head sanity at epoch 2 of the v2 attempt (0.6186 vs 0.6218); at α_dur=0.15 it caught up by epoch 17 and exceeded by epoch 20.
- seeds_run: 1 (single run; seed=42)
- metric_aggregation: single-run
- next_candidates:
  - **`multitask-extended-740`** — same recipe, max_epochs=50 with patience=5 early-stop. Cheap test of "is the win head still improving" → likely yes since best_epoch=30=cap. Expected lift ~+0.001 to +0.003.
  - **`anonymous-aware-modeling-740`** — still the largest residual lever per the persistent 0.022 LOW-HIGH coverage gap. Router head OR per-team-aggregate features over known-player subset. Especially attractive now that the gradient-density lever is no longer "untouched" — combining the two should be additive.
  - **`multitask-alpha-tune-740`** — small grid over α weights (now that we know α_dur=0.15 works, try 0.05 / 0.10 / 0.15 / 0.20 and α_i 0.1 / 0.3 / 0.5). May yield small additional lift.
  - **(Engineering)** Promote `_meta/hardware-investigation-2026-05-21/test_pyarrow_roundtrip.py` to a quarterly RAM-health regression check.

## Follow-up

The hardware fix (disable EXPO, JEDEC 4800 MT/s) unblocked this entire
arc. With the box now stable, the queue is open for several short
follow-ups that should compound on top of this result.

The aux heads are now potentially-useful standalone products for
personal use — would need a lightweight inference wrapper that takes a
draft + 10 player aggregates and outputs (win_prob, duration_curve,
per-slot item top-K).
