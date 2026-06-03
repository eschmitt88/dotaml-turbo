---
kind: paper
title: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"
authors:
  - Kaifeng Bi
  - Lingxi Xie
  - Hengheng Zhang
  - Xin Chen
  - Xiaotao Gu
  - Qi Tian
institutions: ["Huawei Cloud Computing"]
year: 2022
venue: "arXiv:2211.02556 (later published in Nature 2023 as Pangu-Weather)"
peer_reviewed: true
url: "https://arxiv.org/abs/2211.02556"
code_url: null
citations: null
source: "raw/papers/bi2022pangu.pdf"
added: "2026-05-22"
relevance: 4
credibility: 5
status: skimmed
related_experiments: []
related_concepts:
  - tabular-foundation-model
  - attention-bias-positional
tags: [weather-forecast, 3d-transformer, swin-transformer, earth-specific-positional-bias, absolute-position-bias, structured-attention]
---

# Pangu-Weather: A 3D High-Resolution Transformer for Global Weather Forecast

## TL;DR

Pangu-Weather is the first AI-based numerical weather forecaster to
surpass ECMWF operational IFS on all variables and forecast horizons
from 1 hour to 7 days, using a 3D Earth-Specific Transformer (3DEST)
of ~256M parameters trained on 43 years of ERA5 reanalysis. The two
load-bearing technical ideas are (i) an **Earth-Specific Positional
Bias** that adds learnable, *absolute-position-indexed* bias matrices
to attention scores (one sub-matrix per (pressure-level, latitude)
window cohort), and (ii) hierarchical temporal aggregation across
1-/3-/6-/24-hour lead-time models to suppress cumulative iterative
error. The positional-bias innovation is the directly transferable
insight for any domain whose tokens occupy known asymmetric positions.

## Claims

- **First AI system to beat operational IFS on all variables and all
  forecast times 1h–7d** at 0.25° resolution, with single-model RMSE
  for 5-day Z500 of 296.7 vs IFS 333.7 and FourCastNet 462.5
  (Section 4.1, Figure 1, Figure 5).
- **Earth-Specific Positional Bias dramatically outperforms Swin's
  shared relative bias** by giving each (pressure-level, latitude)
  window cohort its own sub-matrix of learnable biases — yielding
  ~527× more bias parameters in the first block but converging
  *faster*, not slower, because the bias encodes a useful absolute-
  position prior (Section 3.3, Figure 3). Critically: "the
  Earth-specific positional bias does not increase the FLOPs of the
  model."
- **3D > 2D input formulation.** Treating height (13 pressure levels)
  as a real spatial dimension and operating on a 3D cube outperforms
  per-level 2D models because cross-height physical processes
  (radiation, convection) can be represented (Section 2.4, Section 3.3).
- **Hierarchical temporal aggregation beats recurrent training.**
  Training four separate forecast models (1h/3h/6h/24h lead time) and
  greedily composing them at inference reduces the number of
  iterations for a 7-day forecast from 168 (with 1h base) to 7 (with
  24h base), cutting cumulative error dramatically (Section 3.4,
  Figure 4) — and avoids the 2× GPU-memory overhead of FourCastNet's
  recurrent f(f(A)) training.
- **Forecast advantage grows with horizon.** "Forecast time gain" over
  operational IFS is >12 hours at every variable, >24 hours for
  specific humidity (Section 4.1.1) — supporting the hypothesis that
  AI methods capture useful patterns that PDE-based methods miss.

## Methods

3DEST is an 8-encoder + 8-decoder Swin-style Transformer with
window-attention over a 8×360×181×C cube formed by patch-embedding
(2×4×4 for upper-air variables, 4×4 for surface variables) and
concatenating along the height axis. Down-/up-sampling between layers
follows Swin. The critical departure from vanilla Swin is the
positional bias: rather than one shared `(2W_pl−1) × 2(W_lat−1) ×
(2W_lon−1)` matrix, the model maintains `M_pl × M_lat` sub-matrices
each of size `W_pl² × W_lat² × (2W_lon−1)` indexed by *which* window
on Earth's sphere the attention is computed within (longitude is
cyclic and shared). Training: AdamW, weight decay 3e-6, DropPath 0.2,
100 epochs × 4 lead-time models on 192 V100 GPUs, batch size 1/GPU.
Loss is plain L1 against ERA5 ground truth at the specified lead time.

## Takeaways for foundation-mvp-740

- **Adopt a (team, slot)-indexed attention bias for our 10-slot draft
  encoder.** The Dota 2 5v5 draft has *exactly* the asymmetric
  positional structure Pangu exploits: position-1 (carry) and
  position-3 (offlaner) on Radiant are not interchangeable, and
  Radiant-carry vs Dire-carry have a known symmetry (mirror), not an
  identity. A learnable bias matrix B[team, slot, team', slot']
  added to QK^T inside the attention layer (zero FLOPs per step, ~200
  extra learnable parameters per head per layer for 10×10 slot-pairs)
  encodes "carry-vs-offlaner is meaningfully different regardless of
  hero ID" as a hard prior the encoder doesn't have to discover from
  data. This is the *single most transferable* idea in this batch.
- **Use position-cohort sub-matrices, not a single shared bias.** The
  Swin → Pangu lesson is that giving *each cohort* (in our case:
  each (team, slot) pair) its own bias sub-matrix beats sharing one
  matrix across cohorts, and that the extra parameters converge
  *faster* not slower because they encode a real prior. Concretely:
  one bias per (team, slot) pair (10 cohorts) is the natural starting
  point; per-cohort bias matrices of shape `[10, 10]` per attention
  head per layer are tiny.
- **Symmetry-aware initialization.** Initialize B such that the
  Radiant↔Dire mirror is built in (B[Radiant, slot_i, Radiant, slot_j]
  = B[Dire, slot_i, Dire, slot_j] at init, B[Radiant, slot_i, Dire,
  slot_j] = B[Dire, slot_i, Radiant, slot_j] at init), then let
  training break the symmetry only as supported by data. Cheap
  regularization against the [[radiant-side-advantage]] confound.
- **Don't bother with the temporal-aggregation trick.** Our forecast
  horizon is one match outcome, not an iterated sequence; Pangu's
  hierarchical aggregation is orthogonal to our setting and not worth
  porting.
- **3D vs 2D framing doesn't translate.** We don't have a "height"
  axis in Dota drafts. The transferable bit is the positional-bias
  trick, not the 3D cube structure.

## Open questions / caveats

- The bias parameters in Pangu are *large* in absolute terms (527×
  Swin's), but Pangu has 256M total parameters and runs on 192 V100s
  — our 77K–5M-param foundation model has a much tighter budget.
  Calculation: 10 slots × 10 slots × N_heads (4-8) × N_layers (4-8)
  = ~1.6K-6.4K extra parameters total, which is negligible against
  our existing 77K-5M scale.
- Pangu's bias is added inside the Swin-window attention; we use
  global attention over 10 tokens (no windowing needed at this
  sequence length). The bias addition still works — it just becomes
  a single global B[slot_i, slot_j] matrix per (team_i, team_j)
  cohort, which is even simpler.
- Pangu's success is heavily about scale (43 yrs of data, 192 GPUs,
  100 epochs). The positional-bias trick is what we can cheaply
  transfer; the rest of the result doesn't promise our scale will
  benefit similarly. The honest framing in the proposal is "this is
  a zero-FLOP architectural prior worth testing for ~$0 cost," not
  "this is why we will hit the ceiling."
- No information on whether the bias parameters themselves overfit
  on small-data regimes; we should monitor train-val gap on the bias
  parameters specifically (low-rank vs full-rank ablation is cheap).

## Trust signals

- **Credibility:** 5 — Huawei Cloud research group; the arXiv tech report
  was subsequently published in Nature (2023), i.e. peer-reviewed at a
  top venue; large-scale reproducible result with public follow-on
  weather-model ecosystem. No code link named in this PDF, hence not a
  full 5-on-every-axis, but venue + peer review carry it.
