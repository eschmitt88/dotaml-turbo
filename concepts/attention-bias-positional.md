---
kind: concept
name: "attention-bias-positional"
status: seeded
added: "2026-05-22"
sources:
  - literature/papers/bi2022pangu.md
  - literature/papers/ghosh2024octo.md
related_concepts:
  - tabular-foundation-model
related_experiments: []
tags: [attention, positional-bias, structured-prior, pangu, swin, earth-specific, zero-flop]
---

# attention-bias-positional

## Definition

A family of attention-layer mechanisms that add **learnable bias
matrices indexed by token (cohort, position)** directly to the
post-softmax-pre-scaling attention logits `QK^T / √D + B`, in lieu of
(or alongside) the additive sinusoidal / RoPE positional encodings
that ride into the value stream. Critically, the bias matrix is *not
the same for all tokens*: it has per-position sub-matrices indexed by
the cohort the token belongs to (in Pangu: the (pressure-level,
latitude) window; in Octo: the (observation-vs-task-vs-readout) token
class via block-attention masking). Pangu-Weather's "Earth-Specific
Positional Bias" is the canonical concrete instance — replacing the
single shared `(2W-1)` Swin bias with `M_pl × M_lat` separate
sub-matrices yields 527× more bias parameters in the first block
while *speeding up* convergence (the bias encodes a real prior the
encoder no longer has to discover).

Distinct from sinusoidal / RoPE / absolute positional encoding: those
mix into the value stream and are not cohort-indexed. Distinct from
attention masking: a mask is a binary {-∞, 0} prior, while a
positional bias is a *learnable real-valued* prior.

## Why it matters here

The 10-slot Dota 2 draft has an obvious cohort structure that this
mechanism captures for free: **(team, slot)** pairs. Position-1
(carry) and position-3 (offlaner) on Radiant are not interchangeable;
Radiant-carry vs Dire-carry is a known symmetry (not identity) we
already capture indirectly via [[radiant-side-advantage]]. Adding a
learnable bias matrix `B[team, slot, team', slot']` (shape `[10, 10]`
per head per layer, ~1.6K-6.4K total parameters for typical
configurations) lets the encoder express "carry-vs-offlaner is
meaningful regardless of hero ID" as a hard prior, at zero FLOPs
per step. Pangu's empirical finding that the extra bias parameters
converge *faster* (because they encode a real prior, not noise)
suggests this is essentially a free architectural lift on top of our
existing 10-token Transformer. Symmetry-aware initialization
(B[Radiant_i, Radiant_j] = B[Dire_i, Dire_j] at init) costs nothing
and bakes in the side-mirror prior.

## Connections

- [[tabular-foundation-model]] — the architecture this mechanism
  augments. The shared encoder spine + per-cohort attention bias is
  the natural pairing for any tabular foundation model whose tokens
  have known positional structure.
- [[radiant-side-advantage]] — the symmetry the bias is initialized
  to respect, and the lever it can break via training data.
