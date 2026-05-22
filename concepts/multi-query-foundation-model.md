---
kind: concept
name: "multi-query-foundation-model"
status: seeded
added: "2026-05-22"
sources:
  - literature/papers/cui2022m6.md
  - literature/papers/somepalli2021saint.md
  - literature/papers/gorishniy2021revisiting.md
related_concepts:
  - tabular-foundation-model
  - uncertainty-weighted-multitask
related_experiments: []
tags: [foundation-model, multi-task, prompt-tuning, option-tuning, late-interaction, recommendation]
---

# multi-query-foundation-model

## Definition

The deployment pattern in which a single pre-trained encoder serves
many downstream tasks via lightweight per-task heads or prompts,
rather than re-training a fresh model per task. M6-Rec (Cui 2022)
is the canonical industrial-recommender exemplar — one encoder, six
disparate task families (retrieval, ranking, CTR, explanation,
zero-shot rec, conversational rec) — adapted via **option tuning**
(a prompt-tuning variant adding only ~1% task-specific parameters)
and **multi-segment late interaction** (cache early-layer
representations, only run late layers per request) for low-latency
serving. SAINT (Somepalli 2021) is the simpler "one pre-trained
encoder, one fine-tuned head per dataset" form.

## Why it matters here

`dotaml-turbo`'s sister repo `dotaml-serve` is the live-prediction
side; if the foundation model produced by `foundation-mvp-740` is to
serve more than just win-prediction in production (e.g. item
recommendation during the draft phase, duration prediction for
matchmaking, KDA prediction for player-facing UI), it should be
designed from day one as a multi-query foundation model — shared
encoder, per-task heads/adapters, late-interaction at serving time.
This is also the framing that justifies the multi-task pre-training
investment: a 5M-param encoder shared across N tasks is cheaper to
maintain than N×5M independent models, and the shared encoder gets
gradient signal from N tasks (which is what made the +0.0018 win-head
lift in [[2026-05-20-rich-supervision-multitask-740]] possible).

## Connections

- [[tabular-foundation-model]] — the architecture being shared.
- [[uncertainty-weighted-multitask]] — how to balance the many heads'
  losses during shared pre-training / fine-tuning.
