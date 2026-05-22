---
kind: concept
name: "task-as-token-prompting"
status: seeded
added: "2026-05-22"
sources:
  - literature/papers/radford2022robust.md
  - literature/papers/ghosh2024octo.md
  - literature/papers/cui2022m6.md
related_concepts:
  - multi-query-foundation-model
  - tabular-foundation-model
related_experiments: []
tags: [multi-query, prompt-tokens, special-tokens, whisper, octo, m6-rec, foundation-model-deployment]
---

# task-as-token-prompting

## Definition

A deployment pattern for a single shared encoder (or encoder-decoder)
serving many downstream tasks, in which the desired task is selected
at inference time by **prepending a special control token** to the
input rather than by routing through a per-task head or per-task
fine-tuned model. Whisper (Radford 2022) is the canonical instance:
one decoder handles ASR / translation / language-ID / VAD / timestamp
prediction by reading a 3-4 token prompt prefix
(`<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`)
that fully specifies the task. Octo (Ghosh 2024) implements the same
idea slightly differently — readout tokens per task class instead of
a prompt prefix in a generative decoder — but the architectural
principle (task selection lives in the input token stream, not in the
parameter set) is the same. M6-Rec (Cui 2022) generalizes further to
"option tuning" — task-specific learnable prefix tokens that fine-tune
~1% of parameters per new task while keeping the trunk frozen.

## Why it matters here

`foundation-mvp-740` will need to serve at least 5 downstream queries
from a single trained encoder: win, duration, items, lineup-eval,
fun-pair. The naive design — one fine-tuned model per query, plus a
production-side switcher — is the path of least resistance but
multiplies maintenance cost and prevents joint gradient signal across
heads. Task-as-token prompting collapses this to one shared encoder
+ a small fixed vocabulary of task tokens (~5-10 special tokens,
~5-10 extra learnable parameter slots) selected per request at
inference time. The Whisper evidence (Section 4.3, Figure 9) is that
this pattern only pays off at scale — at small models / small data,
per-task models can beat the shared design. Our 77K-5M-param regime
is in the "scale-dangerous" zone; the prudent default is per-task
readout-token heads (Octo style) with task-as-token (Whisper style)
as a follow-up when scale supports it. Both belong in the
foundation-mvp-740 design exploration as candidate multi-query
interfaces, evaluated head-to-head.

## Connections

- [[multi-query-foundation-model]] — the deployment shape this
  mechanism implements.
- [[tabular-foundation-model]] — the shared-encoder architecture this
  pattern sits on top of.
