---
kind: paper
title: "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)"
authors:
  - Alec Radford
  - Jong Wook Kim
  - Tao Xu
  - Greg Brockman
  - Christine McLeavey
  - Ilya Sutskever
year: 2022
venue: "arXiv:2212.04356 (OpenAI; subsequently published at ICML 2023)"
url: "https://arxiv.org/abs/2212.04356"
source: "raw/papers/radford2022robust.pdf"
added: "2026-05-22"
relevance: 5
status: skimmed
related_experiments: []
related_concepts:
  - tabular-foundation-model
  - multi-query-foundation-model
  - task-as-token-prompting
tags: [whisper, speech-recognition, task-as-token, multitask, encoder-decoder, openai, weak-supervision, foundation-model]
---

# Whisper: Task-as-Token Prompting for a Shared Encoder-Decoder

## TL;DR

Whisper is a 39M-1550M-param encoder-decoder Transformer trained on
680k hours of weakly-supervised internet (audio, transcript) pairs.
The architectural innovation most relevant to foundation-mvp-740 is
the **task-as-token multitask format**: a single shared decoder
handles transcription, translation, language ID, voice-activity
detection, and timestamp prediction by *prefixing the decoder
sequence with special control tokens* — `<|startoftranscript|>`,
`<|en|>` (language), `<|transcribe|>` or `<|translate|>` (task),
`<|notimestamps|>` (format) — then generating the appropriate
output. No per-task heads, no per-task fine-tuning. The same model
achieves zero-shot WER comparable to supervised models on
LibriSpeech, and matches commercial transcription on long-form audio.
Scale is the story (5-orders-of-magnitude more weak supervision than
SpeechStew), but the **deployment pattern of "one shared decoder,
many tasks via prompt tokens"** is the cleanest known answer to our
multi-query foundation-model design question.

## Claims

- **Zero-shot Whisper closes the human-robustness gap.** On
  LibriSpeech dev-clean: Whisper-Large WER 2.7, matched by
  supervised wav2vec2. On 12 other datasets averaged: Whisper-Large
  WER 12.8, supervised wav2vec2 WER 29.3 — Whisper's effective
  robustness is 55.2% relative WER reduction (Section 3.3,
  Figure 2, Table 2).
- **Task-as-token format unifies many tasks in one decoder.** Section
  2.3 + Figure 1 fully specifies the format: special tokens select
  task (transcribe/translate), language, and format (timestamps /
  no-timestamps). One model decoder handles all of: monolingual
  ASR, multilingual ASR, X→en translation, language ID, VAD,
  word-level timestamps. No architectural changes per task; only
  the prompt prefix changes.
- **Joint multitask + multilingual training shows positive transfer
  at scale.** For small models trained on small compute,
  multitask/multilingual underperforms English-only; but at large
  scale the joint model beats English-only even at matched
  English-task compute (Figure 9, Section 4.3). The lesson: the
  task-as-token format pays off when the encoder is big enough; at
  small scale it hurts.
- **Dataset size matters, and so does its diversity.** Performance
  scales with hours of training data following a power law for
  multilingual ASR and translation (Table 6); the full 680k-hour
  dataset is what enables zero-shot generalization. The 5,140-hour
  SpeechStew baseline is ~130× smaller.
- **No fine-tuning, no augmentation, no regularization.** Trained
  for 2-3 epochs on the full dataset with AdamW + linear LR warmup
  + decay to zero; relies entirely on dataset diversity for
  regularization. Simple architecture (off-the-shelf
  encoder-decoder Transformer); the scale and prompt format are
  the contributions.

## Methods

Encoder: 80-channel log-Mel spectrogram → 2-layer conv stem (stride
2) + sinusoidal positional encoding → N pre-norm transformer
encoder blocks. Decoder: learned positional encoding + tied
input-output embeddings + N pre-norm transformer decoder blocks
with cross-attention to encoder output. Vocabulary: byte-level BPE
(GPT-2 tokenizer for English-only; refit for multilingual).
Multitask format (Section 2.3): each training example's decoder
sequence is
```
[SOT] [<|lang|>] [<|task|>] [<|notimestamps|>?] [text tokens...] [EOT]
```
where `[<|lang|>]` is one of 99 language tokens or `<|nospeech|>`,
`[<|task|>]` is `<|transcribe|>` or `<|translate|>`, and timestamp
mode is optionally selected. Loss is masked over the prompt prefix
and applied only to the output tokens. Inference: prepend the
desired prompt; greedy or beam-search decoding produces the answer.
Training: batch 256 audio segments × 30s, 2^20 updates (2-3 epochs
on 680k hours), AdamW + linear warmup over 2048 steps + linear
decay to zero. Five model sizes: 39M, 74M, 244M, 769M, 1550M.

## Takeaways for foundation-mvp-740

- **Adopt task-as-token prompting as the multi-query interface.**
  This is the cleanest pattern in the literature for our "one
  shared encoder, many downstream queries" requirement. Concrete
  design:
  - Reserve special tokens: `<|predict_win|>`, `<|predict_duration
    |>`, `<|predict_items|>`, `<|predict_lineup_eval|>`,
    `<|predict_fun_pair|>`.
  - At training time, prefix each batch (or each example) with the
    chosen task token; mask training loss over the prefix.
  - One shared encoder + ONE shared head architecture (per output
    type: a regression head and a classification head and an
    item-set head — but reused across tasks of the same output
    type, dispatched by the task token). This is even cleaner than
    Octo's per-task readout-token design when output types overlap.
- **The Whisper recipe scales positively only at large model size.**
  Section 4.3 explicitly: at small scale, task-as-token underperforms
  task-specific models. Our current 77K-5M scale is in the danger
  zone where the unified-decoder design may hurt. Two safer
  options: (a) start with per-task readout-token heads (Octo style)
  and only collapse to a single shared head once we cross the
  scale threshold; (b) use task-as-token but verify against a
  per-task-head baseline at every scale to detect regression.
- **No fine-tuning per task — one model, one training run, all
  queries served.** This is the production-deployment story for
  dotaml-serve. Whisper's "no fine-tuning" property is what makes
  it deployable as a single checkpoint serving heterogeneous
  workloads — the same property we want for the foundation model
  serving win + items + duration to the live frontend.
- **Pure scale on weak supervision, no augmentation.** Whisper's
  "we trained for 2-3 epochs and relied on data diversity for
  regularization" is the simplest possible training recipe; for
  our 60M-match corpus with multiple labels per match, this is
  achievable. Don't over-engineer the training loop before the
  data scale is established.
- **Tokenization specifically: special task tokens cost ~5 extra
  vocab slots, zero runtime cost.** The implementation cost of
  task-as-token is genuinely tiny; the design question is just
  whether per-task heads or per-task tokens compose better with
  the rest of our architecture. Whisper says: at sufficient
  scale, token wins because it generalizes across tasks the
  encoder has never seen co-occurring.

## Open questions / caveats

- Whisper is a *sequence-to-sequence* model with a generative
  decoder; our setting is mostly per-match prediction (single
  scalar or single-set output per match). The task-as-token
  pattern still applies, but we use it to select a *head's
  output mode*, not to drive autoregressive decoding. This is a
  notable architectural difference; the prompting concept
  transfers but the decoder-loop machinery doesn't.
- The "task-as-token underperforms at small scale" finding (Fig 9)
  is empirically important for us — at our current 5M-param scale,
  per-task heads (Octo) may beat task-as-token (Whisper). A clean
  experiment is to build both interfaces and compare on
  foundation-mvp-740 head-to-head.
- Whisper's tasks (transcribe/translate/lang-ID) are tightly
  related (all output text from audio); our tasks (win/duration/
  items/fun-pair) are more heterogeneous in output type. The
  shared-decoder collapse is harder to justify; per-head
  dispatch keyed by task token is the more conservative reading
  of Whisper's lesson for us.
- The dataset-quality filtering in Section 2.1 (removing machine-
  generated transcripts, language-detector validation, fuzzy
  dedup) is the unsexy half of the recipe and is half the
  difference. For our setting the analog is the [[fake-match-
  filtering]] work — Whisper validates that *data quality
  filtering pays off proportionally to data scale*, not just at
  small scale.
- No mention of how to handle adding a *new* task post-training
  without retraining (e.g. "predict 7.41 win"). Whisper retrains
  from scratch when adding tasks. Our multi-query foundation
  model needs the [[ghosh2024octo]] modular pattern (add a new
  readout token + frozen trunk) for that ability. Combine both:
  task-as-token for inference flexibility, Octo modularity for
  post-hoc extension.
