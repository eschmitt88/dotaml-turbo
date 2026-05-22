---
kind: moc
name: "foundation-models"
status: active
added: "2026-05-22"
related_concepts:
  - tabular-foundation-model
  - masked-modeling-tabular
  - uncertainty-weighted-multitask
  - multi-query-foundation-model
  - attention-bias-positional
  - task-as-token-prompting
  - supervised-multitask-pretraining
related_experiments:
  - 2026-05-20-rich-supervision-multitask-740
tags: [moc, foundation-model, tabular-dl, multi-task, multi-query, dotaml-turbo]
---

# foundation-models

Map of Content for the foundation-model cluster in `dotaml-turbo`.
Seven concepts now cluster on this theme; promoted on 2026-05-22 in
response to the second batch of cross-domain FM-paper ingests
(Pangu-Weather, Moirai-MoE, JMP, Octo, Whisper) joining the first
batch (Gorishniy 2021, Somepalli 2021, Kim 2024, Kirchdorfer 2024,
Jiang 2023, Cui 2022, Wang 2025). Together these papers define the
design space for the upcoming `foundation-mvp-740` proposal.

## The seven concepts

### Architecture spine

- [[tabular-foundation-model]] — the shared Transformer encoder over
  per-feature / per-entity tokens, with a `[CLS]` (or readout) head
  feeding downstream tasks. The architectural family that contains
  FT-Transformer (Gorishniy 2021), SAINT (Somepalli 2021), PMAE
  (Kim 2024), HIGFormer (Wang 2025), M6-Rec (Cui 2022), and the
  closest published large-scale neighbor: Octo (Ghosh 2024) at
  27M-93M params.
- [[attention-bias-positional]] — Pangu-Weather's Earth-Specific
  Positional Bias and Octo's block-attention masking — learnable
  bias matrices indexed by token (cohort, position), added to
  attention logits at zero FLOP cost. Directly applicable as a
  (team, slot) bias for our 10-slot draft encoder.

### Pre-training objective

- [[masked-modeling-tabular]] — the self-supervised MAE / denoising
  objective from SAINT (Somepalli 2021) and PMAE (Kim 2024).
  Demoted to *auxiliary* objective by the JMP evidence below; still
  useful for leveraging the unsupervised match tail and for
  anonymity-robustness.
- [[supervised-multitask-pretraining]] — JMP's (Shoghi 2023) evidence
  that joint supervised multi-task pre-training beats both
  self-supervised pre-training and single-task supervised
  pre-training (even at 48× larger data on the single task). Whisper
  (Radford 2022) corroborates at speech-foundation scale. **Settles
  the schedule choice** for foundation-mvp-740: joint multi-task
  supervised pre-training is primary; MAE is auxiliary.

### Multi-task / multi-query machinery

- [[uncertainty-weighted-multitask]] — UW-SO (Kirchdorfer 2024) and
  ForkMerge (Jiang 2023) — analytical loss-balancing without
  per-task α tuning. Enables adding many heads without re-tuning the
  α grid, with ForkMerge providing a safety net against
  negative-transfer auxiliaries.
- [[multi-query-foundation-model]] — M6-Rec's (Cui 2022) one-encoder-
  many-tasks deployment pattern, also implemented at scale by Octo
  and Whisper. The shape we need to serve foundation-mvp-740 across
  win + duration + items + lineup-eval + fun-pair queries from a
  single trained encoder.
- [[task-as-token-prompting]] — Whisper's (Radford 2022) special-
  token prompting and Octo's (Ghosh 2024) readout-token pattern —
  two implementations of the same idea: task selection lives in the
  input token stream, not in the parameter set. The clean interface
  for multi-query deployment.

## How they compose for foundation-mvp-740

The design space the seven concepts span:

1. **Spine**: shared Transformer encoder over per-slot tokens
   (= [[tabular-foundation-model]]).
2. **Positional prior**: learnable (team, slot) bias matrix added
   to attention logits, symmetry-aware initialization
   (= [[attention-bias-positional]]).
3. **Pre-training**: joint multi-task supervised with win / duration
   / items / aux heads; temperature-sampled per-task batches; MAE as
   small auxiliary loss only (= [[supervised-multitask-pretraining]]
   primary, [[masked-modeling-tabular]] auxiliary).
4. **Loss balancing**: UW-SO instead of hand-tuned α; ForkMerge as
   safety net for new auxiliary heads
   (= [[uncertainty-weighted-multitask]]).
5. **Multi-query interface**: per-task readout tokens (Octo) for the
   initial small-scale model; revisit task-as-token (Whisper) when
   scale supports it (= [[task-as-token-prompting]] +
   [[multi-query-foundation-model]]).

Empirical anchor: `rich-supervision-multitask-740` already validated
mechanism (3) at small scale, lifting val_auc 0.6477 → 0.6495 (first
whole-val ceiling movement of the project). The cluster's job now is
to compose (1)+(2)+(3)+(4)+(5) into the foundation-mvp-740 proposal.

## Open questions for the cluster

- **Scale threshold for the task-as-token vs per-task-head choice.**
  Whisper Section 4.3 shows task-as-token underperforms at small
  scale and overtakes at large scale. Where exactly is our
  threshold? Need both interfaces implemented for head-to-head
  evaluation.
- **MAE auxiliary weight.** JMP demotes MAE to auxiliary, but
  doesn't quantify the right weight. Tuning lever for the
  foundation-mvp-740 ablations.
- **Anonymity handling.** 66% of Turbo player slots are anonymous;
  the encoder must be robust to "missing" tokens. MAE pre-training
  helps; per-cohort positional bias may also help (the bias for an
  anonymous slot can be the same as for a missing slot, learnable).
  Not yet clearly designed.
- **Capacity ceiling.** [[player-embedding-prelim-740]]'s 16M-param
  null result was from-scratch. The JMP regularization story says
  pre-training unlocks larger models. Where is the actual capacity
  ceiling under the foundation-model pre-training schedule? Need
  to test 1M / 5M / 15M / 50M head-to-head with the same training
  recipe.
