---
kind: paper
title: "Octo: An Open-Source Generalist Robot Policy"
authors:
  - Dibya Ghosh
  - Homer Walke
  - Karl Pertsch
  - Kevin Black
  - Oier Mees
  - Sudeep Dasari
  - Joey Hejna
  - Tobias Kreiman
  - Ria Doshi
  - Charles Xu
  - Jianlan Luo
  - You Liang Tan
  - Lawrence Yunliang Chen
  - Pannag Sanketi
  - Quan Vuong
  - Ted Xiao
  - Dorsa Sadigh
  - Chelsea Finn
  - Sergey Levine
institutions: ["UC Berkeley", "Stanford University", "Carnegie Mellon University", "Google DeepMind"]
year: 2024
venue: "RSS 2024 (arXiv:2405.12213)"
peer_reviewed: true
url: "https://arxiv.org/abs/2405.12213"
code_url: "https://octo-models.github.io"
citations: null
source: "raw/papers/ghosh2024octo.pdf"
added: "2026-05-22"
relevance: 5
credibility: 5
status: skimmed
related_experiments: []
related_concepts:
  - tabular-foundation-model
  - multi-query-foundation-model
  - task-as-token-prompting
  - attention-bias-positional
tags: [robotics, foundation-model, transformer, readout-tokens, modular-tokenizers, block-attention, octo, oxe, rss2024]
---

# Octo: A Modular Transformer-First Generalist Policy

## TL;DR

Octo is a 27M-93M parameter transformer-based generalist robot policy
pretrained on 800k trajectories from the Open X-Embodiment dataset.
Its load-bearing architectural ideas — **(1)** modular per-modality
tokenizers (lightweight CNN for images, T5-base for language), **(2)**
a block-wise-masked transformer trunk processing the unified token
sequence, **(3)** **learned "readout tokens" that passively read the
sequence and feed task-specific heads**, and **(4)** a diffusion
action head — together produce a policy that can be finetuned to new
robots, new sensors (force-torque), new action spaces (joint vs end-
effector) in <5 hours on a single consumer GPU. The Octo architecture
is the closest published template for what foundation-mvp-740 needs:
modular per-input tokenization + transformer trunk + per-task readout
heads, sized exactly in the 27M-93M range our budget supports.

## Claims

- **The Octo design beats the prior best openly-available generalist
  robot policy RT-1-X (35M)** zero-shot across three robot
  embodiments by ~29% success-rate, and matches RT-2-X (55B) on the
  tested setups despite being ~600× smaller (Section IV-A, Figure 5).
- **Diffusion action head > MSE > discretized actions.** Ablation
  shows MSE drops from 83% to 35%, discrete drops to 18% on the same
  WidowX benchmark (Table II) — the diffusion head's ability to
  model multi-modal action distributions while keeping continuous
  precision is load-bearing.
- **Transformer-first architecture (small CNN patch encoder + large
  ViT trunk) beats ResNet-first (large ResNet + small transformer)**
  when training at the full data scale (83% vs 70%, Table II). But
  ResNet-first wins on small datasets in from-scratch comparisons —
  confirming that the transformer-first design only pays off with
  the pretraining scale that diversity supports.
- **Wider data mix → better policy.** 25-dataset OXE mix (Octo
  default) > 11-dataset RT-X mix (60%) > single-robot Bridge-Data-
  only (43%). Scaling the data mixture is the primary lever.
- **Readout tokens + block attention enable modular post-hoc
  extension.** New observations or new action spaces can be added
  during finetuning by adding new tokens/heads WITHOUT
  re-initializing the trunk. Existing tokens see new tokens only if
  the attention block-mask allows; pretrained weights are wholly
  preserved.

## Methods

Architecture: language tokens go through frozen T5-base (111M, NOT
counted in the 27M/93M backbone count); image tokens go through a
shallow CNN producing patches (ViT-style); learned positional
embeddings are added. The transformer trunk is a vanilla ViT-S
(27M) or ViT-B (93M) with **block-wise causal attention**:
observations at time t attend to task tokens and earlier observation
tokens, but NOT to readout tokens. **Readout tokens** are learned
[CLS]-like tokens inserted per timestep that passively attend to
observations and task tokens (like BERT's CLS) but are never attended
to — they serve as a compact per-timestep summary fed to action
heads. The diffusion action head runs K denoising steps inside a
small head module, conditioned on the readout token embedding.
Training: AdamW, batch 2048, inverse-sqrt LR decay, 300k steps =
14h on TPU v4-128. Finetuning: 50k steps on ~100 in-domain demos,
~5h on a single A5000. Hindsight goal relabeling for goal-image
conditioning; randomly drops language OR goal at training to support
both at inference.

## Takeaways for foundation-mvp-740

- **Use Octo's modular tokenizer + transformer + readout-token + per-
  task head layout as the literal architectural template for
  foundation-mvp-740.** This is the closest published precedent at
  our parameter scale (27-93M is exactly where our roadmap goes
  beyond the current 77K). The layout maps directly:
  - **Task tokens** = (patch ID, language-cohort-of-match, possibly
    a "task-as-token" prefix per [[radford2022robust]]).
  - **Observation tokens** = 10 hero-slot tokens + per-slot player-
    feature tokens (already what `transformer-plus-features-740`
    builds).
  - **Readout tokens** = one per downstream task (win, duration,
    items, lineup-eval, fun-pair). One [CLS]-like token per task
    is cleaner than reusing a single shared [CLS], because each
    head can attend to different subsets of the input via block
    masking.
  - **Per-task heads** = small MLPs on readout-token embeddings;
    diffusion head ONLY if the task target is genuinely
    multi-modal (item-build distribution likely yes; win/duration
    likely no).
- **Block-attention masking is the right way to add new tasks
  post-hoc.** When we want to add a new task (e.g. "7.41 win
  prediction" or "ban suggestion") later, the Octo pattern is: add a
  new readout token, freeze the trunk + existing tokens, finetune
  only the new token + new head. This is what the multi-query
  foundation model framing in [[multi-query-foundation-model]]
  requires architecturally; Octo gives the exact recipe.
- **27M-93M is the right parameter target.** Octo at 93M trained on
  800k trajectories converges in 14h on TPU-128 — for our scale
  (60M matches, RTX 5080) this maps to a 5-15M-param model trained
  in 4-12h on one GPU, well inside budget.yaml. Don't shoot for
  93M on first pass; start at 5M and scale once the design is
  stable. [[player-embedding-prelim-740]]'s 16M-param null result
  is consistent with "model too big for from-scratch task data" —
  the JMP regularization story ([[shoghi2023molecules]]) says
  pre-training unlocks the larger size.
- **Diffusion head is overkill for the win head, useful for the
  item-build head.** Win is single-scalar regression with
  near-Gaussian residuals; MSE head is fine. Item-build is a
  combinatorial multi-modal distribution over 200-item space; the
  Octo result (diffusion vs discrete vs MSE) says diffusion is the
  right choice when the action space is high-dimensional and
  multi-modal. Allocate the complexity only where it pays.
- **Transformer-first NOT ResNet-first.** This is consistent with
  our current MinimalTransformer being a Transformer over patch-
  like tokens, not a giant CNN feeding a small attention layer.
  Confirmed direction; no change.

## Open questions / caveats

- Octo's task tokens are language (instruction) + goal-image. Our
  "task" is which downstream query we want (win / duration / items /
  ...) — a small finite set, not natural language. The
  task-as-token pattern from [[radford2022robust]] is cleaner for
  this: a single special token like `<predict_win>` vs `<predict_
  duration>` prepended to the input. The Whisper-style discrete task
  token plays the same role as Octo's language tokens but is much
  cheaper (no T5 frozen encoder needed).
- Octo's success leans heavily on the 800k-trajectory pre-training
  scale. We have 60M matches but each match is 10 tokens, not a
  long trajectory; the effective compute-per-example is much smaller
  for us. The architectural recipe transfers; the scaling-law
  promise does not directly.
- Octo finetunes the full model on downstream tasks, not just heads.
  For our multi-query setup we want frozen-trunk + per-task-head
  finetuning (the [[cui2022m6]] M6-Rec pattern). The Octo block-
  attention design supports this naturally; just freeze the trunk
  during head training.
- Octo's wrist-camera underperformance is a noted shortcoming
  (Section V); our analog risk is that one task's gradient (likely
  win, the highest-signal one) dominates and other heads get
  starved. The JMP recipe ([[shoghi2023molecules]]: temperature
  sampling + per-task loss normalization) mitigates this.
- Octo uses a frozen T5-base (111M) just to embed language — this is
  not counted in their 27M/93M backbone size. For us there is no
  language modality, so no analogous frozen-encoder cost.

## Trust signals

- **Credibility:** 5 — top robotics-learning labs (UC Berkeley, Stanford,
  CMU, Google DeepMind), peer-reviewed at RSS 2024, fully open-source
  with released models + code (octo-models.github.io), and widely cited
  as a generalist-policy reference. Strong on every axis.
