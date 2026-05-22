---
kind: concept
name: "supervised-multitask-pretraining"
status: seeded
added: "2026-05-22"
sources:
  - literature/papers/shoghi2023molecules.md
  - literature/papers/radford2022robust.md
related_concepts:
  - tabular-foundation-model
  - uncertainty-weighted-multitask
  - masked-modeling-tabular
related_experiments:
  - 2026-05-20-rich-supervision-multitask-740
tags: [pretraining, multi-task, supervised, jmp, temperature-sampling, foundation-model-schedule]
---

# supervised-multitask-pretraining

## Definition

A foundation-model training schedule in which the shared encoder is
**pre-trained from scratch on a portfolio of supervised tasks
simultaneously** (rather than via a self-supervised proxy objective
like MAE / denoising / contrastive, followed by per-task fine-tuning).
Each task is a separate (label-set, prediction-head) pair attached to
the shared encoder; per-task losses are summed (typically with
temperature-sampled batch composition and per-task normalization).
JMP (Shoghi 2023) is the canonical large-scale instance — joint
multi-task training on ~120M atomic systems from 4 chemistry domains
yields +59% average improvement over from-scratch fine-tuning on 40
downstream benchmarks, beats SOTA on 34/40, and crucially beats
*single-task supervised pre-training even when the single task is 48×
larger* (the OC20-only ablation). Whisper (Radford 2022) is a smaller
instance — multilingual + multitask joint training at scale shows
positive transfer that English-only training cannot match.

The mechanism: a richer per-example gradient signal (N tasks × per-
example labels) acts as both a stronger learning signal and a
regularizer that prevents the encoder from overfitting to any single
task's idiosyncrasies. JMP's secondary finding is that this
regularization unlocks the use of *larger* backbones than from-scratch
training can support — JMP-Large (235M) outperforms JMP-Small (30M)
by 21% on average, whereas from-scratch GN-OC-Large *underperforms*
GN-OC-Small by 8% due to overfitting.

## Why it matters here

`rich-supervision-multitask-740` already validated the basic
mechanism for `dotaml-turbo`: adding duration + items + KDA / GPM /
HD aux heads to the shared encoder lifted the win head from 0.6477
(single-task ceiling, anchored across 3 runs within 2e-5) to
**0.6495 — the first whole-val ceiling movement of the project**.
JMP is the literature evidence that this lever is not yet exhausted:
adding more supervised heads (per-player aggregates, talent picks
from `ability_upgrades[]`, first-blood time, tower state, etc.)
should continue to lift the shared encoder, and the regularization
effect should unlock the larger (5M-15M) encoders that
[[player-embedding-prelim-740]]'s 16M-param null result showed cannot
be reached via single-task training.

JMP also closes an open question in the project: **whether to
pre-train MAE-then-fine-tune (the [[masked-modeling-tabular]]
playbook) or to joint-multi-task-pretrain (the JMP playbook)**. The
JMP ablation (-9.9% from single-task even at 48× data; +59% from
joint multi-task over from-scratch) settles the schedule: joint
multi-task supervised pre-training is the primary objective; MAE /
denoising is at most an auxiliary loss, not the primary pre-training
objective.

## Connections

- [[tabular-foundation-model]] — the architecture being trained.
- [[uncertainty-weighted-multitask]] — the loss-balancing machinery
  that makes adding many tasks tractable without per-task α tuning.
- [[masked-modeling-tabular]] — the alternative pre-training paradigm
  this concept supersedes as the primary objective for
  foundation-mvp-740 (downgrades MAE from "pre-train then fine-tune"
  to "optional auxiliary loss alongside supervised heads").
