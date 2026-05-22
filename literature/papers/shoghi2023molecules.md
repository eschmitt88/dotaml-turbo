---
kind: paper
title: "From Molecules to Materials: Pre-training Large Generalizable Models for Atomic Property Prediction"
authors:
  - Nima Shoghi
  - Adeesh Kolluru
  - John R. Kitchin
  - Zachary W. Ulissi
  - C. Lawrence Zitnick
  - Brandon M. Wood
year: 2023
venue: "ICLR 2024 (arXiv:2310.16802)"
url: "https://arxiv.org/abs/2310.16802"
source: "raw/papers/shoghi2023molecules.pdf"
added: "2026-05-22"
relevance: 5
status: skimmed
related_experiments: []
related_concepts:
  - tabular-foundation-model
  - multi-query-foundation-model
  - uncertainty-weighted-multitask
  - supervised-multitask-pretraining
tags: [foundation-model, pretraining, multi-task, supervised-pretraining, jmp, gemnet, atomic-property, iclr2024]
---

# JMP: Joint Multi-domain (Supervised) Pre-training for Atomic Property Prediction

## TL;DR

JMP pre-trains a single GemNet-OC backbone (30M-235M params) jointly
on ~120M atomic systems from 4 disparate datasets (OC20, OC22, ANI-1x,
Transition-1x) by treating each (dataset, label) as a separate
supervised multi-task pre-training task. Fine-tuning the shared
backbone with new prediction heads beats from-scratch training on
all 40 downstream tasks across QM9, rMD17, MD22, SPICE, MatBench,
QMOF — **+59% relative improvement on average, SOTA on 34/40
benchmarks**, and 12× less downstream compute. The central
contribution is empirical evidence that **joint supervised multi-task
pre-training beats both single-task supervised pre-training (OC20
alone tested as -9.9%) and the prior self-supervised denoising
paradigm** for this domain. Pre-training also acts as a regularizer
that lets larger backbones (235M) outperform smaller ones (30M) on
low-data downstream tasks — reversing the overfitting seen in
from-scratch training.

## Claims

- **JMP-Large beats from-scratch GN-OC-Large by 59% on average across
  40 downstream tasks** spanning small molecules, large molecules,
  materials, and MOFs. SOTA on 34/40; matches SOTA on the rest
  (Section 5, Tables 2-5, Figure 2).
- **Joint multi-task pre-training beats single-task pre-training even
  when the single task has 48× more data.** The ablation `OC20 Only`
  (120M OC20 samples, matching JMP's total) underperforms the JMP
  base by 9.9%, "indicating that diverse multi-task pre-training is
  important for generalization" (Section 5.1, Table 6). This is the
  paper's headline scientific finding for our purposes.
- **Pre-training reverses the scaling-vs-overfit pattern.** From
  scratch: GN-OC-L is *8% worse* than GN-OC-S on small downstream
  datasets due to overfitting. With JMP: JMP-L is *21% better* than
  JMP-S on the same datasets (Figure 2a-b). Pre-training acts as a
  strong regularizer that unlocks larger-model benefits.
- **Structure-wise loss averaging is the most impactful design
  choice** in the multi-task setup (+7.7% ablation), preventing
  large-system datasets from dominating the gradient. Combined with
  T=2 temperature sampling, weight decay 0.1, and edge dropout p=0.1,
  the full multi-task recipe contributes +13.2% over the naive
  multi-task baseline (Section 5.1, Table 6).
- **Pre-training cost (34,400 GPU-hours upfront) is recouped via 12×
  faster downstream fine-tuning** (275 GPU-hours total to match
  GN-OC-L's 3,300 GPU-hour scratch performance across all tasks).
  Section 5.2, Figure 4.

## Methods

GemNet-OC message-passing GNN backbone, 30M (small) or 235M (large)
parameters. Pre-training: per-dataset energy + force heads with
per-dataset linear energy reference + RMS force normalization.
Sampling: temperature T=2 (Shaham 2023) — `p_i ∝ (|D_i|/Σ|D_j|)^(1/T)`
to up-weight low-resource datasets without fully balancing. Per-system
loss reduction (NOT per-atom averaging) to prevent large-system
datasets from dominating. Per-dataset λ_E = 1, λ_F = ⟨N⟩_{D_i} (mean
atoms per system) — a heuristic that avoids tuning 2×M loss weights.
Fine-tuning: discard pre-training heads, add fresh randomly-init heads
for downstream task, train end-to-end with cosine LR decay. Tested
PCGrad and other automatic task-weighting strategies in appendix; they
underperformed the fixed-heuristic recipe in this setting.

## Takeaways for foundation-mvp-740

- **Pre-train multi-task supervised, do NOT pre-train MAE-then-
  fine-tune.** This is the single most important architecture
  decision JMP gives us. We have rich-supervision multi-task already
  working ([[2026-05-20-rich-supervision-multitask-740]]: win +
  duration + items + KDA/GPM/HD with hand-picked α's → val_auc
  0.6495). JMP is direct evidence that joint multi-task supervised
  pre-training beats both (a) self-supervised pre-training (MAE,
  denoising) and (b) single-task pre-training, even when the
  single-task data is 48× larger. The foundation-mvp-740 schedule
  should be: **joint multi-task SUPERVISED pre-training over our
  match dataset + MAE as a small auxiliary objective at most, NOT
  MAE → fine-tune.**
- **Per-task linear heads, shared encoder is the right pattern at
  our scale.** JMP confirms what we already showed locally: the
  shared encoder gets ~N× more gradient signal from N task heads,
  and the linear heads cost nothing. Their fine-tuning protocol
  (discard pre-train heads, fresh random heads on downstream task)
  is the exact pattern we'd use if we ever want to add a new task
  post-hoc (e.g. "predict 7.41 win rate from 7.40 features").
- **Temperature sampling for imbalanced task data.** In our setting
  we have 60M matches with full win labels but only the rich-cols
  subset (~30M? need to verify) with item-build / duration / KDA
  labels. Sampling per-task with `T=2` (i.e. partially balance,
  don't fully balance) is the JMP-validated default for ramping up
  low-resource tasks without starving the win head. This is more
  principled than the hand-picked α_w=1.0, α_d=0.15, α_i=0.3,
  α_a=0.1 we used in rich-supervision-multitask-740, and worth
  trying as an alternative weighting scheme.
- **System-wise loss averaging — i.e. per-MATCH not per-event.**
  JMP's structure-wise loss reduction principle translates: when
  some tasks have per-event labels (item picks across N items per
  match) and some are per-match (win), normalize losses to
  per-match before averaging across the batch. We may already be
  doing this implicitly via the head structure; worth checking.
- **Scale-up plan: pre-training acts as regularizer for bigger
  models.** Our current encoder is 77K params (transformer-plus-
  features) to 16M (player-embedding-prelim). [[player-embedding-
  prelim-740]] showed 16M alone yielded zero lift — but the JMP
  data suggests this null result is a *from-scratch* overfitting
  artifact, not a true capacity ceiling. If we joint-multi-task
  pre-train at 1M-5M params, the bigger model may finally pay off
  the way JMP-L pays off vs JMP-S on low-data downstream tasks.

## Open questions / caveats

- JMP's downstream tasks are *transfer* tasks (different chemical
  domains than pre-training). Our setting is closer to *multi-query*
  on a single task family (5v5 matches with shared encoder, different
  heads for win/duration/items). The "OC20-only underperforms by
  9.9%" finding might not transfer if our auxiliary tasks (duration,
  items) don't provide genuinely different supervision than the win
  task — i.e., if items mostly correlate with winning, the multi-task
  win-head lift could be saturated already at our +0.0018. Counter-
  argument: the win head DID lift +0.0018 from adding duration+items,
  so the marginal supervision is real, just smaller than JMP's
  cross-domain lifts.
- JMP uses GemNet-OC (a 3D geometric GNN with directional message
  passing). Our backbone is a vanilla 4-layer Transformer over 10
  tokens. The architectural specifics don't transfer, only the
  pre-training recipe.
- JMP pre-trains for 2 epochs on 120M systems → 240M forward passes.
  We have 60M matches and could plausibly do 10-30 epochs → up to
  1.8B forward passes. Compute-wise this is feasible on the RTX 5080
  if the encoder stays ≤ 5M params (estimated 4-12h wall, well under
  budget.yaml's 24h ceiling).
- JMP discards pre-training heads when fine-tuning. We may NOT need
  to do this — our heads correspond to actual downstream tasks
  (win/duration/items) we want to serve in production, so the
  pre-training heads ARE the inference heads. This is the
  multi-query foundation model pattern from [[cui2022m6]] (M6-Rec),
  not the JMP pattern. We follow M6's deployment but JMP's training.
