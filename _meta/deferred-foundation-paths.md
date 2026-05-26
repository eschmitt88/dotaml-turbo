---
name: deferred-foundation-paths
description: Foundation-model architectural variants we considered but deferred after the v4 diagnostic confirmed the encoder is sound and the val_auc ceiling is data-bound rather than architecture-bound. Pick one of these up if the downstream-query work surfaces a specific representation deficiency that an architectural change would address.
date: 2026-05-26
status: deferred
---

# Deferred foundation-architecture paths

After foundation-v3 (val_auc=0.6462, missed target) → v3-ablations
(both arms NEGATIVE, ruled out duration loss-form and player
embeddings) → v4 (val_auc=0.6471, attribution math closes) → v5 PMAE
(HALTED, over-specialization to per-token reconstruction) → v6 JEPA
(HALTED, representation collapse) → v4 diagnostic (encoder learns
semantically meaningful representations, organizes matches by
predicted outcome along PCA-1 = 0.98 corr with win_pred), the
project converged on: **the v4 encoder is sound; the 0.647 ceiling is
data-bound, not architecture-bound**. Pivoting to downstream queries
on v4. These paths remain valid if a specific representation
deficiency surfaces.

## Path A — v7-rich-skill-features (engineered features)

Extend per-player input feature block from 8 → ~14 features with
item-derived skill proxies (last20_gpm, last20_avg_hd_log1p,
last20_avg_kda, hero-specific variants) and richer hero-novelty
signal (days_since_this_hero_log1p). Same v4 architecture. Tests
whether richer engineered features (no embeddings) can close the
v4 → iso_teambias gap.

- **Cost**: ~10h (3-4h CPU data rebuild + 6h training).
- **Triggers to pick up**: if downstream item-rec queries reveal that
  v4 is missing per-player item-history signal that users wish were
  there. Or if we want one more shot at the val_auc ceiling.
- **Risk**: low. Feature engineering on the same architecture has
  historically lifted val_auc (transformer-plus-features-740 +0.013).

## Path B — v7-mage-lite (variable mask rate on v5 scaffolding)

~50 line change to v5: replace fixed `p_group=0.4` mask rate with
variable rate sampled from truncated Gaussian (μ=0.55, σ=0.25,
range [0.15, 0.95]). Same per-group reconstruction loss. Tests
whether fixed mask rate caused v5's over-specialization
pathology.

- **Cost**: ~10h training (reuses v5 scaffolding verbatim aside from
  the mask scheduler).
- **Triggers to pick up**: if we revisit foundation-pretrain after
  more SSL literature digestion, or if downstream queries need
  conditional generation (start from all-masked, iteratively decode)
  that the trained encoder doesn't naturally support.
- **Risk**: medium. Variable masking is the MAGE innovation that
  most directly addresses the v5 failure mode, but the fundamental
  reconstruction-loss objective still risks over-specialization.

## Path C — v7-cvae (conditional VAE, full generative model)

Encoder q(z | inputs, outputs) → sample z → decoder p(outputs | z,
inputs). Foundation-aligned: the latent IS the rep, decoder is
auxiliary. Explicit probability story (ELBO loss). KL term
mitigates representation collapse explicitly. Standard mitigations
for posterior collapse: KL annealing (β=0 → 1 over 5-10 epochs),
free-bits.

- **Cost**: ~15h (more code than v5/v6, but no data rebuild).
- **Triggers to pick up**: if downstream queries want SAMPLES of
  end-states (e.g., "show me 5 plausible game flavors for this
  lineup") rather than just point estimates. Or if we want a
  principled generative model for the foundation framing.
- **Risk**: medium-high. Posterior collapse is real; KL/recon
  balance is finicky; bigger code change than MAGE-lite.

## Path D — v7-cross-head-conditioning (head couplings)

~+200 line change: each multi-task head's prediction is conditioned
on (or attends to) the other heads' predictions. E.g., item head
attends to win head's prediction; KDA head attends to items head.
Captures some joint structure between heads without full generative
modeling.

- **Cost**: ~8h (architecture change, no data rebuild).
- **Triggers to pick up**: if downstream item-rec queries find that
  predicted items don't make sense given predicted KDA/GPM, or vice
  versa. Tests whether head couplings improve coherence.
- **Risk**: low. Smallest architectural change; explicit signal flow.

## Path E — v7-diffusion (mixed-type joint distribution)

Continuous diffusion (DDPM) on numerical end-stats (KDA, GPM, HD,
NW, dur) + D3PM categorical diffusion for items + binary win.
Models the FULL joint conditional distribution p(end_state | inputs).
Allows sampling, conditional generation, marginal evaluation.

- **Cost**: ~24-36h (significantly larger code, longer training, more
  hyperparameters).
- **Triggers to pick up**: only if downstream queries demand full
  joint distribution modeling (e.g., "given that radiant wins, what
  item builds make sense across both teams jointly?"). Otherwise
  CVAE captures most of the benefit at lower cost.
- **Risk**: high. Most novel in this codebase, longest train, largest
  code surface, highest probability of "different failure mode but
  same underlying signal problem."

## Decision rule

If a specific downstream query reveals a representation deficiency
that a particular path would address, expand that path into a full
proposal at `experiments/_proposals/`. Otherwise these stay deferred.

Do NOT pick one up just because the SSL family looks appealing in
isolation — the v4 diagnostic showed the encoder is sound and the
data signal is the binding constraint. The next ceiling lift, if
any, is more likely to come from richer input features (Path A) or
from accepting v4 as the ceiling and focusing on derived queries.
