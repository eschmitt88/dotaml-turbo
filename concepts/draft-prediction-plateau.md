---
kind: concept
name: "draft-prediction-plateau"
status: growing
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaML.md
  - literature/papers/hodge2017win.md
  - experiments/2026-05-15-plateau-baseline-740/README.md
  - experiments/2026-05-15-plateau-architectures-740/README.md
  - experiments/2026-05-16-transformer-hp-sweep-740/README.md
  - experiments/2026-05-17-player-features-740/README.md
  - experiments/2026-05-18-player-features-prepatch-740/README.md
  - experiments/2026-05-18-transformer-plus-features-740/README.md
related_concepts:
  - draft-only-win-prediction
  - hero-embedding-vs-onehot
related_experiments:
  - 2026-05-15-plateau-baseline-740
  - 2026-05-15-plateau-architectures-740
  - 2026-05-16-transformer-hp-sweep-740
tags: [empirical-finding, capacity-vs-accuracy, hp-robust]
---

# draft-prediction-plateau

## Definition

The observation that, on Dota 2 Turbo draft-only win prediction, test
accuracy and test AUC saturate within a narrow band — once the dataset
is large enough (millions of matches) and basic representation issues
are fixed. Empirically, the band is ≈ 0.619-0.635 AUC across the prior
art's six architectures, NOT a single 0.635 number for all of them.

**Refinement (2026-05-15):** the 0.635 figure is the v5 Transformer's
ceiling specifically. The v3 LightGBM ceiling sits lower at ≈ 0.619.
Replication on the patch-7.40 snapshot under HCE confirmed the v3
LightGBM number to within 0.003 (val_auc 0.6161 vs prior test_auc
0.6189 — see [[2026-05-15-plateau-baseline-740]]). The
**architecture-spread within the plateau** is itself a thing to model:
~0.016 AUC of headroom between the LightGBM and the Transformer
families on the prior art, not noise.

**Refinement (2026-05-15, second experiment):** the architecture-spread
is **family-driven** (Transformer beats FFN), but the **within-FFN
ordering does not reproduce** on patch-7.40 under HCE. See
[[2026-05-15-plateau-architectures-740]]. Three-architecture sweep:

- LightGBM (one-hot, prior baseline): 0.6161
- SimpleFFN (52k params, 64-dim embeds): 0.6217 (+0.006)
- ResidualFFN (225k params, 64-dim embeds): 0.6199 (+0.004) — **lower than SimpleFFN**, inverting prior art
- Transformer (82k params, attention over 11 tokens): 0.6322 (+0.016) — within 0.003 of v6's 0.6354

The Transformer-vs-FFN gap is large and reliable (≥0.011 AUC, regardless
of which FFN you pick); the FFN-internal gap inverted, suggesting
either hyperparameter sensitivity (v5's recipe was tuned on a smaller
pre-7.40 set), an artifact of switching v4 from one-hot to embeddings,
or single-seed noise. A multi-seed FFN sweep is queued to disambiguate.

**Refinement (2026-05-16, third experiment): the Transformer ceiling is
HP-robust.** A 60-trial Optuna TPE+ASHA sweep over a minimal-Transformer
baseline (9-dim search: d_model, n_heads, n_layers, ff_mult, embed_dim,
lr, weight_decay, dropout, batch_size) found best val_auc=0.6318 — within
0.001 of the un-tuned prior Transformer (0.6322). All 5 trials that ran
to convergence cluster in val_auc ∈ [0.6311, 0.6319] (0.0008 spread).
ASHA-pruned trials (55 of 60) had best ep-3 val_auc ≤ 0.6310. See
[[2026-05-16-transformer-hp-sweep-740]]. The ~0.632 Transformer ceiling
is therefore *not* an under-tuned point in a broader HP landscape — it
is a property of (architecture vocabulary × data) on this snapshot.

Implication for ceiling-breaking: further HP tuning is exhausted as a
lever. The remaining levers are structural mutation of the model (LLM-
driven program search à la [[concepts/evolutionary-expansion]]) or
new data features (draft order, lane assignment, hero-pair history,
player MMR). Anything that beats ≈ 0.632 by ≥ 0.005 must originate
from one of those, not from HP search.

## Why it matters here

In the prior-art DotaML repo, six successive model generations spanning
LightGBM, SimpleFFN (47k params), ResidualFFN (228k params), and a
Transformer with learned hero embeddings + masked-input training
(152k params) all land within ~0.04 AUC and ~1pp accuracy of each other.
The v5 README explicitly states "we may be approaching fundamental limits
of hero draft prediction."

For `dotaml-turbo`, this number is the load-bearing baseline. Any new
experiment should be evaluated against three implied tests:

1. **Sanity:** does it match the plateau on the new patch-7.40 snapshot?
2. **Patch effect:** does the plateau itself shift on a larger, more
   recent dataset?
3. **Ceiling:** does any new technique meaningfully exceed it?

A result inside ±0.01 AUC of the **architecture-matched** prior-art
ceiling (LightGBM ≈ 0.619, Transformer ≈ 0.635) should be reported as
"at the plateau for that architecture," not as a successful new model.
A result that exceeds the upper end of the architecture-spread (i.e.
val_auc > ~0.645) is the genuinely interesting case.

After 2026-05-16: HP-search has been ruled out as a lever for the
Transformer architecture vocabulary on this snapshot. Any val_auc > 0.640
result must come from structural mutation, new features, or new data,
not from re-tuning existing architectures.

**Independent attestation (2026-05-17):** Hodge et al. 2017
([[literature/papers/hodge2017win]]) report hero-only Dota 2 win
prediction accuracy of 55-59% across LR and RF on mixed-rank data —
matching our `plateau-baseline-740` val_acc=0.5866 (val_auc=0.6161)
within 0.01. The same paper reports that adding in-game telemetry
(team kills, damage, gold, net worth) lifts accuracy to 75-76% — a
~17 pp gap that demonstrates the broader prediction task has
substantial headroom once feature sets richer than hero-IDs are
admitted. This is independent confirmation that the ~0.62 ceiling is
an information bottleneck, not a model bottleneck, and motivates
extending pre-game features (player identity, draft order) before
investing in further architectural sophistication.

**Refinement (2026-05-17, fourth experiment): the information lever
exists but is bounded by Turbo's anonymous-account fraction.** See
[[2026-05-17-player-features-740]]. Adding ~90 per-player history
features (smoothed overall + hero-specific winrate, recent form,
co-play premade detection, days-since, anonymous flag) to the
LightGBM baseline raised val_auc by only **+0.0067** (0.6161 →
0.6227). This is real signal (heroes-only sanity rebuild matches
plateau-baseline within 0.0001) but **well below the +0.020 target
and well below the architectural Transformer ceiling of 0.6322**. So
on patch-7.40 Turbo, **architecture is a stronger lever than this
specific set of player features**, reversing the ranked-MOBA
intuition from Hodge 2017.

Why the small effect:
- **66% of player-slots are anonymous** (Steam-private profiles;
  account_id ∈ {0, 4294967295}); 12.7% of val matches have ALL 10
  players anonymous. Mean = 6.66 anonymous/match.
- The dominant feature (top-10 importances ALL `pX_smoothed_winrate_hero`)
  depends on per-player-per-hero samples that are inherently sparse
  on a 98-day snapshot.
- Coverage-bucket diagnostic shows monotonic lift (low/med/high val_auc
  = 0.6159/0.6230/0.6296), confirming cold-start is binding. A
  pre-patch ingest would help — but the binding feature is exactly
  the one most affected by metagame drift across patches, so pre-patch
  for hero-specific winrate is risky per Hodge's metagame warnings.

Architecture-matched ceilings on patch-7.40 are therefore:

- LightGBM one-hot:               0.6161 (`plateau-baseline-740`)
- LightGBM + 90 player features:  0.6227 (`player-features-740`)
- SimpleFFN (52k):                0.6217 (`plateau-architectures-740`)
- ResidualFFN (225k):             0.6199 (same)
- Transformer (82k):              **0.6322** (same)
- Transformer HP-tuned (60-trial Optuna): 0.6318 (`transformer-hp-sweep-740`)
- LightGBM + player features, high-coverage val subset: 0.6296

Any val_auc > **~0.640** on this snapshot must come from a NEW
information axis that is neither hero-IDs nor patch-7.40-only player
history (e.g., draft order from `picks_bans[]`, lane/role inference,
hero-pair history, structural mutation of the Transformer, or
larger pre-patch player-coverage with metagame-drift handling).

**Refinement (2026-05-18, fifth experiment): cold-start was NOT the
binding constraint — the casual/anonymous-player tail IS.** See
[[2026-05-18-player-features-prepatch-740]]. Extending the per-player
aggregator with ~127 days of pre-7.40 history (Aug–Dec 15 2025)
raised val_auc by only **+0.0028** (0.6227 → 0.6256), short of the
+0.005 target. The user's pre-experiment prediction that
hero-specific player skill is patch-stable was correct (top-10
features unchanged, no metagame-drift artifact), but the cold-start
hypothesis (low-bucket coverage val matches would gain most from
extended history) FAILED. Instead:

- LOW bucket gained **+0.0014** (least) — because low-bucket players
  had only 2.3% prepatch fraction (4 prepatch games avg). They're
  genuinely casual/new accounts, not active-but-uncached players in
  a cold-start window. Pre-patch data couldn't rescue them because
  they weren't around then either.
- MEDIUM bucket gained +0.0026
- HIGH bucket gained **+0.0043** (most) — active players who already
  had pre-patch presence (24.9% prepatch fraction, 81 games avg) got
  proportionally more new history.

**Big-deal observation:** the HIGH-coverage val_auc reached **0.6339**,
which **beats the architecture-only Transformer ceiling (0.6322)** for
the first time. For the active 1/3 of patch-7.40 val matches, player
features now extract more signal than 82k attention parameters can.
The whole-val ceiling is still bound by the casual/anonymous tail
(66% anonymous-per-match unchanged, 12.6% of val matches have all-10
anonymous).

So the architecture-matched ceiling table now reads:

| approach (whole val) | val_auc |
|---|---|
| LightGBM bag-of-heroes (`plateau-baseline-740`) | 0.6161 |
| LightGBM + patch features (`player-features-740`) | 0.6227 |
| LightGBM + prepatch features (`player-features-prepatch-740`) | **0.6256** |
| SimpleFFN 52k | 0.6217 |
| ResidualFFN 225k | 0.6199 |
| Transformer 82k (`plateau-architectures-740`) | **0.6322** |
| Transformer HP-tuned (`transformer-hp-sweep-740`) | 0.6318 |
| LightGBM + prepatch features, HIGH-coverage subset | **0.6339** ← new |

The architecture-vs-information comparison flipped on the active
subset. The natural next experiment is the COMBINATION
(Transformer + prepatch player features) which has not been tried.
For the casual/anonymous tail, no amount of historical data helps;
that subproblem requires anonymous-aware modeling (per-team
aggregates over the known-player subset, or a separate head).

**Refinement (2026-05-19, sixth experiment): combination is nearly
additive — the architecture-vs-information dichotomy resolves to
"use both".** See [[2026-05-18-transformer-plus-features-740]].
Combining MinimalTransformer (alone: 0.6322) with the 80-dim
per-player feature block (alone via LightGBM: 0.6256), via
`Linear(8, d_model)` projection added per-slot to hero embeddings,
gives val_auc=**0.6452** on whole val. That's +0.0133 over
Transformer-only and +0.0196 over LightGBM-with-features — closely
matching the additive sum of the two individual lifts (which would
predict ~0.6417), confirming the two levers address minimally
redundant information.

Even more strikingly, ALL coverage buckets lifted:

| coverage bucket | prev best (player-features-prepatch) | combined | Δ |
|---|---|---|---|
| low    | 0.6173 | 0.6347 | +0.0174 |
| medium | 0.6256 | 0.6443 | +0.0187 |
| high   | 0.6339 | **0.6560** | +0.0221 |

The HIGH-coverage val_auc of 0.6560 is closing in on Hodge 2017's
75-76% in-game-telemetry ceiling — achieved here with **pre-game info
only**. The LOW-bucket val_auc 0.6347 (mostly-anonymous matches)
**alone beats the architecture-only Transformer's whole-val ceiling
0.6322** — meaning attention extracts substantially more signal even
when most player features are anonymous-priors.

**Updated whole-val scoreboard:**

| approach | val_auc |
|---|---|
| LightGBM bag-of-heroes (`plateau-baseline-740`) | 0.6161 |
| LightGBM + patch features (`player-features-740`) | 0.6227 |
| LightGBM + prepatch features (`player-features-prepatch-740`) | 0.6256 |
| SimpleFFN 52k (`plateau-architectures-740`) | 0.6217 |
| ResidualFFN 225k (same) | 0.6199 |
| Transformer 82k (same) | 0.6322 |
| Transformer HP-tuned (`transformer-hp-sweep-740`) | 0.6318 |
| **Transformer + player features (`transformer-plus-features-740`, 77k)** | **0.6452** ← new |
| LightGBM + prepatch features, HIGH-coverage subset | 0.6339 |
| Combined, HIGH-coverage subset | **0.6560** ← new |
| Combined, n_anon ≤ 1 subset (extrapolated) | ~0.66+ |

Any val_auc > **~0.66** on this snapshot must come from either
(a) anonymous-aware modeling that lifts the bottom-tercile out of the
~0.635 floor, (b) richer player representations (learned embeddings,
hero-pair history), or (c) longer training / bigger architecture on
the combined feature set (the combined model's `best_epoch=14=max`
suggests room).

## Connections

- [[draft-only-win-prediction]] — the task whose ceiling this names.
- [[hero-embedding-vs-onehot]] — both representations hit the same
  ceiling in the prior art, suggesting representation is not the
  bottleneck.
- Hypotheses for the source of the ceiling (to be tested):
  hero-only information genuinely under-determines the outcome;
  label noise from fake matches / queue dodges; player-skill
  variance that the model cannot see; patch instability across the
  training window; calibration vs accuracy trade-off.
