---
kind: proposal
slug: player-embedding-prelim-740
date: 2026-05-19
status: implemented
experiment: experiments/2026-05-19-player-embedding-prelim-740/
hypothesis: "Adding a learned per-player embedding lookup (vocab ≈ 500K most-frequent non-anonymous accounts + 1 'rare' + 1 'anonymous' bucket; embed_dim=32) projected per-slot into the hero+feature sum lifts whole-val val_auc by ≥ 0.0020 over the cleanup-confirmed combined-model reference (target: val_auc ≥ ref+0.002, where ref = `upstream-data-cleanup-740`'s combined val_auc, expected ≈ 0.6477). The active 1/3 of val (HIGH coverage bucket) should benefit disproportionately, since long-tail players still fall into the 'rare' bucket and gain no marginal signal beyond their aggregated features."
rationale: >
  Up to now, every model has represented players solely through
  aggregated history features (smoothed_winrate, smoothed_winrate_hero,
  recent form, days_since, etc.). That representation is bounded by
  the features the aggregator chose to extract — and it cannot capture
  player style or hero-pair preferences that are not in the 8-feature
  schema. A learned per-player embedding is the natural richer
  representation: each frequent account gets a 32-dim slot updated
  end-to-end with the win-prediction objective, and the model is free
  to discover whatever latent attributes turn out to matter (aggression,
  laning preference, role flexibility, MMR-correlated style, etc.).
  Hodge 2017's hero-only ceiling of 55-59% and our own anchored 0.6477
  use only hero-IDs + aggregated features; per-player learned embeddings
  is the obvious next axis. The vocab cutoff (~500K accounts) is set
  to balance memory (32 dim × 500K × 4 bytes = 64 MB forward / ~256 MB
  with Adam state) against long-tail noise; sub-cutoff accounts share
  a single 'rare' embedding, naturally regularized by sample size.
  Anonymous accounts (66% of player-slots) all share a single 'anon'
  embedding, which is the best-trained of all and serves as a
  shrinkage anchor.
reads:
  - "[[concepts/draft-prediction-plateau]]"
  - "[[experiments/2026-05-18-transformer-plus-features-740]]"
  - "[[experiments/2026-05-19-transformer-plus-features-extended-740]]"
  - "[[experiments/2026-05-19-upstream-data-cleanup-740]]"
expected_metric:
  name: val_auc
  target: 0.6497  # reference 0.6477 + 0.002, pending cleanup-confirmed reference
  direction: higher-is-better
design_sketch:
  - **Vocab construction.** Single pass over the clean train parquet (`data/snapshots/7.40-2025-12-16/processed/player_features_prepatch_clean/train.parquet`). Count per-account appearance across all 10 player slots. Take the top-N most-frequent non-anonymous accounts; default N=500_000 (cover ≈85% of non-anonymous player-slots). Allocate vocab indices: 0=anonymous, 1=rare, 2..N+1=frequent. Save vocab as `results/player_vocab.json` (account_id → idx).
  - **Embedding layer.** `nn.Embedding(vocab_size=N+2, embedding_dim=32, padding_idx=None)` initialized N(0, 0.02). Total params ~16 MB at fp32.
  - **Architecture (single change to extended-training model).** Reuse `experiments/2026-05-19-upstream-data-cleanup-740/models.py` MinimalTransformerWithFeatures. Add a `player_embedding` module + a new per-slot projection `player_proj: Linear(32, d_model=64)`. Per-slot input becomes `hero_emb + feature_proj(features) + player_proj(player_emb_lookup(account_idx))`. CLS token unchanged. Total param delta ≈ 16M (embedding) + 2K (projection) over the prior 77K-param model.
  - **Regularization.** Apply `weight_decay=1e-3` only to embedding params (vs 0 for the rest of the model). Test once with `weight_decay=0` on embeddings as a sanity check — the long tail will overfit hard without it, so if `wd=1e-3` is the only viable knob we should know it.
  - **Vocab lookup at data load time.** Modify `data.py:load_arrays` to also emit `account_idx` arrays of shape (n_rows, 10) by looking up each player slot's account_id against the saved vocab. Unknown accounts (val-only accounts that weren't in train) map to the 'rare' bucket.
  - **Two ablations:**
    1. **`baseline_extended_clean`** — sanity replication of the cleanup-confirmed combined model (no embedding, just hero+features). Validates the new data pipeline reproduces the cleanup result.
    2. **`with_player_embedding`** — the primary; adds the embedding lookup + projection.
  - Training recipe identical to `transformer-plus-features-extended-740`: 30-epoch cap, Adam lr=1e-3, batch_size=8192, bf16 autocast, early-stop patience=5 on val_log_loss.
  - HCE strict; same date assertions in data.py.
  - Per-trial subprocess isolation via run_all.sh; MAX_RETRIES=3 per ablation.
  - **Coverage-bucket val_auc diagnostic** carried over from prior — split val into terciles by mean n_games_log1p across the 10 player slots. Critical for interpreting the embedding's contribution: if the LOW bucket moves significantly, the embedding is extracting non-feature signal from sub-cutoff accounts; if only HIGH moves, the embedding is mostly re-encoding what aggregated features already had.
  - **In-vocab-fraction diagnostic.** Per-bucket: fraction of player slots that hit the frequent vocab (not anonymous, not rare). Helps separate "embedding helped because of vocab coverage" from "embedding helped because of architectural capacity".
risks:
  - **Long-tail overfitting.** ~80% of non-anonymous accounts appear ≤5 times in training. Top-500K cutoff plus weight_decay=1e-3 plus shared 'rare' bucket are the three defenses. If overfitting persists (train_auc much higher than val_auc), tighter weight_decay or vocab cutoff at top-100K is the next iteration.
  - **Marginal-information redundancy.** If `smoothed_winrate_hero` already captures most of what a per-player embedding could encode, the gain will be tiny or zero. The HIGH coverage bucket movement is the cleanest signal here; flat HIGH is a strong "redundant signal" result.
  - **Vocab construction cost.** Single pass over 13M train rows for accounts on 10 slots = 130M account-slot pairs to count. ~5 min on CPU with a Counter or pandas value_counts; not a blocker.
  - **Anonymous embedding dominates.** With 66% of player-slots routing to a single embedding, that one row gets ~85K updates per epoch (vs ~10 updates for a typical frequent-vocab account). The 'anon' embedding will converge fast and may swamp the gradient signal for others; if so, the right fix is to FREEZE the anon embedding at a learned position-aware constant (or share with the 'rare' bucket) — flag for iteration if observed.
  - **VRAM.** 32 × 502K × 4 bytes = 64 MB embedding + 192 MB Adam state = 256 MB additional; fits easily on 16 GB.
  - **Memory at vocab-build time.** A Python Counter over 130M account-slot ints can spike to several GB; use numpy unique-counts on a chunked stream to keep peak RAM <8 GB.
  - **Dependency on cleanup completion.** This experiment requires the clean parquet at `data/snapshots/.../player_features_prepatch_clean/`. Implementation must check that path exists before scaffolding; if not, the cleanup experiment hasn't completed yet — fail loudly rather than fall back to the dirty parquet.
related_prior:
  - 2026-05-18-transformer-plus-features-740
  - 2026-05-19-transformer-plus-features-extended-740
  - 2026-05-19-upstream-data-cleanup-740
estimated_runtime: "≈45-60 min on RTX 5080: ~5 min vocab build + ~25-30 min Transformer training (single ablation, +embedding may add ~5% per epoch due to lookup) + ~25 min baseline_extended_clean replication + diagnostic + overhead. Disk delta: ~70 MB for vocab + embedding checkpoint. Well under budget.yaml ceilings."
---

# Player embeddings — the natural richer representation

Every experiment to date has represented players through hand-engineered aggregated features (smoothed winrates, recent form, days since last game, anonymity flag). That representation has carried us from 0.6161 (LightGBM baseline) to 0.6477 (combined Transformer + 8-feature block, extended training), a +0.0316 lift that is genuinely substantial. But it is also bounded: the 8 features are chosen by hand, and any latent signal in player identity that does NOT map onto those 8 axes is invisible to the model. The HIGH-coverage val_auc of 0.6588 hints that for active players, the architecture is already extracting most of the signal the features provide — but the LOW-coverage bucket sits at 0.6367 because anonymous and casual players have features that are mostly zeros and priors.

Per-player learned embeddings is the obvious next axis. Each frequent account gets a learnable 32-dim slot, updated end-to-end with the win-prediction objective. Whatever attributes turn out to matter — aggression, laning preference, role flexibility, MMR-correlated style, hero-pair affinity that the aggregator's per-hero-winrate column can't capture — the embedding can encode it without us naming it in advance. This is the prior-art DotaML repo's v7 motivation made explicit, and it's also what the AIRA2 evolutionary loop would propose for this problem statement.

Three plausible result forks:

- **val_auc ≥ ref+0.002 (CONFIRMED) and HIGH bucket moves the most.** Embeddings are picking up active-player signal beyond what features captured. The next iteration is a wider embedding dim and/or a hierarchical-prior shrinkage toward the anonymous baseline.
- **val_auc ≥ ref+0.002 (CONFIRMED) but LOW bucket moves the most.** Unexpected — would mean the 'rare' bucket and the 'anonymous' embedding are extracting signal that the per-account features cannot. Strong indicator that even shared embeddings have value, motivating a router architecture and/or learned anonymity-aware modeling.
- **val_auc < ref+0.002 (NOT CONFIRMED).** Either redundancy (features already capture the embedding's information) or overfitting (long tail swamping the signal). Either is a real finding; check the train-val gap to disambiguate. If train_auc rises but val_auc doesn't, tighten regularization in a follow-up.

This is a preliminary experiment by design — N=500K vocab, dim=32, single regularization knob, no hierarchical priors, no multi-seed. Establishes a clean reference number that a follow-up can refine. The right time to layer on attention-based player aggregation, hero-pair embeddings, or hierarchical Bayesian shrinkage is AFTER we know whether the plain embedding lookup helps at all.
