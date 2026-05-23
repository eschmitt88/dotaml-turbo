---
kind: proposal
slug: foundation-mvp-740
date: 2026-05-22
status: implemented
experiment: experiments/2026-05-22-foundation-mvp-740/
result: "HYPOTHESIS NOT CONFIRMED. baseline_multitask_repro val_auc=0.6470 (within noise of cleanup anchor 0.6477 — scaling 77K→5M is neutral). foundation_mvp val_auc=0.5058 and foundation_no_patch_token val_auc=0.4984 — both near-random; full design broke training. Diagnostic per the proposal's anticipated result fork: scale isn't the problem (baseline ablation works), the architectural additions are. Likely culprits: UW-SO loss-scale misapplication (T=0.45 + raw loss scales differing 30× → items get ~30× weight over duration), PMAE collapse (mae_loss → 0 mid-training; masking implementation suspected buggy or trivially solvable), possibly (team,team) attention bias interacting badly with multi-task heads. train_win loss INCREASED over training — model actively anti-learning. ~17h compute on stable JEDEC RAM, zero kernel events. Component-isolation v2 experiments needed."
hypothesis: "A ~5M-param Transformer foundation model trained jointly on multi-task heads (win, duration, items, KDA-GPM-hero_damage) with a PMAE auxiliary objective, FT-Transformer skeleton, permutation-equivariant within-team encoding (NO per-slot positional info — `player_slot` is arbitrary lobby order and carries no role/lane semantics in Turbo), patch_id as a learned conditioning token, UW-SO loss weighting, and a shared decoder with task-as-token prompting — pre-trained on ~30-40M Steam-API matches across the available Aug 2025 → Mar 2026 patches — lifts whole-val win val_auc by ≥ 0.003 over the multitask-740 ceiling (target val_auc ≥ 0.6525). It also enables downstream queries from a single trained encoder: hero-pair synergy above naive baseline, lineup-vs-lineup matchup scoring, item recommendation conditioned on net_worth budget, and fun-pair (max-kills) analysis."
rationale: >
  multitask-740 (val_auc=0.6495, +0.0022 over same-data single-head sanity)
  showed the encoder bottleneck is gradient-signal density, not parameter
  count, and that joint multi-task supervision moves the ceiling. The next
  step scales the same thesis into a foundation model: more data (pre-patch
  included), more parameters (5M vs 77K), richer pre-training objectives
  (PMAE auxiliary), correct inductive biases (permutation-equivariance
  within each team; team-membership is the only ordering signal that
  carries semantic content in Turbo), and a shared-decoder multi-query
  design that supports many downstream questions from one trained encoder.

  Twelve foundation-model papers across tabular, recommendation, sports,
  Earth observation, time-series, biology, robotics, and speech converge
  on architectural patterns we adopt directly: FT-Transformer skeleton
  (Gorishniy 2021), per-column proportional MAE masking (Kim 2024, PMAE),
  UW-SO loss weighting (Kirchdorfer 2024), SAINT-style contrastive +
  denoising pre-training as optional v2 (Somepalli 2021), M6-Rec's
  one-encoder-many-prompts framing (Cui 2022), HIGFormer's two-stream
  player + team design (Wang 2025), JMP's evidence that joint supervised
  multi-task pre-training beats unsupervised by ~59% (Shoghi 2023),
  Pangu-Weather's Earth-Specific Positional Bias (Bi 2022) — but
  adapted as a much smaller (team_query, team_key) 2×2 bias rather than
  per-slot, since `player_slot` order in Turbo is arbitrary lobby ordering
  with no semantic content; Moirai-MoE's argument against per-patch
  separate projections (Liu 2024), Octo's modular-tokenizer +
  readout-token + shared-trunk template at the closest published scale
  (Ghosh 2024), Whisper's task-as-token prompting feeding a single
  shared decoder (Radford 2022), and ForkMerge's parameter-averaging
  insurance against negative transfer (Jiang 2023). The "many conditional
  queries from one model" framing — the user's primary motivation — is
  slightly ahead of published tabular FM work but well-precedented in
  recommendation (M6-Rec) and robotics (Octo, Whisper) at adjacent scales.
reads:
  - "[[literature/papers/gorishniy2021revisiting]]"
  - "[[literature/papers/kim2024predict]]"
  - "[[literature/papers/kirchdorfer2024analytical]]"
  - "[[literature/papers/somepalli2021saint]]"
  - "[[literature/papers/cui2022m6]]"
  - "[[literature/papers/wang2025player]]"
  - "[[literature/papers/jiang2023forkmerge]]"
  - "[[literature/papers/bi2022pangu]]"
  - "[[literature/papers/liu2024moirai]]"
  - "[[literature/papers/shoghi2023molecules]]"
  - "[[literature/papers/ghosh2024octo]]"
  - "[[literature/papers/radford2022robust]]"
  - "[[concepts/tabular-foundation-model]]"
  - "[[concepts/masked-modeling-tabular]]"
  - "[[concepts/uncertainty-weighted-multitask]]"
  - "[[concepts/multi-query-foundation-model]]"
  - "[[concepts/attention-bias-positional]]"
  - "[[concepts/task-as-token-prompting]]"
  - "[[concepts/supervised-multitask-pretraining]]"
  - "[[concepts/draft-prediction-plateau]]"
  - "[[mocs/foundation-models]]"
  - "[[experiments/2026-05-20-rich-supervision-multitask-740]]"
  - "[[experiments/2026-05-19-upstream-data-cleanup-740]]"
expected_metric:
  name: val_auc
  target: 0.6525
  direction: higher-is-better
design_sketch:
  - "**Architecture (~5M params).** FT-Transformer skeleton (Gorishniy 2021) with the first-layer first-LayerNorm REMOVED per the paper's note. d_model=256, n_heads=8, n_layers=6, FFN dim 4×d_model=1024, dropout=0.0, GELU, Pre-Norm. ~5.0M trainable params end-to-end."
  - "**Canonical input ordering.** Before any model input, sort each team's 5 (hero, player_features) tuples by `hero_id` ascending. Removes any residual lobby-order signal from `player_slot` (which carries no semantic content in Turbo — see Important data constraint below). Two matches with identical drafts produce identical inputs regardless of original lobby order. Free, no params."
  - "**Tokenization (per match, ~13 tokens).** 10 hero tokens (per-slot hero embedding + team embedding ∈ {radiant, dire}, summed). NO per-slot positional embedding within team — within-team ordering is arbitrary so positional encoding would teach spurious patterns. 1 patch token (learned embedding indexed by patch_id ∈ {patches in train corpus}). 1 lobby token (learned embedding of lobby_type if derivable from raw_json; else dropped). 1 per-slot player-feature token per slot (8-feature MLP → d_model) injected via cross-attention from the corresponding hero token (HIGFormer two-stream pattern). Task-readout tokens (see Heads) appended at decoder time."
  - "**Permutation-equivariance within team.** Because there is no per-slot positional embedding, the encoder is naturally permutation-equivariant within each team's 5 hero tokens. The only ordering signal is the team embedding (radiant vs dire), which IS semantically meaningful. Pangu's per-slot attention bias is NOT adopted (it'd encode arbitrary lobby positions); a much smaller (team_query, team_key) 2×2 bias matrix per attention head IS adopted, since attending across teams is genuinely different from attending within team. ~64 extra params total across 6 layers × 8 heads × 4 entries."
  - "**Patch conditioning (MVP: token; ablate FiLM later).** Patch is one input token; the encoder's attention surfaces patch-conditioned representations as needed. Cheap MVP baseline. v2 follow-up: FiLM on LayerNorm conditioned on patch_id (more expressive); skip per-patch separate projections per Moirai-MoE."
  - "**Heads — shared decoder + task-as-token prompting.** Whisper-style: decoder is a small transformer (2 blocks, d_model=256, n_heads=8) prefixed with a task token from the vocabulary {`<|win|>`, `<|duration|>`, `<|items|slot=k|>`, `<|kda|slot=k|>`, `<|gpm|slot=k|>`, `<|hd|slot=k|>`}. Per-task projection layers: win → 1 logit (BCE); duration → 1 scalar (SmoothL1 on log(seconds), optionally also log_var for distributional); items → 305-dim multi-label (sampled-softmax BCE per slot); KDA/GPM/HD → 1-3 scalars (SmoothL1 on log(1+x), per slot). New query types become config additions."
  - "**Pre-training loss.** L = Σ ω_k(t) · L_task_k + α_mae(t) · L_mae. The ω_k are UW-SO weights with a single tunable temperature T (eliminates the α_d=0.5 → 0.15 fragility from multitask-740). α_mae(t) annealed 1.0 → 0.1 over training so MAE provides early representation but doesn't dominate late epochs."
  - "**PMAE auxiliary objective.** Per-column proportional masking M_j = a·logit(1−p_obs,j) + b, then group-mask whole semantic units (a full player feature block; a full item-list per slot; a hero token; the patch token). Target ~30-50% expected mask rate. NOT uniform 75% (PMAE shows this is wrong for tabular); NOT per-token random (groups force the encoder to recover semantically coherent units)."
  - "**Data.** Train: 2025-08-15 → 2026-02-23, ~30-40M matches across the patches available (7.39 → 7.40 inclusive); confirm patch boundaries at data-load time. Val: 2026-02-24 → 2026-03-09 (same as cleanup-740 / multitask-740, so val_auc is comparable to existing anchors). Test: [2026-03-10, 2026-03-23] sealed per HCE. Player-aggregate features computed leading-window strict (no test-window data leaks)."
  - "**Training.** Adam, lr=1e-3 → warmup 1k steps → cosine to 1e-5 over total steps. Batch_size=512 (smaller than multitask-740's 8192 because the model is larger and the per-batch compute is higher). bf16 autocast. max_epochs=30, early-stop patience=5 on val_win_log_loss. Per-trial subprocess isolation via run_all.sh (defensive — Blackwell torch DataLoader bug is no longer reproducing on JEDEC RAM but the pattern is harmless). Estimated wall: 8-24h."
  - "**Three ablations:**"
  - "  • `baseline_multitask_repro` — replicate multitask-740 design at the new scale (5M params, no PMAE, no patch token, no team-bias, original α-weighting). Anchors the scaling lift independently from architectural changes. Expected val_auc ≥ 0.6495."
  - "  • `foundation_mvp` — PRIMARY: full design above. Target val_auc ≥ 0.6525."
  - "  • `foundation_no_patch_token` — sanity ablation: full design minus the patch token. Validates that cross-patch conditioning matters (or detects that the patch is leaking through player aggregates already)."
  - "**Cross-patch generalization diagnostic.** After training, evaluate `foundation_mvp` separately on val matches grouped by patch_id. Patch-conditioning is genuinely useful if held-out patches don't regress disproportionately."
  - "**Diagnostics (NON-NEGOTIABLE):**"
  - "  • Per-task val metrics: win (AUC, log_loss, brier), duration (MAE on log_seconds, calibration plot), items (mAP@10, mean_precision_at_10, mean_recall_at_10), KDA/GPM/HD (per-dim MSE on normalized targets)."
  - "  • Coverage-bucket win val_auc (low/med/high terciles by mean n_games_log1p) — carried over from prior experiments for direct comparison."
  - "  • Loss component traces per epoch: tr_win, tr_dur, tr_items, tr_kda, tr_mae and the UW-SO ω weights. Reveals if any task is starving or dominating."
  - "  • Train-val gap on the win head — overfit signal."
  - "  • In-vocab item rate per slot for the item head (sanity: ~30% of slots get unique embedding entries was multitask-740's observation; expect similar)."
  - "**ForkMerge insurance.** If at any check (every 5 epochs) val_win_log_loss regresses vs `baseline_multitask_repro`, fork the model and merge by validation-weighted parameter averaging (Jiang 2023). Defensive only; not expected to fire if UW-SO does its job."
risks:
  - "**Negative transfer.** More aux heads + bigger model + new pre-training objectives could degrade the primary win head. Mitigation stack: UW-SO replaces hand-tuned α (auto-balances), `baseline_multitask_repro` ablation anchors the scaling component, ForkMerge insurance if val_win regresses mid-training."
  - "**Compute budget.** 5M-param model × 30-40M matches × 30 epochs × PMAE compute + multi-task heads could push past `budget.yaml`'s 24h ceiling. Mitigation: profile after epoch 1; if extrapolated wall > 18h, scale down to d_model=192 or n_layers=4. Don't run all 3 ablations sequentially — run `foundation_mvp` first; ablations only if primary succeeds."
  - "**Cross-patch confound.** Patches differ in hero balance + item changes + map rules. A single patch_id token may be insufficient. Detected via the cross-patch generalization diagnostic. Mitigation if observed: v2 with FiLM conditioning or Pangu-style (patch, slot) bias."
  - "**MAE schedule on tabular.** PMAE's proportional masking is well-grounded, but the optimal MAE weight schedule + field-group definition is empirical. We may need a brief mask-rate ablation if loss curves look pathological. Default: 40% expected mask rate, group-masked, α_mae annealed 1.0 → 0.1."
  - "**Shared decoder may underfit heterogeneous tasks.** Whisper's task-token pattern works at LLM scale; at 5M params with 4+ distinct output shapes (binary, regression, multi-label, vector), a 2-block decoder may not have enough capacity. Mitigation: profile per-task val curves; if any task plateaus far below its head-per-task multitask-740 baseline, escalate to either a larger decoder or per-task heads."
  - "**Data-build uncertainty.** Pre-patch raw data is on disk but the cross-patch player-aggregate features haven't been computed for the foundation training window yet. The data prep is a new build (~3-4h CPU). Build under the same defensive checkpointing as cleanup-740 to avoid the silent on-disk corruption issue (now hardware-fixed but defense-in-depth is cheap)."
  - "**Items head is the highest-variance prediction.** 305-dim multi-label per slot. May need a dedicated mini-transformer expert (π_0-style, ~300K extra params) instead of a linear projection. Skip in MVP; ablate if mAP@10 stalls below 0.30."
related_prior:
  - 2026-05-20-rich-supervision-multitask-740
  - 2026-05-19-upstream-data-cleanup-740
  - 2026-05-19-transformer-plus-features-extended-740
  - 2026-05-19-player-embedding-prelim-740
estimated_runtime: "≈8-24h on RTX 5080 for `foundation_mvp` alone. Three-ablation sequential run: 24-48h. Disk: ~100MB model checkpoint + ~50MB metrics/plots per ablation. May need a one-time ~3-4h pre-build to compute leading-window player aggregates across the foundation training window (Aug 2025 → Feb 2026). Foundation training itself fits within budget.yaml 24h-per-job ceiling if scoped to the primary ablation; all-three-ablations exceeds and should be sequenced across days."
---

# foundation-mvp-740 — first scaled foundation model for Dota 2 Turbo match modeling

## Where this fits in the arc

The project has moved through three distinct ceilings: LightGBM-baseline (0.6161), Transformer alone (0.6322), Transformer + per-player features (0.6452), extended training (0.6477), then multi-task supervision (0.6495). The pattern over the last four ceilings: incremental, modest, and increasingly hard to break — the 0.6477 number held across three independent runs within 2e-5 before multitask-740 lifted it by a real-but-modest +0.0018.

multitask-740 also produced a useful negative result: a 16M-param player embedding gave ZERO lift over the 77K-param baseline. That ruled out parameter count as the binding constraint and pointed at **gradient-signal density** — the encoder was learning everything the radiant_win label could teach it but no more. multitask-740 then confirmed the theory: adding three auxiliary heads (duration, items, KDA-GPM-damage) gave the encoder ~10× more bits of supervision per match and shifted the ceiling.

This experiment scales that thesis. The foundation framing makes three changes at once:

1. **More data**: pre-patch matches (Aug 2025 → Dec 2025) included alongside patch-7.40. ~30-40M matches across ~3 patches, vs the 13M of patch-7.40 alone we've used so far.
2. **More architectural expressivity**: 5M parameters (vs 77K), 6 transformer layers, and correct inductive biases (permutation-equivariance within each team, team-membership as the only meaningful ordering signal, patch conditioning as a token).
3. **Richer pre-training**: joint multi-task supervised heads + a PMAE auxiliary objective using per-column proportional masking, jointly weighted by UW-SO (eliminating the α-tuning fragility).

The user's primary motivation isn't actually win-prediction ceiling-breaking — it's the *downstream query unlock*. A foundation model with the right structure answers many questions from one trained encoder: hero-pair synergy above the naive winrate baseline, lineup-vs-lineup matchup scoring, item recommendation conditioned on net_worth budget (the cost confound the user explicitly named), ban-set optimization, fun-pair (max-kills) analysis. The win-AUC target (≥ 0.6525) is a sanity check that the foundation is at least as good as the specialized multitask model on its specialized task; the bigger payoff is the new query interface.

## Important data constraint: no slot semantics

A late but load-bearing design correction: `player_slot` in Turbo raw_json
is **arbitrary lobby ordering**, NOT role / lane / pick-order. The values
`0..4` (radiant) and `128..132` (dire) are just team-membership + lobby
position — and lobby position is essentially randomized by matchmaking +
party-order. Two matches with identical drafts can have different slot
orderings; the model should treat them as the same input.

This rules out several architectural moves that initially seemed
attractive (Pangu's per-slot positional bias being the headline example).
The correct inductive bias is **permutation-equivariance within each
team**: the model should treat each team as an unordered set of 5
(hero, player_features) tuples, with the only ordering signal being
team membership (radiant vs dire).

Concretely:

- Sort each team by `hero_id` ascending at load time (canonical input).
- No per-slot positional embedding within team.
- Team embedding (radiant=0, dire=1) is the only positional info.
- Per-attention-head (team_query, team_key) 2×2 bias matrix as the only
  per-block asymmetry — encodes "attending across teams is different
  from attending within team" without making claims about within-team
  position.

This is a *stronger* inductive bias than what multitask-740 had (which
inadvertently used per-slot positional structure even though slots were
arbitrary). Prior experiments' val_auc numbers remain apples-to-apples
because the same wrong-but-consistent indexing was applied across all
ablations; the foundation model fixes it properly.

## What the 12 ingested papers buy us

The literature survey was load-bearing for several design decisions:

- **FT-Transformer + readout-token heads** (Gorishniy 2021, Ghosh 2024 Octo) — the consensus skeleton for tabular FMs at our parameter scale.
- **PMAE per-column proportional masking** (Kim 2024) — closes the question of how to mask. Not uniform 75% (vision); not 15% (BERT); not per-token random (loses semantic units). Field-grouped, ~30-50% expected, proportional to column availability.
- **UW-SO loss weighting** (Kirchdorfer 2024) — single temperature T replaces our 4 hand-tuned α's. We learned in multitask-740 that α_d=0.5 was too high and dropping to 0.15 was needed; UW-SO removes that brittleness.
- **(team_query, team_key) 2×2 attention bias** (Bi 2022 Pangu-Weather, adapted) — Pangu's per-slot bias does NOT transfer because `player_slot` in Turbo is arbitrary lobby order with no semantic content (no role/lane/pick-order info — confirmed with user). We adopt the much smaller (team, team) version of the same pattern: a per-head 2×2 bias encoding "attending across teams is different from attending within team." ~64 params total. The within-team permutation symmetry is preserved by omitting per-slot positional encoding entirely.
- **Single shared decoder + task-as-token prompting** (Radford 2022 Whisper) — instead of head-per-task, prefix the decoder with a task token. Adding new queries becomes a config addition. The user wants to ask many questions; this is the right shape.
- **Joint multi-task supervised pre-training beats unsupervised by 59%** (Shoghi 2023 JMP) — strongest evidence yet that we should train the supervised heads and the MAE auxiliary JOINTLY, not pre-train MAE then fine-tune. We were going to do this anyway; now grounded.
- **No per-patch separate projections** (Liu 2024 Moirai-MoE) — closes one of the patch-handling candidates. Either single shared projection + patch token (MVP), or sparse MoE FFN (overkill). Skip the middle path.
- **Two-stream player + team design** (Wang 2025 HIGFormer) — closest published architecture analog to what we're building. Validates the structure (player history → team aggregate → joint encoder).
- **ForkMerge insurance** (Jiang 2023) — if win-AUC regresses mid-training due to negative transfer, periodic parameter-averaging recovers. Cheap defense, composes with UW-SO.

## Three result forks

- **val_auc ≥ 0.6525 (CONFIRMED).** Foundation framing works. New whole-val ceiling. Downstream query interfaces become the next direction (item recommender with net_worth conditioning, lineup-vs-lineup scoring, hero-pair synergy analysis).
- **val_auc in [0.6495, 0.6525) (FLAT but on-baseline).** Foundation didn't hurt but didn't help the win head either. Still useful: the foundation model is now the substrate for downstream queries — the multitask-740 single-purpose model can't answer "lineup A vs lineup B" or "items conditioned on net_worth" elegantly, but the foundation can. The win-AUC ceiling holds; subsequent work targets the cross-patch generalization story and the multi-query unlock independently.
- **val_auc < 0.6495 (REGRESSION).** Negative transfer or bad scaling. ForkMerge fallback fires, or we drop back to multitask-740 scale and try a smaller architectural variation. The `baseline_multitask_repro` ablation will tell us whether it's the scale or the design that broke things.

## What's deliberately deferred to v2

Several known-good ideas are NOT in the MVP to keep complexity manageable:

- **SAINT-style contrastive auxiliary** (Somepalli 2021) — well-supported, adds ~20% compute. Skip in MVP; ablate later.
- **FiLM patch conditioning** (Bi 2022 / Perez 2017) — possibly better than patch-as-token. MVP uses the simpler token; FiLM is a v2 ablation.
- **π_0-style mini-transformer expert for the items head** (Black 2024) — items is the highest-dim prediction; a small dedicated expert might help. MVP uses a linear projection; escalate only if items mAP@10 stalls.
- **Sparse MoE FFN** (Liu 2024 Moirai-MoE) — useful if cross-patch heterogeneity is large. MVP uses dense FFN; revisit if cross-patch generalization is poor.
- **Hierarchical / heterogeneous-edge graph layers** (Wang 2025 HIGFormer) — full HIGFormer pattern adds substantial complexity. MVP uses just the two-stream design without the graph layers.

## Engineering plan

- **Reuse**: data loaders + clean parquet + rich-cols sidecar from multitask-740. Defensive parquet readers stay (no-op on clean data, insurance against any future read anomalies). All HCE rules and date assertions carry over.
- **New code**: PMAE masking module, UW-SO weight scheduler, (team_query, team_key) 2×2 attention bias module, task-as-token decoder, patch+lobby embedding tables, canonical-sort-by-hero_id preprocessor.
- **Data prep**: a one-time `build_foundation_features.py` that extends the cleanup-740 build to the foundation training window (Aug 2025 → Feb 2026). Uses the same defensive checkpointing. ~3-4h CPU build, ~2-3 GB additional parquet.
- **Sanity smoke**: 1-epoch run on 50k matches confirming all heads emit losses, MAE mask fires, UW-SO weights update, no NaN in any task's val metric, GPU at expected utilization.

The arc this opens — many conditional queries from one foundation, item recommendation with budget conditioning, lineup optimization — was the user's stated motivation back when we discussed scaling beyond multitask-740. The MVP is the minimum demonstration that the framing works; everything downstream branches from this proposal.
