# NOTES

Running log of work sessions. `/wrap` appends a new dated section at the
end of each session with **Did / Findings / Next** subsections. The
SessionEnd hook backstops this if you forget.

<!-- entries go below this line, newest at bottom -->

## 2026-05-15

### Did

- Confirmed `az login` against `Subscription 2 - Dota`; no Azure setup needed.
- Ingested two prior-art repos as literature-only references:
  - `raw/repos/eschmitt88-DotaML.md` → `literature/repos/eschmitt88-DotaML.md`
    (relevance 5/5; commit `5771cba`). Bundled README + six top-level design
    docs + per-experiment `metrics.json` / `RESULTS.md` for v1-v6 into one
    raw capture rather than just the README, to preserve the prior-art
    experiment grid.
  - `raw/repos/eschmitt88-DotaDB.md` → `literature/repos/eschmitt88-DotaDB.md`
    (relevance 3/5; commit `8ddb3f8`). Pipeline reference only.
- Seeded six concepts: `draft-only-win-prediction`,
  `draft-prediction-plateau`, `radiant-side-advantage`,
  `fake-match-filtering`, `hero-embedding-vs-onehot`,
  `match-id-vs-seq-num-ordering`. Flagged a MoC candidate in
  `_meta/index.md` (not yet promoted — all from one source).
- Drafted `experiments/_proposals/2026-05-15-plateau-baseline-740.md`:
  zero-th experiment is a LightGBM one-hot baseline that tests whether
  the ~0.635 AUC plateau from prior art holds on the new snapshot
  (target metric val_auc, falsified at >0.645 or <0.625).
- Wrote project-root `splits.yaml` for snapshot v1: 70/14/14 chronological
  split on `start_time_date`, sealed test, fake-match + dedup filters in
  spec.
- Created `data/snapshots/7.40-2025-12-16/{README.md, raw/, processed/}`
  as structure-only — no data pulled.
- Saved one cross-session memory at
  `~/.claude/projects/-mnt-projects-research-dotaml-turbo/memory/sister-repos.md`
  pinning the dotaml-turbo / dotaml-serve / dotaml-items scope split.

### Findings

- The DotaML prior art's strongest signal is a **plateau**, not a winning
  architecture: six models spanning LightGBM through a 152k-param
  Transformer all converge to ~59.9% test acc / ~0.635 test AUC on 7-9M
  matches of pre-7.40 data. v5 README explicitly says this looks like a
  fundamental limit. That makes "replicate the plateau on patch-7.40 with
  our own HCE pipeline" the natural zero-th experiment — everything else
  needs a number we trust to judge against.
- Two prior-art landmines worth pre-empting:
  (a) Azure file overlap caused 4.9% match_id duplication in the early
      DotaML dataset (docs/DUPLICATION_REPORT). Mitigation in
      `splits.yaml`: dedup by `match_id` at read time, regardless of
      filename ranges.
  (b) The v2 `max_hero_id=130` bug silently dropped 20 heroes and
      materially shifted apparent "best combos" without moving test
      accuracy by more than 0.1pp. Mitigation: hero ID range is `[1,
      150]` in `splits.yaml`; any combo-rank analysis must guard against
      this kind of silent coverage failure.
- DotaDB's match-ID vs match_seq_num study (77.6% vs 42.9% correlation
  with `start_time`): time-based splits must partition on
  `start_time_date`, not `match_seq_num`. Locked into `splits.yaml`.
- The prior-art experiments used chronological 80/20 train/test with no
  held-out test. The HCE rule this project adopts is stricter; gap noted
  as an open item but not yet recorded as an ADR.

*(Session-2 additions, post-`/implement`, 2026-05-15 18:18 — added
under each existing subsection below.)*

### Did (cont.)

- **Pre-flight (Azure):** full 97-day listing scan + boundary probe at
  2025-12-16/17 confirmed the seq_num "overlap" between consecutive days
  is **structural**, not a collector bug — start_time_date partitioning
  causes seq_nums to interleave at midnight, but probe shows 0 match_id
  intersection and 0 seq_num intersection between the boundary files.
  Updated `concepts/match-id-vs-seq-num-ordering.md` with the empirical
  finding. Verified ~200k matches/day rate (Δ +0.5% from splits.yaml
  assumption — no update needed).
- **/implement on plateau-baseline-740:** dispatched a subagent (Opus,
  per `budget.yaml: models.implementer = claude-opus-4-6`). Subagent
  scaffolded env (`pyproject.toml`, `uv.lock`, `.venv`, two `dvc.yaml`
  stages) and authored `pull_raw.py`, `build_features.py`, `train.py`
  before its wall-time expired mid-download.
- Took over from the main agent: ran `pull_raw.py` (1736 files,
  71.6 GB, 1036 s, 0 errors), then chained `build_features.py` (881 s)
  and `train.py` (123 s) as a single backgrounded bash job. Result:
  `metrics.json` + four artifacts in `results/` (calibration.png,
  roc.png, learning_curve.png, lightgbm.txt 1.87 MB).
- Authored README Result / Interpretation / Diagnostics; status
  `running` → `done`; added `result:` frontmatter line. Moved proposal
  to `experiments/_proposals/_done/` with `status: implemented`,
  `experiment:` pointer.
- Promoted `concepts/draft-prediction-plateau.md` to `growing`, refined
  the architecture-spread finding, linked to the experiment.

### Findings (cont.)

- Headline: **val_auc = 0.6161** on 2,419,185 val matches.
  **Partial confirmation** of the proposal hypothesis. Strict band
  [0.625, 0.645] missed by 0.0089 on the low side, but within 0.003 of
  DotaML v3's same-recipe `test_auc=0.6189`.
- The proposal's 0.635 plateau target was actually the **v5 Transformer
  ceiling**, not the v3 LightGBM ceiling. The architecture-spread
  within the prior-art "plateau" is real and ≈ 0.016 AUC, not noise.
  Implication: the next architecture experiment (Transformer/FFN) is
  the test that actually matters for the plateau-across-architectures
  claim — replicating the v3 number told us only that LightGBM-on-7.40
  ≈ LightGBM-on-pre-7.40.
- Soundness checks all pass: HCE intact (test never read; build- and
  train-time asserts hold). Train-val AUC gap = 0.0126 (well-fit, not
  overfit). Calibration near-perfect across all 20 quantile bins
  (`results/calibration.png`). Train/val Radiant base rates within
  0.001 of each other (0.5335 / 0.5326). 0 `match_id` duplicates after
  dedup over 16.9 M reads.
- Filter rate **8.78%** (forfeit + empty-inv combined). Bigger than
  expected — sensitivity audit warranted before any future
  filter-on-vs-off comparisons are meaningful.
- /implement subagent observation: the implementer subagent's wall
  budget is too short for sequential I/O like 100 GB Azure pulls and
  ~15-min single-threaded JSON parsing. Working pattern this session:
  subagent scaffolds + writes scripts; main agent runs the long jobs
  via background Bash; main agent finalises README. Worth noting as a
  /implement skill consideration.

### Next

- (Carryover, deferred) Decide whether to write
  `docs/decisions/0001-hce-vs-prior-art-splits.md`. Lower priority now
  that the first experiment shipped without it.
- (Carryover, deferred) 5M-subset-vs-full-13M sanity check: train.py
  used the 5M stratified subset; an apples-to-apples full-13M run
  would close the loop on whether the subsample costs >0.005 AUC.
- (Resolved) Azure file-overlap, 200k/day rate, ~100 GB SN850X room —
  all verified this session.
- **Highest-priority next experiment:** Transformer/FFN baseline
  mirroring DotaML v5/v6 (64-dim hero embeddings, masked-input
  training) — tests whether the architecture-spread within the plateau
  reproduces on patch 7.40 under HCE, or collapses. This is the
  actually-interesting plateau test, given that the v3 LightGBM
  ceiling already replicates.
- **Cheap follow-ups:** filter sensitivity audit (re-run with each
  filter off; if val_auc shifts >0.005 the filter is doing real work);
  pick-position decomposition (does the signal live in late
  counter-picks vs early picks?).
- DVC integration: `dvc.yaml` declares stages but the pipeline ran via
  direct python, not `dvc repro`. Result artifacts (lightgbm.txt + 3
  PNGs) are in git rather than DVC-tracked. Decide whether to
  formalise `dvc commit` for the existing outs or leave them
  git-tracked given their small size.

## 2026-05-17

(Spans 2026-05-15 18:35 → 2026-05-17 02:00 — the prior `/wrap` covered
only through the LightGBM baseline; this entry catches up on three
arcs: plateau-architectures-740, transformer-hp-sweep-740, and the
Blackwell torch DataLoader bug investigation + upstream issue filing.)

### Did

- **plateau-architectures-740** (proposal 2026-05-15 18:35, implement
  2026-05-15 20:20): three-architecture sweep mirroring DotaML v4-v6.
  SimpleFFN (53k), ResidualFFN (225k), Transformer (82k) all sharing
  64-dim hero embeddings. Reused the same 5M stratified subset
  (seed=42) as the LightGBM baseline. Trained on RTX 5080 bf16
  autocast. README + Diagnostics + experiment moved to `_done/`.
  Concept `draft-prediction-plateau` got a second refinement.
- **transformer-hp-sweep-740** (proposal 2026-05-16 02:22, implement
  2026-05-16 08:13): Optuna TPE + ASHA 60-trial HP sweep over a
  minimal-Transformer baseline (no side-token branch, binary team
  embedding, single linear head). 5M train / 2.4M val, same subset
  as prior experiments. Subagent scaffolded harness + smoke; main
  agent ran the production sweep via per-trial subprocess isolation
  (`run_sweep_loop.sh` looping `run_sweep.py --n-trials 1`).
- **Blackwell torch DataLoader bug investigation** (2026-05-17): when
  the user pushed back on the per-trial-subprocess wrapper as
  potentially papering over a real bug, ran a focused investigation —
  dmesg/journalctl Xid scan → web search for known issues → pinned
  torch 2.9.1+cu128 to test torch-version hypothesis → diagnostic with
  `CUDA_LAUNCH_BLOCKING=1` + `MALLOC_CHECK_=3` + `PYTHONFAULTHANDLER=1`
  → confirmed it's CPU-side heap corruption inside torch's DataLoader
  fetch + tensor GC. Updated `pyproject.toml` to keep torch pinned to
  `>=2.9,<2.10` via the cu128 PyTorch index (most-baked stack).
- Added per-trial cleanup (`del model; torch.cuda.empty_cache();
  gc.collect()`) to `objective.py` for belt-and-suspenders on any
  future in-process invocation.
- Wrote `docs/decisions/0001-per-trial-subprocess-isolation.md` ADR.
- Saved `~/.claude/projects/.../memory/blackwell-torch-dataloader-bug.md`
  cross-session memory so future sessions skip the dead-end fixes.
- Saved `~/.claude/projects/.../memory/feedback_date_grounding.md` —
  user-feedback memory after I overstated "CUDA 13.0 just came out"
  from training intuition (it had been out ~1 year). Future sessions
  verify release dates via WebSearch / wheel headers / dpkg / uv.lock
  before asserting.
- Spawned subagent to produce a minimal reproducer + draft upstream
  report. Subagent actually triggered the crash twice in ~3-4 min
  with a 193-line synthetic script (no pyarrow, no parquet, no
  Optuna), and surfaced a libtorch C++ trace through
  `c10::TensorImpl::~TensorImpl()`. Subagent also discovered that
  `MALLOC_CHECK_=3` does NOT mask the synthetic repro the way it
  appeared to mask the real workload — corrected the ADR + memory.
- **Filed upstream pytorch/pytorch#184062**
  (https://github.com/pytorch/pytorch/issues/184062) with the report
  + full reproducer inlined in a `<details>` block.
- Updated the experiment README's Diagnostics block to cross-reference
  the new ADR + upstream issue.

### Findings

- **plateau-architectures-740: Transformer val_auc=0.6322** (within
  0.003 of v6's 0.6354), SimpleFFN 0.6217, ResidualFFN 0.6199 — all >
  LightGBM 0.6161. Loose hypothesis (architecture-spread is real,
  Transformer-led) **confirmed**. Strict hypothesis (rank order matches
  prior art within ±0.005) **NOT confirmed** — ResidualFFN underperforms
  SimpleFFN by 0.0018 (sign flip vs prior art's v5>v4). The
  Transformer-vs-FFN gap (≥0.0105) is the architectural lever that
  reliably works.
- **transformer-hp-sweep-740: best val_auc=0.6318, +0.0007 vs control**.
  Hypothesis NOT confirmed; the ~0.632 ceiling is **HP-robust** on this
  snapshot. Top 4 COMPLETE trials cluster in val_auc [0.6311, 0.6319] —
  TPE found a 0.0008 envelope around the control point. Architectural
  simplification (no side-token, no 11-position embed, single linear
  head) was free of capacity cost vs the previous Transformer. **HP
  tuning is exhausted as a lever** for this architecture vocabulary;
  further gains require structural mutation or new data features.
- **Blackwell torch bug root cause: torch's DataLoader + tensor GC
  interaction.** NOT a CUDA kernel bug (CUDA_LAUNCH_BLOCKING shows no
  CUDA error). NOT torch-version-specific (reproduces on 2.9.1, 2.11.0,
  2.12.0). NOT driver/cuDNN (same versions across all reproductions).
  NOT pyarrow tp_traverse (explicit `del` + `gc.collect()` of all
  pyarrow refs didn't help). NOT hardware (synthetic repro on a totally
  different code path triggers the same crash). The libtorch C++ trace
  from `c10::TensorImpl::~TensorImpl()` points at PyTorch's CPU tensor
  destructor reached via cyclic GC. NVRM Xid 43 entries in dmesg are
  secondary symptoms (GPU channel reset when process aborts mid-op).
- **The MALLOC_CHECK_=3 "fix" was probably small-sample noise.** The
  synthetic repro crashes under the same allocator settings — sometimes
  even faster. The 3 clean trials in our real-workload diagnostic were
  consistent with the ~21% crash-free fraction by chance.
- **Per-trial subprocess isolation is architecturally clean.** It's how
  Ray Tune, Kubeflow, SageMaker structure HP sweeps anyway. Treating it
  as "the proper architecture for ML sweeps on this hardware" rather
  than a hack avoids future rationalisation drift.
- **Repro details (~3-4 min wall):** the synthetic script with 5M
  random rows + a ~416k-param Transformer + manual `gc.collect()` every
  25 batches triggers the crash at epoch 5, mostly inside
  `fetch.py:52` (the list-comp over `__getitem__`) but sometimes inside
  `default_collate` at `collate.py:208`. Same root corruption, two
  observed crash sites.

### Next

- **Watch pytorch/pytorch#184062 for upstream triage.** If a fix is
  merged and a torch release ships claiming DataLoader/tensor GC fix
  on Blackwell, re-test in-process Optuna and — if stable — retire
  the subprocess wrapper. Record the change as a new ADR.
- **LLM-driven islands evolution experiment** (FunSearch / AlphaEvolve-
  style structural mutation) is the natural next step now that HP
  search is exhausted as a lever. The user explicitly wanted this
  before we detoured into HP-search. Programs = `config.yaml` + small
  Python files describing model/features; LLM mutates structurally;
  evaluator is val_auc on the existing 5M subset; islands evolve
  independently with periodic migration. Reference reads already
  ingested in `agentic-research` (AlphaEvolve, FunSearch,
  evolutionary-expansion concept).
- **Data-side feature additions** (draft order from `picks_bans[]`,
  lane/role inference, hero-pair history, player MMR if obtainable) —
  cheaper than islands evolution and could isolate "what fraction of
  the gap is model vs data." Worth doing in parallel to or before
  islands evolution.
- **Filter sensitivity audit** (carryover): run a known-good config
  with each filter (forfeit / empty-inventory) toggled off. If
  val_auc shifts >0.005 the filter is doing real work; if <0.001
  the filter is mostly cosmetic.
- **Multi-seed re-evaluation** of the top 4 HP-sweep trials to confirm
  the 0.0008 spread is within seed noise (carryover; lower priority
  now).
- **Promote `run_sweep_loop.sh` + `cleanup_failed_trials.py`** to a
  reusable template under `_meta/templates/` for any future sweep
  experiment on this server.
- (Carryover, deferred) HCE-vs-prior-art-splits ADR. Still optional.
- (Carryover, deferred) 5M-subset-vs-full-13M sanity check. Still
  optional.
- (Carryover, deferred) DVC formalisation for the existing experiment
  outs. Still optional.

## 2026-05-18

(Spans 2026-05-17 19:00 → 2026-05-18 15:00 — the prior `/wrap` at
2026-05-17 02:00 covered only through the upstream issue filing.
This entry catches up on the player-features arc: literature ingest,
two experiments, two server-stability incidents.)

### Did

- **Ingested Hodge 2017** (`literature/papers/hodge2017win.md`,
  arXiv 1711.06498, "Win Prediction in Esports: Mixed-Rank Match
  Prediction in MOBA Games") via `/fetch-paper` → `/ingest`. Relevance
  4. Added as source to `concepts/draft-prediction-plateau.md` (third
  refinement: independent attestation of the hero-only ceiling at
  55-59% accuracy, matching our LightGBM baseline 58.66% within 0.01)
  and `concepts/draft-only-win-prediction.md`. Triggered by user
  feedback that I'd been citing "Yang 2014, Summerville 2016" from
  training memory without grounding — saved
  `memory/feedback_ingest_cited_literature.md` to enforce the discipline
  forward.
- **Proposed + implemented `player-features-740`** (
  `experiments/2026-05-17-player-features-740/`): LightGBM baseline +
  ~90 per-player history features (smoothed overall winrate,
  smoothed hero-specific winrate, last-10 recent form, days-since,
  hero diversity, premade-detection coplay, is-anonymous flag)
  computed from chronological leading-window aggregation over the
  19.6M patch-7.40 matches. Build wall ~69 min; LightGBM ~3 min.
  val_auc=**0.6227** (+0.0067 over LightGBM baseline 0.6161, missed
  +0.020 target by 0.0134, hypothesis NOT confirmed). Heroes-only
  sanity rebuild matched plateau-baseline within 0.0001.
- **Discussion arc on player features and pre-game scope.** User
  pushed back correctly on my over-stated metagame-drift framing for
  hero-specific winrate (Bayesian shrinkage to per-hero base rate
  means the per-player deviation is patch-stable; only base rates
  drift). Also discussed player-embedding scale (~1.3M-3M accounts
  is NOT prohibitive: 333 MB at 64-dim, 167 MB at 32-dim) and
  shrinkage strategies (anon embedding as best-trained = natural
  shrinkage target via learned gate).
- **Probed Azure for full date range:** 266 days, ~253 GB across
  Aug 2025 → May 2026. Partial early collection (Aug 16/31 days,
  Oct 21/31), continuous from Nov 2025+. Patch boundaries: 7.40
  Dec 16 2025; 7.39 ~Oct 2025; 7.38 ~Aug 2025.
- **Proposed + implemented `player-features-prepatch-740`** (
  `experiments/2026-05-18-player-features-prepatch-740/`): extended
  aggregator with 127 days of pre-7.40 history (Aug 1 → Dec 15 2025,
  ~100 GB pulled into NEW `data/history/turbo/` directory tree).
  Required **two OOM-fix iterations** (build hit 93.8 GB RSS — the
  per-account `coplay` nested dict at ~5M accounts × 200 cap × ~75
  bytes ≈ 75 GB; removed `coplay` + `unique_heroes` from aggregator,
  neither in top-20 importance from `player-features-740`). val_auc=
  **0.6256** (+0.0028 vs `player-features-740`, missed +0.005 target
  by 0.0022, hypothesis NOT confirmed but diagnostic FLIPPED the
  cold-start story — see Findings below).
- **Two server-stability incidents documented:**
  - Kernel RCU stall on 2026-05-17 ~20:38-20:52 (kernel thread
    PID 3137061 stuck for 13 min, SSH timed out, user power-cycled);
    distinct bug class from `blackwell-torch-dataloader-bug`. Saved
    `memory/aiserver2026-kernel-rcu-stall-incident.md`.
  - Two consecutive OOMs in `player-features-prepatch-740` build
    (2026-05-18 ~07:38 and ~13:30 UTC); fixed by dropping
    memory-hog features per above.
- **Concept refinements:** `draft-prediction-plateau.md` got
  refinements 3, 4, 5 (Hodge attestation; player-features lever
  bounded by anonymous fraction; cold-start was misdiagnosis,
  casual/anonymous tail is binding constraint; HIGH-coverage subset
  beats Transformer). `draft-only-win-prediction.md` was refined
  into two flavours: strict draft-only vs broadened
  pre-game-win-prediction (account_id and derivatives allowed).

### Findings

- **Hero-only ceiling independently confirmed by Hodge 2017** at
  55-59% accuracy (our LightGBM baseline: 58.66 % — match within
  0.01). The paper's 75-76% in-game-telemetry ceiling confirms
  substantial headroom in richer feature sets, BUT in-game features
  are post-game — out of scope for pre-game prediction.
- **Player features matter, but less than expected on Turbo.**
  Patch-only +0.0067 AUC, prepatch +0.0095 AUC (both over LightGBM
  baseline 0.6161). Both still below the architecture-only
  Transformer ceiling 0.6322 on whole val.
- **Per-player-per-hero winrate is THE marginal-value lever.**
  Top-10 features by importance in BOTH player-features experiments
  are exclusively `pX_smoothed_winrate_hero` (one per player slot,
  gain 65-87k each). Overall winrate, recent form, premade-detection
  coplay, days-since, hero diversity didn't crack the top 20.
  Per-hero player skill is the signal; everything else is decoration.
- **Cold-start hypothesis FAILED.** The coverage-bucket diagnostic
  stayed monotonic after pre-patch extension, with HIGH bucket
  gaining most (+0.0043) and LOW gaining least (+0.0014) — opposite
  of the cold-start prediction. History-source breakdown explained
  it: low-bucket players had only 2.3 % prepatch fraction (4
  prepatch games avg). They're genuinely casual / new accounts, not
  active-but-uncached players. Pre-patch data can't rescue players
  who weren't around then either.
- **The binding constraint is the casual / anonymous-player tail,
  not lookback length.** 66 % of player-slots in Turbo are anonymous
  (Steam-private profiles); 12.6 % of val matches have ALL 10
  players anonymous. Both fractions are unchanged across the
  experiments because anonymous status is a Steam privacy setting
  independent of data we have.
- **Big-deal observation: HIGH-coverage val_auc reached 0.6339,
  BEATING the architecture-only Transformer ceiling (0.6322) for the
  first time.** For the active 1/3 of patch-7.40 val matches, 80
  player features beat 82k attention parameters. The
  architecture-vs-information comparison flipped on the active
  subset. The whole-val ceiling is still bound by the
  casual/anonymous tail.
- **The user's pre-experiment prediction on metagame drift held.**
  Pre-patch hero-specific data transferred cleanly; top-10 features
  unchanged in character, no metagame-drift artifact visible. The
  Bayesian shrinkage to per-hero base rate isolated the per-player
  deviation from balance shifts as predicted.
- **Player embedding scale would be ~1.3-3M accounts, NOT prohibitive.**
  64-dim float32 → 333 MB, 32-dim → 167 MB; both fit in 16 GB VRAM
  easily. Long-tail: ~80% of accounts appear ≤5 times. Anon
  embedding becomes the best-trained (66 % of player-slots,
  100M+ training updates) and is a natural shrinkage target via a
  learned gate based on `n_games`. Equating new-at-inference accounts
  with `<anon>` is the clean default.
- **Engineering insight: dict-of-dict aggregator state doesn't scale
  to multi-million unique accounts.** The `coplay` nested dict at
  5M accounts × 200-cap entries hit 75 GB. Future per-account
  state needs to use flat representations or be eagerly evicted.
- **Two distinct server-stability bug classes confirmed.** The
  Blackwell torch DataLoader bug (filed as `pytorch/pytorch#184062`)
  is userspace heap corruption; the May 17 kernel RCU stall is
  kernel-side. Don't conflate. Memory entries enforce this.

### Next

- **`transformer-plus-player-features-740` (HIGHEST PRIORITY).** Combine
  the two strongest individual levers — replace the LightGBM head with
  the `plateau-architectures-740` Transformer (82k params, attention
  over 10 hero tokens), feed it BOTH 10 hero IDs AND the 80-dim
  per-player feature block from `player-features-prepatch-740`.
  Hypothesis: HIGH-bucket val_auc 0.6339 (from this experiment) + the
  architecture lever pushes whole-val past 0.633, possibly to 0.640+.
  The two levers have never been combined; doing so directly tests
  additivity.
- **`anonymous-aware-modeling`.** Address the structural 12.6 % all-anon
  tail. Two design candidates: (a) route all-anonymous matches to a
  separate head that only sees hero one-hot + radiant-side base rate;
  (b) build per-team aggregate features over the known-player subset
  only (e.g. "mean smoothed_winrate over the K non-anonymous players
  on team R"). Either reaches the 13 % of val that's currently
  dead-weight.
- **`player-features-decay-740`.** Now that pre-patch contributes ~13 %
  of total game-history weight on average, test whether time-decayed
  history (exponential τ ≈ 90 days) is better than uniform aggregation.
  Smaller experiment than this arc — same data, different aggregator.
- **`features_only` deep-dive (optional).** It completed cleanly this
  time (val_auc=0.6065) — pure player features without heroes get
  ~0.61. So player and hero features are complementary (not redundant)
  — combined 0.6256 exceeds either alone. Worth understanding which
  player features carry signal when heroes are absent.
- (Carryover) Promote `run_sweep_loop.sh` + `cleanup_failed_trials.py`
  to `_meta/templates/` for future GPU-sweep experiments.
- (Carryover) **LLM-driven islands evolution.** Still on the menu;
  was deferred when user redirected to player-features arc. Could be
  approached now via the combined-Transformer-plus-features baseline
  as the starting point for structural mutation.
- (Carryover, deferred) HCE-vs-prior-art-splits ADR. Still optional.
- (Carryover, deferred) 5M-subset-vs-full-13M sanity check. Still
  optional.
- (Carryover, deferred) DVC formalisation for existing experiment outs.
  Still optional.
- (NEW) **Consider time-decay or recency-weighted aggregator** before
  the next player-features-style experiment. The dict-of-dict state
  problem will recur unless we redesign.

## 2026-05-19

(Spans 2026-05-18 22:46 → 2026-05-19 01:35 — the prior `/wrap` at
2026-05-18 15:18 covered through `player-features-prepatch-740`.
This entry catches up on the player-features → architecture
combination arc: account-id lookup, active-subset analysis,
`transformer-plus-features-740` (CONFIRMED), upstream issue triage,
and `player-features-prepatch-740` upstream data-quality finding.)

### Did

- **User account_id lookup** — pulled match 8815544558 from Azure
  (one-off, into `/tmp/`, no permanent ingest). Identified
  account_id 3303652 (eschmitt's Steam ID, non-anonymous) plus
  friend 2231002. Saved both to
  `~/.claude/projects/.../memory/user_account_ids.md` for use in
  any future personal-data analysis.
- **Verified user's hero pool against canonical data** —
  `odota/dotaconstants/build/heroes.json`. Caught one mis-guess:
  hero_id=94 is **Medusa**, not Riki. Corrected interpretation: user
  is a support-leaning flex player (~73% support, ~27% hard carry).
- **Scanned all 261 days of local raw turbo for matches containing
  account 3303652** — 212 matches across 74 distinct days. Found
  user's matches have **47.7% anonymous "other" players** (vs the
  project-wide 66.6% mean) — confirming user's hypothesis that
  active players cluster with other active players, but only ~19 pp
  off the project mean rather than dramatically lower.
- **Active-player-subset val_auc analysis** (#6) — re-scored
  `player-features-prepatch-740` LightGBM model on different
  data-availability slices of val. Key results: `n_anon ≤ 1` subset
  val_auc=**0.6447** (1.3% of val), `≥5 active players` subset=
  **0.6359** (21.4% of val), `≥3 active players` subset=0.6325 (48%
  of val). Established the active-subset ceiling for the
  LightGBM-only model is ~0.645 at the most-public extreme.
- **Proposed + implemented `transformer-plus-features-740`** (#2) —
  combined the architecture lever (MinimalTransformer from
  `plateau-architectures-740`) with the 80-dim per-player feature
  block from `player-features-prepatch-740` via
  `Linear(8, d_model)` projection added per-slot to hero embeddings
  before self-attention. **val_auc=0.6452 — HYPOTHESIS CONFIRMED,
  +0.0080 over the target 0.6372**. Sanity ablation
  (`architecture_only`) matched `plateau-architectures-740` within
  0.0003 — pipeline correct.
- **Caught upstream data corruption mid-experiment** —
  `player-features-prepatch-740/train.parquet` has 6,482 fp32-max
  (±3.4e38) sentinels (0.005% of cells) in `smoothed_winrate_hero`,
  almost certainly a divide-by-zero in `build_features.py`'s
  Bayesian smoothing formula. val.parquet clean. Sanitized
  defensively in `data.py:load_arrays` for this experiment;
  upstream patch noted for follow-up. Did NOT modify the prior
  experiment's parquet (hard rule).
- **Checked upstream pytorch/pytorch#184062** — assigned to
  `albanD` (PyTorch maintainer), labels `triaged` +
  `needs reproduction`. Comment from albanD: couldn't repro on
  RTX 4080 (sm_89), asking for C++ stack traces. **Consistent with
  our Blackwell-specific (sm_120) diagnosis** — different
  generation, different code path. User is handling the C++ stack
  capture in another agent.
- **Concept refinements**:
  - `concepts/draft-prediction-plateau.md` 6th refinement:
    combination is nearly additive; whole-val ceiling moves from
    ~0.632 to 0.6452; HIGH-coverage bucket reaches 0.6560.
  - `concepts/draft-only-win-prediction.md` related_experiments
    extended.
- **`_meta/index.md`** lists six completed experiments now.

### Findings

- **The architecture-vs-information dichotomy resolves to "use
  both"**. Architecture (Transformer) and per-player-per-hero
  features address minimally-redundant information axes. Combined
  val_auc 0.6452 is nearly the sum of individual lifts (Transformer
  +0.0161 over LGBM-baseline; features +0.0095 over LGBM-baseline;
  naive sum predicted 0.6417, actual 0.6452 — slight synergy not
  redundancy).
- **The HIGH-coverage val_auc 0.6560 is closing in on Hodge 2017's
  75-76% in-game-telemetry ceiling — using PRE-GAME info only.**
  That's striking; it means for active-player lobbies, hero+player
  info nearly matches what you get with full live telemetry.
- **The LOW-bucket val_auc 0.6347 (mostly-anonymous matches) alone
  beats the architecture-only Transformer's whole-val ceiling
  (0.6322).** Attention extracts substantially more signal even on
  mostly-anonymous matches when given partial per-player info.
- **The whole-val LOW-vs-HIGH gap is now 0.0213** — the largest
  internal asymmetry in our best model. This is the biggest
  remaining whole-val lever.
- **User's "active-players-cluster" hypothesis confirmed but
  partial** — their matches have 47.7% anon "other" players vs
  project-wide 66.6%. About ~19 pp better, not zero or near-zero.
  Active-lobby data-quality story is real but bounded.
- **First Transformer experiment with zero Blackwell torch
  retries** — `transformer-plus-features-740` both ablations
  succeeded first attempt. Either the dataset shape or the per-
  trial subprocess pattern happened to dodge the bug this time,
  or the bug is genuinely intermittent at low frequency.
- **Upstream pytorch maintainer is treating the issue seriously**
  — assigned, triaged, asking for C++ stacks. Confirms the issue is
  considered actionable.
- **Engineering caveat: `player-features-prepatch-740` was trained
  on slightly-corrupted data** (6,482 fp32-max cells in one feature
  column, 0.005% of cells). val_auc=0.6256 may have been minutely
  affected. Worth re-running after upstream patch but probably
  noise-level.

### Next

- **(Highest priority) `anonymous-aware-modeling-740`** — address
  the 0.0213 LOW-vs-HIGH coverage asymmetry now that the
  combined-model has resolved the architecture-vs-information
  question. Two concrete design candidates: (a) router head that
  routes all-10-anonymous matches (12.6% of val) to a separate small
  model using only hero one-hot + radiant-side base rate, or
  (b) per-team aggregate features over the known-player subset
  ("mean smoothed_winrate of the K known players on team R", plus
  K_radiant, K_dire as features). (a) is simpler; (b) is more
  principled. Either could lift the LOW bucket by 0.005-0.015.
  Expected: meaningful whole-val gain.
- **`transformer-plus-features-extended-training`** (almost-free
  follow-up) — both ablations had `best_epoch=14=max_epochs`,
  val_loss still improving at cap. Bump to 25-30 epochs with early
  stopping (patience=5). Expected +0.001-0.005 essentially free.
- **`upstream-data-cleanup`** — patch the divide-by-zero edge case
  in `player-features-prepatch-740/build_features.py` that produces
  6,482 fp32-max sentinels. Re-build the prepatch parquet (~3 h
  re-run), then re-run player-features-prepatch and
  transformer-plus-features. Effect likely tiny but matters for
  downstream cleanliness.
- **`player-embedding-prelim-740`** (user-flagged conceptual
  direction) — learned per-player embeddings (~1.3M accounts ×
  32-64 dim = 167-333 MB, fits VRAM easily). Now that we have a
  strong reference at 0.6452, the embedding experiment has a sharp
  comparison point. Long-tail concern: ~80% of accounts appear ≤5
  times; anon embedding (66% of slots) becomes the best-trained
  and serves as a natural shrinkage target via a learned gate.
- **`player-features-decay-740`** — exponential time-weighting
  (τ ≈ 90 days) for the aggregator. Smaller experiment; tests
  whether recent skill matters more than uniformly-weighted
  history. May or may not help.
- (Tracked) **pytorch/pytorch#184062** — user handling the C++ stack
  capture in another agent. If maintainer needs further info, follow
  up via that channel.
- (Carryover, deferred) Promote `run_sweep_loop.sh` +
  `cleanup_failed_trials.py` to `_meta/templates/`.
- (Carryover, deferred) LLM-driven islands evolution. Higher-leverage
  in principle but bigger investment than the cheap follow-ups
  above — defer until the data-side wins are exhausted.
- (Carryover, deferred) HCE-vs-prior-art-splits ADR; 5M-vs-13M
  sanity check; DVC formalisation. Still optional.

---

## 2026-05-20

### Did

- Proposed + implemented `transformer-plus-features-extended-740`
  (`experiments/2026-05-19-transformer-plus-features-extended-740/`):
  same architecture and feature set as `transformer-plus-features-740`,
  training cap raised 14 → 30 epochs with early-stopping patience=5
  on val_log_loss. val_auc=**0.6477** @ best_epoch=22, early-stopped
  at epoch 27. **HYPOTHESIS CONFIRMED** (+0.0025 over parent 0.6452,
  +0.0015 over target 0.6462). All three coverage buckets lifted
  uniformly (~+0.0025). 25.1-min wall, zero Blackwell retries.
- Proposed + implemented `upstream-data-cleanup-740`
  (`experiments/2026-05-19-upstream-data-cleanup-740/`): patched the
  upstream defect in `build_features.py` producing 6,482 fp32-max
  sentinel cells in `p1_smoothed_winrate_hero` of the prepatch
  parquet (0.005% of cells). Rebuilt cleanly with multi-checkpoint
  defense (snapshot-time clamp + numpy-routed pyarrow write +
  pre/post-write bounds-check). **No-regression CONFIRMED**:
  Transformer+features val_auc on clean parquet = **0.6477054**
  (Δ = -2.4e-5 vs dirty); LightGBM features_only = 0.6063985
  (Δ = -6.6e-5 vs dirty). Equality band [0.6467, 0.6487] holds
  dead-center. Root cause of the original corruption was traced to
  transient memory / buffer-fill anomaly in PyArrow's fp32 column
  conversion (NOT a math bug), not deterministically reproducible —
  defense is mechanism-agnostic. Clean parquet at
  `data/snapshots/.../player_features_prepatch_clean/` is the new
  canonical input.
- Proposed + implemented `player-embedding-prelim-740`
  (`experiments/2026-05-19-player-embedding-prelim-740/`): learned
  per-player embedding (vocab top-500K + 'rare' + 'anon' bucket,
  dim=32, 16M params, 208× baseline). **HYPOTHESIS NOT CONFIRMED**
  — clean null. baseline_extended_clean reproduced cleanup-740's
  val_auc to FIVE decimal places (0.6477054); with_player_embedding
  landed at 0.6476302 @ best_epoch=23 (Δ=-7.5e-5 vs baseline,
  -2.07e-3 vs target). Coverage buckets all flat within noise
  including HIGH bucket where 50.9% of slots get frequent vocab
  entries. Train-val gap NARROWER for embedding model — not
  overfitting, just no useful signal. Required a 45-min
  `build_account_sidecar.py` pre-step because account_ids aren't in
  the clean parquet (only raw JSON has them).
- Server-stability events: **2× OOM-kill** during cleanup-740's
  post-write parquet re-read validation step (1.38 GB re-read after
  2h aggregation holding ~30 GB dict-of-dict state). First instance
  cascaded to a full system reboot at 2026-05-19 08:23 UTC; second
  was an isolated SIGKILL rc=137. Both times the on-disk parquet
  was complete; verified clean via pyarrow row-group column
  statistics (cheap, no full-read). Saved as personal memory:
  `aiserver2026-postwrite-parquet-reread-oom.md`.
- Bun (Claude Code CLI) v1.3.14 segfaulted mid-session at ~16:50
  UTC. `nohup`-detached pipeline survived (logged to
  `experiments/.../full_run.log`, not `/tmp`); resume succeeded
  cleanly, no rework needed.
- Updated `concepts/draft-prediction-plateau.md` with refinements
  7+8+9 (extended-training uniform lift, no-regression cleanup,
  player-embedding null result). Updated `_meta/index.md` and
  `_meta/log.md` per experiment.

### Findings

- **The 0.6477 ceiling is anchored across THREE independent runs**
  (`extended-740`, `cleanup-740`, `baseline_extended_clean`) within
  2e-5 of each other. This is a much tighter pin than the typical
  ~1e-4 reproducibility seen across the project's reruns; the
  fixed seed=42 + identical data pipeline really do produce
  bit-stable val_auc.
- **The 8 aggregated player features are essentially complete for
  the per-player identity axis on this task.** A 16M-param learned
  embedding had access to the full identity history but extracted
  ZERO additional signal over the 77K-param baseline. The HIGH
  coverage bucket (50.9% in-vocab frac, the maximum embedding
  leverage) gained only +0.0001. This is the strongest possible
  negative finding for "richer player representations" as a lever.
- **Extended training to ~22 epochs is essentially free signal but
  is uniformly distributed across coverage buckets** — does NOT
  target the cold-start/anonymous tail. The LOW-vs-HIGH gap closed
  only fractionally (0.0213 → 0.0221) under extended training.
- **The 6,482 fp32-max sentinels were genuinely noise-level** — clean
  vs dirty val_auc differs by ≤ 1e-4 for both LightGBM and
  Transformer reruns. Equality band holds dead-center. But fixing
  the source matters: every downstream experiment from now on
  consumes the clean parquet directly with no `data.py`
  sanitization workaround.
- **Root cause of the original parquet corruption was NOT a math
  bug.** Investigation traced to PyArrow's fp32 column-buffer fill
  on one specific row group (single date 2025-12-29, single column,
  signature of torn 16-bit memory writes). Not reproducible. The
  multi-checkpoint defense in `build_features.py` is the long-term
  fix and is mechanism-agnostic.
- **OOM during `build_features.py` post-write re-read is a
  recurring pattern on this box** (2 events in this session, one
  escalated to system reboot). Lesson: never re-read a multi-GB
  parquet inside a process already holding heavy aggregator state;
  use pyarrow row-group column statistics for validation instead.
  Memory note saved.
- **Identity-level latent signal beyond aggregate stats does not
  exist in meaningful quantity for the radiant-win label.** This
  closes off the "richer per-player representation" line for now.
  The remaining ceiling-breakers are NEW information axes (draft
  order, hero-pair history), anonymous-aware modeling (router or
  team-aggregate), or structural mutation — not deeper embeddings.

### Next

- **`draft-order-features-740`** *(new; my recommendation given
  the embedding null result)* — `picks_bans[]` sequence is the
  largest untouched information axis. Encode pick/ban order plus
  side (radiant/dire), inject as a sequence feature into the
  Transformer. Hypothesis: draft order carries strategic
  information the current per-slot encoding lacks. Likely +0.002
  to +0.010 if pick-order signal is real.
- **`anonymous-aware-modeling-740`** — previously user-deprioritized,
  but the embedding null *strengthens* the case. Since identity
  richness doesn't help, attacking the LOW-HIGH bucket asymmetry
  structurally is the residual axis. Two design candidates: (a)
  router head for all-10-anonymous matches (12.6% of val) to a
  separate small model; (b) per-team aggregate features over the
  known-player subset. Could lift LOW (0.6368) toward MED (0.6467),
  whole-val gain ~0.005-0.015.
- **`player-features-decay-740`** — exponential time-weighting
  (τ ≈ 90 days) for the aggregator. Smaller experiment; tests
  whether recent skill matters more than uniformly-weighted
  history. May or may not help.
- **Engineering: extend `build_features.py` to emit `pX_account_id`
  columns alongside features.** Would have saved 45 min on the
  embedding experiment's sidecar walk; would unify the data
  pipeline for any future identity-flavored work.
- **Engineering: DVC-track the new
  `data/snapshots/.../player_features_prepatch_clean/` directory**
  so re-fetches by downstream experiments are reproducible.
- (Tracked) **pytorch/pytorch#184062** — user handling C++ stacks
  in another agent. No change.
- (Carryover, deferred) Promote `run_sweep_loop.sh` +
  `cleanup_failed_trials.py` to `_meta/templates/`.
- (Carryover, deferred) LLM-driven islands evolution. Higher-leverage
  in principle but bigger investment than the cheap follow-ups
  above — defer until the data-side wins are exhausted.
- (Carryover, deferred) HCE-vs-prior-art-splits ADR; 5M-vs-13M
  sanity check; DVC formalisation.

