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


---

## 2026-05-22

### Did

- **Diagnosed and fixed the root cause of every reliability issue in this
  project**: silent RAM bit-flips at DDR5 EXPO 6000 MT/s on non-ECC
  memory. Investigation suite under
  `_meta/hardware-investigation-2026-05-21/`: memtester (dozens of
  single-bit failures), stress-ng (2,110 bit errors in 30 min across 4
  workers), PyArrow round-trip stress (2 silent data corruptions in 100
  iters under 25 GB pressure), kernel BUG + 5h RCU stall during stress.
  After user disabled EXPO in BIOS → JEDEC 4800 MT/s → ALL tests clean.
  Memory note: `aiserver2026-ram-bitflips-root-cause`.
- **Retracted pytorch/pytorch#184062** publicly
  (https://github.com/pytorch/pytorch/issues/184062#issuecomment-4508610051)
  — the "Blackwell torch DataLoader segfault" was actually bit-flipped
  heap state landing in torch's GC path. ADR 0001 marked superseded;
  Blackwell memory note marked superseded; per-trial subprocess pattern
  retained for unrelated reasons.
- **Rebuilt both corrupted parquets on JEDEC RAM**: clean parquet at
  `data/snapshots/.../player_features_prepatch_clean/` (2h11m) + rich-cols
  sidecar at `data/snapshots/.../rich_cols/` (1h34m). Both fully clean,
  all row groups read. Corrupted backups preserved at
  `data/snapshots/.../processed/_corrupted_backup_2026-05-21/`.
- **Implemented `rich-supervision-multitask-740`** end-to-end on stable
  hardware (`experiments/2026-05-20-rich-supervision-multitask-740/`):
  multi-task Transformer with shared encoder + 4 heads (win, duration
  8-bucket, items 305-vocab multi-label, aux KDA/GPM/HD). α weights
  1.0/0.15/0.3/0.1. **val_auc=0.6495 @ best_epoch=30, +0.0018 vs
  cleanup anchor 0.6477054, +0.0022 vs same-data baseline_extended_clean
  sanity at 0.6473.** HYPOTHESIS CONFIRMED. 4h 1m wall, 0 retries, 0
  kernel events.
- **Closeout**: README done, proposal → `_done/`, `_meta/log.md` +
  `_meta/index.md` updated, `concepts/draft-prediction-plateau.md`
  extended with tenth refinement (multitask CONFIRMED, scoreboard
  updated). Investigation log `investigation.log` finalized.

### Findings

- **DDR5 EXPO 6000 MT/s on Ryzen 9950X + 96 GB Corsair CMK96GX5M2E6000C36
  + ASUS X870E HERO with non-ECC RAM produces silent bit-flips** at high
  enough rate (~2,110 errors / 30 min under stress) to cause every
  reliability symptom in this project. Fix: BIOS → Ai Tweaker → Ai
  Overclock Tuner → Auto (JEDEC 4800 MT/s). Loses ~15% memory bandwidth,
  gains rock-solid reliability. Long-term upgrade path: ECC UDIMM kit
  (board supports it).
- **The "Blackwell torch DataLoader bug" was misdiagnosed.** Same RAM
  bit-flips landing in torch's tensor refcount / heap metadata produced
  segfaults inside `DataLoader.fetch` because that's where the most
  per-batch allocation/freeing happens. The synthetic repro could never
  reproduce on a 4080 Laptop because different RAM = different bit-flip
  behavior. Retraction posted; upstream maintainer's time freed.
- **The 0.6477 ceiling is broken**: multitask supervision lifts win
  val_auc to 0.6495 (+0.0022 vs same-data sanity). The
  gradient-density-bottleneck hypothesis (from the embedding-prelim
  NULL) is correct. The encoder had room to learn more from richer
  per-match supervision (~10× more bits than the binary radiant_win
  label provides), and 4 jointly-trained heads (win + duration + items
  + aux KDA-GPM-HD) deliver that.
- **Aux heads are useful standalone products**: duration top1_acc=0.181
  over 8 buckets (vs 0.125 random, ~45% above chance — useful for
  "end early vs scale" intuition), item recommender mAP@10=0.301 with
  mean_precision=0.333 / mean_recall=0.440 (picks 33-44% of actual
  end-game inventory in top-10).
- **Stdout buffering caught me out**: train.py doesn't use `flush=True`
  and Python defaults to block-buffered stdout when redirected through
  nohup. A 30-epoch multitask run produces only ~6 KB of per-epoch
  output, under the 8 KB buffer threshold. The first multitask retry
  was silently progressing for 3h 15min when I killed it thinking it
  was stuck. Re-launched with `python -u`; lesson: **always use `-u`
  for nohup-detached training runs.**
- **No kernel events during ~10h of heavy ML work on JEDEC** today
  (rebuild + multitask training + sanity). Hardware fix validated
  end-to-end on real workload.
- **Turbo drafts are hidden** (user correction): `draft-order-features-740`
  is invalid for this game mode (game_mode=23 doesn't show enemy picks).
  Removed from the queue. Memory note: `turbo-draft-is-hidden`.

### Next

- **`multitask-extended-740`** *(my recommendation: cheapest next win).*
  best_epoch=30 was the cap; the win head was still trending upward.
  Same recipe, raise `max_epochs` to 50 with patience=5 early-stop.
  Wall: ~5-6h. Expected +0.001 to +0.003 free.
- **`anonymous-aware-modeling-740`**. Even more attractive now that
  multitask has established the ceiling moves. Attack the persistent
  0.022 LOW-HIGH bucket gap via (a) router head for all-10-anonymous
  matches (12.6% of val) → separate small model, OR (b) per-team
  aggregate features over known-player subset. Should compound with
  multitask supervision. Wall: ~5-6h.
- **`multitask-alpha-tune-740`**. Small grid over α_dur ∈ {0.05, 0.10,
  0.15, 0.20} × α_i ∈ {0.1, 0.3, 0.5}. May yield small additional
  lift from better-balanced loss weights.
- **`player-features-decay-740`**. Exponential time-weighting on
  aggregator (τ ≈ 90 days). Smaller experiment; tests whether recent
  skill matters more than uniformly-weighted history.
- **Inference wrapper for personal use**: standalone CLI/notebook tool
  that takes (draft + 10 player aggregates) and returns
  (win_prob, duration_curve, per-slot item top-K). Productizes the
  multitask model's three heads for actual personal Dota use.
- **Engineering: extend `build_features.py` to emit `pX_account_id`
  columns** alongside features. Would have saved 45 min on
  embedding-prelim. Unifies the data pipeline for identity-flavored
  work.
- **Engineering: DVC-track** the new
  `data/snapshots/.../player_features_prepatch_clean/` and
  `data/snapshots/.../rich_cols/` directories so re-fetches by
  downstream experiments are reproducible.
- **Engineering: promote** `_meta/hardware-investigation-2026-05-21/test_pyarrow_roundtrip.py`
  to a quarterly RAM-health regression check. Cheap, easy to schedule.
- **(Long-term hardware)** Consider replacing with ECC UDIMM kit
  (X870E HERO supports it). Kingston Server Premier
  KSM56E46BD8KM-48HM is one option. ~$300-500 for 96 GB. Hardware-level
  error detection/correction; would let you safely re-enable EXPO
  6000 OR run more aggressive memory configs without silent bit-flips.
- **Removed from queue**: `draft-order-features-740` — Turbo hides the
  enemy draft, so pick-order encoding isn't valid for this project.
- (Carryover, deferred) Promote `run_sweep_loop.sh` +
  `cleanup_failed_trials.py` to `_meta/templates/`.
- (Carryover, deferred) LLM-driven islands evolution. Higher-leverage
  in principle but bigger investment than the cheap follow-ups above.
- (Carryover, deferred) HCE-vs-prior-art-splits ADR; 5M-vs-13M sanity
  check; DVC formalisation.


## 2026-05-23

### Did

- **2-round literature survey on foundation models across 14 papers**
  (tabular/recommendation/sports first round: FT-Transformer, PMAE,
  UW-SO, SAINT, M6-Rec, HIGFormer, ForkMerge; cross-domain second round:
  Pangu-Weather, Moirai-MoE, JMP, Octo, Whisper). 4 new concepts seeded
  (`tabular-foundation-model`, `masked-modeling-tabular`,
  `uncertainty-weighted-multitask`, `multi-query-foundation-model`), then
  3 more (`attention-bias-positional`, `task-as-token-prompting`,
  `supervised-multitask-pretraining`) → promoted to MoC
  `mocs/foundation-models.md`.
- **Wrote and implemented `foundation-mvp-740`** (5M-param Transformer
  foundation model: FT-Transformer skeleton + permutation-equivariance
  within team + (team,team) attention bias + patch token + UW-SO loss
  weighting + PMAE auxiliary + shared decoder with task-as-token
  prompting). 3 ablations: `baseline_multitask_repro` (works at 0.6470),
  `foundation_mvp` (broken at 0.5058), `foundation_no_patch_token`
  (broken at 0.4984). ~17h compute on stable JEDEC RAM.
- **Wrote v2 proposal `foundation-component-isolation-740`**: 3
  ablations each adding ONE new component on top of the working baseline
  to attribute the failure.
- **Codified "Monitoring long-running ML jobs" discipline** in user-level
  `~/.claude/CLAUDE.md` + brief reference in project CLAUDE.md.
- **User correction: no slot semantics in Turbo.** `player_slot` is
  arbitrary lobby order. Killed the original Pangu-style per-slot
  attention bias from the foundation-mvp proposal; replaced with
  permutation-equivariance within team + (team,team) 2×2 bias.
- **User correction: data source is Steam Web API, not OpenDota.**
  Fixed in 6 in-repo references (CLAUDE.md, concept, index, proposal,
  2 code comments). Public retraction comment didn't reference the
  source so no upstream change needed.

### Findings

- **The foundation-mvp design broke training catastrophically** despite
  all four new components being well-grounded in literature. baseline
  ablation at 0.6470 (close to anchor) confirms the SCALE (77K→5M
  params) is neutral; the additions are the saboteur. Most likely:
  UW-SO loss-scale misapplication (T=0.45 + 30× raw loss-scale variance
  means low-loss tasks like items dominate by ~30× over duration), PMAE
  collapsed to mae_loss=0 mid-training (masking implementation bug),
  possibly (team,team) bias interacting badly with multi-task heads.
- **The proposal's anticipated diagnostic fork worked.**
  `baseline_multitask_repro` was specifically there to disambiguate
  "scale broke it" vs "design broke it." Answer: design. v2 is the
  attribution experiment that fork pointed to.
- **The multi-conditional-queries framing is genuinely ahead of
  published tabular FM literature** per the cross-domain survey. M6-Rec
  (recommendation) and Octo (robotics) are the closest analogs; tabular
  FMs (FT-Transformer, SAINT, TabPFN) all benchmark single-target
  prediction. We're working at the edge of the recipe and should
  expect some design self-invention.
- **Pre-train-then-fine-tune doesn't transfer cleanly to tabular**
  (per the literature). Joint multi-task + auxiliary MAE from scratch
  is the right pattern (JMP shows +59% over unsupervised pre-training).
- **17h of unmonitored training was wasteful** — the foundation_mvp
  failure was visible at epoch 3 (train_win INCREASING, val_auc at
  random). The new monitoring rule should prevent recurrence.

### Next

- **`foundation-component-isolation-740`** (PROPOSED). 3 ablations
  attributing the foundation-mvp failure. Each adds ONE new component
  on top of baseline_multitask_repro. PMAE bug-fix or extra-logging
  BEFORE iso_pmae; UW-SO loss-scale normalization BEFORE iso_uwso.
  Live monitoring per the new rule. ~10-15h total.
- **`foundation-v3`** (depends on v2 results). Re-introduces ONLY the
  components that pass v2, with targeted fixes for the broken ones.
  Plus: maybe add the SAINT contrastive auxiliary (well-supported,
  was deferred), and/or move from patch-token to FiLM patch
  conditioning if cross-patch ablation tells us we need it.
- **Data scale-up** (deferred from foundation-mvp original proposal).
  Once architecture is debugged, the next axis is extending training
  to Aug 2025 → Feb 2026 (~30-40M matches across ~3 patches). Needs
  ~3-4h CPU pre-build of player aggregates + rich-cols sidecar over
  the broader window.
- **Inference wrapper / personal use tool** (deferred from
  multitask-740). Once we have a stable foundation model, build a
  CLI/notebook tool that takes (draft + 10 player aggregates) and
  returns (win_prob, duration_curve, item top-K per slot).
- **(Carryover, deferred)** `anonymous-aware-modeling-740` —
  compounds with foundation if v3 works.
- **(Carryover, deferred)** `player-features-decay-740` — smaller
  experiment, exponential time weighting (τ ≈ 90 days).
- **(Carryover, deferred)** DVC formalization, HCE-vs-prior-art ADR,
  5M-vs-13M sanity check.


## 2026-05-24

### Did

- **Implemented `foundation-component-isolation-740` v2** with live
  monitoring per the new `~/.claude/CLAUDE.md` "Monitoring long-running
  ML jobs" rule. 3 ablations (`iso_uwso`, `iso_pmae`, `iso_teambias`),
  each adding ONE new component on top of the working
  `baseline_multitask_repro` config from foundation-mvp-740.
- **Halted `iso_uwso` at epoch 2** when omega collapsed to 1.000 for
  items and train_win started INCREASING — exactly the foundation-mvp
  failure pattern. Math-deterministic: omega=1.000 means win head's
  gradient is identically zero, model can't recover. Saved ~4-5h of
  wasted compute. The new monitoring discipline worked as designed.
- **iso_pmae completed**: val_auc=0.6464 @ best=21 (Δ=-0.0006 vs anchor
  0.6470). PMAE with EMA-teacher fix is SAFE.
- **iso_teambias completed**: val_auc=0.6493 @ best=14 (Δ=+0.0023 vs
  anchor). (team, team) attention bias is HELPFUL — ~64-param addition
  gives real lift.
- **Rolled up v2 artifacts** + committed `5cbcaec`.
- **Audited code** on user's question: KDA/GPM/HD already use SmoothL1
  regression; **duration is STILL 8-bucket CE** — never got switched.
  Fixing in v3.

### Findings

- **The full diagnostic story of foundation-mvp-740 is now clean.**
  UW-SO (as we implemented it, with or without per-task initial-loss
  normalization) is broken on this multi-task setup. PMAE was broken
  BECAUSE OF Bug A (student=teacher BYOL/JEPA collapse); the EMA-teacher
  fix makes it neutral-safe. (team, team) attention bias is a small but
  genuine win at ~64 params.
- **Live monitoring saved ~4-5h** on this experiment alone. The pattern
  is now reliable: poll log every 30-45 min, look for train loss
  increasing + val at random + multi-task weight collapse + NaN/Inf.
  When the pattern is unambiguous (omega=1.000 means deterministic
  failure), halt at 2 epochs not 3.
- **Duration regression switch was never made** in foundation-mvp or
  component-isolation. The proposal text said "1 scalar SmoothL1 on
  log(seconds)" but the subagent kept multitask-740's 8-bucket CE.
  v3 fixes this.
- **The v3 design is now grounded in evidence**: drop UW-SO; keep
  canonical hero sort + (team, team) bias + PMAE w/ EMA; revert to
  hand-tuned α weights; ADD duration regression; test patch token on
  broader cross-patch data.

### Next

- **`foundation-v3-740`** (PROPOSED). All v2 evidence-driven design
  decisions + duration switched to regression + patch token now
  meaningful on broader cross-patch corpus. Needs ~3-4h CPU prebuild
  to extend player aggregates + rich-cols sidecar over Aug 2025 →
  Feb 2026, then ~6-7h training. Target val_auc ≥ 0.6510.
- **(Long-term)** Inference wrapper / personal-use tool: take
  (draft + 10 player aggregates) → (win_prob, duration-curve via
  regression head, item top-K per slot). Standalone CLI/notebook.
- **(Long-term)** Once foundation works, downstream queries
  (hero-pair synergy, lineup-vs-lineup, item rec conditioned on
  net_worth, fun-pair) per the original foundation framing.
- **(Carryover, deferred)** `anonymous-aware-modeling-740` — compounds
  with foundation.
- **(Carryover, deferred)** `player-features-decay-740` — smaller.
- **(Carryover, deferred)** DVC formalization, HCE-vs-prior-art ADR.


## 2026-05-25

### Did
- **foundation-v3-740 completed**: val_auc=0.6462 @ epoch 25/30 (6.08h
  wall). Clean convergence, NOT another foundation-mvp crash. Coverage
  buckets HIGH=0.6565, MED=0.6450, LOW=0.6364. Missed target 0.6508 by
  0.0046, missed iso_teambias 0.6493 by 0.0031.
- **OOM fix during v3 data build**: `out_cols` dict accumulated all
  emitted rows across 196 days unbounded. Killed at day 123 with
  RSS=91 GB. Implementer added chunked disk-persistent output every
  30 days + stream-concat via ParquetWriter. Same fix applied
  preemptively to `build_rich_cols_extended.py`. RSS plateaued at
  ~46 GB on the retry. Aggregator dict grew at ~0.16 GB/day early
  then ~0.04 GB/day late (unique-player saturation).
- **v3-ablations-740 proposed + implemented + completed**: A1
  (v3_dur_ce) val_auc=0.6349 (Δ=-0.0113 vs v3), A2 (v3_player_emb)
  val_auc=0.6290 (Δ=-0.0172 vs v3, catastrophic overfit). Total
  wall 4.78h (A1 3.32h + A2 1.46h; A2 early-stopped fast).
- **Concept note created + updated**:
  `[[concepts/embedding-vs-features-gradient-competition]]` —
  initially captured the gradient-starvation failure mode from
  embedding-prelim-740, updated post-A2 to document the second
  failure mode (overfit on extended cross-patch data) with
  diagnostics + mitigations for each.
- **Index updated**: 4 new completed experiments (foundation-mvp,
  component-isolation, foundation-v3, v3-ablations).

### Findings
- **Duration regression switch was NOT the v3 regression cause.** A1
  (revert to 8-bucket CE on the v3 stack) val_auc=0.6349 vs v3=0.6462
  — CE is WORSE by 0.0113 in this stack. The dur_top1=0.176 confirms
  the CE head was learning normally; it just doesn't help the win head
  more than regression did.
- **Player embeddings on extended data overfit catastrophically**, not
  the gradient-starvation pattern from embedding-prelim-740. Pattern:
  train_win 0.6812→0.6550 (down) while vl_win 0.6682→0.6840 (up).
  Coverage HIGH bucket hurt MOST (Δ=-0.019), opposite of "embeddings
  help frequent players". Mechanism: enough per-player signal in
  extended train to learn the table, but train (Aug2025-Feb2026
  multi-patch) and val (single-patch 7.40 late Feb / early Mar) differ
  enough that learned vectors don't transfer. Two failure modes now
  documented in the concept note.
- **Live-monitoring discipline keeps paying off**: A1 early-stopped at
  epoch 16 saving ~3h, A2 early-stopped at epoch 7 saving ~5h. The
  patience=5 on vl_win is sometimes too lenient for embedding overfit
  (where vl_win _improves_ early while val_auc declines); concept note
  recommends patience=2 for overfit-mode runs.
- **Remaining suspects for v3's regression vs iso_teambias**: extended
  cross-patch data itself OR PMAE-on-extended-data interaction.
  Duration form and player identity both ruled out via A1+A2.

### Next
- **`v4-iso-teambias-extended-740`** (PROPOSED): run the v2-winner
  architecture (iso_teambias: 7.40-only-style multitask + (team,team)
  bias, no PMAE, no patch token, no UW-SO, 8-bucket CE duration) on
  the EXTENDED cross-patch data. Cleanly tests whether the data
  extension itself is the regression cause vs v3 component composition.
  Reuses existing extended parquets. ~6h wall, single ablation.
- (Carryover) `anonymous-aware-modeling-740` — orthogonal axis, addresses
  the LOW-bucket binding constraint (anonymous tail 0.6364 vs HIGH 0.6565
  = biggest delta in the project).
- (Carryover, deferred) Inference wrapper / personal-use tool.
- (Carryover, deferred) DVC formalization, HCE-vs-prior-art ADR.

### Structured

```yaml
intended_effect: "Two factor-isolation ablations on v3 to attribute the regression vs iso_teambias to either duration loss-form (A1) or per-player identity axis on more data (A2)."
intended_effect_confirmed: yes
diagnostics.leakage_check: "splits.yaml date filter assert_no_test_dates — passed"
diagnostics.overfitting_signal: "A1 train=0.6712 val=0.6611 gap=+0.01 healthy; A2 train=0.655 val=0.684 REVERSED (classic embedding overfit)"
diagnostics.data_quality_issues: "none — reused v3 extended parquets verbatim, post-build row-group stats verified clean during v3"
delta_from_prior: "A1 vs v3=-0.0113; A2 vs v3=-0.0172, vs embedding-prelim-740=-0.0186"
next_candidates:
  - "v4-iso-teambias-extended-740: isolates data-extension effect on v2-winner architecture"
  - "anonymous-aware-modeling-740: orthogonal axis on the binding LOW-coverage constraint"
```

## 2026-05-26

### Did
- **v4-iso-teambias-extended-740 COMPLETED**: val_auc=0.6471 @ epoch
  16/21 (early-stop, 3.95h wall). Outcome (b) confirmed.
- **Attribution math closed**: v3 regression vs iso_teambias = -0.0031
  = -0.0022 (extended-data alone) + -0.0009 (PMAE+patch+dur composition).
  Roughly 70/30 split — extended data is the bigger cost; component
  composition only mildly hurts.
- Status flip + README + index + commit.

### Findings
- **Extended cross-patch data costs ~0.002 val_auc** on the simplest
  known-good architecture. The patch_id token in v3 was supposed to
  bridge the multi-patch train ↔ single-patch val distribution gap;
  did not fully.
- **Component composition costs ~0.001** on top of that. PMAE/patch_token/
  dur_regression interactions are mildly negative but small.
- **Coverage HIGH on v4 = 0.6574** — slightly above v3's 0.6565,
  approaching the transformer-plus-features-extended record (0.6588).
  Extended data does help the HIGH bucket modestly.
- **Real engineering tradeoff**: 7.40-only for max val_auc vs extended
  for cross-patch downstream-query coverage. User's foundation framing
  prefers extended.

### Next
- **`v5-rich-skill-features-740`** (designed below): pursue user's
  foundation direction. Extend per-player input features from 8 → ~14
  with item-derived skill proxies (last20_gpm, last20_hd, etc) and
  richer hero-novelty signal. Tests whether engineered-feature richness
  can close the v4 → iso_teambias gap WITHOUT embeddings.
- (Skipped — user veto) anonymous-aware-modeling. User's main goal is
  the foundation model for their personal-account queries (non-anonymous,
  HIGH coverage), so the anonymous-tail axis isn't valuable.
- (Skipped — outcome attributed) v5-pmae-only-on-v4 was the proposal-
  decision-tree's outcome-(a) next step. We landed in (b), which
  reframes the next-step priority toward richer features rather than
  isolating component interactions.

### Structured

```yaml
intended_effect: "Isolate extended-data factor on the v2-winner architecture to attribute v3's regression vs iso_teambias precisely."
intended_effect_confirmed: yes
diagnostics.leakage_check: "splits.yaml date filter assert_no_test_dates — passed"
diagnostics.overfitting_signal: "train=0.6492 val=0.6549 gap=+0.0057 healthy"
diagnostics.data_quality_issues: "none — reused extended parquets verbatim"
delta_from_prior: "vs v3 +0.0009; vs iso_teambias -0.0022 (the data-extension penalty); vs cleanup_anchor -0.0006"
next_candidates:
  - "v5-rich-skill-features-740: extend per-player features with item-derived skill proxies + hero-novelty (no embeddings; user direction)"
  - "v5-pmae-only-on-v4: cheap PMAE-on-v4 isolation (deprioritized given user direction)"
```

## 2026-05-26 (later)

### Did
- **v5-pretrain-finetune-740 HALTED**: Phase 1 epoch 16/20. Mid-pretrain
  probe trajectory 0.4711 → 0.5237 → 0.5304 → 0.5263 (regression).
  Classic SSL over-specialization to reconstruction. Halt fired per
  pre-committed criterion; saved ~10h. Cost ~2h GPU on aborted pretrain.
- **Concept-level discussion with user** on SSL family tradeoffs
  (reconstruction vs contrastive vs JEPA). User picked Design J — JEPA
  on v5 scaffolding.
- **v6-jepa-pretrain-finetune-740 PROPOSED**: single-change ablation of
  v5 — swap reconstruction → JEPA latent-space prediction, reuse all
  other v5 scaffolding (6-group masking, EMA teacher [finally used],
  mid-pretrain probe). ~12.5h.
- v5 status flip + Diagnostics filled in + index entry added.

### Findings
- **BERT-style raw-target reconstruction on this mask scheme over-trains
  past useful win-discriminative features**: the v5 trajectory is
  diagnostic, not catastrophic — encoder DID find useful features at
  epoch 5, then drifted away as reconstruction loss kept pulling.
- **The EMA teacher infrastructure scaffolded in v5 was UNUSED for the
  loss** (raw-target reconstruction was the BERT-style choice). v6
  actually uses it for what it was designed for — JEPA latent-space
  prediction.
- **Mid-pretrain probe is a load-bearing diagnostic**: without it, we
  would have burned the full 6h pretrain + linear probe + 6h fine-tune
  before knowing the encoder wasn't learning useful features. Worth
  keeping in every future pretrain experiment.

### Next
- **`v6-jepa-pretrain-finetune-740`** (PROPOSED + about to implement):
  swap loss form, keep everything else, test "was reconstruction the
  problem?" Halt criterion same as v5 (mid-probe ≤ 0.51 by ep10 → halt).
- (Deferred) v7-contrastive-player-centric — if v6 also fails, the
  next SSL family to try (sample two matches per non-anonymous player,
  InfoNCE on per-player reps).
- (Deferred) v7-rich-skill-features — pragmatic pivot to engineered
  features if SSL universally fails.

## 2026-05-26 (evening) — pivot to downstream

### Did
- **v6-jepa HALTED at Phase 1 ep11**: classic representation collapse
  (jepa_loss 0.0284 → 0.0014, rep_l2 14.4 → 2.5, mid_probe stuck at
  0.5017). Pairwise cosine DID decrease (0.972 → 0.911 — slot
  differentiation worked) but reps collapsed to low-magnitude
  trivially-predictable subspace. Saved ~10h.
- **Brainstorming detour with user**: diffusion-style ("sharpening the
  outcome distribution") vs marginal multi-task; VAE vs PMAE
  structural+theoretical+tradeoffs comparison; MAGE-style variable
  masking as a cheap test of "did fixed mask rate cause v5 failure?"
- **v4 diagnostic — STRONG positive**:
  - Hero embeddings cluster by role/archetype (carries with carries,
    supports with supports, mids/casters, initiator-tanks). Cosine
    similarity well-spread (mean -0.002, std 0.105, range
    [-0.36, +0.46]). Not collapsed.
  - Sanity check: post-game GPM-diff alone hits 0.9931 val_auc on the
    same data; net_worth-diff 0.9918. Architecture/data work
    perfectly when given strong signal.
  - v4 encoder representation: PCA-1 captures 52% variance and
    correlates +0.98 with v4 win_pred, +0.49 with skill_diff.
    Effective dim ~10-20 (uses capacity, not collapsed).
  - UMAP-x decile analysis: monotonic win_rate from 0.366 (low-x) to
    0.726 (high-x) — the encoder organizes matches along expected
    outcome axis.
  - Linear probe on team-diff-pool: 0.622 val_auc — within 0.012 of
    trained win head (0.634 on this 5k sample). Decoder/cross-attention
    adds almost nothing.
- **Pre-game baseline diagnostics** (univariate / multivariate LR on
  team-mean-diff of 8 player features):
  - smoothed_winrate_hero: 0.5761 (dominant signal)
  - n_games_hero_log1p: 0.5305 (hero novelty matters)
  - plain smoothed_winrate: 0.5084 (matchmaking effect — useless alone)
  - all 8 team-diff features (multivariate LR): 0.5851
- **Saved deferred foundation paths** at
  [[_meta/deferred-foundation-paths]]. Five paths (v7-rich-skill,
  v7-mage-lite, v7-cvae, v7-cross-head, v7-diffusion) with cost +
  pickup-trigger for each.

### Findings
- **The v4 architecture is sound, NOT broken.** Hero embeddings,
  encoder reps, monotonic outcome organization — all behave as a
  trained foundation model should.
- **The val_auc 0.647 ceiling is data-bound, not architecture-bound.**
  Pre-game features have inherent signal limits: matchmaking flattens
  player_winrate, anonymity rate is 66%, game variance is high.
- **Decoder + cross-attention barely outperforms linear probe**
  (+0.012). The encoder is doing essentially all the work.
- **The two SSL failures were genuine SSL design failures**, not
  symptoms of a broken architecture: v5 = per-token reconstruction
  over-specializes; v6 = latent prediction without collapse mitigation.
  Future SSL work needs proper mitigations (VICReg, KL term, variable
  masking) AND must demonstrate value vs a strong baseline.
- **The dominant per-player signal is smoothed_winrate_hero**
  (univariate 0.5761), exactly matching prior LightGBM importance.
  User's matchmaking-flattening hypothesis confirmed: plain
  smoothed_winrate is useless alone.

### Next
- **PIVOT to downstream queries on v4**: build the inference wrapper
  (draft + player aggregates) → (win_prob, duration-curve, item
  top-K, KDA/GPM/HD projections per slot). Then concrete query
  functions: hero-pair synergy, lineup-vs-lineup matchup, item rec
  conditioned on net_worth, fun-pair max-kills.
- (Deferred) The 5 architectural paths above. Pick up only if a
  specific downstream query reveals a representation deficiency.
- (Carryover) DVC formalization, HCE-vs-prior-art ADR.

### Structured

```yaml
intended_effect: "Diagnose whether v4's architecture is the cause of the val_auc ceiling vs the data itself. Verify representations learn expected hero/match semantics."
intended_effect_confirmed: yes
diagnostics.leakage_check: "n/a (read-only inspection)"
diagnostics.overfitting_signal: "v4 train_win=0.6492 vl_win=0.6549 gap=+0.0057 healthy"
diagnostics.data_quality_issues: "none — features in sensible ranges, GPM/NW post-game stats trivially predict win at 0.99 val_auc"
delta_from_prior: "n/a (diagnostic, not training)"
next_candidates:
  - "Downstream inference wrapper + query functions (next-step)"
  - "v7-rich-skill-features (if a query reveals missing per-player item-history signal)"
  - "v7-mage-lite (if we revisit SSL after more lit review)"
```
