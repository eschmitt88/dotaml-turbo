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
