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

### Next

- Decide whether to write `docs/decisions/0001-hce-vs-prior-art-splits.md`
  recording the deliberate departure from the prior-art's 80/20 chronological
  split, before any data is pulled.
- Implement the proposal at `experiments/_proposals/2026-05-15-plateau-baseline-740.md`
  via `/implement` — this triggers the first real data pull from Azure into
  `data/snapshots/7.40-2025-12-16/raw/` (~100 GB). Confirm SN850X has room
  (budget.yaml says 3400 GB free; plenty).
- Verify before pulling: the Azure file-overlap issue described in
  DotaML's `DUPLICATION_REPORT.md` was fully closed before 2025-12-16,
  not just patched for the affected August 2025 days. A quick listing of
  `match_seq_num` ranges in the 7.40 window's filenames will tell.
- Verify the `~200k matches/day` rate on the patch-7.40 window; if real
  volume diverges by >10% from the 19.6M expected total, update
  `expected_match_counts` in `splits.yaml` and note in an ADR.
- Once the baseline runs, decide whether to use a stratified 5M subset
  (memory-safe, mirrors DotaML v3) or fit the full 19.6M; ideally show
  both numbers are within 0.005 AUC of each other.
