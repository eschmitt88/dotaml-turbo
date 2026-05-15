---
kind: concept
name: "match-id-vs-seq-num-ordering"
status: seedling
added: "2026-05-15"
sources:
  - literature/repos/eschmitt88-DotaDB.md
related_concepts:
  - draft-only-win-prediction
related_experiments: []
tags: [data-engineering, time-based-split, steam-api]
---

# match-id-vs-seq-num-ordering

## Definition

In Steam's Dota 2 API, two integer identifiers exist per match:

- **`match_id`**: assigned close to when the match starts. In an empirical
  study by DotaDB (1,124 matches, August 2025) Match ID correlates 77.6%
  with `start_time`.
- **`match_seq_num`**: assigned when Steam's backend processes the match
  data, which can lag the match start by an arbitrary amount. Correlates
  only 42.9% with `start_time`.

The Parquet layout in the Azure Data Lake partitions files by
`year/month/day` of `start_time`, but the filenames carry
`match_seq_num` ranges (`matches_{min_seq}_{max_seq}.parquet`).

## Why it matters here

Time-based splits — train on early window, validate on later window —
**must partition on `start_time` or `start_time_date`**, not on
`match_seq_num`. Partitioning on sequence number leaks: a match played
yesterday with a delayed Steam-side process timestamp could land in a
"future" sequence-number bucket relative to a match played today, and
vice versa.

This rule is load-bearing for HCE in this project: the patch-7.40 window
boundary (2025-12-16 → 2026-03-23) must be enforced on `start_time`, and
the validation/test split within that window must also be by
`start_time`. The matter is settled — there is no ambiguity to resolve,
this concept exists to lock it in.

## Connections

- [[draft-only-win-prediction]] — the task whose splits this concept
  constrains.

## Empirical addendum (2026-05-15 pre-flight scan, patch-7.40 window)

A full scan of all 97 day boundaries in `2025-12-16 → 2026-03-23` found
**every single boundary has cross-day `match_seq_num` overlap** (typical
span 15-25k seq_nums; outliers up to 255k on 2026-03-16→17). This looked
alarming versus the prior-art `DUPLICATION_REPORT` finding of 4.9%
`match_id` duplication. A direct probe of the 2025-12-16/17 boundary
files shows:

- 875 rows in `day16_last`, 10,000 in `day17_first`
- **Zero `match_id` intersection**
- **Zero `match_seq_num` intersection**

So the seq_num "overlap" is **structural, not a bug**: the collector
partitions by `start_time_date`, and matches that started near midnight
get binned to their actual calendar day rather than to a seq_num bucket.
The seq_num sequence interleaves at midnight (Steam doesn't pause its
backend at UTC 00:00) but rows themselves don't duplicate.

The prior-art's 4.9% duplication was a different upstream bug, since
fixed. The downstream guarantee remains: **dedup by `match_id` at read
time**, codified in `splits.yaml` (`dedup_key: match_id`). It's a
near-no-op now but cheap insurance against any future collector
regression.
