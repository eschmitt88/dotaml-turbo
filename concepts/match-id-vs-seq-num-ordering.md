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
- Related but unresolved: when the prior-art DotaML repo wrote
  `DUPLICATION_REPORT.md` about overlapping `match_seq_num` ranges in
  Azure filenames, the implicit assumption was that adjacent sequence
  ranges should not overlap. They mostly don't, but the rare overlap
  produced 4.9% match_id duplication. A safer downstream guarantee:
  deduplicate by `match_id` at read time regardless of filename layout.
