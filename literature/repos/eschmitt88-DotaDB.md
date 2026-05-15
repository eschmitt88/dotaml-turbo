---
kind: repo
name: "eschmitt88/DotaDB"
url: "https://github.com/eschmitt88/DotaDB"
commit: ""
source: "raw/repos/eschmitt88-DotaDB.md"
added: "2026-05-15"
relevance: 3
status: scanned
related_experiments: []
related_concepts:
  - match-id-vs-seq-num-ordering
tags: [prior-art, dota2, data-collection, steam-api, cosmos-db]
---

# eschmitt88/DotaDB

## Purpose

DotaDB is the upstream data-collection pipeline that feeds the Azure
storage that `dotaml-turbo` reads from. Steam Web API → DotaDB collector
→ Azure (Cosmos DB per these docs; the actual Data-Lake Parquet layout
DotaML and dotaml-turbo consume is downstream of these docs, in a stage
not described here). Reference-only — `dotaml-turbo` does not modify
this pipeline.

## Shape

- `src/dotadb/api/steam_client.py` — rate-limited Steam Web API client
  (5 calls/s, 100k daily cap; exponential backoff; 30s timeout).
- `src/dotadb/services/validated_raw_collector.py` — the working
  collector, sequence-based via `GetMatchHistoryBySequenceNum`. Strategic
  75-100-step sequence jumps; full per-match validation before write.
- `src/dotadb/models/match.py` — `FlexibleMatchData` Pydantic model with
  `extra="allow"`. Sets `partition_key = str(game_mode)`; preserves the
  whole API response in `raw_api_data`.
- `src/dotadb/database/cosmos_client.py` — Cosmos DB layer, partitioned
  on `/partition_key`. Turbo = "23".
- Docs: `README.md`, `ARCHITECTURE.md`, `COLLECTION_STRATEGIES.md`.

## Useful bits

- **Collection-time validation.** Before write, the collector confirms 10
  players, KDA fields, item_0..item_5, basic match metadata, and
  `game_mode == 23`. This guarantee transfers to anything reading the
  data lake downstream: `dotaml-turbo` can assume these fields are
  present.
- **Match-ID vs match-seq-num.** An empirical study (1,124 matches,
  August 2025) found Match IDs correlate 77.6% with `start_time` (IDs
  assigned near match start), while `match_seq_num` correlates only
  42.9% (assigned at Steam-side processing, which can be much later than
  the match itself). Time-based splits must partition on `start_time` or
  `start_time_date`, not `match_seq_num`. → seeds
  [[match-id-vs-seq-num-ordering]].
- **Turbo-only filter.** Matches whose `game_mode` cannot be determined
  from the API are dropped, not stored. This eliminates an entire class
  of "unknown game mode" missingness from the data lake.
- **`raw_api_data` field.** The full Steam API response is preserved in
  every Cosmos document. Whether this maps 1:1 to the `raw_json` column
  in the Parquet files DotaML reads is not stated in these docs but is
  consistent with the DotaML field reference (`raw_json` contains the
  complete match JSON).

## Follow-up

**Relevance:** 3/5 — useful prior art on the data pipeline that
`dotaml-turbo` consumes, but it does not shift any model-side concept.
Two genuinely actionable items: split on `start_time` not `seq_num`, and
trust the collection-time validation already done upstream.

Open questions for the new project (not blocking):

- These docs describe Cosmos DB storage, but DotaML reads from an Azure
  Data Lake `matches` container with Parquet files partitioned by
  `year/month/day`. There must be a Cosmos→Parquet export stage; it is
  not visible in this capture. Track down if/when a pipeline change
  becomes relevant — for now treat the Data Lake as the source of truth.
- The README's "Roadmap" lists "Data export: Parquet files for analysis"
  as future work; the existence of the data lake `dotaml-turbo` is
  consuming says this is now done. Status of that pipeline (cadence,
  patch-boundary handling, possible drift from the Cosmos source) is
  worth confirming before the first training run.
- The DotaML `DUPLICATION_REPORT.md` traced 4.9% duplicate match_ids to
  overlapping `match_seq_num` ranges in Azure filenames in 4 days in
  August 2025; the fix described there was applied. Verify before
  2025-12-16 (start of the patch-7.40 window) that the underlying
  collector behavior was corrected, not just the affected days.
