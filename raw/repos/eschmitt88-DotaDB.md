---
source_url: https://github.com/eschmitt88/DotaDB
fetched_at: 2026-05-15
fetched_by: fetch-paper
repo_default_branch: main
repo_private: true
capture_kind: repo-bundle
contents:
  - README.md
  - ARCHITECTURE.md
  - COLLECTION_STRATEGIES.md
---

# eschmitt88/DotaDB — prior-art capture

This is a self-contained snapshot of the DotaDB repo as of 2026-05-15.
DotaDB is the **upstream data-collection sister project** that feeds the
Azure Data Lake (`dota2datalake/matches`) that `dotaml-turbo` reads from.
The new project does not modify DotaDB; this capture is reference-only,
to understand the data schema, collection pipeline, and any filtering
already applied before data lands in the lake.

Two practical signals to extract:

1. **Match-ID vs match-seq-num ordering.** An empirical study in the repo
   (Aug 2025) found Match IDs correlate 77.6% with `start_time` while
   `match_seq_num` correlates only 42.9%. Sequence numbers are assigned
   at processing time, not match start time. Implication: time-based
   splits in `dotaml-turbo` should partition on `start_time` (or
   `start_time_date`), not `match_seq_num`.
2. **Collection-time validation.** Matches are filtered at write time
   (10 players present; KDA fields present; items 0-5 present;
   match metadata complete; `game_mode == 23`). Downstream training
   code does not need to re-check these.

The README still emphasizes Cosmos DB as primary storage; the actual
Parquet/Data-Lake export pipeline that DotaML reads from is not described
in these three docs (likely added later, or implemented in code not
captured here). Not critical for the model-building work.

---

## README.md

```
# DotaDB

A robust Python package for collecting, storing, and analyzing Dota 2 match
data from the Steam Web API. Features intelligent data collection, flexible
schema handling, and Azure Cosmos DB storage with optimized partitioning.

Features:
- Targeted Collection: Specialized Turbo match collector (game_mode 23)
- Resilient API Integration: Handles Steam API reliability issues with
  exponential backoff
- Flexible Schema: Adaptive Pydantic models that evolve with API changes
- Azure Integration: Cosmos DB storage with intelligent partitioning
- Rate Limiting: Respects Steam API limits (5 calls/sec, 100k daily)
- Docker Ready: Containerized deployment support
- Data Integrity: Only stores matches where game_mode can be determined

Architecture:
  Steam Web API → DotaDB Client → Cosmos DB
  (Match History, Match Details, Player Stats)
  (Rate Limiting, Error Handling, Schema Adapt)
  (Partitioned by game_mode, Flexible docs)

Collection strategy: sequence-based using
  GetMatchHistoryBySequenceNum (working, reliable)
  GetMatchDetails (frequently returns 500 errors — avoided)

Config (.env keys):
  STEAM_API_KEY, COSMOS_ENDPOINT, COSMOS_KEY,
  COSMOS_DATABASE_NAME=dota2matches, COSMOS_CONTAINER_NAME=matches,
  RATE_LIMIT_CALLS_PER_SECOND=5, RATE_LIMIT_DAILY_LIMIT=100000

Partition strategy: by game_mode. Turbo = partition_key "23".

Roadmap (incomplete):
  - Support for additional game modes
  - Real-time match tracking
  - Advanced analytics dashboard
  - Data export utilities
  - Machine learning integration
```

## ARCHITECTURE.md (key excerpts)

```
1. Steam API Client
   - Rate limiting (5 calls/s, 100k daily)
   - Exponential backoff retry
   - 30s request timeout
   - Endpoints:
       GET /IDOTA2Match_570/GetMatchHistory/V001/        (reliable)
       GET /IDOTA2Match_570/GetMatchDetails/V001/        (frequent 500s)

2. Data Models — FlexibleMatchData (Pydantic v2, extra="allow")
   - Automatically sets partition_key = str(game_mode).
   - Preserves complete raw API response in raw_api_data.
   - Schema evolution strategy: accept any fields; extract known core
     fields; store everything else.

3. Database — Cosmos DB
   - Partition key: /partition_key (= str(game_mode))
   - Throughput: 400 RU/s
   - Consistency: Session
   - Document size: ~10-50 KB per match
   - Write cost: ~5-10 RU per write

4. Turbo Collector — collection rule:
   for match_summary in match_history:
       details = await steam_client.get_match_details(match_id)
       if not details or 'game_mode' not in details:
           continue                    # skip, no partition key
       if details['game_mode'] == 23:
           await self.store_match(details)
   # i.e. only Turbo matches with confirmed game_mode are stored.

Performance characteristics (reported by the repo):
  - Target throughput: 5 calls/s
  - Actual: ~4.8 calls/s
  - Daily capacity: ~415k API calls
  - Matches/day stored: ~100k (with filtering)

Privacy notes:
  - No PII stored; account_ids are public Steam identifiers.
  - Match data is publicly available via Steam API.

Roadmap mentions "Data Export: Parquet files for analysis" but no
implementation detail in this doc; the Azure Data Lake (`dota2datalake`)
Parquet layout that DotaML reads must come from a later/separate stage.
```

## COLLECTION_STRATEGIES.md (key excerpts)

```
Sequence-based collection is the working strategy.
Entry point: python -m dotadb.sequence_collection_main
Implementation: src/dotadb/services/validated_raw_collector.py

How it works:
1. API endpoint: GetMatchHistoryBySequenceNum with start_at_match_seq_num
2. Data discovery: find recent sequence numbers via GetMatchHistory
   filtered by game_mode=23
3. Strategic jumping: 75-100-step sequence jumps instead of sequential
4. Full validation before write

Validation checks before saving to Cosmos:
  - 10 players present
  - KDA (kills, deaths, assists) per player
  - Item data (item_0 through item_5) per player
  - Match metadata (duration, radiant_win, etc.)
  - game_mode == 23 (Turbo)

Critical bug fix (recorded):
  Steam API client was passing 'match_seq_num': N — wrong.
  Correct parameter is 'start_at_match_seq_num': N.

Match-ID vs match-seq-num correlation study (1,124 matches, Aug 2025):
  - Match IDs correlate 77.6% with start_time   (assigned near start)
  - Sequence numbers correlate 42.9% with start_time (assigned at
    processing time, not match start)
  - 100% order mismatches between sequence-based and chronological
    sorting in recent matches.

Implications recorded by the author:
  - For chronologically recent matches: use Match ID ranges.
  - For systematic collection: use sequence numbers.
  - Sequence numbers preferred for collection because they ensure data
    is fully processed before fetch.

Sequence pattern examples (Aug 2025):
  Seq range 7,070,285,003 → 7,071,388,179 (span ~1.1M, avg gap 982)
  Match-ID range 8,417,564,169 → 8,418,895,313 (span ~1.3M, avg gap 1185)
```
