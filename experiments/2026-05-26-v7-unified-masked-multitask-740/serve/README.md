# v7 serve/ — downstream queries

Tools for testing the v7-unified-masked-multitask trained model on the
downstream queries it was designed for. NOT a production service — see
the parent experiment's README for the model itself and the
"is-this-a-served-model?" framing.

## Files

- `v7_inference.py` — `V7Foundation` class: loads the trained model
  and provides a maskable forward pass + helpers to build input tensors.
- `lookups.py` — hero/item ID ↔ name (OpenDota constants saved locally
  as `heroes.json` / `items.json`), per-account player feature lookup
  from val parquet + sidecar.
- `queries.py` — concrete query functions:
  - `personal_winprob(heroes, account_ids=None)` — P(radiant_win) for a draft
  - `lineup_matchup(radiant, dire, ...)` — P(radiant_win) + predicted duration
  - `hero_pick_rec(known_radiant, known_dire, my_side, account_id, ...)` — top-K heroes via v7's trained hero mask token, single forward pass per candidate
  - `item_rec_marginal_sweep(heroes, my_slot, current_bag, ...)` — DIAGNOSTIC BASELINE: top-K items by marginal win-prob lift (strongly confounded)
  - `item_rec_odds_ratio(heroes, my_slot, current_bag, ...)` — Design A: rank items by P(item|win) / P(item|loss), draft+player context held fixed
  - `build_path(heroes, my_slot, ...)` — Design B: ordered budget-aware progression using odds_ratio + predicted GPM × duration
  - `item_rec_given_win(heroes, my_slot, ...)` — top-K items P(in bag | my-team wins) — descriptive, not prescriptive
  - `win_vs_duration(heroes, duration_minutes=[...])` — sweep duration as input
  - `kills_per_minute_pair(hero_subset, ...)` — predicted K+A per minute for a 1-5 hero subset
- `notebook.qmd` — Quarto notebook demonstrating all queries on
  account 3303652. Render with `quarto render notebook.qmd`.
- `heroes.json` / `items.json` — OpenDota constants snapshot
  (downloaded once; safe to re-fetch if patches add new heroes/items).

## Quick start

```python
import sys
sys.path.insert(0, '/path/to/experiments/2026-05-26-v7-unified-masked-multitask-740')
from serve.v7_inference import V7Foundation
from serve.lookups import hero_id
from serve import queries

f = V7Foundation()  # loads checkpoint once, ~0.3s on cuda

draft = [hero_id(n) for n in [
    'Anti-Mage', 'Drow Ranger', 'Zeus', 'Rubick', 'Mars',
    'Crystal Maiden', 'Shadow Fiend', 'Puck', 'Pudge', 'Sniper']]

wp = queries.personal_winprob(f, heroes=draft)
print(f'P(radiant_win) = {wp:.4f}')
```

## Performance

All queries run in 1-10s on RTX 5080 after the model is loaded
(~0.3s one-time cost):

- `personal_winprob`, `lineup_matchup`: <0.5s (1 forward pass)
- `win_vs_duration` (8 points): ~1s (8 rows batched)
- `item_rec_marginal_sweep`, `item_rec_given_win`: ~1.2s (1 + 150 rows batched)
- `item_rec_odds_ratio`: ~1.2s (2 forward passes, 150-item ranking from outputs)
- `build_path` (6 steps): ~8s (one odds_ratio call per build step)
- `kills_per_minute_pair`: ~1.3s (12 rows batched)
- `hero_pick_rec` (full 148-candidate sweep via hero mask token): ~5s
  (148 rows batched in a single forward pass — single mask-token query
  per candidate, no sampling)

Per-row GPU transfers are the bottleneck if you build inputs in a
python loop; all sweep functions build numpy arrays first and do
a single `.to(device)` call.

## Known limitations

- **Item rec is correlational, not causal.** `item_rec_odds_ratio`
  controls for draft+player context across the two conditional
  probabilities, which is a partial mitigation — but proper causal
  estimation via propensity-score weighting or DR estimators would
  give stronger guarantees. Future work.
- **`build_path` progression is heuristic.** Uses predicted GPM ×
  predicted duration as a gold budget; `expected_minute` is
  `cumulative_cost / GPM` assuming linear gold accumulation. Real
  item timing data (e.g. parsed-replay events) is not available, so
  this is an estimate, not a guaranteed acquisition schedule.
- **Hero pick rec uses v7's trained hero mask token** (`hero_mask_embed`).
  v7's `partial_draft` training scenario explicitly trained the win
  head on partial drafts and converged to highest adaptive sampling
  probability (0.293, up from 0.150 init). Single forward pass per
  candidate, no sampling overhead.
- **Account feature lookup falls back to `ANON_FEATS`** when an account
  isn't found in val. The user's account (3303652) is present.
