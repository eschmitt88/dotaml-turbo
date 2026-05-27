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
  - `hero_pick_rec(known_radiant, known_dire, my_side, account_id, ...)` — top-K heroes via candidate sweep
  - `item_rec_for_winprob(heroes, my_slot, current_bag, ...)` — top-K items by marginal win lift (correlational, not causal)
  - `item_rec_given_win(heroes, my_slot, ...)` — top-K items P(in bag | my-team wins)
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
- `item_rec_for_winprob`, `item_rec_given_win`: ~1.5s (1 + 150 rows batched)
- `kills_per_minute_pair`: ~1.3s (12 rows batched)
- `hero_pick_rec` (full 148-candidate sweep, 16 samples each): ~9s
  (2368 rows batched in a single forward pass)

Per-row GPU transfers are the bottleneck if you build inputs in a
python loop; all sweep functions build numpy arrays first and do
a single `.to(device)` call.

## Known limitations

- **Item rec is correlational, not causal.** Winning teams build
  winning items; the model can't tell which direction the arrow runs.
  Propensity-score-weighted analysis would give cleaner causal
  effects — future work.
- **Item progression (early vs late) is not supported.** rich_cols
  only has final inventories. A build-order heuristic could be
  assembled from item costs + predicted GPM + duration, but isn't
  built here.
- **Hero pick rec uses sampling.** v7 has no trained hero mask token,
  so we sample empirical completions instead of feeding `[MASK]` for
  unknown slots. Results are stochastic; use `n_samples ≥ 16` for
  stable rankings.
- **Account feature lookup falls back to ANON_FEATS** when an account
  isn't found in val. The user's account (3303652) is present.
