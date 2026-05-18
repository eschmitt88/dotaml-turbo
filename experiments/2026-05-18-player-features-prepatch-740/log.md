# player-features-prepatch-740 log

- 2026-05-18: scaffold + smoke complete by MLE-IMPLEMENTER subagent.
  pull_history.py, build_features.py, train.py, run_all.sh, config.yaml
  in place. Smoke test pulled 2 history days (2025-12-14, 2025-12-15) and
  ran build + train against 2 history + 3 patch-7.40 days.
- 04:33 Full run #1 launched. pull_history.py SUCCESS (1409 s, 98.13 GB, 0 errors).
  build_features.py STARTED 04:56, then SEGFAULTED at 05:23 (day 66 of 196).
  Python crash at NULL+1 dereference inside CPython (IP 0x586e04 in python3.12
  binary), not a CUDA / OOM / kernel issue. Per-day timing was ballooning
  before the crash (6.6s/day → 31s/day across 18 days). Cause: per-account
  state structures growing badly at scale.
- 05:26 Full run #2 launched (skipping pull, already done). build_features.py
  ran for 2h 12min then was OOM-killed at 07:38 — Python RSS hit 93.8 GB on
  91 GB-RAM-+-8 GB-swap system. Root cause: per-account `coplay` nested dict
  (5M accounts × 200-cap entries × ~75 bytes = ~75 GB just for coplay).
- 12:30 Surgical fix: removed `coplay` tracking entirely (not in top-20
  importance from player-features-740, so no informational loss) AND
  switched `unique_heroes` set tracking to `len(hero_n[acct])` (saves ~8 GB
  more). Memory ceiling now estimated ~15 GB peak. Feature schema: 8 player
  feats × 10 = 80 cols (vs 90 in player-features-740). Updated train.py
  mirror list. Cleared stale 90-col smoke artifacts.
- 12:37 Full run #3 launched. build_features.py completed in 120 min (rc=0,
  no crashes, no OOM). 3 train.py ablations completed in 8 min total. ALL DONE
  at 14:46.
- 14:50 Aggregated metrics + diagnostics. Headline: val_auc=0.6256
  (+0.0028 over player-features-740 0.6227, +0.0095 over plateau-baseline-740,
  but missed target 0.6277 by 0.0022). Sanity check PASSED (heroes_only
  Δ=-0.0001).
- Coverage-bucket diagnostic was the EYE-OPENER: stayed monotonic (did NOT
  flatten as predicted), HIGH gained most (+0.0043) and LOW gained least
  (+0.0014). History-source breakdown explained: low-bucket players have
  only 2.3% prepatch fraction — they're genuinely casual/new, not active-but-
  uncached. Cold-start NOT binding; casual-player tail IS.
- KEY finding: HIGH-bucket val_auc=0.6339 BEATS the Transformer architectural
  ceiling 0.6322 for the first time. Architecture vs player-features comparison
  is no longer a clear win for architecture on the active player subset.
- README finalized with Result / Interpretation / Diagnostics. status running→done.
