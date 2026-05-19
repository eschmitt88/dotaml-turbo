# Project log

One line per mutation — ingests, new experiments, wrap entries. Written by
skills; read by `/lint`.
2026-05-15 15:09 session_end session=0786da66-2c86-4e4e-869a-e6874b6156c2
2026-05-15 15:15 fetch-paper https://github.com/eschmitt88/DotaML → raw/repos/eschmitt88-DotaML.md
2026-05-15 15:17 ingest raw/repos/eschmitt88-DotaML.md → literature/repos/eschmitt88-DotaML.md (+5 concepts seeded)
2026-05-15 15:22 fetch-paper https://github.com/eschmitt88/DotaDB → raw/repos/eschmitt88-DotaDB.md
2026-05-15 15:23 ingest raw/repos/eschmitt88-DotaDB.md → literature/repos/eschmitt88-DotaDB.md (+1 concept seeded)
2026-05-15 15:28 propose plateau-baseline-740
2026-05-15 16:17 wrap-skip-structured reason=cwd-not-in-experiment
2026-05-15 16:17 wrap ingested DotaML+DotaDB, seeded 6 concepts, proposed plateau-baseline-740, drafted splits.yaml + snapshot dir
2026-05-15 17:24 implement plateau-baseline-740 → experiments/2026-05-15-plateau-baseline-740/ seeds=1 model=claude-opus-4-6 (val_auc=0.6161, partial confirm)
2026-05-15 18:18 wrap-skip-structured reason=cwd-not-in-experiment
2026-05-15 18:18 wrap session-2: pre-flight verified Azure overlap structural + 200k/day rate; ran plateau-baseline-740 to completion (val_auc=0.6161, partial confirm); promoted draft-prediction-plateau concept; refined architecture-spread finding
2026-05-15 18:35 propose plateau-architectures-740
2026-05-15 20:20 implement plateau-architectures-740 → experiments/2026-05-15-plateau-architectures-740/ seeds=1 model=claude-opus-4-6 (Transformer val_auc=0.6322, loose hypothesis confirmed, strict NOT confirmed — ResidualFFN<SimpleFFN inverts prior art)
2026-05-16 02:22 propose transformer-hp-sweep-740
2026-05-16 08:13 implement transformer-hp-sweep-740 → experiments/2026-05-16-transformer-hp-sweep-740/ seeds=1 model=claude-opus-4-6 (best val_auc=0.6318, hypothesis NOT confirmed — Δ vs control +0.0007 over 60 trials; ceiling is architecture-vocabulary-bound)
2026-05-17 01:50 adr docs/decisions/0001-per-trial-subprocess-isolation.md + upstream issue pytorch/pytorch#184062 filed (root cause: torch DataLoader + tensor GC heap corruption on Blackwell, not CUDA/driver; subprocess isolation is the production fix)
2026-05-17 02:00 wrap-skip-structured reason=cwd-not-in-experiment
2026-05-17 02:00 wrap catches up 3 arcs since previous wrap: plateau-architectures-740 (Transformer val_auc=0.6322), transformer-hp-sweep-740 (best 0.6318, HP-search exhausted), Blackwell torch DataLoader bug investigation + ADR 0001 + upstream pytorch/pytorch#184062
2026-05-17 19:02 fetch-paper https://arxiv.org/abs/1711.06498 → raw/papers/hodge2017win.pdf
2026-05-17 19:06 ingest raw/papers/hodge2017win.pdf → literature/papers/hodge2017win.md (relevance=4; sources added to draft-prediction-plateau + draft-only-win-prediction)
2026-05-17 19:57 propose player-features-740 (target val_auc ≥ 0.6361 via per-player history aggregates with HCE-strict leading-window; includes coverage-bucket diagnostic to decide whether a pre-patch ingest follow-up is needed)
2026-05-17 22:31 implement player-features-740 → experiments/2026-05-17-player-features-740/ seeds=1 model=claude-opus-4-6 (val_auc=0.6227, +0.0067 vs baseline but HYPOTHESIS NOT CONFIRMED — missed target 0.6361 by 0.0134, and 0.0095 below Transformer ceiling; coverage-bucket monotonic so cold-start binding; top-10 features all per-player hero winrate; 66% anonymous accounts is the real binding constraint)
2026-05-18 04:18 propose player-features-prepatch-740 (target val_auc ≥ 0.6277 by extending per-player aggregator with ~127 days of pre-7.40 Turbo data; no time-decay this iteration to isolate the "more data" lever; includes new history-source-breakdown diagnostic)
2026-05-18 14:51 implement player-features-prepatch-740 → experiments/2026-05-18-player-features-prepatch-740/ seeds=1 model=claude-opus-4-6 (val_auc=0.6256, +0.0028 vs player-features-740 but missed target 0.6277 by 0.0022 — HYPOTHESIS NOT CONFIRMED. Required 2 OOM-fix iterations to drop coplay+unique_heroes from aggregator (memory hogs, not in top-20 importance). Coverage-bucket stayed monotonic; HIGH gained most (+0.0043) — cold-start NOT binding, casual-player tail IS. HIGH-bucket val_auc=0.6339 beats Transformer ceiling 0.6322 for active 1/3 of val)
2026-05-18 15:18 wrap-skip-structured reason=cwd-not-in-experiment
2026-05-18 15:18 wrap catches up 2 arcs since prior wrap (2026-05-17 02:00): Hodge ingest + player-features-740 (val_auc=0.6227) + player-features-prepatch-740 (val_auc=0.6256, HIGH-coverage subset 0.6339 beats Transformer); cold-start hypothesis FAILED, casual/anonymous tail is binding constraint; +2 server-stability memories (kernel RCU stall, dict-of-dict OOM)
2026-05-18 22:46 propose transformer-plus-features-740 (combine architecture lever + player-features lever; target val_auc ≥ 0.6372 over Transformer-only 0.6322; offline subset analysis showed n_anon ≤ 1 ceiling at 0.6447 motivating the combination)
2026-05-19 00:45 implement transformer-plus-features-740 → experiments/2026-05-18-transformer-plus-features-740/ seeds=1 model=claude-opus-4-6 (val_auc=0.6452, HYPOTHESIS CONFIRMED — +0.0080 over target 0.6372, +0.0133 over Transformer-only 0.6322, +0.0196 over LightGBM+features 0.6256, +0.0291 over LightGBM-baseline 0.6161. All coverage buckets lifted; HIGH bucket 0.6560 closing in on Hodge's 75-76% in-game-telemetry ceiling using PRE-GAME info only. First Transformer experiment with zero Blackwell retries.)
2026-05-19 01:35 wrap-skip-structured reason=cwd-not-in-experiment
2026-05-19 01:35 wrap catches up arc since prior wrap (2026-05-18 15:18): account-id lookup (3303652) + active-subset analysis (n_anon≤1 ceiling 0.6447) + transformer-plus-features-740 CONFIRMED at val_auc=0.6452 (+0.0133 over Transformer-only, combination is nearly additive); HIGH bucket 0.6560; upstream pytorch/pytorch#184062 assigned to albanD needing C++ stacks (user handling separately)
2026-05-15 19:26 session_end session=a226149e-121b-4423-b6a1-c32f78420aae
2026-05-17 20:29 session_end session=553290ae-af55-4204-be1d-6bc1decdeb02
2026-05-17 20:53 session_end session=8f20fc3b-094f-4467-824d-2504f4ae26bc
2026-05-19 01:30 session_end session=fb42480f-67bb-4309-9f4b-23831e75a7a9
2026-05-19 01:31 session_end session=5900a228-185e-4908-a904-5322d3ce88a6
