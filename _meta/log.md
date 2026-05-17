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
2026-05-15 19:26 session_end session=a226149e-121b-4423-b6a1-c32f78420aae
