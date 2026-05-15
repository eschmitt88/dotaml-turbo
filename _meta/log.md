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
