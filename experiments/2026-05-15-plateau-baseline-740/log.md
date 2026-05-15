# Log — plateau-baseline-740

Chronological one-line entries. Per-experiment log, NOT `_meta/log.md`.

- 2026-05-15: scaffold created; config.yaml mirrors DotaML v3 recipe.
- 2026-05-15 16:42: pull_raw.py — 84 train+val days, 1736 parquet files, 86.5 GB; downloaded 71.6 GB in 1036 s (70 MB/s sustained, 0 errors). Test window excluded by HCE guard. Background job from main agent.
- 2026-05-15 17:06: build_features.py — read 16,923,487 raw rows → kept 15,437,578 → filtered 1,485,909 (8.78 % forfeit/empty-inv) → train 13,018,393, val 2,419,185, dup_match_id=0. Wall 881 s. Wrote processed/{train,val}.parquet + processed/build_stats.json.
- 2026-05-15 17:20: train.py — LightGBM 500 rounds on 5M stratified subset, 301-dim sparse. Wall 123 s. val_auc=0.6161, val_acc=0.5866, val_brier=0.2386, train_val_auc_gap=0.0126, val majority-class acc=0.5326. Wrote metrics.json + results/{calibration,roc,learning_curve}.png + results/lightgbm.txt (1.87 MB).
- 2026-05-15 17:24: README finalised with Result/Interpretation/Diagnostics. status: running → done. Conclusion: partial confirmation — val_auc 0.6161 misses the proposal's strict band [0.625, 0.645] but lands within 0.003 of DotaML v3's same-recipe test_auc=0.6189; the 0.635 target was the v5 Transformer's ceiling, not v3 LightGBM's.
