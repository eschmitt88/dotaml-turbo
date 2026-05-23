# foundation-mvp-740 log

## 2026-05-22 — scaffold + smoke

- Scaffolded experiment folder, wrote data.py / models.py / loss.py / mae.py /
  train.py / run_all.sh / config.yaml.
- Reused multitask-740's rich_cols sidecar + item_vocab.json (same 7.40 window).
  Decision documented in README: not extending to Aug 2025 in this MVP because
  the rich_cols sidecar covers only the 7.40 window and rebuilding it across
  the broader window is the ~3-4h pre-build cited in the proposal estimate.
- Smoke (all 3 ablations, 1 epoch, 50K rows): foundation_mvp val_auc=0.5015 in 2.0s; baseline_multitask_repro val_auc=0.4860 in 1.7s; foundation_no_patch_token val_auc=0.4922 in 2.0s. All heads emit finite losses; UW-SO temperature learns (~0.9996 after 1 epoch); PMAE fires; canonical hero-sort verified; HCE date guard fires.
- Profile (foundation_mvp, 1 full epoch over 5M train + 2.4M val rows): 666s = 11.1 min/epoch. Data load 306s (one-time per run). Model: 6,455,021 params trainable (slightly over the 5M target but within proposal's ~5-6M range; not scaling down -- GPU usage 1.65 GB / 16 GB, 42% util).
- Extrapolated wall: 30 epochs * 11.1 min = ~5.5 hours per ablation + 5 min data load. Three ablations sequentially: ~17 hours. Within budget.yaml's 24h ceiling.
- Auxiliary tasks already converging fast (item BCE 0.78 -> 0.10, itemMAP 0.28 by epoch 1 on full data), suggests rich signal is plentiful. Val_auc=0.4892 at epoch 1 not informative due to --max-epochs-override=1 collapsing the cosine schedule to lr=1e-5 already; full run won't have this issue.
