# log — transformer-plus-features-extended-740

## 2026-05-19

- Scaffolded from `experiments/2026-05-18-transformer-plus-features-740/`.
  - Copied `models.py`, `data.py`, `train.py`, `notes.qmd` verbatim.
  - Authored `config.yaml`: identical to parent except `optim.max_epochs`
    14 -> 30 and explicit `optim.early_stopping_metric: val_log_loss`.
  - Edited `train.py`: restricted `--ablation` choices to just
    `transformer_plus_features`; added `delta_vs_transformer_plus_features_740`
    field to metrics dict (parent_anchor=0.6452, target=0.6462). No
    training-loop changes — parent's `train_model` already supports
    `patience` early stopping on val_loss; parent set patience=5 but
    never triggered because max_epochs was the binding cap.
  - Authored `run_all.sh`: single ablation, MAX_RETRIES=3.
  - Authored `README.md` placeholder with frontmatter + Hypothesis + Setup.
- Smoke test PASS (subagent): data loads with sanitization (23 cells),
  HCE date guard live, forward pass clean, metrics.json written,
  1-epoch early-stopping path exits cleanly.
- 02:31 full run launched in background (PID 844773) via
  `nohup bash run_all.sh > /tmp/dotaml_tpfe.log 2>&1 &`.
- 02:57 full run completed (1 attempt, no retries). 25.1-min wall.
  val_auc=0.6477 @ best_epoch=22; early-stopped at epoch 27
  (patience=5 fired). **HYPOTHESIS CONFIRMED**: +0.0025 over parent
  0.6452, +0.0015 over target 0.6462.
- 02:57 README Result + Diagnostics + Follow-up written;
  metrics.json rollup written; proposal moved to _done/ with
  `status: implemented`; `_meta/log.md` + `_meta/index.md` updated;
  `concepts/draft-prediction-plateau.md` extended with seventh
  refinement (uniform-lift / longer-training-now-plateaued).
