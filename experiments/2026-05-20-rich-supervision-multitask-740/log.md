# rich-supervision-multitask-740 — log

Chronological notes; main agent appends to this during the full run.

## Scaffold + smoke (2026-05-20)

- Scaffolded folder from cleanup-740 (`data.py`, `models.py`, `train.py`,
  `config.yaml` adapted) + embedding-prelim's `build_account_sidecar.py`
  pattern for `build_rich_cols.py`.
- Implemented `build_rich_cols.py`, `build_item_vocab.py`, multi-head
  `models.py:MultiHeadTransformer`, multi-task `data.py:load_train_val`,
  multi-task `train.py`.
- Smoke results below (separate entry per step).

## Debug pass — multitask_all 3-attempt failure (2026-05-20)

After full pipeline run, `multitask_all` failed all 3 attempts (sanity passed
val_auc=0.6473). MLE-DEBUGGER subagent diagnosed + fixed:

- **Data gut-check** (`diagnose_targets.py`): all 71 numeric target columns
  in train+val sidecars have 0 nulls, all within physical bounds. p{0..9}_items
  zero-length rate ~0.01% per slot (1k-1.6k of 13M); OOV item rate ~0% on
  20K sample. Data quality is clean — no halt condition.

- **Issue 1 (Arrow length error)**: transient PyArrow `pf.read(columns=cols)`
  anomaly — returned p9_items length 13018391 vs metadata 13018393 on
  attempt 1, but the diagnostic re-read produced the correct 13018393 across
  all 10 item columns. Same family as the cleanup-740 PyArrow buffer-fill
  anomaly. Fix in `data.py:_read_sidecar_tbl`: validate per-column lengths
  against `metadata.num_rows`, retry once, then fall back to per-row-group
  read + `pa.concat_tables()`.

- **Issue 2 (NaN at val)**: attempt 2 ran 2 epochs printing val_auc, crashed
  at epoch-3 eval inside `roc_auc_score`. Targets are clean → NaN was in win
  logits (bf16 autocast spike, plausibly amplified by duration-dominated joint
  loss). Fix in `train.py`: (a) `_masked_smooth_l1_sum` / `_masked_smooth_l1_mean`
  for the aux head — masks non-finite cells in pred and target, divides by
  valid-element count; (b) `torch.nan_to_num(win_logits, posinf=50, neginf=-50)`
  before BCE-sum and sigmoid in `_eval_multitask`; (c) per-dim aux SSE
  accumulator also masks non-finite cells.

- **Issue 3 (alpha_dur)**: dropped 0.5 → 0.15 in `config.yaml:multitask_loss`.
  Rationale per failed-run epoch-1 diagnostics: d×0.5=1.0347 dominated joint
  loss vs w=0.6841; val_auc trailed cleanup-740 trajectory by ~0.003 at
  epoch 2.

- Smoke: TBD by main agent's smoke step before relaunching.
