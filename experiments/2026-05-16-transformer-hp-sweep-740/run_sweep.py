"""Optuna TPE+ASHA sweep runner for transformer-hp-sweep-740.

Usage:
  python run_sweep.py             # full sweep (60 trials, 14 epochs max)
  python run_sweep.py --smoke     # 1-trial control, 3 epochs, 100k/10k subsample

After the sweep, the top-3 trials (by best_val_loss, ties broken by best_val_auc)
are retrained at full epoch budget with checkpoints saved.

The Optuna study uses SQLite at results/optuna.db so it is resumable. Re-running
the script with the same study_name picks up where it left off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import optuna
import torch
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
RESULTS = EXP_DIR / "results"
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"

sys.path.insert(0, str(EXP_DIR))
from data import load_train_val
from models import build_minimal_transformer, count_params
from objective import control_trial_dict, make_objective
from train_one import evaluate, make_loaders, metrics_block, set_seed, train


def build_study(cfg: dict, storage_url: str, study_name: str,
                seed: int) -> optuna.Study:
    opt_cfg = cfg["optuna"]
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=int(opt_cfg["n_startup_trials"]),
        multivariate=bool(opt_cfg.get("multivariate", True)),
        seed=seed,
    )
    # NOTE: Optuna 4.x SuccessiveHalvingPruner has no `max_resource` arg —
    # the maximum is implicit from how many epochs the trial actually reports.
    # We cap the trial's max epochs in the objective via `max_epochs` instead.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=int(opt_cfg["asha_min_resource"]),
        reduction_factor=int(opt_cfg["asha_reduction_factor"]),
        min_early_stopping_rate=0,
        bootstrap_count=0,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",         # val_loss
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def trial_summary(trial: optuna.trial.FrozenTrial) -> dict:
    return {
        "trial_number": trial.number,
        "state": str(trial.state).split(".")[-1],
        "best_val_loss": float(trial.value) if trial.value is not None else None,
        "best_val_auc": trial.user_attrs.get("best_val_auc"),
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "epochs_run": trial.user_attrs.get("epochs_run"),
        "train_seconds": trial.user_attrs.get("train_seconds"),
        "param_count_total": trial.user_attrs.get("param_count_total"),
        "d_model": trial.user_attrs.get("d_model"),
        "n_heads": trial.user_attrs.get("n_heads"),
        "params": trial.params,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="1-trial control, capped epochs, tiny subsample.")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="Override sweep.n_trials.")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Override sweep.trial_max_epochs.")
    ap.add_argument("--skip-top-k", action="store_true",
                    help="Skip the post-sweep top-k retraining pass.")
    ap.add_argument("--retrain-only", action="store_true",
                    help="Skip the sweep entirely; only run top-k retraining "
                         "on whatever trials the SQLite study already has.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    seed = int(cfg["seed"])
    study_name = str(cfg["study_name"])
    if args.smoke:
        study_name = study_name + "-smoke"
    storage_url = "sqlite:///" + str(RESULTS / "optuna.db")
    if args.smoke:
        storage_url = "sqlite:///" + str(RESULTS / "optuna_smoke.db")

    n_trials = int(args.n_trials if args.n_trials is not None
                   else (cfg["smoke"]["n_trials"] if args.smoke else cfg["sweep"]["n_trials"]))
    max_epochs = int(args.max_epochs if args.max_epochs is not None
                     else (cfg["smoke"]["trial_max_epochs"] if args.smoke
                           else cfg["sweep"]["trial_max_epochs"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  (cuda available: {torch.cuda.is_available()})")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name()}")
    print(f"study: {study_name}")
    print(f"storage: {storage_url}")
    print(f"n_trials: {n_trials}  max_epochs: {max_epochs}  smoke: {args.smoke}")

    print("loading data...")
    t0 = time.time()
    smoke_cfg = cfg["smoke"]
    train_ds, val_ds, meta = load_train_val(
        seed=seed,
        n_target=int(cfg["train_subset_size"]),
        splits=splits,
        smoke=args.smoke,
        smoke_n_train=int(smoke_cfg["n_train"]),
        smoke_n_val=int(smoke_cfg["n_val"]),
    )
    print(f"data loaded in {time.time()-t0:.1f}s — train={len(train_ds):,} val={len(val_ds):,}")
    print(f"dates: train={meta['train_date_min']}..{meta['train_date_max']} "
          f"val={meta['val_date_min']}..{meta['val_date_max']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    history_dir = RESULTS / ("trial_histories_smoke" if args.smoke else "trial_histories")

    study = build_study(cfg, storage_url, study_name, seed)

    # Force-pin the control trial as the very first trial of the (fresh) study.
    if len(study.trials) == 0:
        ctrl = control_trial_dict(cfg["control_trial"], cfg["search_space"])
        study.enqueue_trial(ctrl)
        print(f"enqueued control trial: {ctrl}")

    objective = make_objective(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg,
        device=device,
        base_rate_val=meta["radiant_base_rate_val"],
        history_dir=history_dir,
        max_epochs=max_epochs,
    )

    sweep_t0 = time.time()
    sweep_seconds = 0.0
    cuda_dead = False
    if not args.retrain_only:
        # Catch (Exception,) — torch 2.12's AcceleratorError isn't a RuntimeError
        # in all paths, and a single failed trial should mark the trial FAIL and
        # move on. A CUDA device-side assert that POISONS the whole CUDA context
        # (so subsequent ops fail too) won't be recoverable in this process —
        # we let the wrapper script restart, and Optuna's SQLite study resumes.
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                catch=(Exception,),
                gc_after_trial=True,
                show_progress_bar=False,
            )
        except BaseException as e:
            print(f"optimize() raised uncaught: {type(e).__name__}: {e}")
            cuda_dead = True
        sweep_seconds = time.time() - sweep_t0
        print(f"sweep done in {sweep_seconds:.1f}s")
    else:
        print("--retrain-only: skipping sweep, jumping to top-k retraining")

    # Quick CUDA health probe — a device-side assert from a prior trial
    # poisons the CUDA context such that even torch.manual_seed() crashes.
    if device.type == "cuda" and not cuda_dead:
        try:
            torch.cuda.synchronize()
            _probe = torch.zeros(4, device=device).sum().item()
        except BaseException as e:
            print(f"CUDA probe failed post-sweep: {type(e).__name__}: {e}")
            cuda_dead = True
    if cuda_dead:
        print("CUDA context is poisoned — exiting non-zero so the wrapper restarts. "
              "Optuna SQLite study is resumable; next launch picks up.")
        # Still write the metrics aggregate so partial progress is visible.
        partial = {
            "study_name": study_name,
            "smoke": args.smoke,
            "sweep_seconds": sweep_seconds,
            "n_trials_target": n_trials,
            "cuda_dead": True,
            "trials": [trial_summary(t) for t in study.trials],
            **{k: v for k, v in meta.items()},
        }
        (EXP_DIR / "metrics.json").write_text(json.dumps(partial, indent=2))
        return 2  # signal to wrapper: restart needed

    # Aggregate per-trial summary.
    trials_summary = [trial_summary(t) for t in study.trials]

    # Best trial.
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"completed trials: {len(completed)} / {len(study.trials)}")
    if completed:
        best = study.best_trial
        print(f"BEST trial #{best.number}: val_loss={best.value:.4f}  "
              f"val_auc={best.user_attrs.get('best_val_auc'):.4f}  params={best.params}")

    out: dict = {
        "study_name": study_name,
        "smoke": args.smoke,
        "sweep_seconds": sweep_seconds,
        "n_trials_target": n_trials,
        "n_trials_completed": len(completed),
        "n_trials_pruned": sum(1 for t in study.trials
                               if t.state == optuna.trial.TrialState.PRUNED),
        "n_trials_failed": sum(1 for t in study.trials
                               if t.state == optuna.trial.TrialState.FAIL),
        "control_trial_number": 0,
        "trials": trials_summary,
        **{k: v for k, v in meta.items()},
    }

    # Top-k retraining (skipped on smoke).
    if (not args.smoke and not args.skip_top_k and completed
            and (args.retrain_only or n_trials >= 5)):
        top_k = int(cfg["sweep"]["top_k_retrain"])
        retrain_max_epochs = int(cfg["sweep"]["top_k_retrain_max_epochs"])
        retrain_patience = int(cfg["sweep"]["top_k_retrain_patience"])
        ranked = sorted(completed, key=lambda t: (t.value,
                                                  -(t.user_attrs.get("best_val_auc") or 0.0)))
        top_results = []
        for rank, trial in enumerate(ranked[:top_k], start=1):
            print(f"\n=== retraining top-{rank} (trial #{trial.number}) ===")
            # Reconstruct hp dict from the trial's params.
            pairs = [tuple(p) for p in cfg["search_space"]["d_model_n_heads_pairs"]]
            pair_idx = int(trial.params["d_model_n_heads_idx"])
            d_model, n_heads = pairs[pair_idx]
            hp = {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": int(trial.params["n_layers"]),
                "ff_mult": int(trial.params["ff_mult"]),
                "embed_dim": int(trial.params["embed_dim"]),
                "lr": float(trial.params["lr"]),
                "weight_decay": float(trial.params["weight_decay"]),
                "dropout": float(trial.params["dropout"]),
                "batch_size": int(trial.params["batch_size"]),
            }
            set_seed(seed)
            model = build_minimal_transformer(hp, int(cfg["hero"]["vocab_size"])).to(device)
            pcounts = count_params(model)
            result = train(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                hp=hp,
                max_epochs=retrain_max_epochs,
                device=device,
                base_rate_val=meta["radiant_base_rate_val"],
                optuna_trial=None,
                patience=retrain_patience,
                mixed_precision=True,
            )
            ckpt_path = RESULTS / f"top_{rank}.pt"
            torch.save({"state_dict": model.state_dict(), "hp": hp,
                        "trial_number": trial.number}, ckpt_path)
            top_results.append({
                "rank": rank,
                "source_trial_number": trial.number,
                "hp": hp,
                "param_counts": pcounts,
                "best_val_loss": result.best_val_loss,
                "best_val_auc": result.best_val_auc,
                "val_metrics_at_best": result.val_metrics_at_best,
                "train_metrics_at_best": result.train_metrics_at_best,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "train_seconds": result.train_seconds,
                "checkpoint": str(ckpt_path.relative_to(EXP_DIR)),
            })
            print(f"  top-{rank}: val_auc={result.best_val_auc:.4f}  "
                  f"val_loss={result.best_val_loss:.4f}")
        out["top_k_retrained"] = top_results
        (RESULTS / "top_k_metrics.json").write_text(json.dumps(top_results, indent=2))

    metrics_path = EXP_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
