"""Optuna objective for transformer-hp-sweep-740.

Returns final val_loss (lower is better). The Optuna study is configured
with `direction="minimize"` for that reason; val_auc is recorded as a
user attribute on each trial for ranking.

Categorical sampling for (d_model, n_heads) uses a single tuple
categorical so divisibility (d_model % n_heads == 0) is enforced by
construction.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import optuna
import torch

from data import DraftDataset
from models import build_minimal_transformer, count_params
from train_one import set_seed, train


def sample_hp(trial: optuna.Trial, search_space: dict) -> dict:
    pairs = [tuple(p) for p in search_space["d_model_n_heads_pairs"]]
    pair_idx = trial.suggest_categorical("d_model_n_heads_idx", list(range(len(pairs))))
    d_model, n_heads = pairs[pair_idx]
    n_layers = trial.suggest_categorical("n_layers", search_space["n_layers"])
    ff_mult = trial.suggest_categorical("ff_mult", search_space["ff_mult"])
    embed_dim = trial.suggest_categorical("embed_dim", search_space["embed_dim"])
    lr = trial.suggest_float("lr", search_space["lr_low"], search_space["lr_high"], log=True)
    weight_decay = trial.suggest_float(
        "weight_decay",
        search_space["weight_decay_low"],
        search_space["weight_decay_high"],
        log=True,
    )
    dropout = trial.suggest_float(
        "dropout", search_space["dropout_low"], search_space["dropout_high"]
    )
    batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])

    hp = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "ff_mult": ff_mult,
        "embed_dim": embed_dim,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "batch_size": batch_size,
    }
    # Stash the resolved (d_model, n_heads) on the trial for easier inspection.
    trial.set_user_attr("d_model", d_model)
    trial.set_user_attr("n_heads", n_heads)
    return hp


def control_trial_dict(control_cfg: dict, search_space: dict) -> dict:
    """Build the {param_name: value} dict for study.enqueue_trial(...).

    Must use the same parameter names as sample_hp() and the same
    categorical ordering, since Optuna stores the categorical index.
    """
    pairs = [tuple(p) for p in search_space["d_model_n_heads_pairs"]]
    target_pair = tuple(control_cfg["d_model_n_heads"])
    if target_pair not in pairs:
        raise ValueError(f"control pair {target_pair} not in search space pairs")
    pair_idx = pairs.index(target_pair)
    return {
        "d_model_n_heads_idx": pair_idx,
        "n_layers": int(control_cfg["n_layers"]),
        "ff_mult": int(control_cfg["ff_mult"]),
        "embed_dim": int(control_cfg["embed_dim"]),
        "lr": float(control_cfg["lr"]),
        "weight_decay": max(float(control_cfg["weight_decay"]),
                            float(search_space["weight_decay_low"])),
        # If wd is 0, bump to the search-space lower bound (log-uniform can't take 0).
        "dropout": float(control_cfg["dropout"]),
        "batch_size": int(control_cfg["batch_size"]),
    }


def make_objective(
    train_ds: DraftDataset,
    val_ds: DraftDataset,
    cfg: dict,
    device: torch.device,
    base_rate_val: float,
    history_dir: Path,
    max_epochs: int,
):
    search_space = cfg["search_space"]
    vocab_size = int(cfg["hero"]["vocab_size"])
    seed = int(cfg["seed"])

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hp(trial, search_space)
        # Reset seed inside each trial so HP comparisons aren't confounded
        # by Adam state / shuffle order accumulating across trials.
        set_seed(seed)

        try:
            model = build_minimal_transformer(hp, vocab_size).to(device)
        except ValueError as e:
            # Shouldn't happen given our (d_model, n_heads) tuple categorical,
            # but if it does, prune.
            print(f"  build failed: {e}")
            raise optuna.TrialPruned() from e

        pcounts = count_params(model)
        trial.set_user_attr("param_count_total", pcounts["total"])
        trial.set_user_attr("param_count_non_embedding", pcounts["non_embedding"])
        print(f"  trial {trial.number}: hp={hp}  params={pcounts['total']:,}")

        result = train(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            hp=hp,
            max_epochs=max_epochs,
            device=device,
            base_rate_val=base_rate_val,
            optuna_trial=trial,
            patience=None,             # ASHA handles pruning, not patience
            mixed_precision=True,
        )

        # Persist per-trial history regardless of pruning.
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"trial_{trial.number}_history.json"
        history_path.write_text(json.dumps({
            "trial_number": trial.number,
            "hp": hp,
            "param_counts": pcounts,
            "history": result.history,
            "pruned": result.pruned,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "best_val_auc": result.best_val_auc,
            "train_seconds": result.train_seconds,
        }, indent=2))

        trial.set_user_attr("best_val_auc", result.best_val_auc)
        trial.set_user_attr("best_epoch", result.best_epoch)
        trial.set_user_attr("epochs_run", result.epochs_run)
        trial.set_user_attr("train_seconds", result.train_seconds)

        # Explicit per-trial cleanup before the function returns. Drops
        # in-process crash rate when this objective is invoked multiple times
        # in the same Python process (see
        # docs/decisions/0001-per-trial-subprocess-isolation.md — the
        # production sweep wraps each trial in its own subprocess regardless,
        # this is belt-and-suspenders for diagnostic or local-debug runs).
        try:
            best_loss = result.best_val_loss
            pruned = result.pruned
        finally:
            del model, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if pruned:
            raise optuna.TrialPruned()

        return best_loss  # study.direction = minimize

    return objective


__all__ = ["make_objective", "sample_hp", "control_trial_dict"]
