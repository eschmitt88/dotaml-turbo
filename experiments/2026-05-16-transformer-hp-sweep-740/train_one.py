"""Standalone trainer for a single (HPs, n_epochs) tuple.

Used both by objective.py (in-trial training, with optional Optuna trial
for ASHA pruning) and by run_sweep.py's post-sweep top-k retraining.

Forces math SDP backend at module load to avoid torch 2.11 + Blackwell
sm_120 flash/mem-efficient attention crashes on small d_model.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(train_ds, val_ds, batch_size: int):
    """num_workers=0 is the workaround for torch 2.11 + Blackwell DataLoader
    worker segfaults — see plateau-architectures-740 log.md.
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(batch_size * 2, 16384), shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    n = 0
    total_loss = 0.0
    ys, ps = [], []
    for hero_ids, y in loader:
        hero_ids = hero_ids.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            logits = model(hero_ids)
            loss = bce(logits.float(), y.float())
        total_loss += loss.item()
        n += y.size(0)
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.cpu().numpy())
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    return {"loss": total_loss / max(n, 1), "y": y, "p": p}


def metrics_block(y, p, base_rate: float | None = None):
    auc = float(roc_auc_score(y, p))
    pred = (p >= 0.5).astype(int)
    acc = float(accuracy_score(y, pred))
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))
    brier = float(brier_score_loss(y, p))
    out = {"auc": auc, "acc": acc, "log_loss": ll, "brier": brier}
    if base_rate is not None:
        out["majority_class_acc"] = float(max(base_rate, 1 - base_rate))
    return out


@dataclass
class TrainResult:
    history: list
    best_val_loss: float
    best_val_auc: float
    best_epoch: int
    best_state: dict | None
    epochs_run: int
    train_seconds: float
    pruned: bool
    val_metrics_at_best: dict
    train_metrics_at_best: dict


def train(model, train_ds, val_ds, hp: dict, max_epochs: int, device: torch.device,
          base_rate_val: float | None = None, optuna_trial=None,
          patience: int | None = None, mixed_precision: bool = True) -> TrainResult:
    """Train one model. If optuna_trial is provided, report+prune per epoch."""
    import optuna  # local import — only needed if pruning

    batch_size = int(hp["batch_size"])
    lr = float(hp["lr"])
    weight_decay = float(hp["weight_decay"])

    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    autocast_dtype = None
    if mixed_precision and device.type == "cuda":
        autocast_dtype = torch.bfloat16

    history = []
    best_val_loss = math.inf
    best_val_auc = -math.inf
    best_state = None
    best_epoch = -1
    best_val_metrics: dict = {}
    epochs_since_improve = 0
    pruned = False

    train_t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        n_seen = 0
        loss_sum = 0.0
        ep_t0 = time.time()
        for hero_ids, y in train_loader:
            hero_ids = hero_ids.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                logits = model(hero_ids)
                loss = bce(logits.float(), y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_seen += y.size(0)
            loss_sum += loss.item() * y.size(0)
        train_loss = loss_sum / max(n_seen, 1)

        val_eval = evaluate(model, val_loader, device, autocast_dtype)
        val_loss = val_eval["loss"]
        val_metrics = metrics_block(val_eval["y"], val_eval["p"], base_rate=base_rate_val)
        ep_dt = time.time() - ep_t0
        print(f"  epoch {epoch+1}/{max_epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_auc={val_metrics['auc']:.4f}  "
              f"val_acc={val_metrics['acc']:.4f}  ({ep_dt:.1f}s)")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_metrics["auc"],
            "val_acc": val_metrics["acc"],
            "wall_seconds": ep_dt,
        })

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # Optuna ASHA pruning hook.
        if optuna_trial is not None:
            optuna_trial.report(val_loss, step=epoch)
            if optuna_trial.should_prune():
                pruned = True
                break

        # Optional patience early-stop (used for top-k retraining only).
        if patience is not None and epochs_since_improve >= patience:
            print(f"  early stop at epoch {epoch+1} (best epoch {best_epoch})")
            break

    train_seconds = time.time() - train_t0

    # Final eval at the best checkpoint (if not pruned and we have a best state).
    train_metrics: dict = {}
    if best_state is not None and not pruned:
        model.load_state_dict(best_state)
        # Train-set AUC for overfit anchor on a small slice (avoid full re-eval cost).
        try:
            tr_eval = evaluate(model, train_loader, device, autocast_dtype)
            train_metrics = metrics_block(tr_eval["y"], tr_eval["p"])
        except Exception as e:  # noqa: BLE001
            print(f"  train-set eval skipped: {e}")
            train_metrics = {}

    return TrainResult(
        history=history,
        best_val_loss=best_val_loss,
        best_val_auc=best_val_auc,
        best_epoch=best_epoch,
        best_state=best_state,
        epochs_run=history[-1]["epoch"] if history else 0,
        train_seconds=train_seconds,
        pruned=pruned,
        val_metrics_at_best=best_val_metrics,
        train_metrics_at_best=train_metrics,
    )


__all__ = ["train", "TrainResult", "evaluate", "metrics_block", "set_seed", "make_loaders"]
