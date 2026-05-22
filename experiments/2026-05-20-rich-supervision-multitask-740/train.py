"""Train MultiHeadTransformer for one ablation in rich-supervision-multitask-740.

Single-ablation entry point. Selects between:
  --ablation win_only_sanity : only win head active (alpha_d/i/a = 0). Uses
                                the cleanup-740 DataLoader contract
                                (DraftPlusFeaturesDataset, 3-tuple). Should
                                reproduce cleanup-740's 0.6477054 to ~1e-4
                                modulo any clean-vs-sidecar row drop.
  --ablation multitask_all   : all four heads active. Joint loss with config-
                                supplied alpha weights. Early-stop on val
                                WIN log-loss only (not total).

Per-trial subprocess isolation (run_all.sh + MAX_RETRIES=3) is the workaround
for the Blackwell torch DataLoader segfault. Math SDP backend forced at module
load for sm_120.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader  # noqa: E402

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

from data import load_train_val  # noqa: E402
from models import build_model, count_params  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(train_ds, val_ds, batch_size: int):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(batch_size * 2, 16384), shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False,
    )
    return train_loader, val_loader


@dataclass
class TrainResult:
    history: list
    best_val_win_loss: float
    best_val_auc: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    val_metrics_at_best: dict
    val_win_predictions: np.ndarray
    val_win_labels: np.ndarray
    val_dur_predictions: np.ndarray | None
    val_dur_labels: np.ndarray | None


def _eval_win_only(model, loader, device, autocast_dtype, use_features: bool):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    n = 0
    total_loss = 0.0
    ys, ps = [], []
    with torch.no_grad():
        for hero_ids, player_feats, y in loader:
            hero_ids = hero_ids.to(device, non_blocking=True)
            player_feats = player_feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                out = model(hero_ids, player_feats if use_features else None,
                            multitask=False)
                loss = bce(out["win"].float(), y.float())
            total_loss += loss.item()
            n += y.size(0)
            ps.append(torch.sigmoid(out["win"].float()).cpu().numpy())
            ys.append(y.cpu().numpy())
    return (total_loss / max(n, 1),
            np.concatenate(ys), np.concatenate(ps))


def _masked_smooth_l1_sum(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """SmoothL1 with per-element NaN/Inf masking.

    Returns (sum_loss, n_valid). The standard NaN-defense pattern: zero out
    invalid elements in BOTH pred and target before computing the loss, so
    the per-element contribution is zero and the sum is over valid elements
    only. Caller is expected to divide by n_valid (not numel) to keep
    gradient magnitude consistent.
    """
    mask = torch.isfinite(pred) & torch.isfinite(target)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return pred.sum() * 0.0, 0
    p = torch.where(mask, pred, torch.zeros_like(pred))
    t = torch.where(mask, target, torch.zeros_like(target))
    diff = p - t
    abs_diff = diff.abs()
    # SmoothL1 with beta=1.0 (torch default).
    elem_loss = torch.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    # Zero out invalid positions in the loss itself (defense-in-depth).
    elem_loss = torch.where(mask, elem_loss, torch.zeros_like(elem_loss))
    return elem_loss.sum(), n_valid


def _masked_smooth_l1_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-valid-element mean SmoothL1. Returns scalar."""
    s, n = _masked_smooth_l1_sum(pred, target)
    if n == 0:
        # Detach-free zero so grad-flow is preserved (zero grad, no NaN).
        return pred.sum() * 0.0
    return s / float(n)


def _eval_multitask(model, loader, device, autocast_dtype, use_features: bool,
                     alpha: dict, n_dur_buckets: int):
    """Returns dict with win/dur/item/aux preds + labels + per-comp val losses."""
    model.eval()
    bce_sum = nn.BCEWithLogitsLoss(reduction="sum")
    bce_sum_item = nn.BCEWithLogitsLoss(reduction="sum")
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    n = 0
    tot_w = tot_d = tot_i = tot_a = 0.0
    tot_a_n_valid = 0
    ys_w, ps_w = [], []
    ys_d, ps_d = [], []
    # Item/aux metrics are aggregated streaming (per-slot mAP needs per-class
    # accumulation; we compute it as a pooled-over-slots approximation here,
    # since item-vocab is shared across slots and the model has no slot-id).
    # Streaming average precision is expensive; instead we collect a SUBSAMPLE
    # of item preds/labels and compute mAP at the end.
    item_subsample_targets: list[np.ndarray] = []
    item_subsample_logits: list[np.ndarray] = []
    aux_sse = None  # [n_aux] running squared-error sum
    aux_n_per_dim = None  # [n_aux] count of valid (finite) cells per dim
    aux_n = 0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in loader:
            hero_ids, pf, y_win, y_dur, y_item, y_aux = batch
            hero_ids = hero_ids.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            y_win = y_win.to(device, non_blocking=True)
            y_dur = y_dur.to(device, non_blocking=True)
            y_item = y_item.to(device, non_blocking=True)
            y_aux = y_aux.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                out = model(hero_ids, pf if use_features else None, multitask=True)
                # Win: defensively replace any non-finite logits with 0
                # (large bf16 spikes propagate to roc_auc_score otherwise).
                win_logits = out["win"].float()
                win_logits = torch.nan_to_num(win_logits, nan=0.0,
                                                posinf=50.0, neginf=-50.0)
                l_w = bce_sum(win_logits, y_win.float())
                l_d = ce_sum(out["dur"].float(), y_dur)
                # Item: [B, 10, V] -- average loss per element across slots.
                l_i = bce_sum_item(out["item"].float(), y_item.float())
                # Aux: NaN-masked SmoothL1 sum (target nulls or pred-NaN both safe).
                l_a, l_a_n = _masked_smooth_l1_sum(out["aux"].float(),
                                                    y_aux.float())
            tot_w += l_w.item()
            tot_d += l_d.item()
            tot_i += l_i.item()
            tot_a += l_a.item()
            tot_a_n_valid += l_a_n
            bsize = y_win.size(0)
            n += bsize
            ps_w.append(torch.sigmoid(win_logits).cpu().numpy())
            ys_w.append(y_win.cpu().numpy())
            ps_d.append(out["dur"].float().softmax(dim=-1).cpu().numpy())
            ys_d.append(y_dur.cpu().numpy())
            # Subsample for mAP: keep at most ~50k slot-rows total across val.
            keep_b = min(bsize, max(1, int(50000 / max(len(item_subsample_targets) * bsize, 1) + 1)))
            if len(item_subsample_targets) * bsize * 10 < 500_000:
                idx = rng.choice(bsize, size=min(keep_b, bsize), replace=False)
                t_np = y_item[idx].cpu().numpy().reshape(-1, y_item.shape[-1])
                l_np = out["item"][idx].float().cpu().numpy().reshape(-1, y_item.shape[-1])
                item_subsample_targets.append(t_np)
                item_subsample_logits.append(l_np)
            # Aux SSE: per-aux dim sum of squared errors (denorm = standardized;
            # the metric is val MSE on the standardized scale). Mask non-finite
            # cells before squaring so NaN/Inf in pred don't poison the SSE.
            aux_p = out["aux"].float().cpu().numpy()
            aux_t = y_aux.float().cpu().numpy()
            mask_a = np.isfinite(aux_p) & np.isfinite(aux_t)
            err = np.where(mask_a, aux_p - aux_t, 0.0)
            err2 = err ** 2
            err2_flat = err2.reshape(-1, err2.shape[-1])
            mask_flat = mask_a.reshape(-1, mask_a.shape[-1])
            if aux_sse is None:
                aux_sse = np.zeros(err2.shape[-1], dtype=np.float64)
                aux_n_per_dim = np.zeros(err2.shape[-1], dtype=np.int64)
            aux_sse += err2_flat.sum(axis=0)
            aux_n_per_dim += mask_flat.sum(axis=0)
            aux_n += mask_flat.shape[0]
    item_v = int(getattr(model, "item_vocab_size", 1)) or 1
    aux_v = int(getattr(model, "n_aux", 1)) or 1
    val_losses = {
        "win_log_loss_per_row": tot_w / max(n, 1),
        "dur_ce_per_row": tot_d / max(n, 1),
        "item_bce_per_slot_per_class": tot_i / max(n * 10 * item_v, 1),
        # Divide by VALID-element count (NaN-masked) so aux loss isn't
        # artificially depressed by invalid cells.
        "aux_smoothl1_per_slot_per_dim": tot_a / max(tot_a_n_valid, 1),
    }
    # Joint val loss with alpha weighting (matches train objective).
    val_losses["weighted_total"] = (alpha["alpha_win"] * val_losses["win_log_loss_per_row"]
                                     + alpha["alpha_dur"] * val_losses["dur_ce_per_row"]
                                     + alpha["alpha_item"] * val_losses["item_bce_per_slot_per_class"]
                                     + alpha["alpha_aux"] * val_losses["aux_smoothl1_per_slot_per_dim"])
    if aux_sse is not None and aux_n_per_dim is not None:
        aux_mse = (aux_sse / np.maximum(aux_n_per_dim, 1)).tolist()
    else:
        aux_mse = []
    return {
        "win_y": np.concatenate(ys_w), "win_p": np.concatenate(ps_w),
        "dur_y": np.concatenate(ys_d), "dur_p": np.concatenate(ps_d),
        "item_subsample_targets": (np.concatenate(item_subsample_targets, axis=0)
                                    if item_subsample_targets else np.zeros((0, 0))),
        "item_subsample_logits": (np.concatenate(item_subsample_logits, axis=0)
                                   if item_subsample_logits else np.zeros((0, 0))),
        "aux_mse_per_dim": aux_mse,
        "val_losses": val_losses,
    }


def metrics_block(y, p, base_rate: float | None = None) -> dict:
    auc = float(roc_auc_score(y, p))
    pred = (p >= 0.5).astype(int)
    acc = float(accuracy_score(y, pred))
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))
    brier = float(brier_score_loss(y, p))
    out = {"auc": auc, "acc": acc, "log_loss": ll, "brier": brier}
    if base_rate is not None:
        out["majority_class_acc"] = float(max(base_rate, 1 - base_rate))
    return out


def item_map_at_k(targets: np.ndarray, logits: np.ndarray, k: int = 10) -> dict:
    """Pooled mAP@k over the slot-row dimension.

    targets: [M, V] multi-hot
    logits:  [M, V]
    For each row, take the top-k predicted classes and compute precision@k
    and recall@k. mAP is the mean over rows of the average-precision across
    that row's top-k labels.
    """
    if targets.size == 0:
        return {"map_at_k": None, "mean_precision_at_k": None,
                "mean_recall_at_k": None, "n_rows": 0, "k": k}
    M, V = targets.shape
    if V == 0:
        return {"map_at_k": None, "mean_precision_at_k": None,
                "mean_recall_at_k": None, "n_rows": M, "k": k}
    k = min(k, V)
    top_idx = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    # Sort top-k by logit descending.
    sort_order = np.argsort(-np.take_along_axis(logits, top_idx, axis=1), axis=1)
    top_idx = np.take_along_axis(top_idx, sort_order, axis=1)
    hits = np.take_along_axis(targets, top_idx, axis=1).astype(np.float32)
    # Per-row AP@k.
    cum = np.cumsum(hits, axis=1)
    ranks = np.arange(1, k + 1, dtype=np.float32)
    prec_at_i = cum / ranks
    n_pos_per_row = targets.sum(axis=1)
    denom = np.where(n_pos_per_row > 0, np.minimum(n_pos_per_row, k), 1.0)
    ap = (prec_at_i * hits).sum(axis=1) / denom
    map_at_k = float(ap.mean())
    p_at_k = float(hits.sum(axis=1).mean() / k)
    rec_at_k = float((hits.sum(axis=1) / np.maximum(n_pos_per_row, 1)).mean())
    return {"map_at_k": map_at_k, "mean_precision_at_k": p_at_k,
            "mean_recall_at_k": rec_at_k, "n_rows": int(M), "k": int(k)}


def train_model_win_only(model, train_ds, val_ds, hp: dict, max_epochs: int,
                          device, base_rate_val: float | None,
                          use_features: bool, mixed_precision: bool,
                          patience: int | None) -> TrainResult:
    bs = int(hp["batch_size"]); lr = float(hp["lr"]); wd = float(hp["weight_decay"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    autocast_dtype = torch.bfloat16 if (mixed_precision and device.type == "cuda") else None

    history = []
    best_val_loss = math.inf
    best_val_auc = -math.inf
    best_state = None
    best_epoch = -1
    best_eval_metrics: dict = {}
    best_y, best_p = np.array([]), np.array([])
    epochs_since_improve = 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        n_seen = 0; loss_sum = 0.0; ep_t0 = time.time()
        for hero_ids, pf, y in train_loader:
            hero_ids = hero_ids.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                out = model(hero_ids, pf if use_features else None, multitask=False)
                loss = bce(out["win"].float(), y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_seen += y.size(0); loss_sum += loss.item() * y.size(0)
        train_loss = loss_sum / max(n_seen, 1)
        val_loss, y_v, p_v = _eval_win_only(model, val_loader, device,
                                              autocast_dtype, use_features)
        m = metrics_block(y_v, p_v, base_rate=base_rate_val)
        ep_dt = time.time() - ep_t0
        print(f"  epoch {epoch+1}/{max_epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_auc={m['auc']:.4f}  ({ep_dt:.1f}s)")
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_win_log_loss": float(val_loss),
            "val_win_auc": float(m["auc"]),
            "val_win_acc": float(m["acc"]),
            "val_win_brier": float(m["brier"]),
            "wall_seconds": float(ep_dt),
        })
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_auc = m["auc"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval_metrics = {"win": m}
            best_y, best_p = y_v, p_v
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if patience is not None and epochs_since_improve >= patience:
            print(f"  early stop at epoch {epoch+1} (best {best_epoch})")
            break
    train_sec = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(history=history, best_val_win_loss=best_val_loss,
                       best_val_auc=best_val_auc, best_epoch=best_epoch,
                       epochs_run=history[-1]["epoch"] if history else 0,
                       train_seconds=train_sec, val_metrics_at_best=best_eval_metrics,
                       val_win_predictions=best_p, val_win_labels=best_y,
                       val_dur_predictions=None, val_dur_labels=None)


def train_model_multitask(model, train_ds, val_ds, hp: dict, max_epochs: int,
                           device, base_rate_val: float | None,
                           use_features: bool, mixed_precision: bool,
                           patience: int | None, alpha: dict,
                           n_dur_buckets: int) -> TrainResult:
    bs = int(hp["batch_size"]); lr = float(hp["lr"]); wd = float(hp["weight_decay"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    bce_item = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    # NOTE: SmoothL1Loss replaced with _masked_smooth_l1_mean in the train
    # loop so target NaN/Inf or model-output spikes don't poison gradients.
    autocast_dtype = torch.bfloat16 if (mixed_precision and device.type == "cuda") else None
    aw = float(alpha["alpha_win"]); ad = float(alpha["alpha_dur"])
    ai = float(alpha["alpha_item"]); aa = float(alpha["alpha_aux"])

    history = []
    best_val_loss = math.inf
    best_val_auc = -math.inf
    best_state = None
    best_epoch = -1
    best_eval: dict = {}
    epochs_since_improve = 0
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        n_seen = 0
        sum_w = sum_d = sum_i = sum_a = sum_total = 0.0
        ep_t0 = time.time()
        for batch in train_loader:
            hero_ids, pf, y_win, y_dur, y_item, y_aux = batch
            hero_ids = hero_ids.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            y_win = y_win.to(device, non_blocking=True)
            y_dur = y_dur.to(device, non_blocking=True)
            y_item = y_item.to(device, non_blocking=True)
            y_aux = y_aux.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                out = model(hero_ids, pf if use_features else None, multitask=True)
                l_w = bce(out["win"].float(), y_win.float())
                l_d = ce(out["dur"].float(), y_dur)
                l_i = bce_item(out["item"].float(), y_item.float())
                l_a = _masked_smooth_l1_mean(out["aux"].float(), y_aux.float())
                loss = aw * l_w + ad * l_d + ai * l_i + aa * l_a
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bsize = y_win.size(0)
            n_seen += bsize
            sum_w += l_w.item() * bsize
            sum_d += l_d.item() * bsize
            sum_i += l_i.item() * bsize
            sum_a += l_a.item() * bsize
            sum_total += loss.item() * bsize
        train_w = sum_w / max(n_seen, 1)
        train_d = sum_d / max(n_seen, 1)
        train_i = sum_i / max(n_seen, 1)
        train_a = sum_a / max(n_seen, 1)
        train_total = sum_total / max(n_seen, 1)

        eval_out = _eval_multitask(model, val_loader, device, autocast_dtype,
                                    use_features, alpha=alpha, n_dur_buckets=n_dur_buckets)
        vl = eval_out["val_losses"]
        # Early-stop on win component only.
        val_win_loss = vl["win_log_loss_per_row"]
        m_w = metrics_block(eval_out["win_y"], eval_out["win_p"], base_rate=base_rate_val)
        # Duration top-1 accuracy.
        dur_pred_top1 = eval_out["dur_p"].argmax(axis=-1)
        dur_acc = float((dur_pred_top1 == eval_out["dur_y"]).mean())
        dur_brier = float(((eval_out["dur_p"] - np.eye(n_dur_buckets)[eval_out["dur_y"]]) ** 2).sum(axis=1).mean())
        # Item subsample mAP@10.
        item_metrics = item_map_at_k(eval_out["item_subsample_targets"],
                                       eval_out["item_subsample_logits"], k=10)
        ep_dt = time.time() - ep_t0
        print(f"  epoch {epoch+1}/{max_epochs} "
              f"tr[w={train_w:.4f} d={train_d:.4f} i={train_i:.5f} a={train_a:.4f} tot={train_total:.4f}] "
              f"vl_win={val_win_loss:.4f}  val_auc={m_w['auc']:.4f}  "
              f"dur_acc={dur_acc:.4f}  itemMAP@10={item_metrics['map_at_k']}  "
              f"({ep_dt:.1f}s)")
        history.append({
            "epoch": epoch + 1,
            "train_win_loss": train_w, "train_dur_loss": train_d,
            "train_item_loss": train_i, "train_aux_loss": train_a,
            "train_weighted_total": train_total,
            "val_win_log_loss": float(val_win_loss),
            "val_win_auc": float(m_w["auc"]),
            "val_win_brier": float(m_w["brier"]),
            "val_dur_ce_per_row": float(vl["dur_ce_per_row"]),
            "val_dur_acc": float(dur_acc),
            "val_dur_brier": float(dur_brier),
            "val_item_bce_per_slot_per_class": float(vl["item_bce_per_slot_per_class"]),
            "val_item_map_at_10": item_metrics["map_at_k"],
            "val_aux_smoothl1_per_dim": float(vl["aux_smoothl1_per_slot_per_dim"]),
            "val_aux_mse_per_dim": eval_out["aux_mse_per_dim"],
            "val_weighted_total": float(vl["weighted_total"]),
            "wall_seconds": float(ep_dt),
        })
        if val_win_loss < best_val_loss - 1e-6:
            best_val_loss = val_win_loss
            best_val_auc = m_w["auc"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = {
                "win": m_w, "dur_top1_acc": dur_acc, "dur_brier": dur_brier,
                "item_mAP_at_10": item_metrics,
                "aux_mse_per_dim": eval_out["aux_mse_per_dim"],
                "val_component_losses": vl,
            }
            best_y_w, best_p_w = eval_out["win_y"], eval_out["win_p"]
            best_y_d, best_p_d = eval_out["dur_y"], eval_out["dur_p"]
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if patience is not None and epochs_since_improve >= patience:
            print(f"  early stop at epoch {epoch+1} (best {best_epoch})")
            break
    train_sec = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(history=history, best_val_win_loss=best_val_loss,
                       best_val_auc=best_val_auc, best_epoch=best_epoch,
                       epochs_run=history[-1]["epoch"] if history else 0,
                       train_seconds=train_sec, val_metrics_at_best=best_eval,
                       val_win_predictions=best_p_w, val_win_labels=best_y_w,
                       val_dur_predictions=best_p_d, val_dur_labels=best_y_d)


def coverage_bucket_val_auc(val_ds, y_val: np.ndarray, p_val: np.ndarray,
                            feat_names: list[str]) -> dict:
    if "n_games_log1p" not in feat_names:
        return {"error": "n_games_log1p not in feat_names"}
    f_idx = feat_names.index("n_games_log1p")
    coverage = val_ds.player_feats[:, :, f_idx].mean(dim=1).numpy()
    q33, q67 = np.quantile(coverage, [0.333, 0.667])
    buckets = np.digitize(coverage, [q33, q67])
    bucket_aucs = {}
    for b, name in [(0, "low"), (1, "medium"), (2, "high")]:
        mask = buckets == b
        n = int(mask.sum())
        if n < 100:
            bucket_aucs[name] = {"n": n, "val_auc": None, "mean_coverage_log1p": None}
            continue
        yb = y_val[mask]; pb = p_val[mask]
        try:
            auc_b = float(roc_auc_score(yb, pb))
        except ValueError:
            auc_b = None
        bucket_aucs[name] = {"n": n, "val_auc": auc_b,
                              "mean_coverage_log1p": float(coverage[mask].mean())}
    return {"quantile_edges_log1p": [float(q33), float(q67)], "buckets": bucket_aucs}


def plot_calibration(y_true, p_pred, out: Path) -> dict:
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=20, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", lw=1.5, label="model")
    ax.set_xlabel("predicted P(radiant_win)"); ax.set_ylabel("empirical P(radiant_win)")
    ax.set_title("Calibration (val, 20-quantile)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}


def plot_roc(y_true, p_pred, auc, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC={auc:.4f}"); ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (val)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_learning(history: list, out: Path, multitask: bool) -> None:
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    if multitask:
        ax1.plot(ep, [h["train_win_loss"] for h in history], label="train_win", lw=1)
        ax1.plot(ep, [h["val_win_log_loss"] for h in history], label="val_win", lw=1)
    else:
        ax1.plot(ep, [h["train_loss"] for h in history], label="train_loss", lw=1)
        ax1.plot(ep, [h["val_win_log_loss"] for h in history], label="val_loss", lw=1)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_win_auc"] for h in history], "g--", label="val_win_auc", lw=1)
    ax2.set_ylabel("val_win_auc"); ax2.legend(loc="upper right")
    ax1.set_title("Learning curves (win head)"); ax1.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--ablation", required=True,
                    choices=["win_only_sanity", "multitask_all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--metrics-suffix", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    seed = int(cfg["seed"])
    set_seed(seed)

    ab_spec = next((a for a in cfg["transformer_ablations"] if a["name"] == args.ablation), None)
    if ab_spec is None:
        sys.exit(f"unknown ablation {args.ablation}")
    use_features = bool(ab_spec["use_features"])
    multitask = bool(ab_spec.get("multitask", False))

    feat_names = cfg["player_features_transformer"]["feat_names"]
    n_player_feats = int(cfg["player_features_transformer"]["n_player_feats"])
    source_dir = PROJECT_ROOT / cfg["player_features_transformer"]["source_dir"]
    sidecar_dir = PROJECT_ROOT / cfg["rich_cols"]["out_dir"]
    vocab_path = EXP_DIR / cfg["item_vocab"]["vocab_path"]
    aux_targets = cfg["multitask_loss"]["aux_targets"]
    n_dur_buckets = int(cfg["duration_bucket"]["n_buckets"])

    print(f"Ablation: {args.ablation} (use_features={use_features}, multitask={multitask})")
    t0 = time.time()
    n_target = int(cfg["train_subset_size"])
    if args.smoke:
        train_ds, val_ds, meta = load_train_val(
            seed=seed, n_target=n_target, feat_names=feat_names,
            source_dir=source_dir, splits=splits, smoke=True,
            smoke_n_train=int(cfg["transformer_smoke"]["n_train"]),
            smoke_n_val=int(cfg["transformer_smoke"]["n_val"]),
            multitask=multitask, sidecar_dir=sidecar_dir if multitask else None,
            vocab_path=vocab_path if multitask else None,
            aux_targets=aux_targets,
        )
    else:
        train_ds, val_ds, meta = load_train_val(
            seed=seed, n_target=n_target, feat_names=feat_names,
            source_dir=source_dir, splits=splits, smoke=False,
            multitask=multitask, sidecar_dir=sidecar_dir if multitask else None,
            vocab_path=vocab_path if multitask else None,
            aux_targets=aux_targets,
        )
    data_seconds = time.time() - t0
    print(f"Data ready in {data_seconds:.1f}s -- train={len(train_ds):,} val={len(val_ds):,}")
    print(f"  train dates {meta['train_date_min']}..{meta['train_date_max']}")
    print(f"  val   dates {meta['val_date_min']}..{meta['val_date_max']}")

    # Build model. For win_only_sanity we still build the multi-head shell
    # (state-dict shape-stable) but only the win head sees gradient.
    mhp = cfg["transformer_model"]
    item_vocab_size = int(meta.get("item_vocab_size", 0)) or 1
    n_aux = int(meta.get("n_aux", 0)) or 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mhp, vocab_size=int(cfg["hero"]["vocab_size"]),
                        n_player_feats=n_player_feats, use_features=use_features,
                        n_dur_buckets=n_dur_buckets,
                        item_vocab_size=item_vocab_size, n_aux=n_aux)
    model = model.to(device)
    pc = count_params(model)
    print(f"Model: {pc}, device={device}")

    opt_cfg = cfg["transformer_optim"]
    max_epochs = (int(opt_cfg["max_epochs"]) if not args.smoke
                  else int(cfg["transformer_smoke"]["max_epochs"]))
    base_rate_val = meta["radiant_base_rate_val"]
    hp = {"batch_size": int(opt_cfg["batch_size"]), "lr": float(opt_cfg["lr"]),
          "weight_decay": float(opt_cfg["weight_decay"])}
    patience = int(opt_cfg.get("patience", 5)) if not args.smoke else None

    if not multitask:
        tr = train_model_win_only(
            model, train_ds, val_ds, hp, max_epochs=max_epochs, device=device,
            base_rate_val=base_rate_val, use_features=use_features,
            mixed_precision=bool(opt_cfg["mixed_precision"]),
            patience=patience)
    else:
        alpha = cfg["multitask_loss"]
        tr = train_model_multitask(
            model, train_ds, val_ds, hp, max_epochs=max_epochs, device=device,
            base_rate_val=base_rate_val, use_features=use_features,
            mixed_precision=bool(opt_cfg["mixed_precision"]),
            patience=patience, alpha=alpha, n_dur_buckets=n_dur_buckets)
    print(f"Training done in {tr.train_seconds:.1f}s -- best val_auc={tr.best_val_auc:.4f} "
          f"@ epoch {tr.best_epoch}")

    # Coverage-bucket diagnostic on the win head.
    y_val = tr.val_win_labels
    p_val = tr.val_win_predictions
    try:
        cov_info = coverage_bucket_val_auc(val_ds, y_val, p_val, feat_names)
    except Exception as e:  # noqa: BLE001
        cov_info = {"error": f"{type(e).__name__}: {e}"}

    # Plots + checkpoint.
    results_dir = EXP_DIR / cfg["output"]["results_dir"]
    results_dir.mkdir(exist_ok=True, parents=True)
    sfx = args.metrics_suffix or f"_{args.ablation}"
    if args.smoke:
        sfx = cfg["transformer_smoke"]["metrics_suffix"] + f"_{args.ablation}"
    cal = None
    try:
        cal = plot_calibration(y_val, p_val, results_dir / f"calibration{sfx}.png")
        plot_roc(y_val, p_val, tr.best_val_auc, results_dir / f"roc{sfx}.png")
        plot_learning(tr.history, results_dir / f"learning_curve{sfx}.png",
                       multitask=multitask)
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")
    try:
        torch.save(model.state_dict(), results_dir / f"model{sfx}.pt")
    except Exception as e:  # noqa: BLE001
        print(f"checkpoint save skipped: {e}")

    # Anchors / deltas.
    anchors = cfg.get("anchors", {})
    cleanup_anchor = float(anchors.get("cleanup_anchor_val_auc", 0.6477054))
    proposal_target = float(anchors.get("proposal_target_val_auc", 0.6487))

    metrics = {
        "ablation": args.ablation,
        "multitask": multitask,
        "smoke": bool(args.smoke),
        "use_features": use_features,
        "val_auc": float(tr.best_val_auc),
        "val_win_log_loss": float(tr.best_val_win_loss),
        "val_metrics_at_best": tr.val_metrics_at_best,
        "best_epoch": int(tr.best_epoch),
        "epochs_run": int(tr.epochs_run),
        "max_epochs": int(max_epochs),
        "history": tr.history,
        "model_hp": {k: mhp[k] for k in ("embed_dim", "d_model", "n_heads",
                                          "n_layers", "ff_mult", "dropout")},
        "optim_hp": {"batch_size": hp["batch_size"], "lr": hp["lr"],
                     "weight_decay": hp["weight_decay"],
                     "mixed_precision": bool(opt_cfg["mixed_precision"])},
        "multitask_alpha": cfg["multitask_loss"] if multitask else None,
        "param_counts": pc,
        "train_seconds": float(tr.train_seconds),
        "data_seconds": float(data_seconds),
        "n_train_pre_subsample": int(meta["n_train_pre_subsample"]),
        "n_train_post_subsample": int(meta["n_train_post_subsample"]),
        "n_val": int(meta["n_val"]),
        "train_subset_size_target": int(meta["train_subset_size_target"]),
        "train_subset_seed": int(meta["train_subset_seed"]),
        "train_date_min": meta["train_date_min"],
        "train_date_max": meta["train_date_max"],
        "val_date_min": meta["val_date_min"],
        "val_date_max": meta["val_date_max"],
        "radiant_base_rate_train_full": meta["radiant_base_rate_train_full"],
        "radiant_base_rate_train_subsampled": meta["radiant_base_rate_train_subsampled"],
        "radiant_base_rate_val": meta["radiant_base_rate_val"],
        "val_majority_class_acc": max(meta["radiant_base_rate_val"],
                                       1 - meta["radiant_base_rate_val"]),
        "feat_names": list(feat_names),
        "n_player_feats": n_player_feats,
        "item_vocab_size": int(meta.get("item_vocab_size", 0)),
        "n_dur_buckets": int(n_dur_buckets),
        "aux_targets": meta.get("aux_targets"),
        "duration_bucket_edges": meta.get("duration_bucket_edges"),
        "aux_train_mean": meta.get("aux_train_mean"),
        "aux_train_std": meta.get("aux_train_std"),
        "anchors": anchors,
        "delta_vs_cleanup_anchor": float(tr.best_val_auc - cleanup_anchor),
        "delta_vs_proposal_target": float(tr.best_val_auc - proposal_target),
        "coverage_bucket_val_auc": cov_info,
        "calibration": cal,
    }
    out_name = f"metrics{sfx}.json"
    out_path = EXP_DIR / out_name
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
