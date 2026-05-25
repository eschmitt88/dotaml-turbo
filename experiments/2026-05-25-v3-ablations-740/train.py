"""Train FoundationTransformer for v3-ablations-740.

Forked from experiments/2026-05-24-foundation-v3-740/train.py.

Two factor-isolation ablations on the v3 stack (val_auc=0.6462,
-0.0031 below iso_teambias=0.6493):

- v3_dur_ce      : revert duration head to 8-bucket CE (vs v3's SmoothL1
                   regression on log-seconds).
- v3_player_emb  : add per-player identity embedding lookup (~4M params
                   at 128 dim) on top of v3 (duration stays as
                   regression to match v3).

All other v3 knobs preserved: extended cross-patch data, hand-tuned
alpha (1.0/0.15/0.3/0.1), no UW-SO, PMAE EMA-teacher, (team,team) bias,
patch token, canonical hero sort.

Optimizer: AdamW (zero weight-decay on the embedding for A2) lr=1e-3
with 1000-step warmup then cosine to 1e-5. bf16 autocast on CUDA.
Early-stop on val_win_log_loss with patience.
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
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
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
from mae import (  # noqa: E402
    EMATeacher,
    PMAEMasker,
    pmae_reconstruction_loss_logged,
)
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size * 2, 2048),
                             shuffle=False, num_workers=0, pin_memory=False,
                             drop_last=False)
    return train_loader, val_loader


def warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int,
                      base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos


def alpha_mae_schedule(step: int, total_steps: int,
                        alpha_start: float, alpha_end: float) -> float:
    progress = step / max(total_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return alpha_start + (alpha_end - alpha_start) * progress


def _masked_smooth_l1_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0
    p = torch.where(mask, pred, torch.zeros_like(pred))
    t = torch.where(mask, target, torch.zeros_like(target))
    diff = p - t
    abs_diff = diff.abs()
    elem = torch.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    elem = torch.where(mask, elem, torch.zeros_like(elem))
    return elem.sum() / float(mask.sum().item())


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


def _eval_multitask(model, loader, device, autocast_dtype, use_features: bool,
                     use_patch_token: bool, n_dur_buckets: int,
                     use_player_embedding: bool,
                     dur_loss_mode: str,
                     duration_bucket_edges: list[float] | None = None) -> dict:
    """Eval. Supports both CE (dur logits [B, K]) and regression (dur scalar [B])
    duration heads.
    """
    model.eval()
    bce_sum = nn.BCEWithLogitsLoss(reduction="sum")
    bce_item = nn.BCEWithLogitsLoss(reduction="sum")
    n = 0
    tot_w = tot_d_l1 = tot_d_ce = tot_i = 0.0
    tot_d_sq = 0.0
    tot_d_correct = 0
    tot_kda = tot_gpm = tot_hd = 0.0
    tot_kda_n = tot_gpm_n = tot_hd_n = 0
    ys_w, ps_w = [], []
    ys_dur_log, ps_dur_log, ys_dur_bucket = [], [], []
    ps_dur_bucket = []
    item_sub_t, item_sub_l = [], []
    patch_ids_all = []
    account_idx_all = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in loader:
            (hero_ids, pf, patch_id, acct_idx, y_win, y_dur, y_dur_bucket, y_item,
             y_kda, y_gpm, y_hd) = batch
            hero_ids = hero_ids.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            patch_id = patch_id.to(device, non_blocking=True)
            acct_idx = acct_idx.to(device, non_blocking=True)
            y_win = y_win.to(device, non_blocking=True)
            y_dur = y_dur.to(device, non_blocking=True)
            y_dur_bucket_dev = y_dur_bucket.to(device, non_blocking=True)
            y_item = y_item.to(device, non_blocking=True)
            y_kda = y_kda.to(device, non_blocking=True)
            y_gpm = y_gpm.to(device, non_blocking=True)
            y_hd = y_hd.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                out = model(hero_ids, pf if use_features else None,
                            patch_id=patch_id if use_patch_token else None,
                            account_idx=acct_idx if use_player_embedding else None)
                win_logits = torch.nan_to_num(out["win"].float(), nan=0.0,
                                                posinf=50.0, neginf=-50.0)
                l_w = bce_sum(win_logits, y_win.float())
                if dur_loss_mode == "ce":
                    dur_logits = torch.nan_to_num(out["dur"].float(), nan=0.0,
                                                    posinf=50.0, neginf=-50.0)
                    l_d_ce = F.cross_entropy(dur_logits, y_dur_bucket_dev,
                                              reduction="sum")
                    pred_bucket = dur_logits.argmax(dim=-1)
                    tot_d_correct += int((pred_bucket == y_dur_bucket_dev).sum().item())
                    ps_dur_bucket.append(pred_bucket.cpu().numpy())
                    if duration_bucket_edges is not None and len(duration_bucket_edges) > 0:
                        edges = np.asarray(duration_bucket_edges, dtype=np.float64)
                        ext = np.concatenate([[0.0], edges, [7200.0]])
                        mids_arr = 0.5 * (ext[:-1] + ext[1:])
                        n_b = len(mids_arr)
                        pb_np = pred_bucket.cpu().numpy().clip(0, n_b - 1)
                        midpoint_sec = mids_arr[pb_np]
                        pred_log = np.log1p(midpoint_sec).astype(np.float32)
                        ps_dur_log.append(pred_log)
                    else:
                        ps_dur_log.append(np.zeros(len(pred_bucket), dtype=np.float32))
                    tot_d_ce += l_d_ce.item()
                else:
                    dur_pred = torch.nan_to_num(out["dur"].float(), nan=0.0,
                                                  posinf=20.0, neginf=0.0)
                    diff = dur_pred - y_dur.float()
                    ad = diff.abs()
                    elem = torch.where(ad < 1.0, 0.5 * diff * diff, ad - 0.5)
                    l_d_l1 = elem.sum()
                    l_d_sq = (diff * diff).sum()
                    ps_dur_log.append(dur_pred.cpu().numpy())
                    tot_d_l1 += l_d_l1.item()
                    tot_d_sq += l_d_sq.item()
                l_i = bce_item(out["item"].float(), y_item.float())
                def _smooth_l1_sum(p, t):
                    m = torch.isfinite(p) & torch.isfinite(t)
                    nv = int(m.sum().item())
                    if nv == 0:
                        return p.sum() * 0.0, 0
                    p = torch.where(m, p, torch.zeros_like(p))
                    t = torch.where(m, t, torch.zeros_like(t))
                    diff = p - t
                    ad = diff.abs()
                    elem = torch.where(ad < 1.0, 0.5 * diff * diff, ad - 0.5)
                    elem = torch.where(m, elem, torch.zeros_like(elem))
                    return elem.sum(), nv
                l_kda, n_kda = _smooth_l1_sum(out["kda"].float(), y_kda.float())
                l_gpm, n_gpm = _smooth_l1_sum(out["gpm"].float(), y_gpm.float())
                l_hd, n_hd = _smooth_l1_sum(out["hd"].float(), y_hd.float())
            bsize = y_win.size(0)
            n += bsize
            tot_w += l_w.item()
            tot_i += l_i.item()
            tot_kda += l_kda.item(); tot_kda_n += n_kda
            tot_gpm += l_gpm.item(); tot_gpm_n += n_gpm
            tot_hd += l_hd.item();   tot_hd_n += n_hd
            ps_w.append(torch.sigmoid(win_logits).cpu().numpy())
            ys_w.append(y_win.cpu().numpy())
            ys_dur_log.append(y_dur.cpu().numpy())
            ys_dur_bucket.append(y_dur_bucket.cpu().numpy())
            patch_ids_all.append(patch_id.cpu().numpy())
            account_idx_all.append(acct_idx.cpu().numpy())
            if len(item_sub_t) * bsize * 10 < 500_000:
                idx = rng.choice(bsize, size=min(bsize, 64), replace=False)
                t_np = y_item[idx].cpu().numpy().reshape(-1, y_item.shape[-1])
                l_np = out["item"][idx].float().cpu().numpy().reshape(-1, y_item.shape[-1])
                item_sub_t.append(t_np)
                item_sub_l.append(l_np)
    item_v = int(getattr(model, "item_vocab_size", 1)) or 1
    val_losses = {
        "win_log_loss_per_row": tot_w / max(n, 1),
        "item_bce_per_slot_per_class":      tot_i / max(n * 10 * item_v, 1),
        "kda_smoothl1_per_slot": tot_kda / max(tot_kda_n, 1),
        "gpm_smoothl1_per_slot": tot_gpm / max(tot_gpm_n, 1),
        "hd_smoothl1_per_slot":  tot_hd / max(tot_hd_n, 1),
    }
    if dur_loss_mode == "ce":
        val_losses["dur_ce_per_row"] = tot_d_ce / max(n, 1)
        val_losses["dur_top1_acc"] = tot_d_correct / max(n, 1)
        val_losses["dur_smoothl1_per_row_log_seconds"] = 0.0
        val_losses["dur_mse_log_seconds"] = 0.0
        val_losses["dur_mae_log_seconds"] = float(np.mean(np.abs(
            np.concatenate(ps_dur_log) - np.concatenate(ys_dur_log)))) if ps_dur_log else 0.0
    else:
        val_losses["dur_smoothl1_per_row_log_seconds"] = tot_d_l1 / max(n, 1)
        val_losses["dur_mse_log_seconds"]              = tot_d_sq / max(n, 1)
        val_losses["dur_mae_log_seconds"]              = float(np.mean(np.abs(
            np.concatenate(ps_dur_log) - np.concatenate(ys_dur_log)))) if ps_dur_log else 0.0
    return {
        "win_y": np.concatenate(ys_w), "win_p": np.concatenate(ps_w),
        "dur_y_log": np.concatenate(ys_dur_log),
        "dur_p_log": np.concatenate(ps_dur_log) if ps_dur_log else np.zeros(0, dtype=np.float32),
        "dur_y_bucket": np.concatenate(ys_dur_bucket),
        "dur_p_bucket": (np.concatenate(ps_dur_bucket)
                          if ps_dur_bucket else np.zeros(0, dtype=np.int64)),
        "patch_ids": np.concatenate(patch_ids_all),
        "account_idx": np.concatenate(account_idx_all),
        "item_subsample_targets": (np.concatenate(item_sub_t, axis=0) if item_sub_t
                                     else np.zeros((0, 0))),
        "item_subsample_logits": (np.concatenate(item_sub_l, axis=0) if item_sub_l
                                    else np.zeros((0, 0))),
        "val_losses": val_losses,
    }


def post_hoc_duration_bucket_top1(dur_p_log: np.ndarray,
                                    dur_y_bucket: np.ndarray,
                                    duration_bucket_edges: list[float]) -> float:
    """Post-hoc bucket-top1-acc from regression head log preds (A2 only)."""
    if len(dur_p_log) == 0 or len(dur_y_bucket) == 0:
        return float("nan")
    pred_sec = np.expm1(np.clip(dur_p_log.astype(np.float64), 0.0, 20.0))
    edges = np.asarray(duration_bucket_edges, dtype=np.float64)
    pred_bucket = np.digitize(pred_sec, edges).astype(np.int64)
    return float((pred_bucket == dur_y_bucket).mean())


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a = a.astype(np.float64); b = b.astype(np.float64)
    am = a - a.mean(); bm = b - b.mean()
    denom = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((am * bm).sum() / denom)


def per_patch_val_auc(y_win: np.ndarray, p_win: np.ndarray,
                       patch_ids: np.ndarray) -> dict:
    out: dict = {}
    for pid in sorted(set(int(p) for p in np.unique(patch_ids))):
        mask = patch_ids == pid
        n = int(mask.sum())
        if n < 100:
            out[str(pid)] = {"n": n, "val_auc": None}
            continue
        try:
            auc = float(roc_auc_score(y_win[mask], p_win[mask]))
        except ValueError:
            auc = None
        out[str(pid)] = {"n": n, "val_auc": auc}
    return out


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
    if targets.size == 0:
        return {"map_at_k": None, "n_rows": 0, "k": k}
    M, V = targets.shape
    if V == 0:
        return {"map_at_k": None, "n_rows": M, "k": k}
    k = min(k, V)
    top_idx = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    sort_order = np.argsort(-np.take_along_axis(logits, top_idx, axis=1), axis=1)
    top_idx = np.take_along_axis(top_idx, sort_order, axis=1)
    hits = np.take_along_axis(targets, top_idx, axis=1).astype(np.float32)
    cum = np.cumsum(hits, axis=1)
    ranks = np.arange(1, k + 1, dtype=np.float32)
    prec_at_i = cum / ranks
    n_pos_per_row = targets.sum(axis=1)
    denom = np.where(n_pos_per_row > 0, np.minimum(n_pos_per_row, k), 1.0)
    ap = (prec_at_i * hits).sum(axis=1) / denom
    map_at_k = float(ap.mean())
    return {"map_at_k": map_at_k, "n_rows": int(M), "k": int(k)}


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


def topk_in_vocab_val_auc(y_val: np.ndarray, p_val: np.ndarray,
                            account_idx: np.ndarray, anon_idx: int,
                            hash_base_idx: int, threshold: int = 3) -> dict:
    """A2 diagnostic: split val by whether the match has >=threshold slots
    in the 'top-K-frequent' vocab. Returns per-stratum AUC.
    """
    if hash_base_idx > 0:
        in_topk = (account_idx > anon_idx) & (account_idx < hash_base_idx)
    else:
        in_topk = (account_idx > anon_idx)
    n_topk_per_match = in_topk.sum(axis=1)
    high_mask = n_topk_per_match >= threshold
    out: dict = {"threshold": int(threshold),
                 "n_high": int(high_mask.sum()),
                 "n_low": int((~high_mask).sum()),
                 "frac_high": float(high_mask.mean()),
                 "mean_n_topk_per_match": float(n_topk_per_match.mean())}
    for name, mask in [("high_topk", high_mask), ("low_topk", ~high_mask)]:
        n = int(mask.sum())
        if n < 100:
            out[name] = {"n": n, "val_auc": None}
            continue
        try:
            auc = float(roc_auc_score(y_val[mask], p_val[mask]))
        except ValueError:
            auc = None
        out[name] = {"n": n, "val_auc": auc}
    return out


def embedding_diagnostics(model, account_idx_all: np.ndarray,
                           anon_idx: int, hash_base_idx: int,
                           min_appearances: int = 50,
                           n_sample_pairs: int = 10,
                           seed: int = 0) -> dict:
    """A2 diagnostic: L2-norm distribution + cosine-similarity sample."""
    if not hasattr(model, "player_embed") or model.player_embed is None:
        return {"error": "model has no player_embed"}
    W = model.player_embed.weight.detach().cpu().numpy().astype(np.float64)
    norms = np.linalg.norm(W, axis=1)
    qs = np.quantile(norms, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    out: dict = {
        "vocab_size":   int(W.shape[0]),
        "embed_dim":    int(W.shape[1]),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_std":  float(norms.std()),
        "l2_norm_quantiles": {
            "min": float(qs[0]), "p10": float(qs[1]), "p25": float(qs[2]),
            "p50": float(qs[3]), "p75": float(qs[4]), "p90": float(qs[5]),
            "p99": float(qs[6]), "max": float(qs[7]),
        },
        "l2_anon_row":  float(norms[anon_idx]) if anon_idx < W.shape[0] else None,
    }
    if account_idx_all.size > 0:
        flat = account_idx_all.ravel()
        if hash_base_idx > 0:
            mask = (flat > anon_idx) & (flat < hash_base_idx)
        else:
            mask = (flat > anon_idx)
        flat_topk = flat[mask]
        if flat_topk.size > 0:
            uniq, cnts = np.unique(flat_topk, return_counts=True)
            eligible = uniq[cnts >= min_appearances]
            rng = np.random.default_rng(seed)
            if eligible.size >= 2:
                pick_n = min(n_sample_pairs, eligible.size)
                pick = rng.choice(eligible, size=pick_n, replace=False)
                pick_norms = norms[pick]
                P = W[pick]
                Pn = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True),
                                  1e-8, None)
                cos = Pn @ Pn.T
                iu, ju = np.triu_indices(pick_n, k=1)
                cos_pairs = cos[iu, ju]
                out["cosine_sample"] = {
                    "n_pairs":       int(len(cos_pairs)),
                    "n_unique_picked": int(pick_n),
                    "min_appearances": int(min_appearances),
                    "cos_mean":      float(cos_pairs.mean()),
                    "cos_std":       float(cos_pairs.std()),
                    "cos_abs_mean":  float(np.abs(cos_pairs).mean()),
                    "cos_quantiles": [float(np.quantile(cos_pairs, q))
                                       for q in (0.05, 0.25, 0.5, 0.75, 0.95)],
                    "picked_norms":  pick_norms.tolist(),
                }
            else:
                out["cosine_sample"] = {"error": "insufficient eligible rows",
                                          "n_eligible": int(eligible.size)}
        else:
            out["cosine_sample"] = {"error": "no in-topK val account_idx values"}
    return out


def train_one_ablation(model, train_ds, val_ds, hp: dict, max_epochs: int,
                         device, base_rate_val: float | None,
                         use_features: bool, use_patch_token: bool,
                         use_player_embedding: bool,
                         dur_loss_mode: str,
                         mixed_precision: bool, patience: int | None,
                         alpha: dict, n_dur_buckets: int,
                         use_pmae: bool, pmae_cfg: dict,
                         warmup_steps: int, cosine_min_lr: float,
                         duration_bucket_edges: list[float]) -> tuple[TrainResult, dict]:
    bs = int(hp["batch_size"]); lr = float(hp["lr"]); wd = float(hp["weight_decay"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs)
    autocast_dtype = torch.bfloat16 if (mixed_precision and device.type == "cuda") else None

    task_names = ["win", "dur", "item", "kda", "gpm", "hd"]

    fixed_alpha = torch.tensor([
        float(alpha["alpha_win"]), float(alpha["alpha_dur"]), float(alpha["alpha_item"]),
        float(alpha["alpha_kda"]), float(alpha["alpha_gpm"]), float(alpha["alpha_hd"]),
    ], device=device)

    # AdamW with no weight decay on the embedding for A2.
    emb_param_names = {n for n, _ in model.named_parameters() if "player_embed" in n}
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in emb_param_names:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt_groups = [
        {"params": decay_params,      "weight_decay": wd},
        {"params": no_decay_params,   "weight_decay": 0.0},
    ]
    if use_player_embedding:
        print(f"  optimizer: AdamW, {len(decay_params)} decay-param tensors, "
              f"{len(no_decay_params)} no-decay (embedding) param tensors")
    opt = torch.optim.AdamW(opt_groups, lr=lr)

    pmae = PMAEMasker(
        a=float(pmae_cfg.get("a", 0.05)),
        b=float(pmae_cfg.get("b", 0.5)),
        min_rate=float(pmae_cfg.get("min_rate", 0.05)),
        max_rate=float(pmae_cfg.get("max_rate", 0.85)),
        groups=list(pmae_cfg.get("groups", ["player_block", "item_list",
                                              "hero_token", "patch_token"])),
    )
    alpha_mae_start = float(pmae_cfg.get("alpha_mae_start", 1.0))
    alpha_mae_end = float(pmae_cfg.get("alpha_mae_end", 0.1))
    ema_momentum = float(pmae_cfg.get("ema_momentum", 0.996))

    ema_teacher: EMATeacher | None = None
    if use_pmae:
        ema_teacher = EMATeacher(model, momentum=ema_momentum).to(device)
        print(f"  EMA teacher constructed momentum={ema_momentum}")

    bce_w = nn.BCEWithLogitsLoss()
    bce_i = nn.BCEWithLogitsLoss()
    smoothl1_d = nn.SmoothL1Loss()

    total_steps = max(max_epochs * max(1, math.ceil(len(train_ds) / bs)), 1)

    history = []
    best_val_loss = math.inf
    best_val_auc = -math.inf
    best_state = None
    best_epoch = -1
    best_eval: dict = {}
    best_y_w = best_p_w = np.array([])
    best_account_idx_val = np.zeros((0, 10), dtype=np.int64)
    epochs_since_improve = 0
    global_step = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        n_seen = 0
        sum_w = sum_d = sum_i = sum_kda = sum_gpm = sum_hd = sum_mae = 0.0
        sum_total = 0.0
        n_batches = 0
        sum_hero_mask_count = 0
        sum_hero_mask_frac = 0.0
        sum_patch_mask_count = 0
        sum_patch_mask_frac = 0.0
        sum_student_l2 = 0.0
        sum_teacher_l2 = 0.0
        n_mae_batches = 0
        ep_t0 = time.time()
        for batch in train_loader:
            (hero_ids, pf, patch_id, acct_idx, y_win, y_dur, y_dur_bucket, y_item,
             y_kda, y_gpm, y_hd) = batch
            hero_ids = hero_ids.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            patch_id = patch_id.to(device, non_blocking=True)
            acct_idx = acct_idx.to(device, non_blocking=True)
            y_win = y_win.to(device, non_blocking=True)
            y_dur = y_dur.to(device, non_blocking=True)
            y_dur_bucket = y_dur_bucket.to(device, non_blocking=True)
            y_item = y_item.to(device, non_blocking=True)
            y_kda = y_kda.to(device, non_blocking=True)
            y_gpm = y_gpm.to(device, non_blocking=True)
            y_hd = y_hd.to(device, non_blocking=True)

            cur_lr = warmup_cosine_lr(global_step, warmup_steps, total_steps, lr, cosine_min_lr)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            cur_alpha_mae = alpha_mae_schedule(global_step, total_steps,
                                                  alpha_mae_start, alpha_mae_end)

            mask_out = None
            if use_pmae:
                if pf is not None and pf.size(-1) >= 8:
                    is_anon = (pf[:, :, 7] > 0.5)
                else:
                    is_anon = None
                mask_out = pmae(hero_ids,
                                 player_feats=pf,
                                 patch_id=patch_id if use_patch_token else None,
                                 is_anonymous_per_slot=is_anon)
                hero_mask = mask_out["hero_mask"]
                patch_mask = mask_out["patch_mask"]
            else:
                hero_mask = None
                patch_mask = None

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                out = model(hero_ids, pf if use_features else None,
                            patch_id=patch_id if use_patch_token else None,
                            hero_mask=hero_mask, patch_mask=patch_mask,
                            account_idx=acct_idx if use_player_embedding else None)

                l_w = bce_w(out["win"].float(), y_win.float())
                if dur_loss_mode == "ce":
                    l_d = F.cross_entropy(out["dur"].float(), y_dur_bucket)
                else:
                    l_d = smoothl1_d(out["dur"].float(), y_dur.float())
                l_i = bce_i(out["item"].float(), y_item.float())
                l_kda = _masked_smooth_l1_mean(out["kda"].float(), y_kda.float())
                l_gpm = _masked_smooth_l1_mean(out["gpm"].float(), y_gpm.float())
                l_hd = _masked_smooth_l1_mean(out["hd"].float(), y_hd.float())
                losses = torch.stack([l_w, l_d, l_i, l_kda, l_gpm, l_hd])

                sup_loss = (fixed_alpha * losses).sum()

                l_mae = torch.tensor(0.0, device=device)
                pmae_log = None
                if use_pmae and mask_out is not None and ema_teacher is not None:
                    with torch.no_grad():
                        teacher_out = ema_teacher(
                            hero_ids, pf if use_features else None,
                            patch_id=patch_id if use_patch_token else None,
                            hero_mask=None, patch_mask=None,
                            account_idx=acct_idx if use_player_embedding else None,
                        )
                    teacher_enc = teacher_out["encoded"].float()
                    student_enc = out["encoded"].float()
                    pmae_log = pmae_reconstruction_loss_logged(
                        student_enc, teacher_enc,
                        hero_mask=hero_mask, patch_mask=patch_mask,
                    )
                    l_mae = pmae_log["loss"]

                loss = sup_loss + cur_alpha_mae * l_mae

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if ema_teacher is not None:
                ema_teacher.update(model)

            bsize = y_win.size(0)
            n_seen += bsize
            sum_w += l_w.item() * bsize
            sum_d += l_d.item() * bsize
            sum_i += l_i.item() * bsize
            sum_kda += l_kda.item() * bsize
            sum_gpm += l_gpm.item() * bsize
            sum_hd += l_hd.item() * bsize
            sum_mae += float(l_mae.item()) * bsize
            sum_total += loss.item() * bsize
            n_batches += 1
            global_step += 1
            if pmae_log is not None:
                sum_hero_mask_count += pmae_log["hero_mask_count"]
                sum_hero_mask_frac += pmae_log["hero_mask_frac"]
                sum_patch_mask_count += pmae_log["patch_mask_count"]
                sum_patch_mask_frac += pmae_log["patch_mask_frac"]
                sum_student_l2 += pmae_log["student_l2_mean_at_mask"]
                sum_teacher_l2 += pmae_log["teacher_l2_mean_at_mask"]
                n_mae_batches += 1

        tr_w = sum_w / max(n_seen, 1); tr_d = sum_d / max(n_seen, 1)
        tr_i = sum_i / max(n_seen, 1); tr_kda = sum_kda / max(n_seen, 1)
        tr_gpm = sum_gpm / max(n_seen, 1); tr_hd = sum_hd / max(n_seen, 1)
        tr_mae = sum_mae / max(n_seen, 1); tr_total = sum_total / max(n_seen, 1)
        avg_hero_mask_count = sum_hero_mask_count / max(n_mae_batches, 1)
        avg_hero_mask_frac = sum_hero_mask_frac / max(n_mae_batches, 1)
        avg_patch_mask_count = sum_patch_mask_count / max(n_mae_batches, 1)
        avg_patch_mask_frac = sum_patch_mask_frac / max(n_mae_batches, 1)
        avg_student_l2 = sum_student_l2 / max(n_mae_batches, 1)
        avg_teacher_l2 = sum_teacher_l2 / max(n_mae_batches, 1)

        eval_out = _eval_multitask(model, val_loader, device, autocast_dtype,
                                     use_features=use_features,
                                     use_patch_token=use_patch_token,
                                     use_player_embedding=use_player_embedding,
                                     dur_loss_mode=dur_loss_mode,
                                     n_dur_buckets=n_dur_buckets,
                                     duration_bucket_edges=duration_bucket_edges)
        vl = eval_out["val_losses"]
        val_win_loss = vl["win_log_loss_per_row"]
        m_w = metrics_block(eval_out["win_y"], eval_out["win_p"], base_rate=base_rate_val)
        if dur_loss_mode == "ce":
            dur_top1_acc = float(vl["dur_top1_acc"])
            dur_acc_bucket_posthoc = float("nan")
        else:
            dur_top1_acc = post_hoc_duration_bucket_top1(
                eval_out["dur_p_log"], eval_out["dur_y_bucket"],
                duration_bucket_edges
            )
            dur_acc_bucket_posthoc = dur_top1_acc
        dur_pearson = pearson_corr(eval_out["dur_p_log"], eval_out["dur_y_log"])
        dur_mae_log = float(vl["dur_mae_log_seconds"])
        patch_aucs = per_patch_val_auc(eval_out["win_y"], eval_out["win_p"],
                                          eval_out["patch_ids"])
        item_metrics = item_map_at_k(eval_out["item_subsample_targets"],
                                       eval_out["item_subsample_logits"], k=10)
        ep_dt = time.time() - ep_t0
        print(f"  epoch {epoch+1}/{max_epochs}  "
              f"tr[w={tr_w:.4f} d={tr_d:.4f} i={tr_i:.5f} kda={tr_kda:.4f} "
              f"gpm={tr_gpm:.4f} hd={tr_hd:.4f} mae={tr_mae:.4f} tot={tr_total:.4f}]  "
              f"vl_win={val_win_loss:.4f}  val_auc={m_w['auc']:.4f}  "
              f"dur_top1={dur_top1_acc:.4f} dur_mae_log={dur_mae_log:.4f} "
              f"dur_pearson={dur_pearson:.4f}  itemMAP={item_metrics['map_at_k']}  "
              f"lr={cur_lr:.2e}  alpha_mae={cur_alpha_mae:.3f}  ({ep_dt:.1f}s)")
        if use_pmae:
            print(f"    pmae: hero_mask_frac={avg_hero_mask_frac:.3f} "
                  f"(count={avg_hero_mask_count:.1f}) "
                  f"patch_mask_frac={avg_patch_mask_frac:.3f} "
                  f"s_l2={avg_student_l2:.4f} t_l2={avg_teacher_l2:.4f}")
        if use_player_embedding and hasattr(model, "player_embed") and model.player_embed is not None:
            with torch.no_grad():
                W = model.player_embed.weight
                pe_l2 = float(W.pow(2).sum(dim=1).sqrt().mean().item())
                pe_l2_max = float(W.pow(2).sum(dim=1).sqrt().max().item())
                pe_anon = float(W[0].pow(2).sum().sqrt().item())
            print(f"    player_emb: mean_l2={pe_l2:.4f} max_l2={pe_l2_max:.4f} "
                  f"anon_row_l2={pe_anon:.4f}")
        print(f"    patch_auc: {patch_aucs}")
        history.append({
            "epoch": epoch + 1,
            "train_win_loss": tr_w, "train_dur_loss": tr_d, "train_item_loss": tr_i,
            "train_kda_loss": tr_kda, "train_gpm_loss": tr_gpm, "train_hd_loss": tr_hd,
            "train_mae_loss": tr_mae, "train_total_loss": tr_total,
            "val_win_log_loss": float(val_win_loss),
            "val_win_auc": float(m_w["auc"]),
            "val_win_brier": float(m_w["brier"]),
            "val_dur_smoothl1_log_seconds":
                float(vl.get("dur_smoothl1_per_row_log_seconds", 0.0)),
            "val_dur_mse_log_seconds":      float(vl.get("dur_mse_log_seconds", 0.0)),
            "val_dur_mae_log_seconds":      float(dur_mae_log),
            "val_dur_pearson_log":          float(dur_pearson),
            "val_dur_top1_acc":             float(dur_top1_acc),
            "val_dur_ce_per_row":           float(vl.get("dur_ce_per_row", 0.0)),
            "val_item_bce": float(vl["item_bce_per_slot_per_class"]),
            "val_item_map_at_10": item_metrics["map_at_k"],
            "val_kda_smoothl1": float(vl["kda_smoothl1_per_slot"]),
            "val_gpm_smoothl1": float(vl["gpm_smoothl1_per_slot"]),
            "val_hd_smoothl1":  float(vl["hd_smoothl1_per_slot"]),
            "alpha_weights":    [float(a) for a in fixed_alpha.cpu().tolist()],
            "alpha_mae": float(cur_alpha_mae),
            "lr": float(cur_lr),
            "wall_seconds": float(ep_dt),
            "pmae_hero_mask_count": float(avg_hero_mask_count),
            "pmae_hero_mask_frac":  float(avg_hero_mask_frac),
            "pmae_patch_mask_count": float(avg_patch_mask_count),
            "pmae_patch_mask_frac":  float(avg_patch_mask_frac),
            "pmae_student_l2_at_mask": float(avg_student_l2),
            "pmae_teacher_l2_at_mask": float(avg_teacher_l2),
            "ema_momentum": (float(ema_momentum) if use_pmae else None),
            "per_patch_val_auc": patch_aucs,
        })
        if val_win_loss < best_val_loss - 1e-6:
            best_val_loss = val_win_loss
            best_val_auc = m_w["auc"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = {"win": m_w,
                          "dur_mae_log_seconds": dur_mae_log,
                          "dur_pearson_log": dur_pearson,
                          "dur_top1_acc": dur_top1_acc,
                          "dur_top1_acc_posthoc": dur_acc_bucket_posthoc,
                          "per_patch_val_auc": patch_aucs,
                          "item_mAP_at_10": item_metrics,
                          "val_component_losses": vl}
            best_y_w, best_p_w = eval_out["win_y"], eval_out["win_p"]
            best_account_idx_val = eval_out["account_idx"]
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if patience is not None and epochs_since_improve >= patience:
            print(f"  early stop at epoch {epoch+1} (best {best_epoch})")
            break

    train_sec = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    extras = {"task_names": task_names,
              "pmae_used": use_pmae,
              "warmup_steps": warmup_steps,
              "total_steps": total_steps,
              "best_account_idx_val": best_account_idx_val}
    return TrainResult(history=history, best_val_win_loss=best_val_loss,
                       best_val_auc=best_val_auc, best_epoch=best_epoch,
                       epochs_run=history[-1]["epoch"] if history else 0,
                       train_seconds=train_sec, val_metrics_at_best=best_eval,
                       val_win_predictions=best_p_w, val_win_labels=best_y_w), extras


def plot_calibration(y_true, p_pred, out: Path) -> dict:
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=20, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", lw=1.5, label="model")
    ax.set_xlabel("predicted"); ax.set_ylabel("empirical")
    ax.set_title("Calibration (val)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}


def plot_roc(y_true, p_pred, auc, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC={auc:.4f}"); ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (val)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_learning(history: list, out: Path) -> None:
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, [h["train_win_loss"] for h in history], label="train_win", lw=1)
    ax1.plot(ep, [h["val_win_log_loss"] for h in history], label="val_win", lw=1)
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
                    choices=["v3_dur_ce", "v3_player_emb"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--metrics-suffix", default="")
    ap.add_argument("--max-epochs-override", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    seed = int(cfg["seed"])
    set_seed(seed)

    ab_spec = next((a for a in cfg["transformer_ablations"] if a["name"] == args.ablation), None)
    if ab_spec is None:
        sys.exit(f"unknown ablation {args.ablation}")
    use_features = bool(ab_spec["use_features"])
    use_patch_token = bool(ab_spec.get("use_patch_token", True))
    use_team_team_bias = bool(ab_spec.get("use_team_team_bias", True))
    use_pmae = bool(ab_spec.get("use_pmae", True))
    dur_loss_mode = str(ab_spec.get("dur_loss_mode", "regression"))
    use_player_embedding = bool(ab_spec.get("use_player_embedding", False))
    if bool(ab_spec.get("use_uw_so", False)):
        sys.exit("REFUSED: v3-ablations does not support use_uw_so=True.")

    feat_names = cfg["player_features_transformer"]["feat_names"]
    n_player_feats = int(cfg["player_features_transformer"]["n_player_feats"])
    source_dir = PROJECT_ROOT / cfg["player_features_transformer"]["source_dir"]
    sidecar_dir = PROJECT_ROOT / cfg["rich_cols"]["out_dir"]
    vp = cfg["item_vocab"]["vocab_path"]
    vocab_path = (EXP_DIR / vp).resolve() if not Path(vp).is_absolute() else Path(vp)
    aux_targets = cfg["multitask_loss"]["aux_targets"]
    n_dur_buckets = int(cfg["duration_bucket"]["n_buckets"])

    player_cfg = cfg.get("player_embedding", {})
    pv_path: Path | None = None
    acct_train_paths: list[Path] | None = None
    acct_val_paths: list[Path] | None = None
    n_hash_buckets = 0
    hash_base_idx = 0
    if use_player_embedding:
        if not player_cfg.get("enabled", False):
            sys.exit("REFUSED: ablation requests player embedding but config "
                       "player_embedding.enabled is false.")
        pv_path_str = player_cfg["vocab_path"]
        pv_path = ((EXP_DIR / pv_path_str).resolve() if not Path(pv_path_str).is_absolute()
                     else Path(pv_path_str))
        acct_side_cfg = cfg["account_sidecar"]
        acct_train_paths = [PROJECT_ROOT / p for p in acct_side_cfg["train_paths"]]
        acct_val_paths = [PROJECT_ROOT / p for p in acct_side_cfg["val_paths"]]
        if args.smoke:
            smoke_train_paths = acct_side_cfg.get("smoke_train_paths", acct_side_cfg["train_paths"])
            smoke_val_paths = acct_side_cfg.get("smoke_val_paths", acct_side_cfg["val_paths"])
            acct_train_paths = [PROJECT_ROOT / p for p in smoke_train_paths]
            acct_val_paths = [PROJECT_ROOT / p for p in smoke_val_paths]
        n_hash_buckets = int(player_cfg.get("n_hash_buckets", 0))
        hash_base_idx = int(player_cfg.get("hash_base_idx", 0))

    print(f"Ablation: {args.ablation} "
          f"(features={use_features} patch={use_patch_token} team_bias={use_team_team_bias} "
          f"pmae={use_pmae} dur_loss_mode={dur_loss_mode} player_emb={use_player_embedding})")
    t0 = time.time()
    n_target = int(cfg["train_subset_size"])

    canonical_sort = bool(cfg["transformer_model"].get("use_canonical_sort", True))
    default_patch_id = int(cfg["patch"].get("default_patch_id", 1))

    if args.smoke:
        train_ds, val_ds, meta = load_train_val(
            seed=seed, n_target=n_target, feat_names=feat_names,
            source_dir=source_dir, splits=splits, smoke=True,
            smoke_n_train=int(cfg["transformer_smoke"]["n_train"]),
            smoke_n_val=int(cfg["transformer_smoke"]["n_val"]),
            sidecar_dir=sidecar_dir, vocab_path=vocab_path,
            aux_targets=aux_targets, canonical_sort=canonical_sort,
            default_patch_id=default_patch_id,
            account_sidecar_train_paths=acct_train_paths,
            account_sidecar_val_paths=acct_val_paths,
            player_vocab_path=pv_path,
            n_hash_buckets=n_hash_buckets, hash_base_idx=hash_base_idx,
        )
    else:
        train_ds, val_ds, meta = load_train_val(
            seed=seed, n_target=n_target, feat_names=feat_names,
            source_dir=source_dir, splits=splits, smoke=False,
            sidecar_dir=sidecar_dir, vocab_path=vocab_path,
            aux_targets=aux_targets, canonical_sort=canonical_sort,
            default_patch_id=default_patch_id,
            account_sidecar_train_paths=acct_train_paths,
            account_sidecar_val_paths=acct_val_paths,
            player_vocab_path=pv_path,
            n_hash_buckets=n_hash_buckets, hash_base_idx=hash_base_idx,
        )
    data_seconds = time.time() - t0
    print(f"Data ready in {data_seconds:.1f}s -- train={len(train_ds):,} val={len(val_ds):,}")
    print(f"  train dates {meta['train_date_min']}..{meta['train_date_max']}")
    print(f"  val   dates {meta['val_date_min']}..{meta['val_date_max']}")
    print(f"  canonical_sort={meta['canonical_sort']} default_patch_id={meta['default_patch_id']}")
    if use_player_embedding:
        print(f"  account_idx_train_stats: {meta['account_idx_train_stats']}")
        print(f"  account_idx_val_stats: {meta['account_idx_val_stats']}")

    mhp = cfg["transformer_model"]
    item_vocab_size = int(meta.get("item_vocab_size", 0)) or 1
    patch_vocab_size = int(cfg["patch"]["vocab_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_player_embedding:
        pvm = meta.get("player_vocab_meta") or {}
        if "vocab_size_total" in pvm:
            player_vocab_size = int(pvm["vocab_size_total"])
        else:
            top_k = int(pvm.get("n_top_kept", 0))
            player_vocab_size = max(1 + top_k + n_hash_buckets, 1 + top_k, 2)
        print(f"  player_vocab_size={player_vocab_size:,} (top_k={pvm.get('n_top_kept', 0):,} "
              f"+ hash_buckets={n_hash_buckets} + anon)")
    else:
        player_vocab_size = 1

    model = build_model(mhp, vocab_size=int(cfg["hero"]["vocab_size"]),
                          n_player_feats=n_player_feats, use_features=use_features,
                          n_dur_buckets=n_dur_buckets,
                          item_vocab_size=item_vocab_size,
                          patch_vocab_size=patch_vocab_size,
                          use_team_team_bias=use_team_team_bias,
                          use_patch_token=use_patch_token,
                          use_lobby_token=bool(mhp.get("use_lobby_token", False)),
                          dur_loss_mode=dur_loss_mode,
                          use_player_embedding=use_player_embedding,
                          player_vocab_size=player_vocab_size,
                          player_embed_dim=int(player_cfg.get("embed_dim", 128)),
                          player_init_std=float(player_cfg.get("init_std", 0.02)))
    model = model.to(device)
    pc = count_params(model)
    print(f"Model: {pc}, device={device}")

    opt_cfg = cfg["transformer_optim"]
    max_epochs = (int(opt_cfg["max_epochs"]) if not args.smoke
                  else int(cfg["transformer_smoke"]["max_epochs"]))
    if args.max_epochs_override is not None:
        max_epochs = int(args.max_epochs_override)
        print(f"  max_epochs override: {max_epochs}")
    base_rate_val = meta["radiant_base_rate_val"]
    hp = {"batch_size": int(opt_cfg["batch_size"]), "lr": float(opt_cfg["lr"]),
          "weight_decay": float(opt_cfg["weight_decay"])}
    patience = int(opt_cfg.get("patience", 5)) if not args.smoke else None

    alpha = cfg["multitask_loss"]
    alpha_full = {
        "alpha_win": float(alpha["alpha_win"]),
        "alpha_dur": float(alpha["alpha_dur"]),
        "alpha_item": float(alpha["alpha_item"]),
        "alpha_kda": float(alpha.get("alpha_kda", alpha.get("alpha_aux", 0.1))),
        "alpha_gpm": float(alpha.get("alpha_gpm", alpha.get("alpha_aux", 0.1))),
        "alpha_hd":  float(alpha.get("alpha_hd",  alpha.get("alpha_aux", 0.1))),
    }

    duration_bucket_edges = meta.get("duration_bucket_edges") or []
    tr, extras = train_one_ablation(
        model, train_ds, val_ds, hp,
        max_epochs=max_epochs, device=device, base_rate_val=base_rate_val,
        use_features=use_features, use_patch_token=use_patch_token,
        use_player_embedding=use_player_embedding,
        dur_loss_mode=dur_loss_mode,
        mixed_precision=bool(opt_cfg["mixed_precision"]),
        patience=patience, alpha=alpha_full, n_dur_buckets=n_dur_buckets,
        use_pmae=use_pmae, pmae_cfg=cfg["pmae"],
        warmup_steps=int(opt_cfg.get("warmup_steps", 1000)),
        cosine_min_lr=float(opt_cfg.get("cosine_min_lr", 1e-5)),
        duration_bucket_edges=list(duration_bucket_edges),
    )
    print(f"Training done in {tr.train_seconds:.1f}s -- best val_auc={tr.best_val_auc:.4f} "
          f"@ epoch {tr.best_epoch}")

    y_val = tr.val_win_labels
    p_val = tr.val_win_predictions
    try:
        cov_info = coverage_bucket_val_auc(val_ds, y_val, p_val, feat_names)
    except Exception as e:  # noqa: BLE001
        cov_info = {"error": f"{type(e).__name__}: {e}"}

    a2_diag: dict = {}
    if use_player_embedding:
        try:
            a2_diag["topk_in_vocab_val_auc"] = topk_in_vocab_val_auc(
                y_val, p_val, extras["best_account_idx_val"],
                anon_idx=0, hash_base_idx=hash_base_idx, threshold=3
            )
        except Exception as e:  # noqa: BLE001
            a2_diag["topk_in_vocab_val_auc"] = {"error": f"{type(e).__name__}: {e}"}
        try:
            a2_diag["embedding_diagnostics"] = embedding_diagnostics(
                model, extras["best_account_idx_val"],
                anon_idx=0, hash_base_idx=hash_base_idx,
                min_appearances=50, n_sample_pairs=10
            )
        except Exception as e:  # noqa: BLE001
            a2_diag["embedding_diagnostics"] = {"error": f"{type(e).__name__}: {e}"}

    results_dir = EXP_DIR / cfg["output"]["results_dir"]
    results_dir.mkdir(exist_ok=True, parents=True)
    sfx = args.metrics_suffix or f"_{args.ablation}"
    if args.smoke:
        sfx = cfg["transformer_smoke"]["metrics_suffix"] + f"_{args.ablation}"
    cal = None
    try:
        cal = plot_calibration(y_val, p_val, results_dir / f"calibration{sfx}.png")
        plot_roc(y_val, p_val, tr.best_val_auc, results_dir / f"roc{sfx}.png")
        plot_learning(tr.history, results_dir / f"learning_curve{sfx}.png")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")
    try:
        torch.save(model.state_dict(), results_dir / f"model{sfx}.pt")
    except Exception as e:  # noqa: BLE001
        print(f"checkpoint save skipped: {e}")

    anchors = cfg.get("anchors", {})
    v3_anchor = float(anchors.get("foundation_v3_val_auc", 0.6462))
    iso_teambias_anchor = float(anchors.get("iso_teambias_val_auc", 0.6493))
    embedding_prelim_anchor = float(anchors.get("embedding_prelim_val_auc", 0.6476))
    cleanup_anchor = float(anchors.get("cleanup_anchor_val_auc", 0.6477054))
    target = float(anchors.get("proposal_target_v3_ablations_val_auc", 0.6485))
    baseline_repro_anchor = float(anchors.get("baseline_multitask_repro_anchor", 0.6470))

    metrics = {
        "ablation": args.ablation,
        "smoke": bool(args.smoke),
        "use_features": use_features,
        "use_patch_token": use_patch_token,
        "use_team_team_bias": use_team_team_bias,
        "use_pmae": use_pmae,
        "use_uw_so": False,
        "dur_loss_mode": dur_loss_mode,
        "use_player_embedding": use_player_embedding,
        "player_vocab_size": int(player_vocab_size),
        "player_embed_dim": int(player_cfg.get("embed_dim", 128)) if use_player_embedding else None,
        "n_hash_buckets": int(n_hash_buckets),
        "hash_base_idx": int(hash_base_idx),
        "val_auc": float(tr.best_val_auc),
        "val_win_log_loss": float(tr.best_val_win_loss),
        "val_metrics_at_best": tr.val_metrics_at_best,
        "best_epoch": int(tr.best_epoch),
        "epochs_run": int(tr.epochs_run),
        "max_epochs": int(max_epochs),
        "history": tr.history,
        "model_hp": {k: mhp.get(k) for k in ("embed_dim", "d_model", "n_heads",
                                              "n_layers", "ff_mult", "dropout",
                                              "decoder_n_layers", "decoder_n_heads",
                                              "remove_first_layer_first_ln",
                                              "use_canonical_sort")},
        "optim_hp": {"batch_size": hp["batch_size"], "lr": hp["lr"],
                     "weight_decay": hp["weight_decay"],
                     "warmup_steps": int(opt_cfg.get("warmup_steps", 1000)),
                     "cosine_min_lr": float(opt_cfg.get("cosine_min_lr", 1e-5)),
                     "mixed_precision": bool(opt_cfg["mixed_precision"])},
        "pmae_cfg": cfg["pmae"],
        "multitask_alpha": alpha_full,
        "param_counts": pc,
        "task_names": extras["task_names"],
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
        "canonical_sort": meta.get("canonical_sort"),
        "default_patch_id": meta.get("default_patch_id"),
        "train_patch_id_distribution": meta.get("train_patch_id_distribution"),
        "val_patch_id_distribution": meta.get("val_patch_id_distribution"),
        "account_idx_train_stats": meta.get("account_idx_train_stats"),
        "account_idx_val_stats": meta.get("account_idx_val_stats"),
        "player_vocab_meta": meta.get("player_vocab_meta"),
        "anchors": anchors,
        "delta_vs_v3": float(tr.best_val_auc - v3_anchor),
        "delta_vs_iso_teambias": float(tr.best_val_auc - iso_teambias_anchor),
        "delta_vs_embedding_prelim": float(tr.best_val_auc - embedding_prelim_anchor),
        "delta_vs_cleanup_anchor": float(tr.best_val_auc - cleanup_anchor),
        "delta_vs_proposal_target": float(tr.best_val_auc - target),
        "delta_vs_baseline_multitask_repro_anchor":
            float(tr.best_val_auc - baseline_repro_anchor),
        "coverage_bucket_val_auc": cov_info,
        "a2_diagnostics": a2_diag,
        "calibration": cal,
    }
    out_name = f"metrics{sfx}.json"
    out_path = EXP_DIR / out_name
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
