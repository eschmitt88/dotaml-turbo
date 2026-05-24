"""Train FoundationTransformer for one ablation in
foundation-component-isolation-740.

Component-isolation ablations selectable via --ablation:
  - iso_uwso       : baseline_multitask_repro + UW-SO (no PMAE, no patch
                       token, no team-team bias). Tests whether UW-SO alone
                       destabilizes.
  - iso_pmae       : baseline_multitask_repro + PMAE auxiliary objective
                       with EMA-teacher (bug-fix vs foundation-mvp-740).
  - iso_teambias   : baseline_multitask_repro + (team_q, team_k) attention
                       bias only.

Plus the original baseline for reference (smoke only by default):
  - baseline_multitask_repro : known-working 5M anchor (val_auc=0.6470).

Bug-fixes relative to foundation-mvp-740:
  - PMAE teacher is now an EMA copy (was: shared weights -> collapse).
  - UW-SO normalizes per-task loss by initial-epoch L_k_init (was: raw
    losses -> low-magnitude tasks dominated).

Optimizer: Adam, lr=1e-3 with 1000-step warmup then cosine to 1e-5.
bf16 autocast on CUDA. Early-stop on val_win_log_loss with patience.
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
from loss import UWSO  # noqa: E402
from mae import (  # noqa: E402
    EMATeacher,
    PMAEMasker,
    pmae_reconstruction_loss,
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
                     use_patch_token: bool, n_dur_buckets: int) -> dict:
    model.eval()
    bce_sum = nn.BCEWithLogitsLoss(reduction="sum")
    bce_item = nn.BCEWithLogitsLoss(reduction="sum")
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    n = 0
    tot_w = tot_d = tot_i = 0.0
    tot_kda = tot_gpm = tot_hd = 0.0
    tot_kda_n = tot_gpm_n = tot_hd_n = 0
    ys_w, ps_w = [], []
    ys_d, ps_d = [], []
    item_sub_t, item_sub_l = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in loader:
            (hero_ids, pf, patch_id, y_win, y_dur, y_item, y_kda, y_gpm, y_hd) = batch
            hero_ids = hero_ids.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            patch_id = patch_id.to(device, non_blocking=True)
            y_win = y_win.to(device, non_blocking=True)
            y_dur = y_dur.to(device, non_blocking=True)
            y_item = y_item.to(device, non_blocking=True)
            y_kda = y_kda.to(device, non_blocking=True)
            y_gpm = y_gpm.to(device, non_blocking=True)
            y_hd = y_hd.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                out = model(hero_ids, pf if use_features else None,
                            patch_id=patch_id if use_patch_token else None)
                win_logits = torch.nan_to_num(out["win"].float(), nan=0.0,
                                                posinf=50.0, neginf=-50.0)
                l_w = bce_sum(win_logits, y_win.float())
                l_d = ce_sum(out["dur"].float(), y_dur)
                l_i = bce_item(out["item"].float(), y_item.float())
                # KDA/GPM/HD per-slot scalars.
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
            tot_d += l_d.item()
            tot_i += l_i.item()
            tot_kda += l_kda.item(); tot_kda_n += n_kda
            tot_gpm += l_gpm.item(); tot_gpm_n += n_gpm
            tot_hd += l_hd.item();   tot_hd_n += n_hd
            ps_w.append(torch.sigmoid(win_logits).cpu().numpy())
            ys_w.append(y_win.cpu().numpy())
            ps_d.append(out["dur"].float().softmax(dim=-1).cpu().numpy())
            ys_d.append(y_dur.cpu().numpy())
            if len(item_sub_t) * bsize * 10 < 500_000:
                idx = rng.choice(bsize, size=min(bsize, 64), replace=False)
                t_np = y_item[idx].cpu().numpy().reshape(-1, y_item.shape[-1])
                l_np = out["item"][idx].float().cpu().numpy().reshape(-1, y_item.shape[-1])
                item_sub_t.append(t_np)
                item_sub_l.append(l_np)
    item_v = int(getattr(model, "item_vocab_size", 1)) or 1
    val_losses = {
        "win_log_loss_per_row": tot_w / max(n, 1),
        "dur_ce_per_row": tot_d / max(n, 1),
        "item_bce_per_slot_per_class": tot_i / max(n * 10 * item_v, 1),
        "kda_smoothl1_per_slot": tot_kda / max(tot_kda_n, 1),
        "gpm_smoothl1_per_slot": tot_gpm / max(tot_gpm_n, 1),
        "hd_smoothl1_per_slot":  tot_hd / max(tot_hd_n, 1),
    }
    return {
        "win_y": np.concatenate(ys_w), "win_p": np.concatenate(ps_w),
        "dur_y": np.concatenate(ys_d), "dur_p": np.concatenate(ps_d),
        "item_subsample_targets": (np.concatenate(item_sub_t, axis=0) if item_sub_t
                                     else np.zeros((0, 0))),
        "item_subsample_logits": (np.concatenate(item_sub_l, axis=0) if item_sub_l
                                    else np.zeros((0, 0))),
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


def train_one_ablation(model, train_ds, val_ds, hp: dict, max_epochs: int,
                         device, base_rate_val: float | None,
                         use_features: bool, use_patch_token: bool,
                         mixed_precision: bool, patience: int | None,
                         alpha: dict, n_dur_buckets: int,
                         use_uw_so: bool, uw_so_cfg: dict,
                         use_pmae: bool, pmae_cfg: dict,
                         warmup_steps: int, cosine_min_lr: float) -> tuple[TrainResult, dict]:
    """Single ablation training loop. Returns (TrainResult, extras_for_logging)."""
    bs = int(hp["batch_size"]); lr = float(hp["lr"]); wd = float(hp["weight_decay"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs)
    autocast_dtype = torch.bfloat16 if (mixed_precision and device.type == "cuda") else None

    # Loss weighting.
    # Tasks order: [win, dur, item, kda, gpm, hd]
    task_names = ["win", "dur", "item", "kda", "gpm", "hd"]
    n_tasks = len(task_names)
    uw_so = UWSO(n_tasks=n_tasks, T_init=float(uw_so_cfg.get("T_init", 1.0)),
                  learnable_T=bool(uw_so_cfg.get("learnable_T", True))).to(device) if use_uw_so else None

    # Fixed alphas as the fallback weighting when use_uw_so=False.
    fixed_alpha = torch.tensor([
        float(alpha["alpha_win"]), float(alpha["alpha_dur"]), float(alpha["alpha_item"]),
        float(alpha["alpha_kda"]), float(alpha["alpha_gpm"]), float(alpha["alpha_hd"]),
    ], device=device)

    # Optimizer: include UW-SO temperature if learnable.
    params = list(model.parameters())
    if uw_so is not None:
        params = params + list(uw_so.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    # PMAE masker (constructed even when off so we can report stats).
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

    # EMA teacher (bug-fix Bug A from foundation-mvp-740). Only constructed
    # when PMAE is on -- otherwise it would be wasted memory.
    ema_teacher: EMATeacher | None = None
    if use_pmae:
        ema_teacher = EMATeacher(model, momentum=ema_momentum).to(device)
        print(f"  EMA teacher constructed momentum={ema_momentum}")

    bce_w = nn.BCEWithLogitsLoss()
    bce_i = nn.BCEWithLogitsLoss()
    ce_d = nn.CrossEntropyLoss()

    total_steps = max(max_epochs * max(1, math.ceil(len(train_ds) / bs)), 1)

    history = []
    best_val_loss = math.inf
    best_val_auc = -math.inf
    best_state = None
    best_epoch = -1
    best_eval: dict = {}
    best_y_w = best_p_w = np.array([])
    epochs_since_improve = 0
    global_step = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        n_seen = 0
        sum_w = sum_d = sum_i = sum_kda = sum_gpm = sum_hd = sum_mae = 0.0
        sum_total = 0.0
        omega_running = torch.zeros(n_tasks, device=device)
        n_batches = 0
        # PMAE epoch-level diagnostics (Bug A logging).
        sum_hero_mask_count = 0
        sum_hero_mask_frac = 0.0
        sum_patch_mask_count = 0
        sum_patch_mask_frac = 0.0
        sum_student_l2 = 0.0
        sum_teacher_l2 = 0.0
        n_mae_batches = 0
        ep_t0 = time.time()
        for batch in train_loader:
            (hero_ids, pf, patch_id, y_win, y_dur, y_item, y_kda, y_gpm, y_hd) = batch
            hero_ids = hero_ids.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            patch_id = patch_id.to(device, non_blocking=True)
            y_win = y_win.to(device, non_blocking=True)
            y_dur = y_dur.to(device, non_blocking=True)
            y_item = y_item.to(device, non_blocking=True)
            y_kda = y_kda.to(device, non_blocking=True)
            y_gpm = y_gpm.to(device, non_blocking=True)
            y_hd = y_hd.to(device, non_blocking=True)

            cur_lr = warmup_cosine_lr(global_step, warmup_steps, total_steps, lr, cosine_min_lr)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            cur_alpha_mae = alpha_mae_schedule(global_step, total_steps,
                                                  alpha_mae_start, alpha_mae_end)

            # PMAE masking (training-only).
            mask_out = None
            if use_pmae:
                # Use is_anonymous feature column (index 7 by default) as the per-slot anonymity proxy.
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
                            hero_mask=hero_mask, patch_mask=patch_mask)

                l_w = bce_w(out["win"].float(), y_win.float())
                l_d = ce_d(out["dur"].float(), y_dur)
                l_i = bce_i(out["item"].float(), y_item.float())
                l_kda = _masked_smooth_l1_mean(out["kda"].float(), y_kda.float())
                l_gpm = _masked_smooth_l1_mean(out["gpm"].float(), y_gpm.float())
                l_hd = _masked_smooth_l1_mean(out["hd"].float(), y_hd.float())
                losses = torch.stack([l_w, l_d, l_i, l_kda, l_gpm, l_hd])

                # Supervised combined loss.
                if uw_so is not None:
                    sup_loss, omega = uw_so(losses)
                else:
                    omega = fixed_alpha
                    sup_loss = (fixed_alpha * losses).sum()

                # PMAE reconstruction: teacher pass = EMA-tracked copy with
                # stop-gradient (bug-fix for foundation-mvp-740 Bug A).
                l_mae = torch.tensor(0.0, device=device)
                pmae_log = None
                if use_pmae and mask_out is not None and ema_teacher is not None:
                    with torch.no_grad():
                        teacher_out = ema_teacher(
                            hero_ids, pf if use_features else None,
                            patch_id=patch_id if use_patch_token else None,
                            hero_mask=None, patch_mask=None,
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

            # EMA update AFTER opt.step so teacher tracks the just-updated student.
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
            omega_running = omega_running + omega.detach()
            n_batches += 1
            global_step += 1
            # PMAE diagnostics accumulation.
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
        omega_avg = (omega_running / max(n_batches, 1)).cpu().tolist()
        T_now = float(uw_so.T.item()) if uw_so is not None else None
        # PMAE per-epoch diagnostic means.
        avg_hero_mask_count = sum_hero_mask_count / max(n_mae_batches, 1)
        avg_hero_mask_frac = sum_hero_mask_frac / max(n_mae_batches, 1)
        avg_patch_mask_count = sum_patch_mask_count / max(n_mae_batches, 1)
        avg_patch_mask_frac = sum_patch_mask_frac / max(n_mae_batches, 1)
        avg_student_l2 = sum_student_l2 / max(n_mae_batches, 1)
        avg_teacher_l2 = sum_teacher_l2 / max(n_mae_batches, 1)

        eval_out = _eval_multitask(model, val_loader, device, autocast_dtype,
                                     use_features=use_features,
                                     use_patch_token=use_patch_token,
                                     n_dur_buckets=n_dur_buckets)
        vl = eval_out["val_losses"]
        val_win_loss = vl["win_log_loss_per_row"]
        m_w = metrics_block(eval_out["win_y"], eval_out["win_p"], base_rate=base_rate_val)
        dur_pred_top1 = eval_out["dur_p"].argmax(axis=-1)
        dur_acc = float((dur_pred_top1 == eval_out["dur_y"]).mean())
        item_metrics = item_map_at_k(eval_out["item_subsample_targets"],
                                       eval_out["item_subsample_logits"], k=10)
        ep_dt = time.time() - ep_t0
        print(f"  epoch {epoch+1}/{max_epochs}  "
              f"tr[w={tr_w:.4f} d={tr_d:.4f} i={tr_i:.5f} kda={tr_kda:.4f} "
              f"gpm={tr_gpm:.4f} hd={tr_hd:.4f} mae={tr_mae:.4f} tot={tr_total:.4f}]  "
              f"vl_win={val_win_loss:.4f}  val_auc={m_w['auc']:.4f}  "
              f"dur_acc={dur_acc:.4f}  itemMAP={item_metrics['map_at_k']}  "
              f"lr={cur_lr:.2e}  T={T_now}  alpha_mae={cur_alpha_mae:.3f}  ({ep_dt:.1f}s)")
        if use_uw_so:
            print(f"    omega(win,dur,item,kda,gpm,hd) = "
                  f"[{omega_avg[0]:.3f}, {omega_avg[1]:.3f}, {omega_avg[2]:.3f}, "
                  f"{omega_avg[3]:.3f}, {omega_avg[4]:.3f}, {omega_avg[5]:.3f}]")
        if use_pmae:
            print(f"    pmae: hero_mask_frac={avg_hero_mask_frac:.3f} "
                  f"(count={avg_hero_mask_count:.1f}) "
                  f"patch_mask_frac={avg_patch_mask_frac:.3f} "
                  f"s_l2={avg_student_l2:.4f} t_l2={avg_teacher_l2:.4f}")
        history.append({
            "epoch": epoch + 1,
            "train_win_loss": tr_w, "train_dur_loss": tr_d, "train_item_loss": tr_i,
            "train_kda_loss": tr_kda, "train_gpm_loss": tr_gpm, "train_hd_loss": tr_hd,
            "train_mae_loss": tr_mae, "train_total_loss": tr_total,
            "val_win_log_loss": float(val_win_loss),
            "val_win_auc": float(m_w["auc"]),
            "val_win_brier": float(m_w["brier"]),
            "val_dur_ce_per_row": float(vl["dur_ce_per_row"]),
            "val_dur_acc": float(dur_acc),
            "val_item_bce": float(vl["item_bce_per_slot_per_class"]),
            "val_item_map_at_10": item_metrics["map_at_k"],
            "val_kda_smoothl1": float(vl["kda_smoothl1_per_slot"]),
            "val_gpm_smoothl1": float(vl["gpm_smoothl1_per_slot"]),
            "val_hd_smoothl1":  float(vl["hd_smoothl1_per_slot"]),
            "omega_avg": omega_avg,
            "uw_so_T": T_now,
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
        })
        if val_win_loss < best_val_loss - 1e-6:
            best_val_loss = val_win_loss
            best_val_auc = m_w["auc"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_eval = {"win": m_w, "dur_top1_acc": dur_acc,
                          "item_mAP_at_10": item_metrics,
                          "val_component_losses": vl}
            best_y_w, best_p_w = eval_out["win_y"], eval_out["win_p"]
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
              "uw_so_used": use_uw_so,
              "pmae_used": use_pmae,
              "warmup_steps": warmup_steps,
              "total_steps": total_steps}
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
                    choices=["baseline_multitask_repro",
                              "iso_uwso", "iso_pmae", "iso_teambias",
                              "foundation_mvp", "foundation_no_patch_token"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--metrics-suffix", default="")
    ap.add_argument("--max-epochs-override", type=int, default=None,
                    help="override config max_epochs (e.g. for profiling).")
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
    use_uw_so = bool(ab_spec.get("use_uw_so", True))

    feat_names = cfg["player_features_transformer"]["feat_names"]
    n_player_feats = int(cfg["player_features_transformer"]["n_player_feats"])
    source_dir = PROJECT_ROOT / cfg["player_features_transformer"]["source_dir"]
    sidecar_dir = PROJECT_ROOT / cfg["rich_cols"]["out_dir"]
    # vocab_path may be relative to EXP_DIR ("../..."); resolve from there.
    vp = cfg["item_vocab"]["vocab_path"]
    vocab_path = (EXP_DIR / vp).resolve() if not Path(vp).is_absolute() else Path(vp)
    aux_targets = cfg["multitask_loss"]["aux_targets"]
    n_dur_buckets = int(cfg["duration_bucket"]["n_buckets"])

    print(f"Ablation: {args.ablation} "
          f"(features={use_features} patch={use_patch_token} team_bias={use_team_team_bias} "
          f"pmae={use_pmae} uw_so={use_uw_so})")
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
        )
    else:
        train_ds, val_ds, meta = load_train_val(
            seed=seed, n_target=n_target, feat_names=feat_names,
            source_dir=source_dir, splits=splits, smoke=False,
            sidecar_dir=sidecar_dir, vocab_path=vocab_path,
            aux_targets=aux_targets, canonical_sort=canonical_sort,
            default_patch_id=default_patch_id,
        )
    data_seconds = time.time() - t0
    print(f"Data ready in {data_seconds:.1f}s -- train={len(train_ds):,} val={len(val_ds):,}")
    print(f"  train dates {meta['train_date_min']}..{meta['train_date_max']}")
    print(f"  val   dates {meta['val_date_min']}..{meta['val_date_max']}")
    print(f"  canonical_sort={meta['canonical_sort']} default_patch_id={meta['default_patch_id']}")

    mhp = cfg["transformer_model"]
    item_vocab_size = int(meta.get("item_vocab_size", 0)) or 1
    patch_vocab_size = int(cfg["patch"]["vocab_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mhp, vocab_size=int(cfg["hero"]["vocab_size"]),
                          n_player_feats=n_player_feats, use_features=use_features,
                          n_dur_buckets=n_dur_buckets,
                          item_vocab_size=item_vocab_size,
                          patch_vocab_size=patch_vocab_size,
                          use_team_team_bias=use_team_team_bias,
                          use_patch_token=use_patch_token,
                          use_lobby_token=bool(mhp.get("use_lobby_token", False)))
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
    # Patch alpha names to include kda/gpm/hd from config or default.
    alpha_full = {
        "alpha_win": float(alpha["alpha_win"]),
        "alpha_dur": float(alpha["alpha_dur"]),
        "alpha_item": float(alpha["alpha_item"]),
        "alpha_kda": float(alpha.get("alpha_kda", alpha.get("alpha_aux", 0.1))),
        "alpha_gpm": float(alpha.get("alpha_gpm", alpha.get("alpha_aux", 0.1))),
        "alpha_hd":  float(alpha.get("alpha_hd",  alpha.get("alpha_aux", 0.1))),
    }

    tr, extras = train_one_ablation(
        model, train_ds, val_ds, hp,
        max_epochs=max_epochs, device=device, base_rate_val=base_rate_val,
        use_features=use_features, use_patch_token=use_patch_token,
        mixed_precision=bool(opt_cfg["mixed_precision"]),
        patience=patience, alpha=alpha_full, n_dur_buckets=n_dur_buckets,
        use_uw_so=use_uw_so, uw_so_cfg=cfg["uw_so"],
        use_pmae=use_pmae, pmae_cfg=cfg["pmae"],
        warmup_steps=int(opt_cfg.get("warmup_steps", 1000)),
        cosine_min_lr=float(opt_cfg.get("cosine_min_lr", 1e-5)),
    )
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
        plot_learning(tr.history, results_dir / f"learning_curve{sfx}.png")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")
    try:
        torch.save(model.state_dict(), results_dir / f"model{sfx}.pt")
    except Exception as e:  # noqa: BLE001
        print(f"checkpoint save skipped: {e}")

    anchors = cfg.get("anchors", {})
    cleanup_anchor = float(anchors.get("cleanup_anchor_val_auc", 0.6477054))
    target = float(anchors.get("proposal_target_val_auc", 0.6525))
    multitask_anchor = float(anchors.get("rich_supervision_multitask_740_val_auc", 0.6495))
    baseline_repro_anchor = float(anchors.get("baseline_multitask_repro_anchor", 0.6470))

    metrics = {
        "ablation": args.ablation,
        "smoke": bool(args.smoke),
        "use_features": use_features,
        "use_patch_token": use_patch_token,
        "use_team_team_bias": use_team_team_bias,
        "use_pmae": use_pmae,
        "use_uw_so": use_uw_so,
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
        "uw_so_cfg": cfg["uw_so"],
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
        "anchors": anchors,
        "delta_vs_cleanup_anchor": float(tr.best_val_auc - cleanup_anchor),
        "delta_vs_multitask_anchor": float(tr.best_val_auc - multitask_anchor),
        "delta_vs_proposal_target": float(tr.best_val_auc - target),
        "delta_vs_baseline_multitask_repro_anchor":
            float(tr.best_val_auc - baseline_repro_anchor),
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
