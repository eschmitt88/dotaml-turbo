"""Three-phase training entry point for v5-pretrain-finetune-740.

Usage:
  python -u train.py --phase pretrain   [--smoke]
  python -u train.py --phase probe      [--smoke]
  python -u train.py --phase finetune   [--smoke]

Phase 1 (pretrain) writes results/pretrain_encoder.pt + per-group loss
history + mid-pretrain probe history.

Phase 2A (probe) loads the encoder, freezes it, trains a single linear
win head with KDA/GPM/HD/items FULLY MASKED via mask tokens. Writes
metrics_linear_probe.json.

Phase 2B (finetune) loads the encoder, UNFREEZES it, attaches v4-style
multi-task heads, trains with two-param-group AdamW (encoder lr=1e-5,
heads lr=1e-3). KDA/GPM/HD/items still FULLY MASKED at the encoder;
their PREDICTION heads are trained from rich_cols targets (same as v4).
Writes metrics_finetune.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader  # noqa: E402

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

from data import load_train_val  # noqa: E402
from mae import (  # noqa: E402
    EMATeacherV5,
    PRETRAIN_GROUPS,
    SixGroupMasker,
    per_group_reconstruction_losses,
)
from models import (  # noqa: E402
    build_model_v5,
    count_params,
    encoder_param_names,
)


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


def per_patch_val_auc(y_win, p_win, patch_ids) -> dict:
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


def coverage_bucket_val_auc(val_ds, y_val, p_val, feat_names) -> dict:
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
        try:
            auc_b = float(roc_auc_score(y_val[mask], p_val[mask]))
        except ValueError:
            auc_b = None
        bucket_aucs[name] = {"n": n, "val_auc": auc_b,
                              "mean_coverage_log1p": float(coverage[mask].mean())}
    return {"quantile_edges_log1p": [float(q33), float(q67)], "buckets": bucket_aucs}


def anchors_deltas(val_auc: float, anchors: dict) -> dict:
    """Compute deltas vs every anchor in the config block."""
    out: dict = {"anchors": dict(anchors)}
    name_map = [
        ("delta_vs_v4",                "v4_val_auc"),
        ("delta_vs_iso_teambias",      "iso_teambias_val_auc"),
        ("delta_vs_v3",                "v3_val_auc"),
        ("delta_vs_cleanup",           "cleanup_anchor_val_auc"),
        ("delta_vs_baseline_repro",    "baseline_multitask_repro_val_auc"),
        ("delta_vs_iso_pmae",          "iso_pmae_val_auc"),
        ("delta_vs_target",            "proposal_target_val_auc"),
    ]
    for delta_key, anchor_key in name_map:
        if anchor_key in anchors:
            out[delta_key] = float(val_auc - float(anchors[anchor_key]))
    return out


# =============================================================================
# Phase 1: self-supervised pre-train
# =============================================================================

def run_pretrain(cfg: dict, splits: dict, smoke: bool) -> int:
    set_seed(int(cfg["seed"]))
    pre_cfg = cfg["pretrain"]
    if smoke:
        pre_cfg = {**pre_cfg, **cfg["smoke"]["pretrain"]}

    print(f"[pretrain] smoke={smoke} max_epochs={pre_cfg['max_epochs']} "
          f"p_group={pre_cfg['p_group']} ema={pre_cfg['ema_momentum']}")

    # Load data.
    feat_names = cfg["player_features_transformer"]["feat_names"]
    n_player_feats = int(cfg["player_features_transformer"]["n_player_feats"])
    source_dir = PROJECT_ROOT / cfg["player_features_transformer"]["source_dir"]
    sidecar_dir = PROJECT_ROOT / cfg["rich_cols"]["out_dir"]
    vp = cfg["item_vocab"]["vocab_path"]
    vocab_path = (EXP_DIR / vp).resolve() if not Path(vp).is_absolute() else Path(vp)
    aux_targets = cfg["multitask_loss"]["aux_targets"]
    canonical_sort = bool(cfg["transformer_model"].get("use_canonical_sort", True))
    default_patch_id = int(cfg["patch"].get("default_patch_id", 1))

    t0 = time.time()
    smoke_n_train = int(cfg["smoke"]["pretrain"].get("n_train", 50_000))
    smoke_n_val = int(cfg["smoke"]["pretrain"].get("n_val", 5_000))
    train_ds, val_ds, meta = load_train_val(
        seed=int(cfg["seed"]),
        n_target=int(cfg["train_subset_size"]),
        feat_names=feat_names, source_dir=source_dir, splits=splits,
        smoke=smoke, smoke_n_train=smoke_n_train, smoke_n_val=smoke_n_val,
        sidecar_dir=sidecar_dir, vocab_path=vocab_path,
        aux_targets=aux_targets, canonical_sort=canonical_sort,
        default_patch_id=default_patch_id,
    )
    print(f"[pretrain] data ready in {time.time() - t0:.1f}s — "
          f"train={len(train_ds):,} val={len(val_ds):,}")

    # Build model.
    mhp = cfg["transformer_model"]
    item_vocab_size = int(meta.get("item_vocab_size", 0)) or 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_v5(mhp, vocab_size=int(cfg["hero"]["vocab_size"]),
                            n_player_feats=n_player_feats,
                            n_dur_buckets=int(cfg["duration_bucket"]["n_buckets"]),
                            item_vocab_size=item_vocab_size,
                            use_team_team_bias=bool(mhp.get("use_team_team_bias", True)),
                            dur_loss_mode=cfg.get("dur_loss_mode", "ce"))
    model = model.to(device)
    print(f"[pretrain] model: {count_params(model)}")

    masker = SixGroupMasker(p_group=float(pre_cfg["p_group"]),
                              groups=list(pre_cfg["groups"]))
    ema = EMATeacherV5(model, momentum=float(pre_cfg["ema_momentum"])).to(device)
    print(f"[pretrain] EMA teacher constructed momentum={pre_cfg['ema_momentum']}")

    bs = int(pre_cfg["batch_size"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs)
    opt = torch.optim.Adam(model.parameters(), lr=float(pre_cfg["lr"]),
                            weight_decay=float(pre_cfg["weight_decay"]))
    autocast_dtype = (torch.bfloat16 if (bool(pre_cfg["mixed_precision"])
                                          and device.type == "cuda") else None)
    lw = pre_cfg["loss_weights"]
    weights = {g: float(lw[g]) for g in PRETRAIN_GROUPS}

    total_steps = max(int(pre_cfg["max_epochs"]) * max(1, math.ceil(len(train_ds) / bs)), 1)
    warmup_steps = int(pre_cfg["warmup_steps"])
    base_lr = float(pre_cfg["lr"])
    cos_min_lr = float(pre_cfg["cosine_min_lr"])

    history: list = []
    probe_history: list = []
    probe_at = set(int(e) for e in pre_cfg["probe_at_epochs"])
    global_step = 0

    for epoch in range(int(pre_cfg["max_epochs"])):
        model.train()
        ep_t0 = time.time()
        sum_w_loss = {g: 0.0 for g in PRETRAIN_GROUPS}
        sum_total = 0.0
        sum_n_mask = {g: 0 for g in PRETRAIN_GROUPS}
        n_batches = 0
        avg_mask_rate = {g: 0.0 for g in PRETRAIN_GROUPS}
        n_seen = 0
        for batch in train_loader:
            (hero_ids, pf, _patch_id, sc_in,
             y_win, y_dur, y_dur_bucket, items,
             y_kda_t, y_gpm_t, y_hd_t) = batch
            hero_ids = hero_ids.to(device, non_blocking=True)
            pf = pf.to(device, non_blocking=True)
            sc_in = sc_in.to(device, non_blocking=True)
            items = items.to(device, non_blocking=True)
            B = hero_ids.size(0)
            n_seen += B

            cur_lr = warmup_cosine_lr(global_step, warmup_steps, total_steps,
                                        base_lr, cos_min_lr)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            mask_dict = masker(B, device)
            for g in PRETRAIN_GROUPS:
                avg_mask_rate[g] += float(mask_dict[g].float().mean().item())

            # Target inputs (UN-masked raw values).
            target_inputs = {
                "player_block": pf,                       # [B, 10, F]
                "hero_token":   hero_ids,                 # [B, 10] long
                "item_list":    items,                    # [B, 10, V]
                "kda":          sc_in[:, :, 0],           # [B, 10]
                "gpm":          sc_in[:, :, 1],
                "hd":           sc_in[:, :, 2],
            }

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                pred = model.forward_pretrain(hero_ids, pf, items, sc_in, mask_dict)
                losses = per_group_reconstruction_losses(pred, target_inputs, mask_dict)
                total = pred["encoded"].sum() * 0.0
                for g in PRETRAIN_GROUPS:
                    total = total + weights[g] * losses[g]

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            ema.update(model)

            for g in PRETRAIN_GROUPS:
                sum_w_loss[g] += float(losses[g].item()) * B
                sum_n_mask[g] += int(losses["_n_mask"][g])
            sum_total += float(total.item()) * B
            n_batches += 1
            global_step += 1

        ep_dt = time.time() - ep_t0
        avg_loss = {g: sum_w_loss[g] / max(n_seen, 1) for g in PRETRAIN_GROUPS}
        avg_mask_rate = {g: avg_mask_rate[g] / max(n_batches, 1) for g in PRETRAIN_GROUPS}
        ep_summary = (f"[pretrain] epoch {epoch+1}/{pre_cfg['max_epochs']}  "
                      f"total={sum_total / max(n_seen, 1):.4f}  "
                      f"lr={cur_lr:.2e}  ({ep_dt:.1f}s)")
        print(ep_summary)
        print("    per-group losses: " +
              " ".join(f"{g}={avg_loss[g]:.4f}" for g in PRETRAIN_GROUPS))
        print("    mask rates: " +
              " ".join(f"{g}={avg_mask_rate[g]:.3f}" for g in PRETRAIN_GROUPS))

        ep_record = {
            "epoch": epoch + 1,
            "per_group_loss": avg_loss,
            "per_group_mask_rate": avg_mask_rate,
            "per_group_n_mask": sum_n_mask,
            "total_loss": sum_total / max(n_seen, 1),
            "lr": float(cur_lr),
            "wall_seconds": float(ep_dt),
        }
        history.append(ep_record)

        # Mid-pretrain probe?
        if (epoch + 1) in probe_at:
            probe_t0 = time.time()
            probe_auc = run_mid_probe(model, train_ds, val_ds, device, pre_cfg, smoke=smoke)
            probe_dt = time.time() - probe_t0
            print(f"    mid_probe[epoch {epoch+1}]: val_auc={probe_auc:.4f} ({probe_dt:.1f}s)")
            probe_history.append({"epoch": epoch + 1,
                                    "mid_probe_val_auc": float(probe_auc),
                                    "wall_seconds": float(probe_dt)})

    # Save encoder + histories.
    results_dir = EXP_DIR / cfg["output"]["results_dir"]
    results_dir.mkdir(exist_ok=True, parents=True)
    enc_out = EXP_DIR / pre_cfg["output_encoder"]
    enc_out.parent.mkdir(exist_ok=True, parents=True)
    if smoke:
        enc_out = enc_out.with_name(enc_out.stem + "_smoke.pt")
    torch.save(model.state_dict(), enc_out)
    print(f"[pretrain] wrote {enc_out}")

    hist_out = EXP_DIR / pre_cfg["output_history"]
    if smoke:
        hist_out = hist_out.with_name(hist_out.stem + "_smoke.json")
    hist_out.write_text(json.dumps({"history": history,
                                       "meta": {k: meta[k] for k in
                                                 ("train_date_min", "train_date_max",
                                                  "val_date_min", "val_date_max",
                                                  "n_train_post_subsample", "n_val",
                                                  "scalar_inputs_train_mean",
                                                  "scalar_inputs_train_std",
                                                  "scalar_input_names",
                                                  "item_vocab_size",
                                                  "train_patch_id_distribution",
                                                  "val_patch_id_distribution")}},
                                       indent=2))
    print(f"[pretrain] wrote {hist_out}")
    probe_out = EXP_DIR / pre_cfg["output_probe_history"]
    if smoke:
        probe_out = probe_out.with_name(probe_out.stem + "_smoke.json")
    probe_out.write_text(json.dumps({"mid_probe_history": probe_history}, indent=2))
    print(f"[pretrain] wrote {probe_out}")
    return 0


def run_mid_probe(model, train_ds, val_ds, device, pre_cfg, smoke: bool = False) -> float:
    """Linear-probe diagnostic during pre-training.

    1. Snapshot encoder state (we'll restore at end so pre-training is unaffected).
    2. Subsample probe_n_train rows from train_ds and probe_n_val from val_ds.
    3. Construct a fresh nn.Linear(d_model, 1) head.
    4. Freeze encoder, train ONLY the head for probe_epochs at probe_lr.
    5. Eval val_auc, return it.
    """
    rng = np.random.default_rng(0)
    n_tr = min(int(pre_cfg["probe_n_train"]), len(train_ds))
    n_va = min(int(pre_cfg["probe_n_val"]), len(val_ds))
    tr_idx = rng.choice(len(train_ds), size=n_tr, replace=False)
    va_idx = rng.choice(len(val_ds), size=n_va, replace=False)
    tr_sub = torch.utils.data.Subset(train_ds, tr_idx.tolist())
    va_sub = torch.utils.data.Subset(val_ds, va_idx.tolist())

    bs = max(int(pre_cfg["batch_size"]), 256)
    tr_ld = DataLoader(tr_sub, batch_size=bs, shuffle=True, num_workers=0)
    va_ld = DataLoader(va_sub, batch_size=bs * 2, shuffle=False, num_workers=0)

    d_model = model.d_model
    head = nn.Linear(d_model, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=float(pre_cfg["probe_lr"]))
    bce = nn.BCEWithLogitsLoss()
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    # Mask dict: all 4 rich groups FULLY MASKED (inference-distribution match).
    def _inf_mask(B: int) -> dict:
        # During the probe we mask item_list/kda/gpm/hd for ALL examples.
        # player_block and hero_token are always unmasked.
        zb = torch.zeros(B, dtype=torch.bool, device=device)
        ob = torch.ones(B, dtype=torch.bool, device=device)
        return {"player_block": zb.clone(), "hero_token": zb.clone(),
                "item_list": ob.clone(), "kda": ob.clone(),
                "gpm": ob.clone(), "hd": ob.clone()}

    # Freeze encoder.
    enc_state = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    try:
        for _ep in range(int(pre_cfg["probe_epochs"])):
            head.train()
            for batch in tr_ld:
                (hero_ids, pf, _patch_id, _sc_in,
                 y_win, _y_dur, _y_dur_bucket, _items,
                 _y_kda_t, _y_gpm_t, _y_hd_t) = batch
                hero_ids = hero_ids.to(device); pf = pf.to(device)
                y_win = y_win.to(device).float()
                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                     enabled=autocast_dtype is not None):
                    with torch.no_grad():
                        memory, _ = model.encode(hero_ids, pf,
                                                   items_input=None,
                                                   scalar_inputs=None,
                                                   mask_dict=_inf_mask(hero_ids.size(0)))
                        pooled = model.pooled(memory).float()
                    logits = head(pooled).squeeze(-1)
                    loss = bce(logits.float(), y_win)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        head.eval()
        ys = []; ps = []
        with torch.no_grad():
            for batch in va_ld:
                (hero_ids, pf, _patch_id, _sc_in,
                 y_win, _y_dur, _y_dur_bucket, _items,
                 _y_kda_t, _y_gpm_t, _y_hd_t) = batch
                hero_ids = hero_ids.to(device); pf = pf.to(device)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                     enabled=autocast_dtype is not None):
                    memory, _ = model.encode(hero_ids, pf, items_input=None,
                                                scalar_inputs=None,
                                                mask_dict=_inf_mask(hero_ids.size(0)))
                    pooled = model.pooled(memory).float()
                    logits = head(pooled).squeeze(-1).float()
                ps.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(y_win.cpu().numpy())
        y_arr = np.concatenate(ys); p_arr = np.concatenate(ps)
        try:
            auc = float(roc_auc_score(y_arr, p_arr))
        except ValueError:
            auc = float("nan")
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(enc_state.get(n, True))
        model.train()
    return auc


# =============================================================================
# Phase 2A: linear probe
# =============================================================================

def _build_inference_mask(B: int, device) -> dict:
    """All 4 rich groups FULLY MASKED; player_block + hero_token unmasked."""
    zb = torch.zeros(B, dtype=torch.bool, device=device)
    ob = torch.ones(B, dtype=torch.bool, device=device)
    return {"player_block": zb.clone(), "hero_token": zb.clone(),
            "item_list": ob.clone(), "kda": ob.clone(),
            "gpm": ob.clone(), "hd": ob.clone()}


def run_linear_probe(cfg: dict, splits: dict, smoke: bool) -> int:
    set_seed(int(cfg["seed"]))
    probe_cfg = cfg["linear_probe"]
    feat_names = cfg["player_features_transformer"]["feat_names"]
    n_player_feats = int(cfg["player_features_transformer"]["n_player_feats"])
    source_dir = PROJECT_ROOT / cfg["player_features_transformer"]["source_dir"]
    sidecar_dir = PROJECT_ROOT / cfg["rich_cols"]["out_dir"]
    vp = cfg["item_vocab"]["vocab_path"]
    vocab_path = (EXP_DIR / vp).resolve() if not Path(vp).is_absolute() else Path(vp)
    aux_targets = cfg["multitask_loss"]["aux_targets"]
    canonical_sort = bool(cfg["transformer_model"].get("use_canonical_sort", True))
    default_patch_id = int(cfg["patch"].get("default_patch_id", 1))

    t0 = time.time()
    smoke_n_train = int(cfg["smoke"]["finetune"].get("n_train", 50_000))
    smoke_n_val = int(cfg["smoke"]["finetune"].get("n_val", 5_000))
    train_ds, val_ds, meta = load_train_val(
        seed=int(cfg["seed"]),
        n_target=int(cfg["train_subset_size"]),
        feat_names=feat_names, source_dir=source_dir, splits=splits,
        smoke=smoke, smoke_n_train=smoke_n_train, smoke_n_val=smoke_n_val,
        sidecar_dir=sidecar_dir, vocab_path=vocab_path,
        aux_targets=aux_targets, canonical_sort=canonical_sort,
        default_patch_id=default_patch_id,
    )
    print(f"[probe] data ready in {time.time() - t0:.1f}s — "
          f"train={len(train_ds):,} val={len(val_ds):,}")

    mhp = cfg["transformer_model"]
    item_vocab_size = int(meta.get("item_vocab_size", 0)) or 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_v5(mhp, vocab_size=int(cfg["hero"]["vocab_size"]),
                            n_player_feats=n_player_feats,
                            n_dur_buckets=int(cfg["duration_bucket"]["n_buckets"]),
                            item_vocab_size=item_vocab_size,
                            use_team_team_bias=bool(mhp.get("use_team_team_bias", True)),
                            dur_loss_mode=cfg.get("dur_loss_mode", "ce"))
    enc_path = EXP_DIR / probe_cfg["encoder_path"]
    if smoke:
        enc_path = enc_path.with_name(enc_path.stem + "_smoke.pt")
    if not enc_path.exists():
        raise SystemExit(f"[probe] encoder checkpoint not found: {enc_path}")
    state = torch.load(enc_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"[probe] loaded encoder from {enc_path} "
          f"(missing={len(missing) if missing else 0} "
          f"unexpected={len(unexpected) if unexpected else 0})")
    model = model.to(device)

    # Freeze all model params.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    head = nn.Linear(model.d_model, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=float(probe_cfg["lr"]))
    bce = nn.BCEWithLogitsLoss()
    bs = int(probe_cfg["batch_size"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs)
    autocast_dtype = torch.bfloat16 if (bool(probe_cfg["mixed_precision"])
                                          and device.type == "cuda") else None
    max_epochs = int(probe_cfg["max_epochs"])
    if smoke:
        max_epochs = 1

    history = []
    best_auc = -math.inf
    best_epoch = -1
    for epoch in range(max_epochs):
        head.train()
        ep_t0 = time.time()
        tr_loss = 0.0; n_seen = 0
        for batch in train_loader:
            (hero_ids, pf, _patch_id, _sc_in,
             y_win, _y_dur, _y_dur_bucket, _items,
             _y_kda_t, _y_gpm_t, _y_hd_t) = batch
            hero_ids = hero_ids.to(device); pf = pf.to(device)
            y_win = y_win.to(device).float()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                with torch.no_grad():
                    memory, _ = model.encode(hero_ids, pf, items_input=None,
                                               scalar_inputs=None,
                                               mask_dict=_build_inference_mask(hero_ids.size(0), device))
                    pooled = model.pooled(memory).float()
                logits = head(pooled).squeeze(-1)
                loss = bce(logits.float(), y_win)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * y_win.size(0); n_seen += y_win.size(0)

        # Val.
        head.eval()
        ys = []; ps = []
        patches = []
        with torch.no_grad():
            for batch in val_loader:
                (hero_ids, pf, patch_id, _sc_in,
                 y_win, _y_dur, _y_dur_bucket, _items,
                 _y_kda_t, _y_gpm_t, _y_hd_t) = batch
                hero_ids = hero_ids.to(device); pf = pf.to(device)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                     enabled=autocast_dtype is not None):
                    memory, _ = model.encode(hero_ids, pf, items_input=None,
                                                scalar_inputs=None,
                                                mask_dict=_build_inference_mask(hero_ids.size(0), device))
                    pooled = model.pooled(memory).float()
                    logits = head(pooled).squeeze(-1).float()
                ps.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(y_win.cpu().numpy())
                patches.append(patch_id.cpu().numpy())
        y_arr = np.concatenate(ys); p_arr = np.concatenate(ps)
        patch_arr = np.concatenate(patches)
        m = metrics_block(y_arr, p_arr, base_rate=meta["radiant_base_rate_val"])
        if m["auc"] > best_auc:
            best_auc = m["auc"]; best_epoch = epoch + 1
            best_y = y_arr; best_p = p_arr; best_patch = patch_arr
        ep_dt = time.time() - ep_t0
        print(f"[probe] epoch {epoch+1}/{max_epochs}  "
              f"tr_loss={tr_loss/max(n_seen,1):.4f}  "
              f"val_auc={m['auc']:.4f}  ({ep_dt:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": tr_loss / max(n_seen, 1),
                          "val_auc": m["auc"], "val_log_loss": m["log_loss"],
                          "val_brier": m["brier"], "wall_seconds": ep_dt})

    cov = coverage_bucket_val_auc(val_ds, best_y, best_p, feat_names)
    patch_aucs = per_patch_val_auc(best_y, best_p, best_patch)
    anchors = cfg["anchors"]
    out = {
        "phase": "linear_probe",
        "smoke": bool(smoke),
        "val_auc":        float(best_auc),
        "best_epoch":     int(best_epoch),
        "epochs_run":     int(max_epochs),
        "history":        history,
        "coverage_bucket_val_auc": cov,
        "per_patch_val_auc":       patch_aucs,
        "model_hp":       {k: mhp.get(k) for k in
                             ("embed_dim", "d_model", "n_heads", "n_layers", "ff_mult")},
        "n_train":        int(len(train_ds)),
        "n_val":          int(len(val_ds)),
        "train_date_min": meta["train_date_min"],
        "train_date_max": meta["train_date_max"],
        "val_date_min":   meta["val_date_min"],
        "val_date_max":   meta["val_date_max"],
        "radiant_base_rate_val": meta["radiant_base_rate_val"],
        "train_patch_id_distribution": meta["train_patch_id_distribution"],
        "val_patch_id_distribution":   meta["val_patch_id_distribution"],
    }
    out.update(anchors_deltas(float(best_auc), anchors))
    out_path = EXP_DIR / probe_cfg["output_metrics"]
    if smoke:
        out_path = out_path.with_name(out_path.stem + "_smoke.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[probe] wrote {out_path}")
    return 0


# =============================================================================
# Phase 2B: full multi-task fine-tune
# =============================================================================

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


def run_finetune(cfg: dict, splits: dict, smoke: bool) -> int:
    set_seed(int(cfg["seed"]))
    ft_cfg = cfg["finetune"]
    if smoke:
        ft_cfg = {**ft_cfg, **cfg["smoke"]["finetune"]}

    feat_names = cfg["player_features_transformer"]["feat_names"]
    n_player_feats = int(cfg["player_features_transformer"]["n_player_feats"])
    source_dir = PROJECT_ROOT / cfg["player_features_transformer"]["source_dir"]
    sidecar_dir = PROJECT_ROOT / cfg["rich_cols"]["out_dir"]
    vp = cfg["item_vocab"]["vocab_path"]
    vocab_path = (EXP_DIR / vp).resolve() if not Path(vp).is_absolute() else Path(vp)
    aux_targets = cfg["multitask_loss"]["aux_targets"]
    canonical_sort = bool(cfg["transformer_model"].get("use_canonical_sort", True))
    default_patch_id = int(cfg["patch"].get("default_patch_id", 1))

    t0 = time.time()
    smoke_n_train = int(cfg["smoke"]["finetune"].get("n_train", 50_000))
    smoke_n_val = int(cfg["smoke"]["finetune"].get("n_val", 5_000))
    train_ds, val_ds, meta = load_train_val(
        seed=int(cfg["seed"]),
        n_target=int(cfg["train_subset_size"]),
        feat_names=feat_names, source_dir=source_dir, splits=splits,
        smoke=smoke, smoke_n_train=smoke_n_train, smoke_n_val=smoke_n_val,
        sidecar_dir=sidecar_dir, vocab_path=vocab_path,
        aux_targets=aux_targets, canonical_sort=canonical_sort,
        default_patch_id=default_patch_id,
    )
    print(f"[finetune] data ready in {time.time() - t0:.1f}s — "
          f"train={len(train_ds):,} val={len(val_ds):,}")

    mhp = cfg["transformer_model"]
    item_vocab_size = int(meta.get("item_vocab_size", 0)) or 1
    n_dur_buckets = int(cfg["duration_bucket"]["n_buckets"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_v5(mhp, vocab_size=int(cfg["hero"]["vocab_size"]),
                            n_player_feats=n_player_feats,
                            n_dur_buckets=n_dur_buckets,
                            item_vocab_size=item_vocab_size,
                            use_team_team_bias=bool(mhp.get("use_team_team_bias", True)),
                            dur_loss_mode=cfg.get("dur_loss_mode", "ce"))
    enc_path = EXP_DIR / ft_cfg["encoder_path"]
    if smoke:
        enc_path = enc_path.with_name(enc_path.stem + "_smoke.pt")
    if not enc_path.exists():
        raise SystemExit(f"[finetune] encoder checkpoint not found: {enc_path}")
    state = torch.load(enc_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print(f"[finetune] loaded encoder from {enc_path}; model: {count_params(model)}")

    # Pre-finetune L2 norm of encoder weights — catastrophic-forgetting check.
    enc_names = set(encoder_param_names(model))
    def _enc_l2() -> float:
        total = 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in enc_names:
                    total += float((p.float() ** 2).sum().item())
        return float(total ** 0.5)
    l2_pre = _enc_l2()
    print(f"[finetune] encoder L2-norm pre-finetune: {l2_pre:.4f}")

    # Two-group optimizer.
    enc_params = [p for n, p in model.named_parameters() if n in enc_names and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                    if n not in enc_names and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": enc_params,  "lr": float(ft_cfg["encoder_lr"]),
         "weight_decay": float(ft_cfg["encoder_weight_decay"])},
        {"params": head_params, "lr": float(ft_cfg["head_lr"]),
         "weight_decay": float(ft_cfg["head_weight_decay"])},
    ])
    print(f"[finetune] optimizer: {len(enc_params)} encoder tensors "
          f"(lr={ft_cfg['encoder_lr']}), {len(head_params)} head tensors "
          f"(lr={ft_cfg['head_lr']})")

    alpha = cfg["multitask_loss"]
    fixed_alpha = torch.tensor([
        float(alpha["alpha_win"]), float(alpha["alpha_dur"]),
        float(alpha["alpha_item"]),
        float(alpha["alpha_kda"]), float(alpha["alpha_gpm"]),
        float(alpha["alpha_hd"]),
    ], device=device)

    bs = int(ft_cfg["batch_size"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs)
    autocast_dtype = (torch.bfloat16 if (bool(ft_cfg["mixed_precision"])
                                            and device.type == "cuda") else None)
    bce_w = nn.BCEWithLogitsLoss()
    bce_i = nn.BCEWithLogitsLoss()

    max_epochs = int(ft_cfg["max_epochs"])
    if smoke:
        max_epochs = 1
    patience = int(ft_cfg.get("patience", 5)) if not smoke else None
    warmup_steps = int(ft_cfg["warmup_steps"])
    total_steps = max(max_epochs * max(1, math.ceil(len(train_ds) / bs)), 1)

    history = []
    best_val_loss = math.inf
    best_val_auc = -math.inf
    best_state = None
    best_epoch = -1
    best_y = best_p = best_patch = None
    epochs_since_improve = 0
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        ep_t0 = time.time()
        sum_w = sum_d = sum_i = sum_kda = sum_gpm = sum_hd = 0.0
        sum_total = 0.0
        n_seen = 0
        for batch in train_loader:
            (hero_ids, pf, _patch_id, _sc_in,
             y_win, y_dur, y_dur_bucket, items,
             y_kda_t, y_gpm_t, y_hd_t) = batch
            hero_ids = hero_ids.to(device); pf = pf.to(device)
            y_win = y_win.to(device); y_dur_bucket = y_dur_bucket.to(device)
            items = items.to(device)
            y_kda_t = y_kda_t.to(device); y_gpm_t = y_gpm_t.to(device); y_hd_t = y_hd_t.to(device)
            B = hero_ids.size(0); n_seen += B

            # Cosine LR per param group: encoder vs head have different bases.
            enc_lr = warmup_cosine_lr(global_step, warmup_steps, total_steps,
                                         float(ft_cfg["encoder_lr"]),
                                         float(ft_cfg["encoder_cosine_min_lr"]))
            head_lr = warmup_cosine_lr(global_step, warmup_steps, total_steps,
                                          float(ft_cfg["head_lr"]),
                                          float(ft_cfg["head_cosine_min_lr"]))
            opt.param_groups[0]["lr"] = enc_lr
            opt.param_groups[1]["lr"] = head_lr

            mask_dict = _build_inference_mask(B, device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                out = model.forward_multitask(hero_ids, pf, items_input=None,
                                                 scalar_inputs=None,
                                                 mask_dict=mask_dict)
                l_w = bce_w(out["win"].float(), y_win.float())
                l_d = F.cross_entropy(out["dur"].float(), y_dur_bucket)
                l_i = bce_i(out["item"].float(), items.float())
                l_kda = _masked_smooth_l1_mean(out["kda"].float(), y_kda_t.float())
                l_gpm = _masked_smooth_l1_mean(out["gpm"].float(), y_gpm_t.float())
                l_hd = _masked_smooth_l1_mean(out["hd"].float(), y_hd_t.float())
                losses = torch.stack([l_w, l_d, l_i, l_kda, l_gpm, l_hd])
                total = (fixed_alpha * losses).sum()

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

            sum_w += l_w.item() * B
            sum_d += l_d.item() * B
            sum_i += l_i.item() * B
            sum_kda += l_kda.item() * B
            sum_gpm += l_gpm.item() * B
            sum_hd += l_hd.item() * B
            sum_total += total.item() * B
            global_step += 1

        # Val.
        model.eval()
        ys = []; ps = []; patches = []
        tot_val_win = 0.0; n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                (hero_ids, pf, patch_id, _sc_in,
                 y_win, _y_dur, _y_dur_bucket, _items,
                 _y_kda_t, _y_gpm_t, _y_hd_t) = batch
                hero_ids = hero_ids.to(device); pf = pf.to(device)
                y_win_d = y_win.to(device)
                B = hero_ids.size(0)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                     enabled=autocast_dtype is not None):
                    out = model.forward_multitask(hero_ids, pf, items_input=None,
                                                     scalar_inputs=None,
                                                     mask_dict=_build_inference_mask(B, device))
                    win_logits = out["win"].float()
                    l_w = F.binary_cross_entropy_with_logits(win_logits, y_win_d.float(),
                                                                reduction="sum")
                ps.append(torch.sigmoid(win_logits).cpu().numpy())
                ys.append(y_win.cpu().numpy())
                patches.append(patch_id.cpu().numpy())
                tot_val_win += float(l_w.item()); n_val += B
        y_arr = np.concatenate(ys); p_arr = np.concatenate(ps); patch_arr = np.concatenate(patches)
        m = metrics_block(y_arr, p_arr, base_rate=meta["radiant_base_rate_val"])
        val_win_loss = tot_val_win / max(n_val, 1)
        ep_dt = time.time() - ep_t0
        print(f"[finetune] epoch {epoch+1}/{max_epochs}  "
              f"tr[w={sum_w/max(n_seen,1):.4f} d={sum_d/max(n_seen,1):.4f} "
              f"i={sum_i/max(n_seen,1):.5f} kda={sum_kda/max(n_seen,1):.4f} "
              f"gpm={sum_gpm/max(n_seen,1):.4f} hd={sum_hd/max(n_seen,1):.4f}]  "
              f"vl_win={val_win_loss:.4f}  val_auc={m['auc']:.4f}  "
              f"enc_lr={enc_lr:.2e} head_lr={head_lr:.2e}  ({ep_dt:.1f}s)")

        history.append({
            "epoch": epoch + 1,
            "train_win_loss": sum_w / max(n_seen, 1),
            "train_dur_loss": sum_d / max(n_seen, 1),
            "train_item_loss": sum_i / max(n_seen, 1),
            "train_kda_loss": sum_kda / max(n_seen, 1),
            "train_gpm_loss": sum_gpm / max(n_seen, 1),
            "train_hd_loss":  sum_hd / max(n_seen, 1),
            "train_total_loss": sum_total / max(n_seen, 1),
            "val_win_log_loss": float(val_win_loss),
            "val_win_auc": float(m["auc"]),
            "val_win_brier": float(m["brier"]),
            "encoder_lr": float(enc_lr),
            "head_lr":    float(head_lr),
            "wall_seconds": float(ep_dt),
        })
        if val_win_loss < best_val_loss - 1e-6:
            best_val_loss = val_win_loss
            best_val_auc = m["auc"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_y = y_arr; best_p = p_arr; best_patch = patch_arr
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if patience is not None and epochs_since_improve >= patience:
            print(f"[finetune] early stop at epoch {epoch+1} (best {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    l2_post = _enc_l2()
    print(f"[finetune] encoder L2-norm post-finetune: {l2_post:.4f} "
          f"(delta {l2_post - l2_pre:+.4f}, frac {l2_post/max(l2_pre,1e-9):+.4f}x)")

    cov = coverage_bucket_val_auc(val_ds, best_y, best_p, feat_names)
    patch_aucs = per_patch_val_auc(best_y, best_p, best_patch)
    anchors = cfg["anchors"]
    out = {
        "phase": "finetune",
        "smoke": bool(smoke),
        "val_auc":        float(best_val_auc),
        "val_win_log_loss": float(best_val_loss),
        "best_epoch":     int(best_epoch),
        "epochs_run":     int(history[-1]["epoch"]) if history else 0,
        "max_epochs":     int(max_epochs),
        "history":        history,
        "coverage_bucket_val_auc": cov,
        "per_patch_val_auc":       patch_aucs,
        "encoder_l2_norm_pre_finetune":  float(l2_pre),
        "encoder_l2_norm_post_finetune": float(l2_post),
        "encoder_l2_norm_delta":         float(l2_post - l2_pre),
        "encoder_l2_norm_frac":          float(l2_post / max(l2_pre, 1e-9)),
        "model_hp":       {k: mhp.get(k) for k in
                             ("embed_dim", "d_model", "n_heads", "n_layers", "ff_mult")},
        "param_counts":   count_params(model),
        "multitask_alpha": dict(alpha),
        "optim_hp":       {
            "encoder_lr":     float(ft_cfg["encoder_lr"]),
            "encoder_cosine_min_lr": float(ft_cfg["encoder_cosine_min_lr"]),
            "encoder_weight_decay":  float(ft_cfg["encoder_weight_decay"]),
            "head_lr":        float(ft_cfg["head_lr"]),
            "head_cosine_min_lr":    float(ft_cfg["head_cosine_min_lr"]),
            "head_weight_decay":     float(ft_cfg["head_weight_decay"]),
            "batch_size":     int(bs), "warmup_steps": int(warmup_steps),
            "max_epochs":     int(max_epochs),
        },
        "n_train":        int(len(train_ds)),
        "n_val":          int(len(val_ds)),
        "train_date_min": meta["train_date_min"],
        "train_date_max": meta["train_date_max"],
        "val_date_min":   meta["val_date_min"],
        "val_date_max":   meta["val_date_max"],
        "radiant_base_rate_val": meta["radiant_base_rate_val"],
        "train_patch_id_distribution": meta["train_patch_id_distribution"],
        "val_patch_id_distribution":   meta["val_patch_id_distribution"],
    }
    out.update(anchors_deltas(float(best_val_auc), anchors))
    out_path = EXP_DIR / ft_cfg["output_metrics"]
    if smoke:
        out_path = out_path.with_name(out_path.stem + "_smoke.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[finetune] wrote {out_path}")

    # Also drop a top-level metrics.json (validation-split search signal).
    if not smoke:
        top_metrics = EXP_DIR / "metrics.json"
        top_metrics.write_text(json.dumps(out, indent=2))
        print(f"[finetune] wrote {top_metrics}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--phase", required=True, choices=["pretrain", "probe", "finetune"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    if args.phase == "pretrain":
        return run_pretrain(cfg, splits, smoke=args.smoke)
    elif args.phase == "probe":
        return run_linear_probe(cfg, splits, smoke=args.smoke)
    elif args.phase == "finetune":
        return run_finetune(cfg, splits, smoke=args.smoke)
    else:
        raise SystemExit(f"unknown phase: {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
