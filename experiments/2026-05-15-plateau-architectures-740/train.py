"""Train ONE architecture (--arch simple_ffn|residual_ffn|transformer).

HCE rule: only validation metrics are written. No final_metrics.json.

Outputs:
  - results/{arch}_metrics.json   — per-arch val metrics + run metadata
  - results/{arch}.pt              — trained model state_dict
  - results/{arch}_history.json    — per-epoch train/val loss + AUC

Use --smoke for a tiny end-to-end check (1 epoch, 50k train, 5k val).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
RESULTS = EXP_DIR / "results"
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"

sys.path.insert(0, str(EXP_DIR))
from data import load_train_val
from models import build_model, count_params


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(train_ds, val_ds, batch_size: int, num_workers: int,
                 pin_memory: bool):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(batch_size * 2, 16384), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    model.eval()
    losses, ys, ps = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    n = 0
    total_loss = 0.0
    for hero_ids, side_bit, y in loader:
        hero_ids = hero_ids.to(device, non_blocking=True)
        side_bit = side_bit.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            logits = model(hero_ids, side_bit)
            loss = bce(logits.float(), y.float())
        total_loss += loss.item()
        n += y.size(0)
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.cpu().numpy())
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    return {
        "loss": total_loss / max(n, 1),
        "y": y,
        "p": p,
    }


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


def train_one(arch: str, cfg: dict, smoke: bool, device: torch.device) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)

    splits = yaml.safe_load(SPLITS_PATH.read_text())
    smoke_cfg = cfg["smoke"]
    optim_cfg = cfg["optim"]
    hero_cfg = cfg["hero"]
    arch_cfg = cfg["architectures"][arch]

    print(f"[{arch}] loading data (smoke={smoke})...")
    t0 = time.time()
    train_ds, val_ds, meta = load_train_val(
        seed=seed,
        n_target=int(cfg["train_subset_size"]),
        splits=splits,
        smoke=smoke,
        smoke_n_train=int(smoke_cfg["n_train"]),
        smoke_n_val=int(smoke_cfg["n_val"]),
    )
    print(f"[{arch}] data loaded in {time.time()-t0:.1f}s — "
          f"train={len(train_ds):,} val={len(val_ds):,}")
    print(f"[{arch}] dates: train={meta['train_date_min']}..{meta['train_date_max']} "
          f"val={meta['val_date_min']}..{meta['val_date_max']}")

    batch_size = int(smoke_cfg["batch_size"]) if smoke else int(optim_cfg["batch_size"])
    max_epochs = int(smoke_cfg["max_epochs"]) if smoke else int(optim_cfg["max_epochs"])
    train_loader, val_loader = make_loaders(
        train_ds, val_ds,
        batch_size=batch_size,
        num_workers=int(optim_cfg.get("num_workers", 0)) if not smoke else 0,
        pin_memory=bool(optim_cfg.get("pin_memory", False)),
    )

    model = build_model(arch, hero_cfg, arch_cfg).to(device)
    pcounts = count_params(model)
    print(f"[{arch}] params: {pcounts}")

    opt = torch.optim.Adam(model.parameters(),
                           lr=float(optim_cfg["lr"]),
                           weight_decay=float(optim_cfg.get("weight_decay", 0.0)))
    bce = nn.BCEWithLogitsLoss()

    autocast_dtype = None
    if bool(optim_cfg.get("mixed_precision", False)) and device.type == "cuda":
        autocast_dtype = torch.bfloat16  # bf16 doesn't need a GradScaler

    history = []
    best_val_loss = math.inf
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    patience = int(optim_cfg.get("patience", 5))

    train_t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        n_seen = 0
        loss_sum = 0.0
        ep_t0 = time.time()
        for hero_ids, side_bit, y in train_loader:
            hero_ids = hero_ids.to(device, non_blocking=True)
            side_bit = side_bit.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                logits = model(hero_ids, side_bit)
                loss = bce(logits.float(), y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_seen += y.size(0)
            loss_sum += loss.item() * y.size(0)
        train_loss = loss_sum / max(n_seen, 1)

        val_eval = evaluate(model, val_loader, device, autocast_dtype)
        val_loss = val_eval["loss"]
        val_metrics = metrics_block(val_eval["y"], val_eval["p"],
                                    base_rate=meta["radiant_base_rate_val"])
        ep_dt = time.time() - ep_t0
        print(f"[{arch}] epoch {epoch+1}/{max_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_metrics['auc']:.4f}  val_acc={val_metrics['acc']:.4f}  "
              f"({ep_dt:.1f}s)")

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
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience and not smoke:
                print(f"[{arch}] early stop at epoch {epoch+1} (best epoch {best_epoch})")
                break

    train_seconds = time.time() - train_t0

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval at the best checkpoint.
    val_eval = evaluate(model, val_loader, device, autocast_dtype)
    val_metrics = metrics_block(val_eval["y"], val_eval["p"],
                                base_rate=meta["radiant_base_rate_val"])
    # Optional train-set AUC for overfit anchor (use the same train_loader, no shuffle needed).
    train_eval = evaluate(model, train_loader, device, autocast_dtype)
    train_metrics = metrics_block(train_eval["y"], train_eval["p"])

    out = {
        "arch": arch,
        "val_auc": val_metrics["auc"],
        "val_acc": val_metrics["acc"],
        "val_log_loss": val_metrics["log_loss"],
        "val_brier": val_metrics["brier"],
        "train_auc": train_metrics["auc"],
        "train_acc": train_metrics["acc"],
        "train_log_loss": train_metrics["log_loss"],
        "train_val_auc_gap": train_metrics["auc"] - val_metrics["auc"],
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "train_seconds": train_seconds,
        "param_counts": pcounts,
        "model": arch,
        "smoke": bool(smoke),
        "val_majority_class_acc": val_metrics.get("majority_class_acc"),
        **{k: v for k, v in meta.items()},
    }
    return out, history, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["simple_ffn", "residual_ffn", "transformer"])
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny 1-epoch end-to-end test (50k train / 5k val).")
    ap.add_argument("--save", action="store_true",
                    help="Save model checkpoint (auto-on for non-smoke).")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Override config.optim.num_workers (workaround for "
                         "torch 2.11+Blackwell DataLoader worker segfaults).")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="Override config.optim.max_epochs (workaround to exit "
                         "before torch 2.11+Blackwell intermittent crashes).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.num_workers is not None:
        cfg["optim"]["num_workers"] = args.num_workers
    if args.max_epochs is not None:
        cfg["optim"]["max_epochs"] = args.max_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  (cuda available: {torch.cuda.is_available()})")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name()}")

    out, history, model = train_one(args.arch, cfg, smoke=args.smoke, device=device)

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    metrics_path = RESULTS / f"{args.arch}{suffix}_metrics.json"
    history_path = RESULTS / f"{args.arch}{suffix}_history.json"
    metrics_path.write_text(json.dumps(out, indent=2))
    history_path.write_text(json.dumps(history, indent=2))
    print(f"wrote {metrics_path}")
    print(f"wrote {history_path}")

    if args.save or not args.smoke:
        ckpt_path = RESULTS / f"{args.arch}{suffix}.pt"
        torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                    "config": cfg}, ckpt_path)
        print(f"wrote {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} MB)")

    print(f"FINAL [{args.arch}{' SMOKE' if args.smoke else ''}]: "
          f"val_auc={out['val_auc']:.4f}  val_acc={out['val_acc']:.4f}  "
          f"val_log_loss={out['val_log_loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
