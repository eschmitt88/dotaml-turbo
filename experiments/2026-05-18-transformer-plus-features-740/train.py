"""Train MinimalTransformerWithFeatures for one ablation.

Single-ablation entry point. Selects between:
  --ablation architecture_only         : Transformer, NO player-feature injection.
                                         Sanity check vs plateau-architectures-740 (~0.6322).
  --ablation transformer_plus_features : PRIMARY. Transformer + Linear(8, d_model) per slot.

Each invocation runs in a fresh Python process — per-trial subprocess isolation
is the workaround for the Blackwell + torch 2.9-2.12 DataLoader GC segfault
documented in docs/decisions/0001-per-trial-subprocess-isolation.md.

Math SDP backend is forced at module load to avoid sm_120 flash/mem-eff
attention crashes (same workaround as transformer-hp-sweep-740/train_one.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
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

# Force math SDP backend on Blackwell — must run before any model is built.
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

# Local imports (after the SDP toggle).
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
    """num_workers=0 — Blackwell + torch 2.9 DataLoader worker segfault workaround."""
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
def evaluate(model, loader, device, autocast_dtype, use_features: bool):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    n = 0
    total_loss = 0.0
    ys, ps = [], []
    for hero_ids, player_feats, y in loader:
        hero_ids = hero_ids.to(device, non_blocking=True)
        player_feats = player_feats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            logits = model(hero_ids, player_feats if use_features else None)
            loss = bce(logits.float(), y.float())
        total_loss += loss.item()
        n += y.size(0)
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.cpu().numpy())
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    return {"loss": total_loss / max(n, 1), "y": y, "p": p}


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


@dataclass
class TrainResult:
    history: list
    best_val_loss: float
    best_val_auc: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    val_metrics_at_best: dict
    val_predictions: np.ndarray
    val_labels: np.ndarray


def train_model(model, train_ds, val_ds, hp: dict, max_epochs: int,
                device: torch.device, base_rate_val: float | None,
                use_features: bool, mixed_precision: bool = True,
                patience: int | None = None) -> TrainResult:
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
    best_eval: dict = {}
    epochs_since_improve = 0

    train_t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        n_seen = 0
        loss_sum = 0.0
        ep_t0 = time.time()
        for hero_ids, player_feats, y in train_loader:
            hero_ids = hero_ids.to(device, non_blocking=True)
            player_feats = player_feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                logits = model(hero_ids, player_feats if use_features else None)
                loss = bce(logits.float(), y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_seen += y.size(0)
            loss_sum += loss.item() * y.size(0)
        train_loss = loss_sum / max(n_seen, 1)

        val_eval = evaluate(model, val_loader, device, autocast_dtype, use_features)
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
            best_eval = val_eval
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if patience is not None and epochs_since_improve >= patience:
            print(f"  early stop at epoch {epoch+1} (best epoch {best_epoch})")
            break

    train_seconds = time.time() - train_t0
    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        history=history,
        best_val_loss=best_val_loss,
        best_val_auc=best_val_auc,
        best_epoch=best_epoch,
        epochs_run=history[-1]["epoch"] if history else 0,
        train_seconds=train_seconds,
        val_metrics_at_best=best_val_metrics,
        val_predictions=best_eval.get("p", np.array([])),
        val_labels=best_eval.get("y", np.array([])),
    )


def coverage_bucket_val_auc(val_ds, y_val: np.ndarray, p_val: np.ndarray,
                            feat_names: list[str]) -> dict:
    """Reproduce the coverage-bucket diagnostic from
    player-features-prepatch-740/train.py: bucket val by mean p*_n_games_log1p
    across 10 slots → low / medium / high tercile, report AUC in each.
    """
    if "n_games_log1p" not in feat_names:
        return {"error": "n_games_log1p not in feat_names"}
    f_idx = feat_names.index("n_games_log1p")
    # val_ds.player_feats shape: [N, 10, F]
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
        yb = y_val[mask]
        pb = p_val[mask]
        try:
            auc_b = float(roc_auc_score(yb, pb))
        except ValueError:
            auc_b = None
        bucket_aucs[name] = {
            "n": n,
            "val_auc": auc_b,
            "mean_coverage_log1p": float(coverage[mask].mean()),
        }
    return {
        "quantile_edges_log1p": [float(q33), float(q67)],
        "buckets": bucket_aucs,
    }


def plot_calibration(y_true, p_pred, out: Path) -> dict:
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=20, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", lw=1.5, label="model")
    ax.set_xlabel("predicted P(radiant_win)")
    ax.set_ylabel("empirical P(radiant_win)")
    ax.set_title("Calibration (val, 20-quantile)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)
    return {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}


def plot_roc(y_true, p_pred, auc, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (val)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def plot_learning(history: list, out: Path) -> None:
    if not history:
        return
    ep = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    va = [h["val_auc"] for h in history]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ep, tr, label="train_loss", lw=1)
    ax1.plot(ep, vl, label="val_loss", lw=1)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(ep, va, "g--", label="val_auc", lw=1)
    ax2.set_ylabel("val_auc"); ax2.legend(loc="upper right")
    ax1.set_title("Learning curves"); ax1.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--ablation", required=True,
                    choices=["architecture_only", "transformer_plus_features"])
    ap.add_argument("--smoke", action="store_true",
                    help="Use 50k train / 5k val and 1-epoch cap (pipeline test only).")
    ap.add_argument("--metrics-suffix", default="",
                    help="Suffix appended to metrics filename. e.g. '_architecture_only'.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    seed = int(cfg["seed"])
    set_seed(seed)

    # Resolve ablation spec.
    ab_spec = next((a for a in cfg["ablations"] if a["name"] == args.ablation), None)
    if ab_spec is None:
        sys.exit(f"unknown ablation {args.ablation}")
    use_features = bool(ab_spec["use_features"])

    feat_names = cfg["player_features"]["feat_names"]
    n_player_feats = int(cfg["player_features"]["n_player_feats"])
    if len(feat_names) != n_player_feats:
        sys.exit(f"feat_names len {len(feat_names)} != n_player_feats {n_player_feats}")
    source_dir = PROJECT_ROOT / cfg["player_features"]["source_dir"]

    # Load data.
    print(f"Ablation: {args.ablation} (use_features={use_features})")
    t0 = time.time()
    n_target = int(cfg["train_subset_size"])
    if args.smoke:
        train_ds, val_ds, meta = load_train_val(
            seed=seed, n_target=n_target, feat_names=feat_names,
            source_dir=source_dir, splits=splits, smoke=True,
            smoke_n_train=int(cfg["smoke"]["n_train"]),
            smoke_n_val=int(cfg["smoke"]["n_val"]),
        )
    else:
        train_ds, val_ds, meta = load_train_val(
            seed=seed, n_target=n_target, feat_names=feat_names,
            source_dir=source_dir, splits=splits, smoke=False,
        )
    data_seconds = time.time() - t0
    print(f"Data ready in {data_seconds:.1f}s — train={len(train_ds):,} val={len(val_ds):,}")
    print(f"  train dates {meta['train_date_min']}..{meta['train_date_max']}")
    print(f"  val   dates {meta['val_date_min']}..{meta['val_date_max']}")
    print(f"  radiant base rate (train sub) {meta['radiant_base_rate_train_subsampled']:.4f}, "
          f"val {meta['radiant_base_rate_val']:.4f}")

    # Build model.
    mhp = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mhp, vocab_size=int(cfg["hero"]["vocab_size"]),
                        n_player_feats=n_player_feats, use_features=use_features)
    model = model.to(device)
    pc = count_params(model)
    print(f"Model: {pc}, device={device}, dtype=bf16-autocast")

    # Train.
    opt = cfg["optim"]
    max_epochs = int(opt["max_epochs"]) if not args.smoke else int(cfg["smoke"]["max_epochs"])
    base_rate_val = meta["radiant_base_rate_val"]
    hp = {"batch_size": int(opt["batch_size"]), "lr": float(opt["lr"]),
          "weight_decay": float(opt["weight_decay"])}
    patience = int(opt.get("patience", 5)) if not args.smoke else None
    tr = train_model(model, train_ds, val_ds, hp, max_epochs=max_epochs,
                     device=device, base_rate_val=base_rate_val,
                     use_features=use_features,
                     mixed_precision=bool(opt["mixed_precision"]),
                     patience=patience)
    print(f"Training done in {tr.train_seconds:.1f}s — best val_auc={tr.best_val_auc:.4f} "
          f"@ epoch {tr.best_epoch}")

    # Diagnostics on best-epoch predictions.
    y_val = tr.val_labels
    p_val = tr.val_predictions
    coverage_info = None
    try:
        coverage_info = coverage_bucket_val_auc(val_ds, y_val, p_val, feat_names)
    except Exception as e:  # noqa: BLE001
        coverage_info = {"error": f"{type(e).__name__}: {e}"}

    # Plots.
    results_dir = EXP_DIR / cfg["output"]["results_dir"]
    results_dir.mkdir(exist_ok=True, parents=True)
    sfx = args.metrics_suffix or f"_{args.ablation}"
    if args.smoke:
        sfx = cfg["smoke"]["metrics_suffix"] + f"_{args.ablation}"
    cal = None
    try:
        cal = plot_calibration(y_val, p_val, results_dir / f"calibration{sfx}.png")
        plot_roc(y_val, p_val, tr.best_val_auc, results_dir / f"roc{sfx}.png")
        plot_learning(tr.history, results_dir / f"learning_curve{sfx}.png")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped: {e}")

    # Save best-epoch checkpoint (small — ~82k params).
    try:
        ckpt_path = results_dir / f"model{sfx}.pt"
        torch.save(model.state_dict(), ckpt_path)
    except Exception as e:  # noqa: BLE001
        print(f"checkpoint save skipped: {e}")

    # Anchors / deltas.
    anchors = cfg.get("anchors", {})
    arch_only_anchor = float(anchors.get("plateau_architectures_740_val_auc", 0.6322))
    feats_only_anchor = float(anchors.get(
        "player_features_prepatch_740_features_only_val_auc", 0.6256))
    target = float(anchors.get("proposal_target_val_auc", 0.6372))

    architecture_only_sanity = None
    if args.ablation == "architecture_only" and not args.smoke:
        delta = tr.best_val_auc - arch_only_anchor
        passed = abs(delta) <= 0.005
        architecture_only_sanity = {
            "anchor_val_auc": arch_only_anchor,
            "this_val_auc": tr.best_val_auc,
            "delta": delta,
            "passed_within_0.005": bool(passed),
        }
        verdict = "PASS" if passed else "FAIL"
        print(f"ARCHITECTURE-ONLY SANITY CHECK: {verdict} (Δ={delta:+.4f})")

    metrics = {
        "ablation": args.ablation,
        "smoke": bool(args.smoke),
        "use_features": use_features,
        "val_auc": float(tr.best_val_auc),
        "val_loss": float(tr.best_val_loss),
        "val_metrics_at_best": tr.val_metrics_at_best,
        "best_epoch": int(tr.best_epoch),
        "epochs_run": int(tr.epochs_run),
        "max_epochs": int(max_epochs),
        "history": tr.history,
        "model_hp": {k: mhp[k] for k in ("embed_dim", "d_model", "n_heads",
                                          "n_layers", "ff_mult", "dropout")},
        "optim_hp": {"batch_size": hp["batch_size"], "lr": hp["lr"],
                     "weight_decay": hp["weight_decay"],
                     "mixed_precision": bool(opt["mixed_precision"])},
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
        "anchors": anchors,
        "delta_vs_plateau_architectures_740": tr.best_val_auc - arch_only_anchor,
        "delta_vs_features_only_lgbm_prepatch": tr.best_val_auc - feats_only_anchor,
        "delta_vs_proposal_target": tr.best_val_auc - target,
        "coverage_bucket_val_auc": coverage_info,
        "calibration": cal,
        "architecture_only_sanity_check": architecture_only_sanity,
    }
    out_name = f"metrics{sfx}.json"
    out = EXP_DIR / out_name
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
