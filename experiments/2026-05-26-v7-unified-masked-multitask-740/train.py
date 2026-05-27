"""Train FoundationTransformerV7 for v7-unified-masked-multitask-740.

Per-batch scenario sampling + per-scenario loss weighting + periodic
probe suite + adaptive sampling update. See proposal for the design
rationale; see config.yaml for the tunable scenario distribution.

Anchors block (written to metrics_v7.json):
  v4 anchor (PRIMARY pure_pregame): 0.6471
  iso_teambias:                     0.6493
  baseline_multitask_repro:         0.6470
  proposal_pure_pregame_target:     0.6471
  proposal_items_cond_target:       0.80
  proposal_duration_cond_target:    0.68

Usage:
  python -u train.py --config config.yaml --ablation v7_unified [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
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
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

from data import load_train_val  # noqa: E402
from mae import HEAD_NAMES, ScenarioSampler  # noqa: E402
from models import build_model, count_params  # noqa: E402
from probes import ProbeSuite  # noqa: E402


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


def _smoothl1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean SmoothL1; NaN-safe."""
    m = torch.isfinite(pred) & torch.isfinite(target)
    if not m.any():
        return pred.sum() * 0.0
    p = torch.where(m, pred, torch.zeros_like(pred))
    t = torch.where(m, target, torch.zeros_like(target))
    diff = p - t
    ad = diff.abs()
    elem = torch.where(ad < 1.0, 0.5 * diff * diff, ad - 0.5)
    elem = torch.where(m, elem, torch.zeros_like(elem))
    return elem.sum() / float(m.sum().item())


@dataclass
class TrainResult:
    history: list
    best_pure_pregame_auc: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    halted: bool
    halt_reasons: list


def _eval_pure_pregame_auc(model, loader, device, autocast_dtype) -> float:
    """Quick pure_pregame val_auc -- used for early-stop signal."""
    from probes import _build_masks
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            (hero_ids, pf, _patch_id, _acct, items,
             kills, deaths, assists, gpm, hd,
             dur_log, y_win) = batch
            B = hero_ids.size(0)
            hero_ids = hero_ids.to(device); pf = pf.to(device); items = items.to(device)
            kills = kills.to(device); deaths = deaths.to(device); assists = assists.to(device)
            gpm = gpm.to(device); hd = hd.to(device)
            dur_log = dur_log.to(device); y_win = y_win.to(device)
            win_idx = y_win.long()
            masks = _build_masks(B, device, {
                "items": "mask", "kills": "mask", "deaths": "mask", "assists": "mask",
                "gpm": "mask", "hd": "mask", "duration": "mask", "win": "mask"})
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                out = model(hero_ids, pf, items, kills, deaths, assists,
                            gpm, hd, dur_log, win_idx, masks=masks)
                p = torch.sigmoid(out["win"].float())
            ys.append(y_win.cpu().numpy())
            ps.append(p.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def train_v7(model, train_ds, val_ds, hp: dict, max_epochs: int, device,
              mixed_precision: bool, patience: int | None,
              sampler: ScenarioSampler, probe_suite: ProbeSuite,
              probe_every_epochs: int, halt_at_epoch: int,
              warmup_steps: int, cosine_min_lr: float,
              probe_history_path: Path, sampling_history_path: Path,
              smoke: bool) -> TrainResult:
    bs = int(hp["batch_size"]); lr = float(hp["lr"]); wd = float(hp["weight_decay"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, bs)
    autocast_dtype = torch.bfloat16 if (mixed_precision and device.type == "cuda") else None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    bce_w = nn.BCEWithLogitsLoss()
    bce_i = nn.BCEWithLogitsLoss()

    total_steps = max(max_epochs * max(1, math.ceil(len(train_ds) / bs)), 1)
    print(f"  total_steps={total_steps:,} warmup_steps={warmup_steps:,}")

    history = []
    best_pure_pregame = -math.inf
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    global_step = 0
    halted = False
    halt_reasons: list = []
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        ep_t0 = time.time()
        # Loss running sums per head + scenario count.
        sum_per_head = {h: 0.0 for h in HEAD_NAMES}
        n_per_head = {h: 0 for h in HEAD_NAMES}
        scenario_counter: Counter = Counter()
        n_batches = 0
        sum_total = 0.0

        for batch in train_loader:
            (hero_ids, pf, _patch_id, _acct, items,
             kills, deaths, assists, gpm, hd,
             dur_log, y_win) = batch
            hero_ids = hero_ids.to(device); pf = pf.to(device); items = items.to(device)
            kills = kills.to(device); deaths = deaths.to(device); assists = assists.to(device)
            gpm = gpm.to(device); hd = hd.to(device)
            dur_log = dur_log.to(device); y_win = y_win.to(device)
            win_idx = y_win.long()
            B = hero_ids.size(0)

            scenario = sampler.sample_batch_scenario()
            scenario_counter[scenario] += 1
            masks = sampler.apply_mask(B, device, scenario, win_idx=win_idx)
            lw = sampler.loss_weights(scenario)

            cur_lr = warmup_cosine_lr(global_step, warmup_steps, total_steps,
                                         lr, cosine_min_lr)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                out = model(hero_ids, pf, items, kills, deaths, assists,
                            gpm, hd, dur_log, win_idx, masks=masks)
                # Per-head losses (all 8 heads always computed).
                # NOTE: kills/deaths/assists/gpm/hd targets log1p-transformed
                # to keep scales commensurate (raw hd ~30000 would dominate
                # multi-task sum). Heads output predictions in log1p space;
                # downstream query code does expm1() to recover raw values.
                # Matches duration head which is already log_seconds.
                l_win  = bce_w(out["win"].float(), y_win.float())
                l_dur  = _smoothl1(out["dur"].float(), dur_log.float())
                l_item = bce_i(out["item"].float(), items.float())
                l_kil  = _smoothl1(out["kills"].float(), torch.log1p(kills.float()))
                l_dea  = _smoothl1(out["deaths"].float(), torch.log1p(deaths.float()))
                l_ass  = _smoothl1(out["assists"].float(), torch.log1p(assists.float()))
                l_gpm  = _smoothl1(out["gpm"].float(), torch.log1p(gpm.float()))
                l_hd   = _smoothl1(out["hd"].float(), torch.log1p(hd.float()))
                losses = {"win": l_win, "dur": l_dur, "items": l_item,
                          "kills": l_kil, "deaths": l_dea, "assists": l_ass,
                          "gpm": l_gpm, "hd": l_hd}
                total = sum(lw[h] * losses[h] for h in HEAD_NAMES)

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

            for h in HEAD_NAMES:
                sum_per_head[h] += losses[h].item() * B
                n_per_head[h] += B
            sum_total += total.item() * B
            n_batches += 1
            global_step += 1

        per_head_avg = {h: sum_per_head[h] / max(n_per_head[h], 1) for h in HEAD_NAMES}
        total_avg = sum_total / max(sum(n_per_head.values()) / len(HEAD_NAMES), 1)
        ep_dt = time.time() - ep_t0
        scen_str = " ".join(f"{s}={scenario_counter[s]}" for s in sampler.SCENARIOS)
        print(f"  epoch {epoch+1}/{max_epochs} ({ep_dt:.1f}s) lr={cur_lr:.2e}")
        print(f"    losses: " +
              " ".join(f"{h}={per_head_avg[h]:.4f}" for h in HEAD_NAMES) +
              f"  total={total_avg:.4f}")
        print(f"    scenarios: {scen_str}")

        # Quick pure_pregame eval each epoch (cheap, single mask spec).
        pure_pregame_auc = _eval_pure_pregame_auc(model, val_loader, device, autocast_dtype)
        print(f"    pure_pregame_val_auc={pure_pregame_auc:.4f}")

        # Probe suite every N epochs (and at epoch 1 to seed adaptive update).
        probe_results = None
        adaptive_snapshot = None
        if ((epoch + 1) % probe_every_epochs == 0) or (epoch == 0):
            ps_t0 = time.time()
            probe_results = probe_suite.run(model)
            print(f"    PROBE SUITE ({time.time() - ps_t0:.1f}s):")
            for name, val in probe_results.items():
                print(f"      {name}: {val:.4f}")
            # Map probe-key -> scenario-key for sampling update.
            scenario_probes = {
                "pure_pregame":      probe_results["pure_pregame"],
                "partial_draft":     probe_results["partial_draft"],
                "duration_cond":     probe_results["duration_cond"],
                "items_cond":        probe_results["items_cond"],
                "outcome_cond":      probe_results["outcome_cond"],
                "partial_items":     probe_results["partial_items"],
                "kills_pair_probe":  probe_results["kills_pair_probe"],
            }
            adaptive_snapshot = sampler.update_probs(scenario_probes)
            print(f"    sampling probs after update: " +
                  " ".join(f"{k}={v:.3f}" for k, v in sampler.probs.items()))
            # Persist histories.
            try:
                with probe_history_path.open("a") as f:
                    f.write(json.dumps({"epoch": epoch + 1, "results": probe_results}) + "\n")
                with sampling_history_path.open("a") as f:
                    f.write(json.dumps({"epoch": epoch + 1, "snapshot": adaptive_snapshot}) + "\n")
            except Exception as e:  # noqa: BLE001
                print(f"    history write skipped: {e}")
            # Halt check (skip during smoke).
            if not smoke:
                hd_dec = probe_suite.halt_decision(probe_results, epoch=epoch + 1,
                                                     halt_at_epoch=halt_at_epoch)
                if hd_dec["halt"]:
                    halted = True
                    halt_reasons = hd_dec["reasons"]
                    print(f"    HALT at epoch {epoch+1}: {halt_reasons}")
                    history.append({
                        "epoch": epoch + 1,
                        "per_head_loss_avg": per_head_avg,
                        "train_total_loss": total_avg,
                        "wall_seconds": ep_dt,
                        "lr": cur_lr,
                        "scenarios": dict(scenario_counter),
                        "pure_pregame_val_auc": float(pure_pregame_auc),
                        "probe_results": probe_results,
                        "sampling_snapshot": adaptive_snapshot,
                        "halt_triggered": True,
                        "halt_reasons": halt_reasons,
                    })
                    break

        history.append({
            "epoch": epoch + 1,
            "per_head_loss_avg": per_head_avg,
            "train_total_loss": total_avg,
            "wall_seconds": ep_dt,
            "lr": cur_lr,
            "scenarios": dict(scenario_counter),
            "pure_pregame_val_auc": float(pure_pregame_auc),
            "probe_results": probe_results,
            "sampling_snapshot": adaptive_snapshot,
        })

        if pure_pregame_auc > best_pure_pregame + 1e-6:
            best_pure_pregame = pure_pregame_auc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if patience is not None and epochs_since_improve >= patience:
            print(f"  early stop at epoch {epoch+1} (best {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_sec = time.time() - t0
    return TrainResult(history=history,
                        best_pure_pregame_auc=best_pure_pregame,
                        best_epoch=best_epoch,
                        epochs_run=history[-1]["epoch"] if history else 0,
                        train_seconds=train_sec,
                        halted=halted, halt_reasons=halt_reasons)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--ablation", required=True, choices=["v7_unified"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--metrics-suffix", default="")
    ap.add_argument("--max-epochs-override", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())
    seed = int(cfg["seed"])
    set_seed(seed)

    feat_names = cfg["player_features"]["feat_names"]
    n_player_feats = int(cfg["player_features"]["n_player_feats"])
    source_dir = PROJECT_ROOT / cfg["player_features"]["source_dir"]
    sidecar_dir = PROJECT_ROOT / cfg["rich_cols"]["out_dir"]
    vp = cfg["item_vocab"]["vocab_path"]
    vocab_path = (EXP_DIR / vp).resolve() if not Path(vp).is_absolute() else Path(vp)

    print(f"Ablation: {args.ablation} smoke={args.smoke}")
    print(f"  source_dir={source_dir}")
    print(f"  sidecar_dir={sidecar_dir}")
    print(f"  vocab_path={vocab_path}")
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
            canonical_sort=canonical_sort, default_patch_id=default_patch_id)
    else:
        train_ds, val_ds, meta = load_train_val(
            seed=seed, n_target=n_target, feat_names=feat_names,
            source_dir=source_dir, splits=splits, smoke=False,
            sidecar_dir=sidecar_dir, vocab_path=vocab_path,
            canonical_sort=canonical_sort, default_patch_id=default_patch_id)
    data_seconds = time.time() - t0
    print(f"Data ready in {data_seconds:.1f}s -- train={len(train_ds):,} val={len(val_ds):,}")
    print(f"  train dates {meta['train_date_min']}..{meta['train_date_max']}")
    print(f"  val   dates {meta['val_date_min']}..{meta['val_date_max']}")

    mhp = cfg["transformer_model"]
    item_vocab_size = int(meta.get("item_vocab_size", 0)) or 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(mhp, vocab_size=int(cfg["hero"]["vocab_size"]),
                          n_player_feats=n_player_feats,
                          item_vocab_size=item_vocab_size)
    model = model.to(device)
    pc = count_params(model)
    print(f"Model: {pc}, device={device}")

    opt_cfg = cfg["transformer_optim"]
    max_epochs = (int(opt_cfg["max_epochs"]) if not args.smoke
                  else int(cfg["transformer_smoke"]["max_epochs"]))
    if args.max_epochs_override is not None:
        max_epochs = int(args.max_epochs_override)
        print(f"  max_epochs override: {max_epochs}")
    hp = {"batch_size": int(opt_cfg["batch_size"]), "lr": float(opt_cfg["lr"]),
          "weight_decay": float(opt_cfg["weight_decay"])}
    patience = int(opt_cfg.get("patience", 8)) if not args.smoke else None

    # Build ScenarioSampler from config.
    sc_cfg = cfg["scenarios"]
    initial_probs = {s: float(d["init_prob"]) for s, d in sc_cfg["distribution"].items()}
    loss_weights = {s: dict(d.get("loss_weights", {})) for s, d in sc_cfg["distribution"].items()}
    initial_targets = {s: float(d["probe_target"]) for s, d in sc_cfg["distribution"].items()
                        if d.get("probe_target") is not None}
    sampler = ScenarioSampler(initial_probs=initial_probs,
                               loss_weights=loss_weights,
                               initial_targets=initial_targets,
                               seed=seed)
    # Re-normalize probs in case YAML totals don't exactly sum to 1.
    tot = sum(sampler.probs.values())
    sampler.probs = {k: v / tot for k, v in sampler.probs.items()}

    # Build ProbeSuite.
    probe_cfg = cfg["probes"]
    halt_thresholds = {name: spec for name, spec in probe_cfg["halt_thresholds"].items()}
    autocast_dtype = (torch.bfloat16 if (bool(opt_cfg["mixed_precision"])
                                            and device.type == "cuda") else None)
    probe_suite = ProbeSuite(val_ds=val_ds, device=device,
                                autocast_dtype=autocast_dtype,
                                fixed_subset_size=int(probe_cfg["fixed_subset_size"])
                                if not args.smoke
                                else int(probe_cfg.get("smoke_subset_size", 2_000)),
                                seed=int(probe_cfg.get("seed", 42)),
                                batch_size=int(probe_cfg.get("batch_size", 1024)),
                                halt_thresholds=halt_thresholds)
    print(f"  probe subset: {len(probe_suite.subset):,} rows (of {len(val_ds):,})")
    print(f"  initial scenario probs: " +
          " ".join(f"{k}={v:.3f}" for k, v in sampler.probs.items()))

    results_dir = EXP_DIR / cfg["output"]["results_dir"]
    results_dir.mkdir(exist_ok=True, parents=True)
    sfx = args.metrics_suffix or f"_{args.ablation}"
    if args.smoke:
        sfx = cfg["transformer_smoke"]["metrics_suffix"] + f"_{args.ablation}"
    probe_history_path = EXP_DIR / f"probe_suite_history{sfx}.json"
    sampling_history_path = EXP_DIR / f"adaptive_sampling_history{sfx}.json"
    # Reset history files for fresh runs.
    probe_history_path.write_text("")
    sampling_history_path.write_text("")

    tr = train_v7(model, train_ds, val_ds, hp,
                    max_epochs=max_epochs, device=device,
                    mixed_precision=bool(opt_cfg["mixed_precision"]),
                    patience=patience, sampler=sampler, probe_suite=probe_suite,
                    probe_every_epochs=int(probe_cfg.get("every_epochs", 2)),
                    halt_at_epoch=int(probe_cfg.get("halt_at_epoch", 10)),
                    warmup_steps=int(opt_cfg.get("warmup_steps", 1000)),
                    cosine_min_lr=float(opt_cfg.get("cosine_min_lr", 1e-5)),
                    probe_history_path=probe_history_path,
                    sampling_history_path=sampling_history_path,
                    smoke=args.smoke)
    print(f"Training done in {tr.train_seconds:.1f}s "
          f"-- best pure_pregame_val_auc={tr.best_pure_pregame_auc:.4f} @ epoch {tr.best_epoch}")

    # Final probe-suite pass on the best checkpoint for canonical metrics.
    final_probes = probe_suite.run(model)
    print(f"FINAL PROBE SUITE: {final_probes}")

    # Checkpoint save.
    try:
        torch.save(model.state_dict(), results_dir / f"pretrain_encoder{sfx}.pt")
    except Exception as e:  # noqa: BLE001
        print(f"checkpoint save skipped: {e}")

    anchors = cfg.get("anchors", {})
    v4_anchor = float(anchors.get("v4_val_auc", 0.6471))
    iso_teambias = float(anchors.get("iso_teambias_val_auc", 0.6493))
    baseline_repro = float(anchors.get("baseline_multitask_repro_val_auc", 0.6470))
    target_pp = float(anchors.get("proposal_pure_pregame_target", 0.6471))
    target_items_cond = float(anchors.get("proposal_items_cond_target", 0.80))
    target_dur_cond = float(anchors.get("proposal_duration_cond_target", 0.68))

    metrics = {
        "ablation": args.ablation,
        "smoke": bool(args.smoke),
        "val_auc_pure_pregame": float(tr.best_pure_pregame_auc),
        "final_probe_results": final_probes,
        "best_epoch": int(tr.best_epoch),
        "epochs_run": int(tr.epochs_run),
        "max_epochs": int(max_epochs),
        "halted": bool(tr.halted),
        "halt_reasons": list(tr.halt_reasons),
        "train_seconds": float(tr.train_seconds),
        "data_seconds": float(data_seconds),
        "param_counts": pc,
        "history": tr.history,
        "scenario_distribution_final": sampler.probs,
        "scenario_distribution_initial": sampler.initial_probs,
        "scenario_loss_weights": sampler.loss_weights_per_scenario,
        "scenario_targets": sampler.targets,
        "model_hp": {k: mhp.get(k) for k in ("embed_dim", "d_model", "n_heads", "n_layers",
                                              "ff_mult", "dropout", "decoder_n_layers",
                                              "decoder_n_heads", "use_team_team_bias",
                                              "remove_first_layer_first_ln",
                                              "use_canonical_sort")},
        "optim_hp": {"batch_size": hp["batch_size"], "lr": hp["lr"],
                      "weight_decay": hp["weight_decay"],
                      "warmup_steps": int(opt_cfg.get("warmup_steps", 1000)),
                      "cosine_min_lr": float(opt_cfg.get("cosine_min_lr", 1e-5)),
                      "mixed_precision": bool(opt_cfg["mixed_precision"])},
        "n_train_pre_subsample": int(meta["n_train_pre_subsample"]),
        "n_train_post_subsample": int(meta["n_train_post_subsample"]),
        "n_val": int(meta["n_val"]),
        "train_date_min": meta["train_date_min"], "train_date_max": meta["train_date_max"],
        "val_date_min": meta["val_date_min"], "val_date_max": meta["val_date_max"],
        "radiant_base_rate_val": meta["radiant_base_rate_val"],
        "feat_names": list(feat_names),
        "n_player_feats": n_player_feats,
        "item_vocab_size": int(meta.get("item_vocab_size", 0)),
        "canonical_sort": meta.get("canonical_sort"),
        "default_patch_id": meta.get("default_patch_id"),
        "train_patch_id_distribution": meta.get("train_patch_id_distribution"),
        "val_patch_id_distribution": meta.get("val_patch_id_distribution"),
        "anchors": anchors,
        "delta_pure_pregame_vs_v4": float(tr.best_pure_pregame_auc - v4_anchor),
        "delta_pure_pregame_vs_iso_teambias": float(tr.best_pure_pregame_auc - iso_teambias),
        "delta_pure_pregame_vs_baseline_repro": float(tr.best_pure_pregame_auc - baseline_repro),
        "delta_pure_pregame_vs_target": float(tr.best_pure_pregame_auc - target_pp),
        "delta_items_cond_vs_target": float(final_probes.get("items_cond", float("nan")) - target_items_cond),
        "delta_duration_cond_vs_target": float(final_probes.get("duration_cond", float("nan")) - target_dur_cond),
    }
    # Two output names: canonical metrics_v7 + per-ablation suffix.
    out_path = EXP_DIR / f"metrics{sfx}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    if not args.smoke:
        canonical = EXP_DIR / "metrics_v7.json"
        canonical.write_text(json.dumps(metrics, indent=2))
        # Also dump to dvc-tracked metrics.json so the project picks it up.
        (EXP_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"Wrote {canonical}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
