"""Train LightGBM player-features-740: heroes + per-player leading-window feats.

Mirrors plateau-baseline-740 hyperparameters for direct A/B. Adds dense
player features alongside the sparse 300-dim hero one-hot + 1-bit side.

HCE: reads only augmented processed/{train,val}.parquet under
player_features/. Never reads test split. Writes metrics.json.

Ablations:
  --ablation heroes_only        : sparse hero one-hot + side bit ONLY (must
                                  reproduce plateau-baseline-740 within 0.001)
  --ablation features_only      : dense player feats ONLY (pure player signal)
  --ablation heroes_plus_features (default): both
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = PROJECT_ROOT / "data/snapshots/7.40-2025-12-16"
RESULTS = EXP_DIR / "results"
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"

FEAT_NAMES_PER_PLAYER = [
    "n_games_log1p",
    "smoothed_winrate",
    "smoothed_winrate_hero",
    "last10_winrate",
    "days_since_last_log1p",
    "n_games_hero_log1p",
    "hero_diversity_log1p",
    "coplay_mean",
    "is_anonymous",
]
N_PLAYERS = 10


def player_feat_cols() -> list[str]:
    return [f"p{p}_{f}" for p in range(N_PLAYERS) for f in FEAT_NAMES_PER_PLAYER]


def build_sparse_heroes(tbl, hero_min: int = 1, hero_max: int = 150,
                        add_side_bit: bool = True) -> sp.csr_matrix:
    """One-hot 300-dim hero features + 1 Radiant-side indicator (constant)."""
    n = tbl.num_rows
    hero_dim = hero_max - hero_min + 1  # 150
    feat_dim = 2 * hero_dim + (1 if add_side_bit else 0)

    r_cols = [tbl.column(f"r{i}").to_numpy(zero_copy_only=False) for i in range(5)]
    d_cols = [tbl.column(f"d{i}").to_numpy(zero_copy_only=False) for i in range(5)]

    rows = np.repeat(np.arange(n, dtype=np.int64), 10 + (1 if add_side_bit else 0))
    r_idx = np.stack(r_cols, axis=1).astype(np.int64) - hero_min
    d_idx = np.stack(d_cols, axis=1).astype(np.int64) - hero_min + hero_dim
    side_col = np.full((n, 1), 2 * hero_dim, dtype=np.int64)
    if add_side_bit:
        cols = np.concatenate([r_idx, d_idx, side_col], axis=1).reshape(-1)
    else:
        cols = np.concatenate([r_idx, d_idx], axis=1).reshape(-1)

    if cols.min() < 0 or cols.max() >= feat_dim:
        raise ValueError(f"feature col index out of range: [{cols.min()}, {cols.max()}]")
    data = np.ones_like(cols, dtype=np.float32)
    coo = sp.coo_matrix((data, (rows, cols)), shape=(n, feat_dim))
    csr = coo.tocsr()
    csr.sum_duplicates()
    return csr


def build_dense_player_feats(tbl) -> np.ndarray:
    """Stack 90 dense player feature columns into a (n, 90) float32 array."""
    cols = player_feat_cols()
    arrs = []
    for c in cols:
        a = tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
        arrs.append(a)
    X = np.stack(arrs, axis=1)
    return X


def stratified_subsample(y: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y)
    if n_target >= n:
        return rng.permutation(n)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    p_pos = len(pos_idx) / n
    n_pos = int(round(n_target * p_pos))
    n_neg = n_target - n_pos
    pos_pick = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_pick = rng.choice(neg_idx, size=n_neg, replace=False)
    out = np.concatenate([pos_pick, neg_pick])
    rng.shuffle(out)
    return out


def date_range(tbl) -> tuple[str, str]:
    sds = tbl.column("start_time_date").to_numpy(zero_copy_only=False)
    return (str(np.min(sds)), str(np.max(sds)))


def assert_no_test_dates(tbl, splits: dict, name: str) -> None:
    import datetime as dt
    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    sds = tbl.column("start_time_date").to_pylist()
    bad = [s for s in sds if test_lo <= dt.date.fromisoformat(s) <= test_hi]
    if bad:
        sys.exit(f"REFUSED: {name} contains test-window dates {bad[:3]}... HCE violated.")


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


def plot_learning(evals: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for split_name, metrics in evals.items():
        for metric_name, vals in metrics.items():
            ax.plot(vals, label=f"{split_name}/{metric_name}", lw=1)
    ax.set_xlabel("boosting round"); ax.set_ylabel("metric")
    ax.set_title("Learning curves")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    ap.add_argument("--ablation", default="heroes_plus_features",
                    choices=["heroes_only", "features_only", "heroes_plus_features"])
    ap.add_argument("--smoke", action="store_true",
                    help="Use *_smoke.parquet and reduce num_boost_round.")
    ap.add_argument("--metrics-suffix", default="",
                    help="Suffix for metrics filename (e.g. _ablation_heroes_only).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    seed = int(cfg["seed"])
    feat_cfg = cfg["features"]
    hero_min = int(feat_cfg["hero_id_min"])
    hero_max = int(feat_cfg["hero_id_max"])
    add_side = bool(feat_cfg["add_radiant_side_bit"])
    pf_dir = PROJECT_ROOT / cfg["player_features"]["out_dir"]

    suffix = "_smoke" if args.smoke else ""
    train_path = pf_dir / f"train{suffix}.parquet"
    val_path = pf_dir / f"val{suffix}.parquet"

    print(f"Reading augmented parquet: {train_path.name}, {val_path.name}")
    train_tbl = pq.read_table(train_path)
    val_tbl = pq.read_table(val_path)
    print(f"  train rows: {train_tbl.num_rows:,}")
    print(f"  val   rows: {val_tbl.num_rows:,}")

    # Smoke fallback: if val is empty (because smoke covers only early days
    # before the val window starts), carve a 10% tail off train_smoke as a
    # pseudo-val so the pipeline can be exercised end-to-end. This is a
    # pipeline-correctness check only — NOT a real validation signal.
    if args.smoke and val_tbl.num_rows == 0:
        n_t = train_tbl.num_rows
        n_v = max(1000, n_t // 10)
        print(f"  SMOKE: val parquet empty; carving tail {n_v:,} rows off "
              f"train as pseudo-val (pipeline test only).")
        val_tbl = train_tbl.slice(n_t - n_v, n_v)
        train_tbl = train_tbl.slice(0, n_t - n_v)
        print(f"  train rows (post-carve): {train_tbl.num_rows:,}")
        print(f"  val   rows (post-carve): {val_tbl.num_rows:,}")

    assert_no_test_dates(train_tbl, splits, "train")
    assert_no_test_dates(val_tbl, splits, "val")

    train_dr = date_range(train_tbl)
    val_dr = date_range(val_tbl)
    print(f"  train dates: {train_dr}")
    print(f"  val   dates: {val_dr}")

    y_train_full = train_tbl.column("radiant_win").to_numpy(zero_copy_only=False).astype(np.int8)
    y_val = val_tbl.column("radiant_win").to_numpy(zero_copy_only=False).astype(np.int8)

    radiant_base_train = float(y_train_full.mean())
    radiant_base_val = float(y_val.mean())
    print(f"  Radiant base rate — train: {radiant_base_train:.4f}, val: {radiant_base_val:.4f}")

    # Subsample
    n_train_pre = int(train_tbl.num_rows)
    n_target = int(cfg["train_subset_size"])
    if args.smoke:
        n_target = int(cfg["smoke"].get("max_matches", n_target))
    sub_idx = stratified_subsample(y_train_full, n_target, seed)
    train_tbl_sub = train_tbl.take(sub_idx)
    y_train = y_train_full[sub_idx]
    print(f"  Subsampled train: {len(y_train):,} (target={n_target:,})")

    # Build features
    print(f"Building features (ablation={args.ablation})...")
    t0 = time.time()
    feature_names: list[str] = []
    if args.ablation == "heroes_only":
        Xtr = build_sparse_heroes(train_tbl_sub, hero_min, hero_max, add_side)
        Xva = build_sparse_heroes(val_tbl, hero_min, hero_max, add_side)
        hero_dim = hero_max - hero_min + 1
        feature_names = (
            [f"r_hero_{h}" for h in range(hero_min, hero_max + 1)]
            + [f"d_hero_{h}" for h in range(hero_min, hero_max + 1)]
            + (["side_bit"] if add_side else [])
        )
    elif args.ablation == "features_only":
        Xtr_dense = build_dense_player_feats(train_tbl_sub)
        Xva_dense = build_dense_player_feats(val_tbl)
        Xtr = sp.csr_matrix(Xtr_dense.astype(np.float32))
        Xva = sp.csr_matrix(Xva_dense.astype(np.float32))
        feature_names = player_feat_cols()
    else:
        Xtr_h = build_sparse_heroes(train_tbl_sub, hero_min, hero_max, add_side)
        Xva_h = build_sparse_heroes(val_tbl, hero_min, hero_max, add_side)
        Xtr_d = build_dense_player_feats(train_tbl_sub)
        Xva_d = build_dense_player_feats(val_tbl)
        Xtr = sp.hstack([Xtr_h, sp.csr_matrix(Xtr_d)], format="csr").astype(np.float32)
        Xva = sp.hstack([Xva_h, sp.csr_matrix(Xva_d)], format="csr").astype(np.float32)
        hero_dim = hero_max - hero_min + 1
        feature_names = (
            [f"r_hero_{h}" for h in range(hero_min, hero_max + 1)]
            + [f"d_hero_{h}" for h in range(hero_min, hero_max + 1)]
            + (["side_bit"] if add_side else [])
            + player_feat_cols()
        )

    print(f"  Xtr: {Xtr.shape}, Xva: {Xva.shape}, build={time.time()-t0:.1f}s")

    # LightGBM
    mcfg = cfg["model"]
    n_rounds = int(mcfg["num_boost_round"])
    if args.smoke:
        n_rounds = int(cfg["smoke"]["num_boost_round"])
    params = dict(
        objective=mcfg["objective"],
        metric=mcfg["metric"],
        learning_rate=mcfg["learning_rate"],
        num_leaves=mcfg["num_leaves"],
        feature_fraction=mcfg["feature_fraction"],
        bagging_fraction=mcfg["bagging_fraction"],
        min_data_in_leaf=mcfg["min_data_in_leaf"],
        reg_alpha=mcfg["reg_alpha"],
        reg_lambda=mcfg["reg_lambda"],
        num_threads=mcfg["num_threads"],
        verbose=mcfg["verbose"],
        seed=seed,
        feature_pre_filter=False,
    )
    print(f"LightGBM params: {params}, num_boost_round={n_rounds}")
    dtrain = lgb.Dataset(Xtr, label=y_train, free_raw_data=False,
                         feature_name=feature_names)
    dval = lgb.Dataset(Xva, label=y_val, reference=dtrain, free_raw_data=False)

    evals_result: dict = {}
    print(f"Training {n_rounds} rounds...")
    t0 = time.time()
    booster = lgb.train(
        params, dtrain, num_boost_round=n_rounds,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.record_evaluation(evals_result),
                   lgb.log_evaluation(period=max(10, n_rounds // 10))],
    )
    train_seconds = time.time() - t0

    p_train = booster.predict(Xtr)
    p_val = booster.predict(Xva)
    train_auc = float(roc_auc_score(y_train, p_train))
    val_auc = float(roc_auc_score(y_val, p_val))
    train_acc = float(accuracy_score(y_train, (p_train >= 0.5).astype(int)))
    val_acc = float(accuracy_score(y_val, (p_val >= 0.5).astype(int)))
    train_ll = float(log_loss(y_train, p_train))
    val_ll = float(log_loss(y_val, p_val))
    val_brier = float(brier_score_loss(y_val, p_val))

    print(f"  train: auc={train_auc:.4f} acc={train_acc:.4f} logloss={train_ll:.4f}")
    print(f"  val:   auc={val_auc:.4f} acc={val_acc:.4f} logloss={val_ll:.4f} brier={val_brier:.4f}")

    # Heroes-only sanity check
    anchors = cfg.get("anchors", {})
    baseline_auc = float(anchors.get("plateau_baseline_740_val_auc", 0.6161))
    heroes_only_check = None
    if args.ablation == "heroes_only" and not args.smoke:
        delta = val_auc - baseline_auc
        passed = abs(delta) <= 0.001
        heroes_only_check = {
            "baseline_val_auc": baseline_auc,
            "this_val_auc": val_auc,
            "delta": delta,
            "passed_within_0.001": passed,
        }
        verdict = "PASS" if passed else "FAIL"
        print(f"  HEROES-ONLY SANITY CHECK: {verdict} (Δ={delta:+.4f})")

    # Coverage-bucketed val_auc
    coverage_bucket_info = None
    if args.ablation != "heroes_only":
        n_games_cols = [f"p{p}_n_games_log1p" for p in range(N_PLAYERS)]
        ngs = np.stack(
            [val_tbl.column(c).to_numpy(zero_copy_only=False) for c in n_games_cols],
            axis=1,
        )
        coverage = ngs.mean(axis=1)
        q33, q67 = np.quantile(coverage, [0.333, 0.667])
        buckets = np.digitize(coverage, [q33, q67])  # 0,1,2
        bucket_aucs = {}
        for b, name in [(0, "low"), (1, "medium"), (2, "high")]:
            mask = buckets == b
            n = int(mask.sum())
            if n < 100:
                bucket_aucs[name] = {"n": n, "val_auc": None,
                                      "mean_coverage_log1p": None}
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
        coverage_bucket_info = {
            "quantile_edges_log1p": [float(q33), float(q67)],
            "buckets": bucket_aucs,
        }

    # Anonymous-per-match histogram in val
    anon_arr = val_tbl.column("n_anonymous_in_match").to_numpy(zero_copy_only=False)
    anon_hist = {str(i): int((anon_arr == i).sum()) for i in range(11)}

    # Feature importance (top-20)
    gain = booster.feature_importance(importance_type="gain")
    names_iter = feature_names if feature_names else [f"f{i}" for i in range(len(gain))]
    pairs = sorted(zip(names_iter, gain.tolist()), key=lambda x: -x[1])[:20]
    feature_importance_top20 = [{"name": n, "gain": float(g)} for n, g in pairs]

    # Plots
    RESULTS.mkdir(exist_ok=True, parents=True)
    sfx = args.metrics_suffix if args.metrics_suffix else ""
    cal = plot_calibration(y_val, p_val, RESULTS / f"calibration{sfx}.png")
    plot_roc(y_val, p_val, val_auc, RESULTS / f"roc{sfx}.png")
    plot_learning(evals_result, RESULTS / f"learning_curve{sfx}.png")

    # Save booster
    booster.save_model(str(RESULTS / f"lightgbm{sfx}.txt"))

    # Build metrics.json
    metrics = {
        "ablation": args.ablation,
        "smoke": bool(args.smoke),
        "val_auc": val_auc,
        "val_acc": val_acc,
        "val_log_loss": val_ll,
        "val_brier": val_brier,
        "train_auc": train_auc,
        "train_acc": train_acc,
        "train_log_loss": train_ll,
        "train_val_auc_gap": train_auc - val_auc,
        "n_train_pre_subsample": n_train_pre,
        "n_train_post_subsample": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_subset_size_target": n_target,
        "train_subset_seed": seed,
        "radiant_base_rate_train_full": radiant_base_train,
        "radiant_base_rate_train_subsampled": float(y_train.mean()),
        "radiant_base_rate_val": radiant_base_val,
        "val_majority_class_acc": max(radiant_base_val, 1 - radiant_base_val),
        "train_date_min": train_dr[0], "train_date_max": train_dr[1],
        "val_date_min": val_dr[0], "val_date_max": val_dr[1],
        "model": "lightgbm",
        "num_boost_round": n_rounds,
        "learning_rate": mcfg["learning_rate"],
        "num_leaves": mcfg["num_leaves"],
        "feature_dim": int(Xtr.shape[1]),
        "train_seconds": train_seconds,
        "calibration_quantile_bins": 20,
        "calibration": cal,
        "anchors": anchors,
        "delta_vs_plateau_baseline_740_val_auc": val_auc - baseline_auc,
        "delta_vs_plateau_architectures_740_val_auc": val_auc - float(
            anchors.get("plateau_architectures_740_val_auc", 0.6322)),
        "delta_vs_transformer_hp_sweep_740_val_auc": val_auc - float(
            anchors.get("transformer_hp_sweep_740_val_auc", 0.6318)),
        "delta_vs_proposal_target_val_auc": val_auc - float(
            anchors.get("proposal_target_val_auc", 0.6361)),
        "feature_importance_top20": feature_importance_top20,
        "coverage_bucket_val_auc": coverage_bucket_info,
        "anonymous_per_match_hist_val": anon_hist,
        "heroes_only_sanity_check": heroes_only_check,
    }
    out_name = f"metrics{sfx}.json"
    if args.smoke and not args.metrics_suffix:
        out_name = "metrics_smoke.json"
    out = EXP_DIR / out_name
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
