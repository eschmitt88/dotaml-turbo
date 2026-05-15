"""Train LightGBM plateau-baseline-740, write metrics + plots.

HCE rule: this script reads only data/snapshots/.../processed/{train,val}.parquet.
It must NEVER read the test parquet (which doesn't even exist on disk).
Writes metrics.json (validation metrics) — no final_metrics.json.

Mirrors DotaML v3 recipe (300-dim one-hot heroes + 1 Radiant-side bit,
500 boosting rounds, lr 0.1, 31 leaves), trained on 5M-row stratified
subsample of train.
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
PROCESSED = SNAPSHOT_DIR / "processed"
RESULTS = EXP_DIR / "results"
SPLITS_PATH = PROJECT_ROOT / "splits.yaml"


def build_sparse_features(tbl, hero_min: int = 1, hero_max: int = 150,
                          add_side_bit: bool = True) -> sp.csr_matrix:
    """One-hot 300-dim hero features + 1 Radiant-side indicator (always 1).

    The Radiant-side bit is constant across rows (every row's perspective is
    Radiant); we still include it for proposal fidelity. It carries no
    discriminative signal but does not break LightGBM.
    """
    n = tbl.num_rows
    hero_dim = hero_max - hero_min + 1  # 150
    feat_dim = 2 * hero_dim + (1 if add_side_bit else 0)

    # Pull hero columns into numpy arrays.
    r_cols = [tbl.column(f"r{i}").to_numpy(zero_copy_only=False) for i in range(5)]
    d_cols = [tbl.column(f"d{i}").to_numpy(zero_copy_only=False) for i in range(5)]

    # Build COO indices.
    rows = np.repeat(np.arange(n, dtype=np.int64), 10 + (1 if add_side_bit else 0))
    nnz_per_row = 10 + (1 if add_side_bit else 0)

    # Hero columns: radiant heroes go to columns [hero_id - hero_min];
    # dire heroes go to columns [hero_dim + hero_id - hero_min].
    r_idx = np.stack(r_cols, axis=1).astype(np.int64) - hero_min
    d_idx = np.stack(d_cols, axis=1).astype(np.int64) - hero_min + hero_dim
    # Side bit at column = 2*hero_dim
    side_col = np.full((n, 1), 2 * hero_dim, dtype=np.int64)

    if add_side_bit:
        cols = np.concatenate([r_idx, d_idx, side_col], axis=1).reshape(-1)
    else:
        cols = np.concatenate([r_idx, d_idx], axis=1).reshape(-1)

    # Validate
    if cols.min() < 0 or cols.max() >= feat_dim:
        raise ValueError(f"feature col index out of range: [{cols.min()}, {cols.max()}], dim={feat_dim}")

    data = np.ones_like(cols, dtype=np.float32)
    coo = sp.coo_matrix((data, (rows, cols)), shape=(n, feat_dim))
    csr = coo.tocsr()
    # Multiple ones in same cell shouldn't happen (no duplicate heroes within a team)
    # but if a hero somehow appears twice in radiant, sum_duplicates collapses to 2.
    csr.sum_duplicates()
    return csr


def stratified_subsample(y: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """Return indices that stratify on y to roughly preserve class balance."""
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


def date_range_for_split(tbl, name: str) -> tuple[str, str]:
    sds = tbl.column("start_time_date").to_numpy(zero_copy_only=False)
    return (str(np.min(sds)), str(np.max(sds)))


def assert_no_test_dates(tbl_train, tbl_val, splits: dict) -> None:
    import datetime as dt
    test_lo = dt.date.fromisoformat(splits["test_start_date"])
    test_hi = dt.date.fromisoformat(splits["test_end_date"])
    for name, tbl in (("train", tbl_train), ("val", tbl_val)):
        sds = tbl.column("start_time_date").to_pylist()
        bad = [s for s in sds if test_lo <= dt.date.fromisoformat(s) <= test_hi]
        if bad:
            sys.exit(f"REFUSED: {name} split contains test-window dates {bad[:3]}... — HCE rule.")


def plot_calibration(y_true, p_pred, out: Path) -> dict:
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=20, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", lw=1.5, label="model")
    ax.set_xlabel("predicted P(radiant_win)")
    ax.set_ylabel("empirical P(radiant_win)")
    ax.set_title("Calibration (val, 20-quantile)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}


def plot_roc(y_true, p_pred, auc, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (val)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_learning(evals: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for split_name, metrics in evals.items():
        for metric_name, vals in metrics.items():
            ax.plot(vals, label=f"{split_name}/{metric_name}", lw=1)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("metric")
    ax.set_title("Learning curves")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP_DIR / "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load(SPLITS_PATH.read_text())

    seed = int(cfg["seed"])
    feat_cfg = cfg["features"]
    hero_min = int(feat_cfg["hero_id_min"])
    hero_max = int(feat_cfg["hero_id_max"])
    add_side = bool(feat_cfg["add_radiant_side_bit"])

    print("Reading processed parquet...")
    train_tbl = pq.read_table(PROCESSED / "train.parquet")
    val_tbl = pq.read_table(PROCESSED / "val.parquet")
    print(f"  train rows: {train_tbl.num_rows:,}")
    print(f"  val   rows: {val_tbl.num_rows:,}")

    assert_no_test_dates(train_tbl, val_tbl, splits)

    train_dr = date_range_for_split(train_tbl, "train")
    val_dr = date_range_for_split(val_tbl, "val")
    print(f"  train dates: {train_dr}")
    print(f"  val   dates: {val_dr}")

    y_train_full = train_tbl.column("radiant_win").to_numpy(zero_copy_only=False).astype(np.int8)
    y_val = val_tbl.column("radiant_win").to_numpy(zero_copy_only=False).astype(np.int8)

    radiant_base_train = float(y_train_full.mean())
    radiant_base_val = float(y_val.mean())
    print(f"  Radiant base rate — train: {radiant_base_train:.4f}, val: {radiant_base_val:.4f}")

    # Subsample train.
    n_train_pre = int(train_tbl.num_rows)
    n_target = int(cfg["train_subset_size"])
    sub_idx = stratified_subsample(y_train_full, n_target, seed)
    train_tbl_sub = train_tbl.take(sub_idx)
    y_train = y_train_full[sub_idx]
    print(f"  Subsampled train: {len(y_train):,} (target={n_target:,})")

    # Build features.
    print("Building sparse features...")
    t0 = time.time()
    X_train = build_sparse_features(train_tbl_sub, hero_min, hero_max, add_side)
    X_val = build_sparse_features(val_tbl, hero_min, hero_max, add_side)
    print(f"  X_train: {X_train.shape}, nnz/row≈{X_train.nnz/X_train.shape[0]:.1f}, "
          f"X_val: {X_val.shape}, build={time.time()-t0:.1f}s")

    # Sanity: every row should have exactly nnz_per_row hits (10 heroes + 1 side).
    expected_nnz = 10 + (1 if add_side else 0)
    actual_nnz_per_row = X_train.getnnz(axis=1)
    if not (actual_nnz_per_row.min() >= expected_nnz - 1 and actual_nnz_per_row.max() <= expected_nnz):
        # Allow 1 less if a hero collides between teams (impossible) or duplicate hero on a team.
        print(f"  WARN: nnz per row range [{actual_nnz_per_row.min()}, {actual_nnz_per_row.max()}]; expected {expected_nnz}")

    # LightGBM.
    mcfg = cfg["model"]
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
    print("LightGBM params:", params)

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    evals_result: dict = {}
    print(f"Training {mcfg['num_boost_round']} rounds...")
    t0 = time.time()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=int(mcfg["num_boost_round"]),
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.record_evaluation(evals_result), lgb.log_evaluation(period=50)],
    )
    train_seconds = time.time() - t0
    print(f"  done in {train_seconds:.0f}s")

    # Predict.
    print("Predicting...")
    p_train = booster.predict(X_train)
    p_val = booster.predict(X_val)

    train_auc = float(roc_auc_score(y_train, p_train))
    val_auc = float(roc_auc_score(y_val, p_val))
    train_acc = float(accuracy_score(y_train, (p_train >= 0.5).astype(int)))
    val_acc = float(accuracy_score(y_val, (p_val >= 0.5).astype(int)))
    train_logloss = float(log_loss(y_train, p_train))
    val_logloss = float(log_loss(y_val, p_val))
    val_brier = float(brier_score_loss(y_val, p_val))

    print(f"  train: auc={train_auc:.4f} acc={train_acc:.4f} logloss={train_logloss:.4f}")
    print(f"  val:   auc={val_auc:.4f} acc={val_acc:.4f} logloss={val_logloss:.4f} brier={val_brier:.4f}")

    # Side-conditional acc — what does Radiant base-rate predict alone?
    base_acc_val = max(radiant_base_val, 1 - radiant_base_val)
    print(f"  val majority-class acc: {base_acc_val:.4f}")

    # Plots.
    RESULTS.mkdir(exist_ok=True, parents=True)
    cal_path = RESULTS / "calibration.png"
    roc_path = RESULTS / "roc.png"
    learn_path = RESULTS / "learning_curve.png"
    cal = plot_calibration(y_val, p_val, cal_path)
    plot_roc(y_val, p_val, val_auc, roc_path)
    plot_learning(evals_result, learn_path)
    print(f"  wrote {cal_path}, {roc_path}, {learn_path}")

    # Save the booster for future reference.
    model_path = RESULTS / "lightgbm.txt"
    booster.save_model(str(model_path))
    print(f"  wrote {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")

    # Build metrics.json. Validation is the search signal.
    metrics = {
        # Headline (used by /iterate, /lint).
        "val_auc": val_auc,
        "val_acc": val_acc,
        "val_log_loss": val_logloss,
        "val_brier": val_brier,
        # Anchor / overfitting check.
        "train_auc": train_auc,
        "train_acc": train_acc,
        "train_log_loss": train_logloss,
        "train_val_auc_gap": train_auc - val_auc,
        # Counts.
        "n_train_pre_subsample": n_train_pre,
        "n_train_post_subsample": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_subset_size_target": n_target,
        "train_subset_seed": seed,
        # Sanity.
        "radiant_base_rate_train_full": radiant_base_train,
        "radiant_base_rate_train_subsampled": float(y_train.mean()),
        "radiant_base_rate_val": radiant_base_val,
        "val_majority_class_acc": base_acc_val,
        # Date ranges (HCE leakage anchor).
        "train_date_min": train_dr[0],
        "train_date_max": train_dr[1],
        "val_date_min": val_dr[0],
        "val_date_max": val_dr[1],
        # Run metadata.
        "model": "lightgbm",
        "num_boost_round": int(mcfg["num_boost_round"]),
        "learning_rate": mcfg["learning_rate"],
        "num_leaves": mcfg["num_leaves"],
        "feature_dim": int(X_train.shape[1]),
        "train_seconds": train_seconds,
        "calibration_quantile_bins": 20,
        "calibration": cal,
        # Comparison anchor.
        "prior_art_dotaml_v3_test_auc": 0.6189,
        "prior_art_dotaml_v3_test_acc": 0.5882,
        "delta_val_auc_vs_v3_test": val_auc - 0.6189,
        "delta_val_acc_vs_v3_test": val_acc - 0.5882,
    }
    out = EXP_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
