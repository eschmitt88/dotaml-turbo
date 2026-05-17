"""Run all three architectures sequentially, then aggregate metrics.

Usage:
    nohup .venv/bin/python experiments/2026-05-15-plateau-architectures-740/run_all.py \\
        > /tmp/dotaml_arch.log 2>&1 &

Estimated wall: ~90 min on RTX 5080 (3 arches × ~30 min each, sometimes less
because early stopping kicks in).

HCE rule: only validation metrics. No final_metrics.json is ever written here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
RESULTS = EXP_DIR / "results"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

ARCHES = ["simple_ffn", "residual_ffn", "transformer"]

PRIOR_ART = {
    "simple_ffn":   {"label": "DotaML v4 SimpleFFN",   "test_auc": 0.6285, "params": 47000},
    "residual_ffn": {"label": "DotaML v5 ResidualFFN", "test_auc": 0.6310, "params": 228000},
    "transformer":  {"label": "DotaML v6 Transformer", "test_auc": 0.6354, "params": 150000},
}

LIGHTGBM_BASELINE_VAL_AUC = 0.6160890687449119  # plateau-baseline-740/metrics.json


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_lines = []
    overall_t0 = time.time()
    per_arch: dict[str, dict] = {}

    for arch in ARCHES:
        print(f"\n{'='*70}\n=== {arch} ===\n{'='*70}", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [str(PYTHON), str(EXP_DIR / "train.py"), "--arch", arch],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        dt_s = time.time() - t0
        if proc.returncode != 0:
            print(f"!!! {arch} FAILED with rc={proc.returncode} after {dt_s:.0f}s", flush=True)
            log_lines.append(f"{arch}: FAILED rc={proc.returncode} ({dt_s:.0f}s)")
            continue
        log_lines.append(f"{arch}: OK ({dt_s:.0f}s)")
        m_path = RESULTS / f"{arch}_metrics.json"
        if not m_path.exists():
            print(f"!!! {arch} returned 0 but {m_path} missing", flush=True)
            continue
        per_arch[arch] = json.loads(m_path.read_text())

    overall_seconds = time.time() - overall_t0

    # Build comparison table.
    table_rows = [
        ["arch", "params_total", "val_auc", "val_acc", "val_log_loss", "Δ_vs_lgbm",
         "prior_art_test_auc", "Δ_vs_prior"]
    ]
    for arch in ARCHES:
        if arch not in per_arch:
            table_rows.append([arch, "—", "FAILED", "—", "—", "—", "—", "—"])
            continue
        m = per_arch[arch]
        prior = PRIOR_ART[arch]["test_auc"]
        table_rows.append([
            arch,
            m["param_counts"]["total"],
            round(m["val_auc"], 4),
            round(m["val_acc"], 4),
            round(m["val_log_loss"], 4),
            round(m["val_auc"] - LIGHTGBM_BASELINE_VAL_AUC, 4),
            prior,
            round(m["val_auc"] - prior, 4),
        ])

    # Pairwise gap analysis (rank-order hypothesis).
    pair_gaps = {}
    if all(a in per_arch for a in ARCHES):
        s = per_arch["simple_ffn"]["val_auc"]
        r = per_arch["residual_ffn"]["val_auc"]
        t = per_arch["transformer"]["val_auc"]
        pair_gaps = {
            "transformer_vs_residual": t - r,
            "residual_vs_simple": r - s,
            "transformer_vs_simple": t - s,
            "rank_order_holds": (t >= r >= s) and (s > LIGHTGBM_BASELINE_VAL_AUC),
        }

    combined = {
        "lightgbm_baseline_val_auc": LIGHTGBM_BASELINE_VAL_AUC,
        "per_arch": per_arch,
        "comparison_table": table_rows,
        "pair_gaps": pair_gaps,
        "prior_art": PRIOR_ART,
        "overall_seconds": overall_seconds,
        "log": log_lines,
        # Headline (copies for /iterate-friendly access).
        "val_auc_simple_ffn":   per_arch.get("simple_ffn",   {}).get("val_auc"),
        "val_auc_residual_ffn": per_arch.get("residual_ffn", {}).get("val_auc"),
        "val_auc_transformer":  per_arch.get("transformer",  {}).get("val_auc"),
        # Pick the best arch's val_auc as the headline metric (rank-ordered for /iterate).
        "val_auc": max(
            (m["val_auc"] for m in per_arch.values() if "val_auc" in m),
            default=None,
        ),
        "best_arch": max(per_arch, key=lambda a: per_arch[a]["val_auc"]) if per_arch else None,
    }

    out = EXP_DIR / "metrics.json"
    out.write_text(json.dumps(combined, indent=2))
    print(f"\nWrote {out}")
    print(f"Total wall: {overall_seconds/60:.1f} min")
    print("Per-arch summary:")
    for row in table_rows:
        print("  " + "  ".join(str(c) for c in row))
    return 0 if per_arch else 1


if __name__ == "__main__":
    sys.exit(main())
