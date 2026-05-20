"""Phase C: replay Phase 5 benchmark comparison across BTC/ETH/SOL.

For each Bybit-spot asset, compute per-test-day F1 of four detectors
against persistence-labeled ground truth:

    - raw_lomn      : |LOMN stat| >= 4.0  (the Section-3 threshold)
    - lee_mykland   : nonparametric, K=270, alpha=0.05
    - lomn_xgb      : XGBoost trained on this asset (53 features)
    - pure_ml       : XGBoost without LOMN features (51 features)

Matches the Phase 5 methodology (TOLERANCE_S=60, persistence positives
on full labeled set, train on first 12 days / test on last 2). The
output is the same 4-row F1 comparison table from Section 5.3, now
extended to 3 columns (one per asset).

Outputs:
    results/phase_c_ext/benchmark_f1_per_asset.csv
    results/phase_c_ext/benchmark_f1.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.benchmarks.f1_evaluation import f1_match
from src.benchmarks.lee_mykland import detect_jumps
from src.lomn.detector import block_minima, optimal_block_size, robust_scale
from src.realdata.phase_c_runner import (
    DEFAULT_SYMBOLS,
    build_symbol_dataset,
    train_eval_split,
)
from src.realdata.train_xgb import (
    BUCKET_DERIVED_COLS,
    BUCKET_RAW_COLS,
    FEATURE_COLS,
    FEATURE_COLS_L20,
)

LOG = logging.getLogger("phase-c-bench")

TOLERANCE_S = 60
RAW_LOMN_THRESHOLD = 4.0
ML_PROBA_THRESHOLD = 0.5
LM_K = 270
LM_ALPHA = 0.05
TEST_N_DAYS = 2


def _train_pure_ml(labeled: pd.DataFrame, days_train: list[str]) -> tuple[xgb.XGBClassifier, list[str]]:
    """XGBoost on FEATURE_COLS minus the two LOMN ones; same hyperparams as runner."""
    cols = [c for c in FEATURE_COLS_L20 if not c.startswith("f_lomn_")]
    train = labeled[labeled["day"].isin(days_train) & (labeled["label"] != -1)]
    Xtr = train[cols].values
    ytr = train["label"].values.astype(int)
    if int(ytr.sum()) == 0 or int((1 - ytr).sum()) == 0:
        raise RuntimeError("degenerate pure_ml training set")
    pos_w = (1 - ytr).sum() / max(1, int(ytr.sum()))
    val = int(0.85 * len(ytr))
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=pos_w, eval_metric="logloss",
        random_state=42, n_jobs=-1, early_stopping_rounds=20,
    )
    model.fit(Xtr[:val], ytr[:val],
              eval_set=[(Xtr[val:], ytr[val:])], verbose=False)
    return model, cols


def benchmark_one_asset(
    symbol: str, book_dir: Path, trades_dir: Path,
) -> dict:
    LOG.info("====== %s ======", symbol)
    labeled, _ = build_symbol_dataset(symbol, book_dir, trades_dir)
    res = train_eval_split(labeled, n_test_days=TEST_N_DAYS)
    if "error" in res:
        return {"error": res["error"]}
    test_days = res["test_days"]
    train_days = res["train_days"]
    xgb_model = res["model"]
    feat_cols = res["feat_cols"]
    LOG.info("%s test days: %s", symbol, test_days)

    pure_model, pure_cols = _train_pure_ml(labeled, train_days)

    # Per-day evaluation on test
    rows = []
    for d in test_days:
        # Truth positions on this day (obs indices of persistence positives)
        truth = labeled[(labeled["day"] == d) & (labeled["label"] == 1)]["obs_idx"].values.astype(int)
        # Subset of labeled candidates on this day
        g = labeled[labeled["day"] == d]
        if len(g) == 0:
            continue
        # raw LOMN
        det_lomn = g["obs_idx"].values[g["f_lomn_abs_std"] >= RAW_LOMN_THRESHOLD].astype(int)
        # LOMN+XGB
        p_xgb = xgb_model.predict_proba(g[feat_cols].values)[:, 1]
        det_xgb = g["obs_idx"].values[p_xgb >= ML_PROBA_THRESHOLD].astype(int)
        # pure ML
        p_pure = pure_model.predict_proba(g[pure_cols].values)[:, 1]
        det_pure = g["obs_idx"].values[p_pure >= ML_PROBA_THRESHOLD].astype(int)
        # Lee-Mykland on the day's log_mid (full 86,401 grid)
        book_path = book_dir / symbol.lower() / f"resampled_1s_{d}.parquet"
        book = pd.read_parquet(book_path)
        lm = detect_jumps(book["log_mid"].values, K=LM_K, alpha=LM_ALPHA)
        det_lm = lm["detected_obs_idx"].astype(int)

        for method, det in [
            ("raw_lomn", det_lomn),
            ("lee_mykland", det_lm),
            ("lomn_xgb", det_xgb),
            ("pure_ml", det_pure),
        ]:
            stats = f1_match(det, truth, tol=TOLERANCE_S)
            rows.append({"method": method, "day": d, "symbol": symbol,
                         "n_detected": int(len(det)), "n_truth": int(len(truth)),
                         **stats})

    per_day = pd.DataFrame(rows)
    # Aggregate across test days
    agg = per_day.groupby("method").agg(
        TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"),
        n_truth=("n_truth", "sum"), n_detected=("n_detected", "sum"),
    )
    agg["precision"] = agg["TP"] / (agg["TP"] + agg["FP"]).clip(lower=1)
    agg["recall"] = agg["TP"] / (agg["TP"] + agg["FN"]).clip(lower=1)
    agg["F1"] = 2 * agg["precision"] * agg["recall"] / (
        agg["precision"] + agg["recall"]
    ).clip(lower=1e-9)
    return {
        "test_days": test_days,
        "per_day": per_day,
        "agg": agg.reset_index(),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--trades-dir", type=Path, default=Path("data/historical"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase_c_ext"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_day_rows = []
    for sym in DEFAULT_SYMBOLS:
        out = benchmark_one_asset(sym, args.book_dir, args.trades_dir)
        if "error" in out:
            LOG.warning("%s: %s", sym, out["error"])
            continue
        per_day_rows.append(out["per_day"])
        for _, r in out["agg"].iterrows():
            rows.append({
                "symbol": sym,
                "method": r["method"],
                "TP": int(r["TP"]), "FP": int(r["FP"]), "FN": int(r["FN"]),
                "n_truth": int(r["n_truth"]), "n_detected": int(r["n_detected"]),
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "F1": float(r["F1"]),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "benchmark_f1_per_asset.csv", index=False)
    pd.concat(per_day_rows, ignore_index=True).to_csv(
        args.out_dir / "benchmark_per_day.csv", index=False
    )
    print("\n=== F1 per asset per method ===")
    pivot = summary.pivot(index="method", columns="symbol", values="F1").reindex(
        ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== FP per asset per method ===")
    fps = summary.pivot(index="method", columns="symbol", values="FP").reindex(
        ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]
    )
    print(fps.to_string(float_format=lambda x: f"{x:.0f}"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    methods_order = ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]
    colors = {"BTCUSDT": "#264653", "ETHUSDT": "#2a9d8f", "SOLUSDT": "#e76f51"}
    width = 0.25
    x = np.arange(len(methods_order))
    for i, sym in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
        sub = summary[summary["symbol"] == sym].set_index("method").reindex(methods_order)
        axes[0].bar(x + (i - 1) * width, sub["F1"], width, label=sym, color=colors[sym])
        axes[1].bar(x + (i - 1) * width, sub["FP"], width, label=sym, color=colors[sym])
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods_order, rotation=12)
    axes[0].set_ylabel("F1 (test days)")
    axes[0].set_title("F1 score by method × asset")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_xticks(x); axes[1].set_xticklabels(methods_order, rotation=12)
    axes[1].set_ylabel("False positives over 2 test days")
    axes[1].set_yscale("log")
    axes[1].set_title("False positives (log scale)")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Phase C benchmarks — F1 and FP across BTC / ETH / SOL")
    fig.tight_layout()
    fig.savefig(args.out_dir / "benchmark_f1.png", dpi=150)
    plt.close(fig)

    # JSON
    j = {
        "F1_by_method_x_asset": pivot.to_dict(),
        "FP_by_method_x_asset": fps.to_dict(),
    }
    (args.out_dir / "benchmark_summary.json").write_text(json.dumps(j, indent=2))


if __name__ == "__main__":
    main()
