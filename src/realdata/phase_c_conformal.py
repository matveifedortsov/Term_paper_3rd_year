"""Phase C: split-conformal wrapper per asset.

Replicates Phase A6 per asset on Bybit spot. For each of BTC/ETH/SOL:
    1. Split labeled days into fit (first 11), cal (next 2), test (last 2)
    2. Train XGBoost on fit
    3. Compute non-conformity scores on cal, find q-hat at alpha=0.10
    4. Apply prediction-set rule to test; report coverage, abstain rate,
       singleton precision

Output: results/phase_c_ext/conformal_per_asset.{csv,json,png}
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

from src.benchmarks.conformal import (
    evaluate_sets,
    predict_sets,
    split_conformal_threshold,
)
from src.realdata.phase_c_runner import DEFAULT_SYMBOLS, build_symbol_dataset
from src.realdata.train_xgb import FEATURE_COLS_L20, select_feature_cols

LOG = logging.getLogger("phase-c-conformal")
ALPHA = 0.10
ML_THRESHOLD = 0.5


def conformal_one_asset(symbol: str, labeled: pd.DataFrame) -> dict:
    feat_cols = select_feature_cols(labeled)
    labeled_valid = labeled[labeled["label"] != -1].copy()
    days = sorted(labeled_valid["day"].unique())
    if len(days) < 5:
        return {"error": "need >=5 days"}
    fit_days = days[:-4]
    cal_days = days[-4:-2]
    test_days = days[-2:]
    LOG.info("%s fit_days=%s | cal_days=%s | test_days=%s",
             symbol, fit_days, cal_days, test_days)

    fit = labeled_valid[labeled_valid["day"].isin(fit_days)]
    cal = labeled_valid[labeled_valid["day"].isin(cal_days)]
    test = labeled_valid[labeled_valid["day"].isin(test_days)]

    Xfit = fit[feat_cols].values; yfit = fit["label"].values.astype(int)
    if int(yfit.sum()) == 0 or int((1 - yfit).sum()) == 0:
        return {"error": "degenerate fit set"}

    pos_w = (1 - yfit).sum() / max(1, int(yfit.sum()))
    val = int(0.85 * len(yfit))
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=pos_w, eval_metric="logloss",
        random_state=42, n_jobs=-1, early_stopping_rounds=20,
    )
    model.fit(Xfit[:val], yfit[:val], eval_set=[(Xfit[val:], yfit[val:])], verbose=False)

    Xcal = cal[feat_cols].values; ycal = cal["label"].values.astype(int)
    Xte  = test[feat_cols].values; yte  = test["label"].values.astype(int)

    pcal = model.predict_proba(Xcal)
    pte = model.predict_proba(Xte)
    q_hat = split_conformal_threshold(pcal, ycal, alpha=ALPHA)

    metrics_cal = evaluate_sets(pcal, ycal, q_hat)
    metrics_te = evaluate_sets(pte, yte, q_hat)

    # Singleton operating-point performance
    sets = predict_sets(pte, q_hat)
    singletons = sets.sum(axis=1) == 1
    yhat_single = sets[:, 1].astype(int)
    if int(singletons.sum()) > 0:
        s_y = yte[singletons]; s_p = yhat_single[singletons]
        tp = int(((s_p == 1) & (s_y == 1)).sum())
        fp = int(((s_p == 1) & (s_y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
    else:
        tp = fp = 0; prec = 1.0
    fn = int((yte == 1).sum() - tp)
    rec = tp / (tp + fn) if (tp + fn) else 1.0

    return {
        "fit_days": fit_days, "cal_days": cal_days, "test_days": test_days,
        "n_fit": int(len(yfit)), "n_cal": int(len(ycal)), "n_test": int(len(yte)),
        "q_hat": float(q_hat),
        "cal_metrics": metrics_cal, "test_metrics": metrics_te,
        "singleton_precision": float(prec), "singleton_recall_global": float(rec),
        "singleton_TP": tp, "singleton_FP": fp, "singleton_FN_global": fn,
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

    results = {}
    rows = []
    for sym in DEFAULT_SYMBOLS:
        LOG.info("====== %s ======", sym)
        labeled, _ = build_symbol_dataset(sym, args.book_dir, args.trades_dir)
        r = conformal_one_asset(sym, labeled)
        if "error" in r:
            LOG.warning("%s skipped: %s", sym, r["error"])
            continue
        results[sym] = r
        rows.append({
            "symbol": sym,
            "n_test": r["n_test"],
            "q_hat": r["q_hat"],
            "coverage": r["test_metrics"]["coverage"],
            "avg_set_size": r["test_metrics"]["avg_set_size"],
            "abstain_rate": r["test_metrics"]["abstain_rate"],
            "singleton_share": r["test_metrics"]["size_one_pct"],
            "singleton_precision": r["singleton_precision"],
            "singleton_TP": r["singleton_TP"],
            "singleton_FP": r["singleton_FP"],
            "singleton_FN_global": r["singleton_FN_global"],
        })

    (args.out_dir / "conformal_per_asset.json").write_text(json.dumps(results, indent=2))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "conformal_per_asset.csv", index=False)
    print("\n=== Conformal summary per asset (alpha = 0.10) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot — coverage and singleton precision per asset
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"BTCUSDT": "#264653", "ETHUSDT": "#2a9d8f", "SOLUSDT": "#e76f51"}
    x = np.arange(len(summary))
    bars1 = axes[0].bar(x, summary["coverage"], color=[colors[s] for s in summary["symbol"]])
    axes[0].axhline(1.0 - ALPHA, color="red", ls="--", lw=1,
                    label=f"nominal {1 - ALPHA:.2f}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(summary["symbol"])
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0.7, 1.02)
    axes[0].set_title("Marginal coverage (target 1-α = 0.90)")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
    for b, v in zip(bars1, summary["coverage"]):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

    bars2 = axes[1].bar(x, summary["singleton_precision"],
                        color=[colors[s] for s in summary["symbol"]])
    axes[1].set_xticks(x); axes[1].set_xticklabels(summary["symbol"])
    axes[1].set_ylabel("Singleton precision")
    axes[1].set_ylim(0.6, 1.02)
    axes[1].set_title("Singleton precision (when classifier commits)")
    axes[1].grid(axis="y", alpha=0.3)
    for b, v in zip(bars2, summary["singleton_precision"]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)

    fig.suptitle("Phase C — split-conformal wrapper per asset")
    fig.tight_layout()
    fig.savefig(args.out_dir / "conformal_per_asset.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
