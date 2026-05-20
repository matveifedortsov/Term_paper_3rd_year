"""Phase C statistical significance — per-asset bootstrap + McNemar + DeLong.

For each asset (BTCUSDT, ETHUSDT, SOLUSDT) we replay Phase C's training
split, save the test-set scores (XGB and raw_LOMN), and compute:

    1. 5000-iter paired bootstrap CIs on F1 (XGB and raw LOMN) and their
       paired difference (XGB - raw_LOMN).
    2. McNemar test on paired correctness at the 0.5 threshold.
    3. DeLong test on paired AUC difference.

Output: results/phase_c/significance.json + significance_table.csv.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

from src.realdata.phase_c_runner import (
    DEFAULT_SYMBOLS,
    build_symbol_dataset,
    train_eval_split,
)
from src.benchmarks.significance import (
    _bootstrap_f1,
    delong_paired,
    mcnemar,
)

LOG = logging.getLogger("phase-c-sig")
ML_THRESHOLD = 0.5
RAW_LOMN_THRESHOLD = 4.0
N_BOOTSTRAP = 5000


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    out_dir = Path("results/phase_c")
    out_dir.mkdir(parents=True, exist_ok=True)
    book_dir = Path("data/interim")
    trades_dir = Path("data/historical")

    results = {}
    table_rows = []

    for symbol in DEFAULT_SYMBOLS:
        LOG.info("====== %s ======", symbol)
        labeled, _ = build_symbol_dataset(symbol, book_dir, trades_dir)
        res = train_eval_split(labeled, n_test_days=2)
        if "error" in res:
            LOG.warning("%s skipped: %s", symbol, res["error"])
            continue

        model = res["model"]
        Xte = res["test_X"]
        yte = res["test_y"]
        feat_cols = res["feat_cols"]

        # Test set predictions
        p_xgb = model.predict_proba(Xte)[:, 1]
        # Find f_lomn_abs_std column
        idx_lomn = feat_cols.index("f_lomn_abs_std")
        raw_lomn = Xte[:, idx_lomn]

        # Bootstrap CIs
        yhat = {
            "xgb": (p_xgb >= ML_THRESHOLD).astype(int),
            "raw_lomn": (raw_lomn >= RAW_LOMN_THRESHOLD).astype(int),
        }
        rng = np.random.default_rng(20260520 + hash(symbol) % 1000)
        boots = _bootstrap_f1(yte, yhat, n_iter=N_BOOTSTRAP, rng=rng)
        diff = boots["xgb"] - boots["raw_lomn"]
        p_two = 2.0 * min(float(np.mean(diff > 0)), float(np.mean(diff < 0)))

        ci_xgb = (float(np.quantile(boots["xgb"], 0.025)),
                  float(np.quantile(boots["xgb"], 0.975)))
        ci_lomn = (float(np.quantile(boots["raw_lomn"], 0.025)),
                   float(np.quantile(boots["raw_lomn"], 0.975)))
        ci_diff = (float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975)))

        # McNemar
        mc = mcnemar(yte, yhat["xgb"], yhat["raw_lomn"])

        # DeLong
        dl = delong_paired(p_xgb, raw_lomn, yte)

        results[symbol] = {
            "n_test": int(len(yte)),
            "F1_xgb_CI": ci_xgb,
            "F1_raw_lomn_CI": ci_lomn,
            "F1_diff_xgb_minus_raw_CI": ci_diff,
            "F1_diff_bootstrap_p": p_two,
            "mcnemar_xgb_vs_raw_lomn": mc,
            "delong_AUC_xgb_vs_raw_lomn": dl,
        }
        table_rows.append({
            "symbol": symbol,
            "n_test": int(len(yte)),
            "F1_xgb_mean":      float(np.mean(boots["xgb"])),
            "F1_xgb_ci_lo":     ci_xgb[0],
            "F1_xgb_ci_hi":     ci_xgb[1],
            "F1_raw_mean":      float(np.mean(boots["raw_lomn"])),
            "F1_raw_ci_lo":     ci_lomn[0],
            "F1_raw_ci_hi":     ci_lomn[1],
            "F1_diff_mean":     float(np.mean(diff)),
            "F1_diff_ci_lo":    ci_diff[0],
            "F1_diff_ci_hi":    ci_diff[1],
            "F1_diff_p":        p_two,
            "mcnemar_chi2":     mc["chi2"],
            "mcnemar_p":        mc["p_value"],
            "delong_auc_diff":  dl["diff"],
            "delong_p":         dl["p_value"],
        })
        LOG.info("%s F1 CI XGB %.3f-%.3f raw %.3f-%.3f diff %.3f-%.3f p=%.4f",
                 symbol, ci_xgb[0], ci_xgb[1], ci_lomn[0], ci_lomn[1],
                 ci_diff[0], ci_diff[1], p_two)
        LOG.info("%s McNemar chi2=%.2f p=%.4f", symbol, mc["chi2"], mc["p_value"])
        LOG.info("%s DeLong auc_diff=%.4f p=%.4f", symbol, dl["diff"], dl["p_value"])

    (out_dir / "significance.json").write_text(json.dumps(results, indent=2))
    df = pd.DataFrame(table_rows)
    df.to_csv(out_dir / "significance_table.csv", index=False)
    print("\n=== Phase C significance ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
