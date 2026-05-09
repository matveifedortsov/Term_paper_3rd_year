"""Regime stratification: does ML refinement help more in calm or volatile hours?

For each hour on each test day, compute realized variance from 1Hz log-mid
returns. Categorize hours into three vol regimes by tercile of hourly RV
across the test period. For each regime, compute F1 separately for
raw_lomn, lomn_xgb, lee_mykland, and pure_ml.

Hypothesis: ML refinement should help most in low-vol hours, where false
positives dominate the LOMN candidate stream. In high-vol hours, every
detector tends to agree because real signal swamps noise.
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

from src.benchmarks.f1_evaluation import (
    detections_lee_mykland,
    detections_lomn_xgb,
    detections_pure_ml,
    detections_raw_lomn,
    f1_match,
    load_book,
    train_pure_ml,
)
from src.realdata.train_xgb import FEATURE_COLS, TEST_DAYS

LOG = logging.getLogger("regime")
TOLERANCE_S = 60


def hourly_rv(book: pd.DataFrame) -> np.ndarray:
    """Realized variance over each of 24 one-hour windows."""
    log_mid = book["log_mid"].values
    r = np.diff(log_mid)  # length 86399
    rv = np.empty(24)
    for h in range(24):
        s = h * 3600
        e = (h + 1) * 3600
        rv[h] = float(np.sum(r[s:min(e, len(r))] ** 2))
    return rv


def hour_of_obs_idx(idx: np.ndarray) -> np.ndarray:
    return (np.asarray(idx) // 3600).astype(int)


def f1_in_hours(
    detected_obs: np.ndarray, truth_obs: np.ndarray, hours: list[int]
) -> dict:
    in_hr = lambda arr: arr[np.isin(hour_of_obs_idx(arr), hours)]
    return f1_match(in_hr(detected_obs), in_hr(truth_obs), tol=TOLERANCE_S)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--xgb-model", type=Path, default=Path("results/phase3/xgb_lomn_refiner.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase6"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    pure_model, pure_cols = train_pure_ml(feats)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_model)

    # ---- Compute hourly RV across all test hours, classify into terciles ----
    rv_rows = []
    for d in TEST_DAYS:
        book = load_book(args.book_dir, d)
        rv = hourly_rv(book)
        for h, v in enumerate(rv):
            rv_rows.append({"day": d, "hour": h, "rv": float(v)})
    rv_df = pd.DataFrame(rv_rows)
    q33 = rv_df["rv"].quantile(1.0 / 3.0)
    q67 = rv_df["rv"].quantile(2.0 / 3.0)
    rv_df["regime"] = np.where(
        rv_df["rv"] <= q33, "low",
        np.where(rv_df["rv"] <= q67, "mid", "high"),
    )
    LOG.info("regime cutoffs: low<=%.2e  mid<=%.2e", q33, q67)
    rv_df.to_csv(args.out_dir / "hourly_rv.csv", index=False)

    regime_hours = {
        regime: rv_df[rv_df["regime"] == regime]
        for regime in ["low", "mid", "high"]
    }

    rows = []
    for d in TEST_DAYS:
        book = load_book(args.book_dir, d)
        truth = feats[(feats["day"] == d) & (feats["label"] == 1)]["obs_idx"].values.astype(int)
        det_lomn = detections_raw_lomn(feats, d)
        det_lm = detections_lee_mykland(book)
        det_xgb = detections_lomn_xgb(feats, d, xgb_model)
        det_pure = detections_pure_ml(feats, d, pure_model, pure_cols)

        for regime in ["low", "mid", "high"]:
            hours_today = regime_hours[regime].query(f"day == '{d}'")["hour"].tolist()
            for method, det in [
                ("raw_lomn", det_lomn),
                ("lee_mykland", det_lm),
                ("lomn_xgb", det_xgb),
                ("pure_ml", det_pure),
            ]:
                stats = f1_in_hours(det, truth, hours_today)
                stats.update(method=method, day=d, regime=regime,
                             n_hours=len(hours_today),
                             n_truth=stats["TP"] + stats["FN"],
                             n_detected=stats["TP"] + stats["FP"])
                rows.append(stats)

    per_day_regime = pd.DataFrame(rows)
    per_day_regime.to_csv(args.out_dir / "regime_per_day.csv", index=False)

    # Aggregate over test days within each regime
    agg = per_day_regime.groupby(["regime", "method"]).agg(
        TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"),
        n_hours=("n_hours", "sum"),
    )
    agg["precision"] = agg["TP"] / (agg["TP"] + agg["FP"]).clip(lower=1)
    agg["recall"] = agg["TP"] / (agg["TP"] + agg["FN"]).clip(lower=1)
    agg["F1"] = 2 * agg["precision"] * agg["recall"] / (
        agg["precision"] + agg["recall"]
    ).clip(lower=1e-9)
    agg = agg.reset_index()
    agg.to_csv(args.out_dir / "regime_summary.csv", index=False)

    print("\n=== F1 by realized-vol regime (test days) ===")
    pivot = agg.pivot(index="method", columns="regime", values="F1").reindex(
        index=["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"],
        columns=["low", "mid", "high"],
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    methods = ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]
    colors = {"raw_lomn": "#264653", "lee_mykland": "#e76f51",
              "lomn_xgb": "#2a9d8f", "pure_ml": "#f4a261"}
    width = 0.2
    x = np.arange(3)
    for i, m in enumerate(methods):
        f1s = [pivot.loc[m, r] for r in ["low", "mid", "high"]]
        ax.bar(x + (i - 1.5) * width, f1s, width=width, label=m, color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels(["low-vol", "mid-vol", "high-vol"])
    ax.set_ylabel("F1 (test days)")
    ax.set_title("F1 by realized-vol regime (terciles of hourly RV)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "regime_f1.png", dpi=150)
    plt.close(fig)

    # Lift table: lomn_xgb F1 - raw_lomn F1, by regime
    lift = (
        pivot.loc["lomn_xgb"] - pivot.loc["raw_lomn"]
    ).to_frame(name="F1_lift_xgb_vs_raw")
    print("\n=== F1 lift of LOMN+XGB over raw LOMN, by regime ===")
    print(lift.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
