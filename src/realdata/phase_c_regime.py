"""Phase C: regime stratification per asset.

Replicates Phase 6 Item 4 (Section 5.6 of the paper) per asset:

    - Split test hours into terciles by hourly realized variance.
    - Compute F1 per (regime, method) for each of BTC/ETH/SOL.
    - Report the lift of LOMN+XGB over raw_LOMN by regime.

This is the cleanest test of whether the +0.182 low-vol lift on Binance
futures BTC (Phase 6) replicates on Bybit spot and generalises across
assets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks.f1_evaluation import f1_match
from src.benchmarks.lee_mykland import detect_jumps
from src.benchmarks.regime_analysis import hour_of_obs_idx, hourly_rv
from src.realdata.phase_c_runner import (
    DEFAULT_SYMBOLS,
    build_symbol_dataset,
    train_eval_split,
)

LOG = logging.getLogger("phase-c-regime")

TOLERANCE_S = 60
RAW_LOMN_THRESHOLD = 4.0
ML_PROBA_THRESHOLD = 0.5
LM_K = 270
LM_ALPHA = 0.05


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
    for sym in DEFAULT_SYMBOLS:
        LOG.info("====== %s ======", sym)
        labeled, _ = build_symbol_dataset(sym, args.book_dir, args.trades_dir)
        res = train_eval_split(labeled, n_test_days=2)
        if "error" in res:
            continue
        model = res["model"]; feat_cols = res["feat_cols"]
        test_days = res["test_days"]

        # Compute hourly RV across the test days for this asset
        rv_rows = []
        for d in test_days:
            book = pd.read_parquet(args.book_dir / sym.lower() / f"resampled_1s_{d}.parquet")
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
        regime_hours = {r: rv_df[rv_df["regime"] == r] for r in ("low", "mid", "high")}

        for d in test_days:
            book = pd.read_parquet(args.book_dir / sym.lower() / f"resampled_1s_{d}.parquet")
            truth = labeled[(labeled["day"] == d) & (labeled["label"] == 1)]["obs_idx"].values.astype(int)
            g = labeled[labeled["day"] == d]
            p_xgb = model.predict_proba(g[feat_cols].values)[:, 1]
            det_xgb = g["obs_idx"].values[p_xgb >= ML_PROBA_THRESHOLD].astype(int)
            det_lomn = g["obs_idx"].values[g["f_lomn_abs_std"] >= RAW_LOMN_THRESHOLD].astype(int)
            det_lm = detect_jumps(book["log_mid"].values, K=LM_K, alpha=LM_ALPHA)[
                "detected_obs_idx"
            ].astype(int)

            for regime in ("low", "mid", "high"):
                hours_today = regime_hours[regime].query(f"day == '{d}'")["hour"].tolist()
                in_hr = lambda arr, hh=hours_today: arr[np.isin(hour_of_obs_idx(arr), hh)]
                for method, det in [
                    ("raw_lomn", det_lomn),
                    ("lomn_xgb", det_xgb),
                    ("lee_mykland", det_lm),
                ]:
                    stats = f1_match(in_hr(det), in_hr(truth), tol=TOLERANCE_S)
                    rows.append({
                        "symbol": sym, "day": d, "regime": regime,
                        "method": method,
                        "n_hours": len(hours_today),
                        "n_truth": stats["TP"] + stats["FN"],
                        **stats,
                    })

    per_day = pd.DataFrame(rows)
    per_day.to_csv(args.out_dir / "regime_per_asset_per_day.csv", index=False)

    # Aggregate by (symbol, regime, method)
    agg = per_day.groupby(["symbol", "regime", "method"]).agg(
        TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"),
    )
    agg["precision"] = agg["TP"] / (agg["TP"] + agg["FP"]).clip(lower=1)
    agg["recall"] = agg["TP"] / (agg["TP"] + agg["FN"]).clip(lower=1)
    agg["F1"] = 2 * agg["precision"] * agg["recall"] / (
        agg["precision"] + agg["recall"]
    ).clip(lower=1e-9)
    agg = agg.reset_index()
    agg.to_csv(args.out_dir / "regime_per_asset_summary.csv", index=False)

    print("\n=== F1 per (asset, regime, method) ===")
    pivot = agg.pivot_table(
        index=["symbol", "method"], columns="regime", values="F1",
    ).reindex(columns=["low", "mid", "high"])
    pivot = pivot.reindex([(s, m) for s in DEFAULT_SYMBOLS
                           for m in ("raw_lomn", "lomn_xgb", "lee_mykland")])
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== F1 lift of XGB over raw_LOMN by regime ===")
    lift_rows = []
    for sym in DEFAULT_SYMBOLS:
        sub = agg[agg["symbol"] == sym]
        for regime in ("low", "mid", "high"):
            xgb = sub[(sub["regime"] == regime) & (sub["method"] == "lomn_xgb")]["F1"].iloc[0]
            raw = sub[(sub["regime"] == regime) & (sub["method"] == "raw_lomn")]["F1"].iloc[0]
            lift_rows.append({"symbol": sym, "regime": regime,
                              "F1_xgb": xgb, "F1_raw_lomn": raw, "lift": xgb - raw})
    lift_df = pd.DataFrame(lift_rows)
    lift_df.to_csv(args.out_dir / "regime_lift_per_asset.csv", index=False)
    print(lift_df.pivot(index="symbol", columns="regime", values="lift")
          .reindex(columns=["low", "mid", "high"]).to_string(
              float_format=lambda x: f"{x:+.4f}"))

    # Plot — grouped bar
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.27
    x = np.arange(3)
    colors = {"BTCUSDT": "#264653", "ETHUSDT": "#2a9d8f", "SOLUSDT": "#e76f51"}
    for i, sym in enumerate(DEFAULT_SYMBOLS):
        sub = lift_df[lift_df["symbol"] == sym].set_index("regime").reindex(["low", "mid", "high"])
        ax.bar(x + (i - 1) * width, sub["lift"], width, label=sym, color=colors[sym])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["low-vol", "mid-vol", "high-vol"])
    ax.set_ylabel("F1 lift  (LOMN+XGB  –  raw LOMN)")
    ax.set_title("Phase C regime stratification — XGB lift over raw LOMN per asset")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "regime_lift_per_asset.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
