"""Phase C: MLE Merton calibration per asset, H2 stability test.

Replicates Phase 4 (compare_jump_sets.py) per asset on Bybit spot.
For each of BTC/ETH/SOL we estimate (lambda, mu_J, sigma_J, mu, sigma)
day by day under three jump-set definitions:

    raw_lomn       : |LOMN stat| >= 4.0
    ml_refined     : XGBoost predict_proba >= 0.5
    persist_truth  : persistence-z >= 5.0 (label == 1)

Then compute the across-day STD of each parameter — H2 says
ml_refined should give lower variance. The Phase 4 result on Binance
futures BTC was MIXED: ML cut mu_J/sigma_J std by ~30% but increased
lambda std by 35%. This test asks whether that pattern is consistent
across assets.
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

from src.calibration.mle import calibrate_per_day
from src.realdata.phase_c_runner import (
    DEFAULT_SYMBOLS,
    build_symbol_dataset,
    train_eval_split,
)
from src.realdata.train_xgb import select_feature_cols

LOG = logging.getLogger("phase-c-mle")
RAW_LOMN_THRESHOLD = 4.0
ML_PROBA_THRESHOLD = 0.5


def jset(rows: pd.DataFrame, days: list[str]) -> dict[str, pd.DataFrame]:
    """Group candidate rows into a {day -> DataFrame[obs_idx, jump_size]} mapping."""
    out: dict[str, pd.DataFrame] = {d: pd.DataFrame(columns=["obs_idx", "jump_size"]) for d in days}
    for d, g in rows.groupby("day"):
        out[d] = pd.DataFrame({
            "obs_idx": g["obs_idx"].values.astype(int),
            "jump_size": g["delta_M"].values.astype(float),
        })
    return out


def calibrate_one_asset(symbol: str, book_dir: Path, trades_dir: Path) -> dict:
    LOG.info("====== %s ======", symbol)
    labeled, _ = build_symbol_dataset(symbol, book_dir, trades_dir)

    days = sorted(labeled["day"].unique())
    # Load books
    days_book: dict[str, pd.DataFrame] = {}
    for d in days:
        f = book_dir / symbol.lower() / f"resampled_1s_{d}.parquet"
        if f.exists():
            days_book[d] = pd.read_parquet(f, columns=["ts", "log_mid"])

    # Train XGB on this asset (same as runner)
    res = train_eval_split(labeled, n_test_days=2)
    if "error" in res:
        return {"error": res["error"]}
    model = res["model"]; feat_cols = res["feat_cols"]

    # Three jump-set definitions over all 15 days
    raw_rows = labeled[labeled["f_lomn_abs_std"] >= RAW_LOMN_THRESHOLD]
    persist_rows = labeled[labeled["label"] == 1]
    proba = model.predict_proba(labeled[feat_cols].values)[:, 1]
    ml_rows = labeled[proba >= ML_PROBA_THRESHOLD]

    sets = {
        "raw_lomn":      jset(raw_rows, days),
        "ml_refined":    jset(ml_rows, days),
        "persist_truth": jset(persist_rows, days),
    }

    out_dfs = []
    for name, s in sets.items():
        df = calibrate_per_day(days_book, s)
        df["jump_set"] = name
        out_dfs.append(df)
    params = pd.concat(out_dfs, ignore_index=True)
    params["symbol"] = symbol

    # Variance (std) by jump_set
    summary = params.groupby("jump_set").agg(
        n_jumps_total=("n_jumps", "sum"),
        lambda_mean=("lambda_hat", "mean"),
        lambda_std=("lambda_hat", "std"),
        mu_J_mean=("mu_J_hat", "mean"),
        mu_J_std=("mu_J_hat", "std"),
        sigma_J_mean=("sigma_J_hat", "mean"),
        sigma_J_std=("sigma_J_hat", "std"),
        sigma_mean=("sigma_hat", "mean"),
        sigma_std=("sigma_hat", "std"),
    ).reindex(["raw_lomn", "ml_refined", "persist_truth"])

    raw = summary.loc["raw_lomn"]; ml = summary.loc["ml_refined"]

    def pct(a, b):
        return 100 * (a - b) / a if a > 0 else float("nan")

    h2 = {
        "lambda_std_reduction_pct":   pct(raw["lambda_std"], ml["lambda_std"]),
        "mu_J_std_reduction_pct":     pct(raw["mu_J_std"],   ml["mu_J_std"]),
        "sigma_J_std_reduction_pct":  pct(raw["sigma_J_std"], ml["sigma_J_std"]),
        "sigma_std_reduction_pct":    pct(raw["sigma_std"],  ml["sigma_std"]),
    }
    return {
        "symbol": symbol,
        "params_per_day": params,
        "summary": summary.reset_index(),
        "h2": h2,
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

    all_params = []
    all_summaries = []
    h2_all = {}
    for sym in DEFAULT_SYMBOLS:
        res = calibrate_one_asset(sym, args.book_dir, args.trades_dir)
        if "error" in res:
            LOG.warning("%s skipped: %s", sym, res["error"])
            continue
        all_params.append(res["params_per_day"])
        s = res["summary"].copy(); s["symbol"] = sym
        all_summaries.append(s)
        h2_all[sym] = res["h2"]

    pd.concat(all_params, ignore_index=True).to_csv(
        args.out_dir / "mle_per_asset_per_day.csv", index=False
    )
    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_df.to_csv(args.out_dir / "mle_per_asset_summary.csv", index=False)
    (args.out_dir / "mle_h2_per_asset.json").write_text(json.dumps(h2_all, indent=2))

    # Pretty print H2 across assets
    h2_df = pd.DataFrame(h2_all).T  # rows = assets, cols = h2 fields
    print("\n=== H2 reduction (raw_lomn -> ml_refined) per asset, % ===")
    print(h2_df.to_string(float_format=lambda x: f"{x:+.1f}"))

    # Plot
    fig, ax = plt.subplots(figsize=(11, 4.8))
    metrics = ["lambda_std_reduction_pct", "mu_J_std_reduction_pct",
               "sigma_J_std_reduction_pct", "sigma_std_reduction_pct"]
    labels = ["λ", "μ_J", "σ_J", "σ"]
    colors = {"BTCUSDT": "#264653", "ETHUSDT": "#2a9d8f", "SOLUSDT": "#e76f51"}
    width = 0.27
    x = np.arange(len(metrics))
    for i, sym in enumerate(DEFAULT_SYMBOLS):
        if sym not in h2_all:
            continue
        vals = [h2_all[sym][m] for m in metrics]
        ax.bar(x + (i - 1) * width, vals, width, label=sym, color=colors[sym])
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("% reduction of across-day std\n(positive = ML refines)")
    ax.set_title("Phase C — H2 stability test per asset (raw_lomn -> ml_refined)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "mle_h2_per_asset.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
