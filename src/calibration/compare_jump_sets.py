"""Per-day calibration under three jump-set definitions and H2 test.

Three jump sets, each calibrated separately:
    raw_lomn        : all candidates with abs_std >= RAW_THRESHOLD (default 4.0)
    ml_refined      : XGBoost predict_proba >= ML_THRESHOLD (default 0.5)
    persist_truth   : persistence-based gold positives (label == 1)

For each jump set we estimate (lambda, mu_J, sigma_J, mu, sigma) per
day (15 days), then compute the across-day standard deviation of each
parameter. H2 says ml_refined should produce *lower* across-day std
than raw_lomn (more stable estimates).

Headline output: results/phase4/h2_variance_comparison.csv
                 results/phase4/per_day_params.csv
                 results/phase4/lambda_per_day.png
                 results/phase4/sigma_per_day.png
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
from src.realdata.train_xgb import FEATURE_COLS

LOG = logging.getLogger("compare-cal")

RAW_THRESHOLD = 4.0
ML_THRESHOLD = 0.5

DAYS = [
    "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18", "2024-03-19",
    "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24",
    "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29",
]


def load_books(book_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for d in DAYS:
        f = book_dir / f"resampled_1s_{d}.parquet"
        if f.exists():
            out[d] = pd.read_parquet(f, columns=["ts", "log_mid"])
    return out


def load_features(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def jump_set_from_lomn(features: pd.DataFrame, threshold: float) -> dict[str, pd.DataFrame]:
    sel = features[features["f_lomn_abs_std"] >= threshold]
    return _group_by_day(sel)


def jump_set_from_ml(
    features: pd.DataFrame,
    model: xgb.XGBClassifier,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    X = features[FEATURE_COLS].values
    proba = model.predict_proba(X)[:, 1]
    sel = features[proba >= threshold].copy()
    return _group_by_day(sel)


def jump_set_from_label(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sel = features[features["label"] == 1]
    return _group_by_day(sel)


def _group_by_day(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for d, g in df.groupby("day"):
        out[d] = pd.DataFrame({
            "obs_idx": g["obs_idx"].values,
            # delta_M is the signed jump-size estimate from LOMN
            "jump_size": g["f_lomn_signed"].values * g["f_lomn_abs_std"].rdiv(
                g["f_lomn_abs_std"]
            ).fillna(0).astype(float),
        })
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--candidates-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--model", type=Path, default=Path("results/phase3/xgb_lomn_refiner.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase4"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    books = load_books(args.book_dir)
    LOG.info("loaded %d days of book data", len(books))

    feats = load_features(args.features)

    # The features file holds f_lomn_signed (standardized) but not the
    # raw delta_M. Rebuild by joining against the candidate files.
    cand_frames = []
    for d in DAYS:
        f = args.candidates_dir / f"lomn_candidates_{d}.parquet"
        if f.exists():
            cand_frames.append(pd.read_parquet(f, columns=["ts", "obs_idx", "delta_M", "abs_std", "day"]))
    cands = pd.concat(cand_frames, ignore_index=True)
    feats = feats.merge(cands[["day", "obs_idx", "delta_M"]], on=["day", "obs_idx"], how="left")

    # Jump set helpers using signed delta_M directly
    def jset(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {d: pd.DataFrame(columns=["obs_idx", "jump_size"])
                                         for d in DAYS}
        for d, g in rows.groupby("day"):
            out[d] = pd.DataFrame({
                "obs_idx": g["obs_idx"].values.astype(int),
                "jump_size": g["delta_M"].values.astype(float),
            })
        return out

    raw_rows = feats[feats["f_lomn_abs_std"] >= RAW_THRESHOLD]
    persist_rows = feats[feats["label"] == 1]
    LOG.info("raw_lomn jumps : %d  persist_truth jumps : %d",
             len(raw_rows), len(persist_rows))

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    proba = model.predict_proba(feats[FEATURE_COLS].values)[:, 1]
    feats = feats.assign(p_ml=proba)
    ml_rows = feats[feats["p_ml"] >= ML_THRESHOLD]
    LOG.info("ml_refined jumps : %d", len(ml_rows))

    sets = {
        "raw_lomn":      jset(raw_rows),
        "ml_refined":    jset(ml_rows),
        "persist_truth": jset(persist_rows),
    }

    all_params = []
    for name, s in sets.items():
        df = calibrate_per_day(books, s)
        df["jump_set"] = name
        all_params.append(df)
    params = pd.concat(all_params, ignore_index=True)
    params.to_csv(args.out_dir / "per_day_params.csv", index=False)

    # ---------- H2: across-day std of each parameter ----------
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
        mu_mean=("mu_hat", "mean"),
        mu_std=("mu_hat", "std"),
    ).reindex(["raw_lomn", "ml_refined", "persist_truth"])
    summary.to_csv(args.out_dir / "h2_variance_comparison.csv")
    print("\n=== Per-jump-set parameter mean and across-day std ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4g}"))

    # H2 numeric: relative reduction in std going raw -> ml
    raw = summary.loc["raw_lomn"]
    ml = summary.loc["ml_refined"]
    h2 = {
        "lambda_std_reduction_pct":
            100.0 * (raw["lambda_std"] - ml["lambda_std"]) / raw["lambda_std"]
            if raw["lambda_std"] > 0 else float("nan"),
        "mu_J_std_reduction_pct":
            100.0 * (raw["mu_J_std"] - ml["mu_J_std"]) / raw["mu_J_std"]
            if raw["mu_J_std"] > 0 else float("nan"),
        "sigma_J_std_reduction_pct":
            100.0 * (raw["sigma_J_std"] - ml["sigma_J_std"]) / raw["sigma_J_std"]
            if raw["sigma_J_std"] > 0 else float("nan"),
        "sigma_std_reduction_pct":
            100.0 * (raw["sigma_std"] - ml["sigma_std"]) / raw["sigma_std"]
            if raw["sigma_std"] > 0 else float("nan"),
    }
    (args.out_dir / "h2_summary.json").write_text(json.dumps(h2, indent=2))
    print("\n=== H2 (lower is more stable; positive = ML wins) ===")
    print(json.dumps(h2, indent=2))

    # ---------- Plots ----------
    for col, ylab, fname in [
        ("lambda_hat", "lambda (jumps/day)", "lambda_per_day.png"),
        ("sigma_hat",  "sigma (daily, log-price)", "sigma_per_day.png"),
        ("sigma_J_hat", "sigma_J (jump-size std)", "sigma_J_per_day.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for name, color, marker in [
            ("raw_lomn", "#e76f51", "o"),
            ("ml_refined", "#2a9d8f", "s"),
            ("persist_truth", "#264653", "^"),
        ]:
            sub = params[params["jump_set"] == name].sort_values("day")
            ax.plot(sub["day"], sub[col], lw=1.4, marker=marker, ms=6,
                    label=name, color=color)
        ax.set_ylabel(ylab)
        ax.set_xlabel("day")
        ax.set_title(f"{col} per day, by jump-set definition")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / fname, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
