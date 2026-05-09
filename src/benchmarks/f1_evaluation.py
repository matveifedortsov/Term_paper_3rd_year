"""Unified F1 / precision-recall comparison of jump detectors.

Compares jump time-stamps emitted by:
    raw_lomn        -- LOMN block-min, |stat| >= threshold
    lomn_xgb        -- LOMN candidates filtered by trained XGBoost
    lee_mykland     -- LM nonparametric test
    pure_ml         -- XGBoost without LOMN features (book + trade only)

Ground truth is the persistence-based positive set: candidates whose
forward-looking |log_mid(tau+30) - log_mid(tau-30)| / scale >= 5.

Matching: a detection is a true positive if it falls within
TOLERANCE_S seconds of a ground-truth jump. Matching is greedy
nearest-neighbor and one-to-one (no double counting).

Scope: only TEST days (Mar 27-29) are evaluated for F1.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.benchmarks.lee_mykland import detect_jumps_df
from src.realdata.train_xgb import FEATURE_COLS, TEST_DAYS

LOG = logging.getLogger("f1")

TOLERANCE_S = 60  # +/- seconds for matching detection to ground truth
RAW_LOMN_THRESHOLD = 4.0
ML_PROBA_THRESHOLD = 0.5
LM_K = 270
LM_ALPHA = 0.05


def load_book(book_dir: Path, day: str) -> pd.DataFrame:
    return pd.read_parquet(book_dir / f"resampled_1s_{day}.parquet")


def truth_for_day(features: pd.DataFrame, day: str) -> np.ndarray:
    g = features[(features["day"] == day) & (features["label"] == 1)]
    return g["obs_idx"].values.astype(int)


def f1_match(detected: np.ndarray, truth: np.ndarray, tol: int) -> dict:
    """Greedy NN matching with tolerance.

    detected, truth are int arrays of obs indices on a 1-second grid.
    Returns TP, FP, FN, precision, recall, F1.
    """
    if len(truth) == 0 and len(detected) == 0:
        return {"TP": 0, "FP": 0, "FN": 0, "precision": 1.0, "recall": 1.0, "F1": 1.0}
    if len(truth) == 0:
        return {"TP": 0, "FP": int(len(detected)), "FN": 0,
                "precision": 0.0 if len(detected) else 1.0,
                "recall": 1.0, "F1": 0.0}
    if len(detected) == 0:
        return {"TP": 0, "FP": 0, "FN": int(len(truth)),
                "precision": 1.0, "recall": 0.0, "F1": 0.0}

    detected = np.sort(detected.astype(int))
    truth = np.sort(truth.astype(int))
    matched_truth = np.zeros(len(truth), dtype=bool)
    tp = 0
    fp = 0
    for d in detected:
        idx = np.searchsorted(truth, d)
        candidates = []
        if idx < len(truth):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = -1
        best_d = tol + 1
        for c in candidates:
            if matched_truth[c]:
                continue
            dist = abs(int(truth[c]) - int(d))
            if dist <= tol and dist < best_d:
                best = c
                best_d = dist
        if best >= 0:
            matched_truth[best] = True
            tp += 1
        else:
            fp += 1
    fn = int((~matched_truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": float(precision), "recall": float(recall), "F1": float(f1)}


def detections_raw_lomn(features: pd.DataFrame, day: str) -> np.ndarray:
    g = features[(features["day"] == day) & (features["f_lomn_abs_std"] >= RAW_LOMN_THRESHOLD)]
    return g["obs_idx"].values.astype(int)


def detections_lomn_xgb(
    features: pd.DataFrame, day: str, model: xgb.XGBClassifier
) -> np.ndarray:
    g = features[features["day"] == day]
    if len(g) == 0:
        return np.empty(0, dtype=int)
    proba = model.predict_proba(g[FEATURE_COLS].values)[:, 1]
    return g["obs_idx"].values[proba >= ML_PROBA_THRESHOLD].astype(int)


def detections_lee_mykland(book: pd.DataFrame) -> np.ndarray:
    rows = detect_jumps_df(book, K=LM_K, alpha=LM_ALPHA)
    return rows["obs_idx"].values.astype(int)


def detections_pure_ml(
    features: pd.DataFrame, day: str, model: xgb.XGBClassifier, cols: list[str]
) -> np.ndarray:
    """Pure-ML baseline: XGBoost on non-LOMN features only.

    The 'pure ML' here uses the same candidate set as LOMN (so we can
    evaluate F1 on the same tau timestamps), but the classifier is not
    given f_lomn_abs_std or f_lomn_signed.
    """
    g = features[features["day"] == day]
    if len(g) == 0:
        return np.empty(0, dtype=int)
    proba = model.predict_proba(g[cols].values)[:, 1]
    return g["obs_idx"].values[proba >= ML_PROBA_THRESHOLD].astype(int)


def train_pure_ml(features: pd.DataFrame) -> tuple[xgb.XGBClassifier, list[str]]:
    no_lomn_cols = [c for c in FEATURE_COLS if not c.startswith("f_lomn_")]
    train_days = sorted(features["day"].unique())[:-3]  # all but last 3 = test
    train = features[features["day"].isin(train_days) & (features["label"] != -1)]
    Xtr = train[no_lomn_cols].values
    ytr = train["label"].values.astype(int)
    pos_w = (1 - ytr).sum() / max(1, ytr.sum())
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=pos_w,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
        early_stopping_rounds=20,
    )
    val_split = int(0.85 * len(ytr))
    model.fit(
        Xtr[:val_split], ytr[:val_split],
        eval_set=[(Xtr[val_split:], ytr[val_split:])],
        verbose=False,
    )
    return model, no_lomn_cols


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
    p.add_argument("--out-dir", type=Path, default=Path("results/phase5"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features)

    # Train pure-ML baseline
    LOG.info("training pure-ML baseline (no LOMN features)")
    pure_ml_model, pure_ml_cols = train_pure_ml(features)
    LOG.info("pure-ML features: %s", pure_ml_cols)

    # Load XGBoost LOMN refiner
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_model)

    # Per-day F1 on test days
    rows = []
    detect_records = []
    for day in TEST_DAYS:
        book = load_book(args.book_dir, day)
        truth = truth_for_day(features, day)
        n_truth = len(truth)

        for method, det in [
            ("raw_lomn",    detections_raw_lomn(features, day)),
            ("lee_mykland", detections_lee_mykland(book)),
            ("lomn_xgb",    detections_lomn_xgb(features, day, xgb_model)),
            ("pure_ml",     detections_pure_ml(features, day, pure_ml_model, pure_ml_cols)),
        ]:
            stats = f1_match(det, truth, tol=TOLERANCE_S)
            stats.update(method=method, day=day, n_detected=int(len(det)),
                         n_truth=int(n_truth))
            rows.append(stats)
            for d in det:
                detect_records.append({
                    "day": day, "method": method, "obs_idx": int(d),
                })

    per_day = pd.DataFrame(rows)
    per_day.to_csv(args.out_dir / "f1_per_day.csv", index=False)

    summary = per_day.groupby("method").agg(
        n_truth_total=("n_truth", "sum"),
        n_detected_total=("n_detected", "sum"),
        TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"),
    )
    summary["precision"] = summary["TP"] / (summary["TP"] + summary["FP"]).clip(lower=1)
    summary["recall"] = summary["TP"] / (summary["TP"] + summary["FN"]).clip(lower=1)
    summary["F1"] = 2 * summary["precision"] * summary["recall"] / (
        summary["precision"] + summary["recall"]
    ).clip(lower=1e-9)
    order = ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]
    summary = summary.reindex(order)
    summary.to_csv(args.out_dir / "f1_summary.csv")
    print("\n=== F1 comparison (test days Mar 27-29) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    pd.DataFrame(detect_records).to_csv(args.out_dir / "all_detections.csv", index=False)


if __name__ == "__main__":
    main()
