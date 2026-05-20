"""Train and evaluate the XGBoost LOMN-refinement classifier.

Time-based split: first TRAIN_DAYS used for fit, last TEST_DAYS held
out. Class imbalance handled via scale_pos_weight. Reports:
    - Precision/recall/F1 on test
    - ROC AUC, PR AUC
    - FPR reduction at matched recall vs raw LOMN stat as the baseline
      score (this is the headline H1 test from the paper)
    - Feature importance and SHAP summary

Outputs go to results/phase3/.
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

LOG = logging.getLogger("train")

FEATURE_COLS = [
    "f_spread",
    "f_dspread_60s",
    "f_obi_l1",
    "f_log_mid",
    "f_lomn_abs_std",
    "f_lomn_signed",
    "f_dt_prev_cand",
    "f_realvar_60s",
    "f_bipower_60s",
    "f_realkurt_60s",
    "f_jump_ratio",
    "f_volume_pm5s",
    "f_signed_flow_pm5s",
    "f_n_trades_pm5s",
]

# Bucket / L20 features produced by features.build_features_for_day when
# the book has the L20 schema. Keep the names stable for downstream tests.
BUCKET_BP_EDGES = (0, 1, 2, 5, 10, 25, 50, 100, 500)
BUCKET_RAW_COLS = [
    f"{side}_{int(BUCKET_BP_EDGES[i])}_{int(BUCKET_BP_EDGES[i + 1])}bp"
    for i in range(len(BUCKET_BP_EDGES) - 1)
    for side in ("bid", "ask")
]
BUCKET_DERIVED_COLS = [
    *[f"f_imb_{int(BUCKET_BP_EDGES[i])}_{int(BUCKET_BP_EDGES[i + 1])}bp"
      for i in range(len(BUCKET_BP_EDGES) - 1)],
    *[f"f_cumimb_{int(BUCKET_BP_EDGES[i + 1])}bp"
      for i in range(len(BUCKET_BP_EDGES) - 1)],
    "f_total_depth_bid_100bp",
    "f_total_depth_ask_100bp",
    "f_book_slope_bid",
    "f_book_slope_ask",
    "f_book_skew",
    "f_inner5_share_bid",
    "f_inner5_share_ask",
]
FEATURE_COLS_L20 = FEATURE_COLS + BUCKET_RAW_COLS + BUCKET_DERIVED_COLS


def select_feature_cols(df) -> list[str]:
    """Return FEATURE_COLS_L20 if all bucket cols exist, else FEATURE_COLS."""
    bucket_cols = BUCKET_RAW_COLS + BUCKET_DERIVED_COLS
    if all(c in df.columns for c in bucket_cols):
        return FEATURE_COLS_L20
    return FEATURE_COLS

TRAIN_DAYS = [
    "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18",
    "2024-03-19", "2024-03-20", "2024-03-21", "2024-03-22",
    "2024-03-23", "2024-03-24", "2024-03-25", "2024-03-26",
]
TEST_DAYS = ["2024-03-27", "2024-03-28", "2024-03-29"]


def fpr_at_recall(y_true: np.ndarray, score: np.ndarray, target_recall: float) -> dict:
    fpr_arr, tpr_arr, thr = roc_curve(y_true, score)
    above = np.where(tpr_arr >= target_recall)[0]
    if not above.size:
        return {"recall": float("nan"), "fpr": float("nan"), "threshold": float("nan")}
    i = above[0]
    return {
        "recall": float(tpr_arr[i]),
        "fpr": float(fpr_arr[i]),
        "threshold": float(thr[i]),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase3"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.src)
    labeled = df[df["label"] != -1].copy()
    train_mask = labeled["day"].isin(TRAIN_DAYS)
    test_mask = labeled["day"].isin(TEST_DAYS)
    Xtr = labeled.loc[train_mask, FEATURE_COLS].values
    ytr = labeled.loc[train_mask, "label"].values.astype(int)
    Xte = labeled.loc[test_mask, FEATURE_COLS].values
    yte = labeled.loc[test_mask, "label"].values.astype(int)

    LOG.info("train: n=%d  pos=%d  neg=%d", len(ytr), int(ytr.sum()), int((1 - ytr).sum()))
    LOG.info("test : n=%d  pos=%d  neg=%d", len(yte), int(yte.sum()), int((1 - yte).sum()))

    pos_w = float((1 - ytr).sum()) / max(1, int(ytr.sum()))
    LOG.info("scale_pos_weight = %.3f", pos_w)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_w,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    val_split = int(0.85 * len(ytr))
    Xtr2, Xva = Xtr[:val_split], Xtr[val_split:]
    ytr2, yva = ytr[:val_split], ytr[val_split:]
    model.fit(Xtr2, ytr2, eval_set=[(Xva, yva)], verbose=False)

    # ----- Evaluation on test -----
    p_test = model.predict_proba(Xte)[:, 1]
    raw_lomn = labeled.loc[test_mask, "f_lomn_abs_std"].values

    auc_xgb = roc_auc_score(yte, p_test)
    auc_lomn = roc_auc_score(yte, raw_lomn)
    ap_xgb = average_precision_score(yte, p_test)
    ap_lomn = average_precision_score(yte, raw_lomn)

    fpr_xgb = fpr_at_recall(yte, p_test, 0.90)
    fpr_lomn = fpr_at_recall(yte, raw_lomn, 0.90)
    fpr_reduction = (
        100.0 * (fpr_lomn["fpr"] - fpr_xgb["fpr"]) / fpr_lomn["fpr"]
        if fpr_lomn["fpr"] > 0 else float("nan")
    )

    # Confusion matrix at default 0.5 threshold for completeness
    yhat = (p_test >= 0.5).astype(int)
    cm = confusion_matrix(yte, yhat)

    metrics = {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "scale_pos_weight": pos_w,
        "roc_auc_xgb": float(auc_xgb),
        "roc_auc_raw_lomn": float(auc_lomn),
        "pr_auc_xgb": float(ap_xgb),
        "pr_auc_raw_lomn": float(ap_lomn),
        "fpr_at_recall_90_xgb": fpr_xgb,
        "fpr_at_recall_90_raw_lomn": fpr_lomn,
        "fpr_reduction_pct": fpr_reduction,
        "confusion_matrix_thr_0_5": {
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    LOG.info("=== metrics ===")
    LOG.info(json.dumps(metrics, indent=2))

    # ----- Plots -----
    fig, ax = plt.subplots(figsize=(7, 5))
    fpr_x, tpr_x, _ = roc_curve(yte, p_test)
    fpr_l, tpr_l, _ = roc_curve(yte, raw_lomn)
    ax.plot(fpr_x, tpr_x, label=f"XGBoost (AUC={auc_xgb:.3f})", lw=2)
    ax.plot(fpr_l, tpr_l, label=f"Raw LOMN (AUC={auc_lomn:.3f})", lw=2, ls="--")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8)
    ax.axhline(0.9, color="red", lw=0.6, ls=":", label="90% recall")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: XGBoost refinement vs raw LOMN (test = Mar 27-29)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    pr_x, rc_x, _ = precision_recall_curve(yte, p_test)
    pr_l, rc_l, _ = precision_recall_curve(yte, raw_lomn)
    ax.plot(rc_x, pr_x, label=f"XGBoost (AP={ap_xgb:.3f})", lw=2)
    ax.plot(rc_l, pr_l, label=f"Raw LOMN (AP={ap_lomn:.3f})", lw=2, ls="--")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall: XGBoost refinement vs raw LOMN")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "pr.png", dpi=150)
    plt.close(fig)

    # Feature importance
    booster = model.get_booster()
    fmap = {f"f{i}": name for i, name in enumerate(FEATURE_COLS)}
    booster.feature_names = list(fmap.values())
    importance = booster.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(
            [(name, importance.get(name, 0.0)) for name in FEATURE_COLS],
            columns=["feature", "gain"],
        )
        .sort_values("gain", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(imp_df["feature"], imp_df["gain"], color="#2a9d8f")
    ax.set_xlabel("XGBoost gain (sum)")
    ax.set_title("Feature importance (gain)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    imp_df.iloc[::-1].to_csv(args.out_dir / "feature_importance.csv", index=False)

    # Save model
    model.save_model(args.out_dir / "xgb_lomn_refiner.json")
    LOG.info("model + plots saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
