"""Optuna hyperparameter tuning for the LOMN-refinement XGBoost model.

Design notes (small-sample-aware):

  - Search runs on the TRAIN period only (Mar 15 to Mar 26). The test
    period (Mar 27 to Mar 29) is never seen during tuning.
  - TimeSeriesSplit with 5 expanding-window folds on the train slice.
  - Trial objective: median ROC-AUC across folds (robust to one bad fold).
  - TPE sampler, MedianPruner aborts obvious losers early.
  - Search space is intentionally moderate to avoid noise-fitting:
        max_depth, learning_rate, min_child_weight, gamma,
        reg_alpha, reg_lambda, subsample, colsample_bytree.
    n_estimators is NOT tuned: early stopping picks it per fold.
  - After search, refit best params on full train and evaluate on test.

Outputs (results/phase3/):
    optuna_history.png    Trial-by-trial best-so-far AUC
    optuna_params.json    Best params + search-time mean/median CV AUC
    metrics_tuned.json    Tuned model: AUC, AP, FPR@90, vs default and raw LOMN
    roc_tuned.png         ROC overlay (tuned, default, raw)
    pr_tuned.png          PR overlay
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

from src.realdata.train_xgb import FEATURE_COLS, TEST_DAYS, TRAIN_DAYS, fpr_at_recall

LOG = logging.getLogger("tune")

N_TRIALS = 50
N_SPLITS = 5
EARLY_STOPPING = 30
RANDOM_STATE = 42


def build_objective(Xtr: np.ndarray, ytr: np.ndarray, ts_train: np.ndarray):
    """Closure that returns Optuna objective: median CV ROC-AUC."""
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    pos_w = float((1 - ytr).sum()) / max(1, int(ytr.sum()))

    # Pre-sort the training data by time so TimeSeriesSplit folds are chronological
    order = np.argsort(ts_train, kind="mergesort")
    Xs, ys = Xtr[order], ytr[order]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        fold_aucs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(Xs)):
            X_tr, X_va = Xs[tr_idx], Xs[va_idx]
            y_tr, y_va = ys[tr_idx], ys[va_idx]
            if y_va.sum() == 0 or y_va.sum() == len(y_va):
                # degenerate fold: AUC undefined
                continue
            model = xgb.XGBClassifier(
                n_estimators=600,
                scale_pos_weight=pos_w,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                early_stopping_rounds=EARLY_STOPPING,
                **params,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            p_va = model.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, p_va))
            trial.report(float(np.median(fold_aucs)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not fold_aucs:
            return 0.0
        # Use median for robustness to one bad fold
        return float(np.median(fold_aucs))

    return objective


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase3"))
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
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
    ts_train = labeled.loc[train_mask, "ts"].astype("int64").values * 1_000_000

    LOG.info("train n=%d (pos=%d)  test n=%d (pos=%d)",
             len(ytr), int(ytr.sum()), len(yte), int(yte.sum()))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=RANDOM_STATE, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = build_objective(Xtr, ytr, ts_train)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    LOG.info("optuna done in %.1fs (%d completed, %d pruned)",
             elapsed,
             sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
             sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED))

    best = study.best_params
    LOG.info("best CV median AUC = %.4f", study.best_value)
    LOG.info("best params: %s", best)

    # ---------- Refit best on full train, eval on test ----------
    pos_w = float((1 - ytr).sum()) / max(1, int(ytr.sum()))
    val_split = int(0.85 * len(ytr))
    Xtr2, Xva = Xtr[:val_split], Xtr[val_split:]
    ytr2, yva = ytr[:val_split], ytr[val_split:]
    tuned = xgb.XGBClassifier(
        n_estimators=600,
        scale_pos_weight=pos_w,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING,
        **best,
    )
    tuned.fit(Xtr2, ytr2, eval_set=[(Xva, yva)], verbose=False)
    p_test_tuned = tuned.predict_proba(Xte)[:, 1]

    # Default model for comparison (same code path as Phase 3)
    default = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=pos_w,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    default.fit(Xtr2, ytr2, eval_set=[(Xva, yva)], verbose=False)
    p_test_default = default.predict_proba(Xte)[:, 1]

    raw_lomn = labeled.loc[test_mask, "f_lomn_abs_std"].values

    metrics = {
        "n_trials": args.n_trials,
        "n_completed": int(sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)),
        "n_pruned": int(sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)),
        "search_time_s": elapsed,
        "best_cv_median_auc": float(study.best_value),
        "best_params": best,
        "test_metrics": {
            "raw_lomn": {
                "roc_auc": float(roc_auc_score(yte, raw_lomn)),
                "pr_auc": float(average_precision_score(yte, raw_lomn)),
                "fpr_at_recall_90": fpr_at_recall(yte, raw_lomn, 0.90),
            },
            "xgb_default": {
                "roc_auc": float(roc_auc_score(yte, p_test_default)),
                "pr_auc": float(average_precision_score(yte, p_test_default)),
                "fpr_at_recall_90": fpr_at_recall(yte, p_test_default, 0.90),
            },
            "xgb_tuned": {
                "roc_auc": float(roc_auc_score(yte, p_test_tuned)),
                "pr_auc": float(average_precision_score(yte, p_test_tuned)),
                "fpr_at_recall_90": fpr_at_recall(yte, p_test_tuned, 0.90),
            },
        },
    }
    (args.out_dir / "metrics_tuned.json").write_text(json.dumps(metrics, indent=2))
    (args.out_dir / "optuna_params.json").write_text(json.dumps({
        "best_params": best,
        "best_cv_auc": float(study.best_value),
    }, indent=2))

    LOG.info("=== test set comparison ===")
    LOG.info(json.dumps({k: {"roc_auc": v["roc_auc"], "fpr@90": v["fpr_at_recall_90"]["fpr"]}
                        for k, v in metrics["test_metrics"].items()}, indent=2))

    # ---------- Plots ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    completed_vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(completed_vals)
    ax.plot(np.arange(len(completed_vals)) + 1, completed_vals, "o", ms=3,
            color="#888", alpha=0.6, label="trial AUC")
    ax.plot(np.arange(len(best_so_far)) + 1, best_so_far, lw=2,
            color="#e76f51", label="best so far")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("median CV ROC-AUC (train period only)")
    ax.set_title(f"Optuna search history ({len(completed_vals)} completed trials)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "optuna_history.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, score, ls in [
        ("XGBoost tuned", p_test_tuned, "-"),
        ("XGBoost default", p_test_default, ":"),
        ("Raw LOMN", raw_lomn, "--"),
    ]:
        fpr, tpr, _ = roc_curve(yte, score)
        auc = roc_auc_score(yte, score)
        ax.plot(fpr, tpr, ls=ls, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.6)
    ax.axhline(0.9, color="red", lw=0.5, ls=":", label="90% recall")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: tuned XGBoost vs default vs raw LOMN")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "roc_tuned.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, score, ls in [
        ("XGBoost tuned", p_test_tuned, "-"),
        ("XGBoost default", p_test_default, ":"),
        ("Raw LOMN", raw_lomn, "--"),
    ]:
        pr, rc, _ = precision_recall_curve(yte, score)
        ap = average_precision_score(yte, score)
        ax.plot(rc, pr, ls=ls, lw=2, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall: tuned XGBoost vs default vs raw LOMN")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "pr_tuned.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
