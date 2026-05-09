"""Split-conformal wrapper around the LOMN+XGBoost classifier.

Method: Vovk-Shafer-Lei split-conformal classification with
non-conformity score s_i = 1 - p_i[true_class_i]. Given target
miscoverage alpha, the q-hat threshold is the (1 - alpha)-quantile of
calibration scores (with the (n+1)/n correction). At test time we
output a *prediction set* containing every class whose score is below
q-hat. The procedure has the marginal coverage guarantee

    P(true label in prediction set) >= 1 - alpha

under exchangeability between calibration and test data (Vovk et al.
2005; Lei & Wasserman 2014; Romano, Sesia & Candes 2020).

For binary classification the prediction set is one of:
    {} (empty, "abstain"), {0}, {1}, or {0, 1} (uncertain).
We treat empty and {0,1} as 'abstain' for downstream FP control.

Typical use: train classifier on train days; calibrate on a held-out
slice of train (or on validation); evaluate prediction sets on test
days; report empirical coverage and average set size.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import config
from src.realdata.train_xgb import FEATURE_COLS

LOG = logging.getLogger("conformal")


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = 0.10           # target miscoverage
    cal_fraction: float = 0.20    # share of train days used for calibration
    seed: int = 42


def split_conformal_threshold(
    p_cal: np.ndarray, y_cal: np.ndarray, alpha: float,
) -> float:
    """1 - alpha quantile of nonconformity scores with finite-sample correction.

    Score for sample i: 1 - p_i[true_class_i]. Returns q-hat such that
    a prediction set {c : 1 - p_c <= q-hat} has marginal coverage
    >= 1 - alpha.
    """
    n = len(p_cal)
    if n < 5:
        return 1.0  # degenerate; predict the all-classes set
    s = 1.0 - p_cal[np.arange(n), y_cal]
    # Empirical 1-alpha quantile with (n+1)/n correction
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(s)[k - 1])


def predict_sets(p: np.ndarray, q_hat: float) -> np.ndarray:
    """Return a (n, K) Boolean matrix; entry [i, c] is True if c in set."""
    return (1.0 - p) <= q_hat


def evaluate_sets(p: np.ndarray, y: np.ndarray, q_hat: float) -> dict:
    """Empirical coverage, average size, and breakdown."""
    in_set = predict_sets(p, q_hat)
    coverage = float(in_set[np.arange(len(y)), y].mean())
    set_sizes = in_set.sum(axis=1)
    abstain = (set_sizes == 0) | (set_sizes == p.shape[1])
    return {
        "coverage": coverage,
        "avg_set_size": float(set_sizes.mean()),
        "abstain_rate": float(abstain.mean()),
        "size_zero_pct": float((set_sizes == 0).mean()),
        "size_one_pct":  float((set_sizes == 1).mean()),
        "size_two_pct":  float((set_sizes == 2).mean()),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = config()
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path,
                   default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--xgb-model", type=Path,
                   default=Path("results/phase3/xgb_lomn_refiner.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/phase_a6"))
    p.add_argument("--alpha", type=float,
                   default=float(cfg["conformal"].get("alpha", 0.10)))
    p.add_argument("--cal-frac", type=float,
                   default=float(cfg["conformal"].get("calibration_fraction", 0.20)))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_days = cfg["split"]["train_days"]
    test_days = cfg["split"]["test_days"]

    feats = pd.read_parquet(args.features)
    labeled = feats[feats["label"] != -1].copy()

    # Time-respecting calibration split: the latest cal_frac of train days
    n_cal_days = max(1, int(round(len(train_days) * args.cal_frac)))
    cal_days = train_days[-n_cal_days:]
    fit_days = train_days[:-n_cal_days]
    LOG.info("fit_days=%s | cal_days=%s | test_days=%s",
             fit_days, cal_days, test_days)

    fit_set  = labeled[labeled["day"].isin(fit_days)]
    cal_set  = labeled[labeled["day"].isin(cal_days)]
    test_set = labeled[labeled["day"].isin(test_days)]

    LOG.info("counts: fit=%d cal=%d test=%d",
             len(fit_set), len(cal_set), len(test_set))
    if len(cal_set) < 30 or len(test_set) < 10:
        raise SystemExit("not enough labeled samples for split-conformal")

    # ----- Re-train XGB on the FIT (not full train) split for honesty -----
    Xfit = fit_set[FEATURE_COLS].values
    yfit = fit_set["label"].values.astype(int)
    pos_w = (1 - yfit).sum() / max(1, yfit.sum())
    val = int(0.85 * len(yfit))
    model = xgb.XGBClassifier(
        n_estimators=cfg["xgb"]["n_estimators"],
        max_depth=cfg["xgb"]["max_depth"],
        learning_rate=cfg["xgb"]["learning_rate"],
        subsample=cfg["xgb"]["subsample"],
        colsample_bytree=cfg["xgb"]["colsample_bytree"],
        scale_pos_weight=pos_w,
        eval_metric="logloss",
        random_state=cfg["xgb"]["random_state"],
        n_jobs=-1,
        early_stopping_rounds=cfg["xgb"]["early_stopping_rounds"],
    )
    model.fit(Xfit[:val], yfit[:val], eval_set=[(Xfit[val:], yfit[val:])], verbose=False)

    # ----- Calibrate -----
    Xcal = cal_set[FEATURE_COLS].values
    ycal = cal_set["label"].values.astype(int)
    pcal = model.predict_proba(Xcal)
    q_hat = split_conformal_threshold(pcal, ycal, alpha=args.alpha)
    LOG.info("q_hat = %.4f at target alpha=%.2f", q_hat, args.alpha)

    # ----- Evaluate prediction sets on test -----
    Xte = test_set[FEATURE_COLS].values
    yte = test_set["label"].values.astype(int)
    pte = model.predict_proba(Xte)
    sets = predict_sets(pte, q_hat)

    metrics_test = evaluate_sets(pte, yte, q_hat)
    LOG.info("=== test prediction sets (alpha=%.2f) ===", args.alpha)
    for k, v in metrics_test.items():
        LOG.info("  %-15s = %.4f", k, v)
    metrics_cal = evaluate_sets(pcal, ycal, q_hat)

    # ----- Sweep alpha for the coverage-vs-size tradeoff plot -----
    rows = []
    for a in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        q_a = split_conformal_threshold(pcal, ycal, alpha=a)
        m_te = evaluate_sets(pte, yte, q_a)
        rows.append({"alpha_target": a, "q_hat": q_a, **m_te})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(args.out_dir / "conformal_alpha_sweep.csv", index=False)

    # ----- Singleton-confidence performance for FP control story -----
    singletons = sets.sum(axis=1) == 1
    singleton_yhat = sets[:, 1].astype(int)
    if singletons.sum() > 0:
        s_y = yte[singletons]
        s_p = singleton_yhat[singletons]
        tp = int(((s_p == 1) & (s_y == 1)).sum())
        fp = int(((s_p == 1) & (s_y == 0)).sum())
        fn = int((yte == 1).sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        singleton_perf = {"singleton_n": int(singletons.sum()),
                          "TP": tp, "FP": fp, "FN_global": fn,
                          "precision_singletons": prec,
                          "recall_global": rec,
                          "F1_singletons_only": f1}
    else:
        singleton_perf = {"singleton_n": 0}

    out = {
        "config": {"alpha": args.alpha,
                   "calibration_fraction": args.cal_frac,
                   "fit_days": fit_days, "cal_days": cal_days, "test_days": test_days},
        "calibration_metrics": metrics_cal,
        "test_metrics": metrics_test,
        "q_hat": q_hat,
        "singleton_perf": singleton_perf,
    }
    (args.out_dir / "conformal_summary.json").write_text(json.dumps(out, indent=2))
    LOG.info("singleton perf: %s", json.dumps(singleton_perf, indent=2))

    # Plot: coverage vs alpha
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(sweep["alpha_target"], 1 - sweep["alpha_target"], "k--",
               lw=1, label="nominal 1-α")
    ax[0].plot(sweep["alpha_target"], sweep["coverage"], "o-",
               color="#2a9d8f", label="empirical")
    ax[0].set_xlabel("target miscoverage α")
    ax[0].set_ylabel("empirical coverage")
    ax[0].set_title("Marginal coverage")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(sweep["alpha_target"], sweep["avg_set_size"], "s-", color="#e76f51",
               label="avg set size")
    ax[1].plot(sweep["alpha_target"], sweep["abstain_rate"], "^-", color="#264653",
               label="abstain rate")
    ax[1].set_xlabel("target miscoverage α")
    ax[1].set_ylabel("set size / abstain")
    ax[1].set_title("Set-size and abstention vs α")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "conformal_coverage.png", dpi=150)
    plt.close(fig)
    LOG.info("plots saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
