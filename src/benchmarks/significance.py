"""Statistical significance for the F1/AUC comparisons.

Three tests:

1. Bootstrap percentile CIs on F1 for each method.
   We resample the persistence-labeled candidate set on the test days
   with replacement (paired across methods), recompute F1 on each
   resample, and take the 2.5/97.5 percentiles.

2. McNemar's test for paired correctness.
   For each labeled candidate on the test days, each method either
   classifies it correctly or not (relative to the persistence label).
   The 2x2 contingency table over disagreements between method A and
   method B (b = A correct, B wrong; c = A wrong, B correct) gives
   McNemar's chi-square (b - c)^2 / (b + c) on 1 df.

3. DeLong's test for paired AUC differences.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from src.benchmarks.f1_evaluation import (
    load_book,
    train_pure_ml,
)
from src.benchmarks.lee_mykland import detect_jumps
from src.realdata.train_xgb import FEATURE_COLS, TEST_DAYS

LOG = logging.getLogger("sig")

TOLERANCE_S = 60
N_BOOTSTRAP = 5000
PROBA_THRESHOLD = 0.5
RAW_LOMN_THRESHOLD = 4.0
LM_K = 270
LM_ALPHA = 0.05


def lm_score_at_candidate(
    L_array: np.ndarray, obs_idx: int, tol: int = TOLERANCE_S
) -> float:
    lo = max(0, obs_idx - tol)
    hi = min(len(L_array), obs_idx + tol + 1)
    seg = L_array[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(seg.max()) if len(seg) else 0.0


def _bootstrap_f1(
    methods_y_true: np.ndarray,
    methods_y_pred: dict[str, np.ndarray],
    n_iter: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = len(methods_y_true)
    out = {m: np.empty(n_iter) for m in methods_y_pred}
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt = methods_y_true[idx]
        for m, yp in methods_y_pred.items():
            yp_b = yp[idx]
            tp = int(((yp_b == 1) & (yt == 1)).sum())
            fp = int(((yp_b == 1) & (yt == 0)).sum())
            fn = int(((yp_b == 0) & (yt == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / (tp + fn) if (tp + fn) else 1.0
            out[m][b] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


def mcnemar(yt: np.ndarray, yA: np.ndarray, yB: np.ndarray) -> dict:
    correctA = (yA == yt)
    correctB = (yB == yt)
    b = int((correctA & ~correctB).sum())
    c = int((~correctA & correctB).sum())
    if b + c == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1.0 - norm.cdf(np.sqrt(chi2))
    p_two = 2.0 * min(p, 1.0 - p)
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(p_two)}


def _placement_values(scores_pos: np.ndarray, scores_neg: np.ndarray):
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    sort_neg = np.sort(scores_neg)
    V10 = np.searchsorted(sort_neg, scores_pos, side="right") / n_neg
    sort_pos = np.sort(scores_pos)
    V01 = (n_pos - np.searchsorted(sort_pos, scores_neg, side="left")) / n_pos
    return V10, V01


def delong_paired(scores_A: np.ndarray, scores_B: np.ndarray, labels: np.ndarray) -> dict:
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return {"auc_A": float("nan"), "auc_B": float("nan"),
                "diff": float("nan"), "se": float("nan"),
                "z": float("nan"), "p_value": float("nan")}

    posA = scores_A[pos_mask]; negA = scores_A[neg_mask]
    posB = scores_B[pos_mask]; negB = scores_B[neg_mask]
    V10A, V01A = _placement_values(posA, negA)
    V10B, V01B = _placement_values(posB, negB)
    aucA = V10A.mean()
    aucB = V10B.mean()
    cov10 = float(np.cov(V10A, V10B, ddof=1)[0, 1]) if n_pos > 1 else 0.0
    cov01 = float(np.cov(V01A, V01B, ddof=1)[0, 1]) if n_neg > 1 else 0.0
    var10A = float(V10A.var(ddof=1)) if n_pos > 1 else 0.0
    var10B = float(V10B.var(ddof=1)) if n_pos > 1 else 0.0
    var01A = float(V01A.var(ddof=1)) if n_neg > 1 else 0.0
    var01B = float(V01B.var(ddof=1)) if n_neg > 1 else 0.0
    var_diff = (var10A + var10B - 2 * cov10) / n_pos + (var01A + var01B - 2 * cov01) / n_neg
    var_diff = max(var_diff, 0.0)
    se = float(np.sqrt(var_diff))
    z = (aucA - aucB) / se if se > 0 else float("inf")
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return {"auc_A": float(aucA), "auc_B": float(aucB),
            "diff": float(aucA - aucB), "se": se,
            "z": float(z), "p_value": float(p)}


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
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    test = feats[feats["day"].isin(TEST_DAYS) & (feats["label"] != -1)].copy()
    LOG.info("test labeled candidates: %d", len(test))

    pure_model, pure_cols = train_pure_ml(feats)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_model)

    scores: dict[str, np.ndarray] = {
        "raw_lomn": test["f_lomn_abs_std"].values.astype(float),
        "lomn_xgb": xgb_model.predict_proba(test[FEATURE_COLS].values)[:, 1],
        "pure_ml":  pure_model.predict_proba(test[pure_cols].values)[:, 1],
    }

    LOG.info("computing LM scores around each labeled candidate")
    cached_L: dict[str, np.ndarray] = {}
    lm_scores = np.empty(len(test))
    for i, (_, row) in enumerate(test.reset_index(drop=True).iterrows()):
        d = row["day"]
        if d not in cached_L:
            book = load_book(args.book_dir, d)
            res = detect_jumps(book["log_mid"].values, K=LM_K, alpha=LM_ALPHA)
            cached_L[d] = res["L"]
        lm_scores[i] = lm_score_at_candidate(cached_L[d], int(row["obs_idx"]))
    scores["lee_mykland"] = lm_scores

    labels = test["label"].values.astype(int)

    yhat = {
        "raw_lomn": (scores["raw_lomn"] >= RAW_LOMN_THRESHOLD).astype(int),
        "lomn_xgb": (scores["lomn_xgb"] >= PROBA_THRESHOLD).astype(int),
        "pure_ml":  (scores["pure_ml"]  >= PROBA_THRESHOLD).astype(int),
    }
    first_book = load_book(args.book_dir, TEST_DAYS[0])
    lm_cv = float(detect_jumps(first_book["log_mid"].values, K=LM_K, alpha=LM_ALPHA)["critical_value"])
    yhat["lee_mykland"] = (lm_scores >= lm_cv).astype(int)

    point = {}
    for m, y in yhat.items():
        tp = int(((y == 1) & (labels == 1)).sum())
        fp = int(((y == 1) & (labels == 0)).sum())
        fn = int(((y == 0) & (labels == 1)).sum())
        tn = int(((y == 0) & (labels == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        point[m] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "precision": prec, "recall": rec, "F1": f1}

    rng = np.random.default_rng(20260509)
    boots = _bootstrap_f1(labels, yhat, n_iter=args.n_bootstrap, rng=rng)
    ci = {
        m: {
            "F1_mean":     float(np.mean(b)),
            "F1_lo_2.5":   float(np.quantile(b, 0.025)),
            "F1_hi_97.5":  float(np.quantile(b, 0.975)),
        }
        for m, b in boots.items()
    }
    diff = {}
    for m in ["lomn_xgb", "pure_ml", "lee_mykland"]:
        d_b = boots[m] - boots["raw_lomn"]
        p_two = 2.0 * min(float(np.mean(d_b > 0)), float(np.mean(d_b < 0)))
        diff[f"{m}_minus_raw_lomn"] = {
            "diff_mean":    float(np.mean(d_b)),
            "diff_lo_2.5":  float(np.quantile(d_b, 0.025)),
            "diff_hi_97.5": float(np.quantile(d_b, 0.975)),
            "p_two_sided_vs_zero": p_two,
        }

    mcn = {}
    for m in ["lomn_xgb", "pure_ml", "lee_mykland"]:
        mcn[f"{m}_vs_raw_lomn"] = mcnemar(labels, yhat[m], yhat["raw_lomn"])

    delong = {}
    for m in ["lomn_xgb", "pure_ml", "lee_mykland"]:
        delong[f"{m}_vs_raw_lomn"] = delong_paired(scores[m], scores["raw_lomn"], labels)

    out = {
        "n_test_labeled": int(len(test)),
        "n_bootstrap": args.n_bootstrap,
        "point_estimates": point,
        "bootstrap_F1_CI": ci,
        "bootstrap_F1_difference_vs_raw_lomn": diff,
        "mcnemar_vs_raw_lomn": mcn,
        "delong_AUC_vs_raw_lomn": delong,
    }
    (args.out_dir / "significance.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
