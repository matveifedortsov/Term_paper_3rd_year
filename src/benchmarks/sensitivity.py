"""Sensitivity sweep: do the headline F1 numbers depend on free parameters?

We sweep three design choices, each holding the others at their default
values, and report F1 for raw_lomn and lomn_xgb at each setting:

    1. LOMN candidate threshold        in {1.5, 2.0, 2.5, 3.0}
       (the |std-stat| floor for emitting a candidate)
    2. LOMN block-constant c           in {0.5, 0.75, 1.0, 1.5, 2.0}
       (h_n = round(c * n^{1/3}))
    3. Persistence positive threshold  in {3.0, 4.0, 5.0, 6.0}
       (sigma multiples for "this is a real jump")

For each setting, we re-run the relevant downstream stages and recompute
F1 on the test days against the (potentially relabeled) ground truth.

Sweeps 1 and 3 reuse cached candidates with threshold 2.0; sweep 2
re-runs the LOMN detector with a new block constant.
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

from src.benchmarks.f1_evaluation import f1_match
from src.lomn.detector import block_minima, optimal_block_size, robust_scale
from src.realdata.label import label_features
from src.realdata.train_xgb import FEATURE_COLS, TEST_DAYS, TRAIN_DAYS

LOG = logging.getLogger("sens")
TOLERANCE_S = 60
RAW_LOMN_THR_DEFAULT = 4.0
ML_PROBA_THR = 0.5

DAYS = TRAIN_DAYS + TEST_DAYS


def _train_xgb(features: pd.DataFrame) -> xgb.XGBClassifier:
    train = features[features["day"].isin(TRAIN_DAYS) & (features["label"] != -1)]
    if len(train) < 30:
        raise RuntimeError(f"too few training rows: {len(train)}")
    Xtr = train[FEATURE_COLS].values
    ytr = train["label"].values.astype(int)
    pos_w = (1 - ytr).sum() / max(1, ytr.sum())
    val_split = int(0.85 * len(ytr))
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=pos_w, eval_metric="logloss",
        random_state=42, n_jobs=-1, early_stopping_rounds=20,
    )
    model.fit(Xtr[:val_split], ytr[:val_split],
              eval_set=[(Xtr[val_split:], ytr[val_split:])], verbose=False)
    return model


def _f1_on_test(features: pd.DataFrame, model: xgb.XGBClassifier,
                raw_thr: float) -> dict:
    test = features[features["day"].isin(TEST_DAYS)].copy()
    truth_idx_per_day = {
        d: features[(features["day"] == d) & (features["label"] == 1)]["obs_idx"].values.astype(int)
        for d in TEST_DAYS
    }

    out = {}
    for method in ["raw_lomn", "lomn_xgb"]:
        tp_tot = fp_tot = fn_tot = 0
        for d in TEST_DAYS:
            g = test[test["day"] == d]
            if len(g) == 0:
                truth = truth_idx_per_day[d]
                tp_tot += 0
                fp_tot += 0
                fn_tot += len(truth)
                continue
            if method == "raw_lomn":
                det = g["obs_idx"].values[g["f_lomn_abs_std"] >= raw_thr].astype(int)
            else:
                proba = model.predict_proba(g[FEATURE_COLS].values)[:, 1]
                det = g["obs_idx"].values[proba >= ML_PROBA_THR].astype(int)
            stats = f1_match(det, truth_idx_per_day[d], tol=TOLERANCE_S)
            tp_tot += stats["TP"]
            fp_tot += stats["FP"]
            fn_tot += stats["FN"]
        prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 1.0
        rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[method] = {"TP": tp_tot, "FP": fp_tot, "FN": fn_tot,
                       "precision": prec, "recall": rec, "F1": f1}
    return out


# ---------------- Sweep helpers ----------------

def sweep_candidate_threshold(features_full: pd.DataFrame, scale_per_day: pd.Series) -> pd.DataFrame:
    """Vary the LOMN |stat| floor for emitting a candidate."""
    rows = []
    for thr in [1.5, 2.0, 2.5, 3.0]:
        sub = features_full[features_full["f_lomn_abs_std"] >= thr].copy()
        if len(sub) < 30:
            continue
        labeled = label_features(sub, scale_per_day=scale_per_day)
        try:
            model = _train_xgb(labeled)
        except RuntimeError as e:
            LOG.warning("threshold %.2f: %s", thr, e)
            continue
        f1 = _f1_on_test(labeled, model, raw_thr=max(thr + 0.5, RAW_LOMN_THR_DEFAULT))
        rows.append({
            "param": "candidate_threshold", "value": thr,
            "n_candidates": len(sub),
            "F1_raw_lomn": f1["raw_lomn"]["F1"],
            "F1_lomn_xgb": f1["lomn_xgb"]["F1"],
            "TP_xgb": f1["lomn_xgb"]["TP"],
            "FP_xgb": f1["lomn_xgb"]["FP"],
        })
        LOG.info("threshold=%.2f F1_raw=%.3f F1_xgb=%.3f",
                 thr, f1["raw_lomn"]["F1"], f1["lomn_xgb"]["F1"])
    return pd.DataFrame(rows)


def sweep_persistence_threshold(features_full: pd.DataFrame, scale_per_day: pd.Series) -> pd.DataFrame:
    """Vary the persistence-z threshold for POSITIVE label."""
    rows = []
    from src.realdata import label as label_mod
    orig_pos = label_mod.POS_PERSIST_Z
    orig_neg = label_mod.NEG_PERSIST_Z
    try:
        for pos_z in [3.0, 4.0, 5.0, 6.0]:
            label_mod.POS_PERSIST_Z = pos_z
            label_mod.NEG_PERSIST_Z = min(2.0, pos_z - 1.0)
            labeled = label_mod.label_features(features_full, scale_per_day=scale_per_day)
            try:
                model = _train_xgb(labeled)
            except RuntimeError as e:
                LOG.warning("pos_z=%.1f: %s", pos_z, e)
                continue
            f1 = _f1_on_test(labeled, model, raw_thr=RAW_LOMN_THR_DEFAULT)
            rows.append({
                "param": "persistence_pos_z", "value": pos_z,
                "n_pos": int((labeled["label"] == 1).sum()),
                "n_neg": int((labeled["label"] == 0).sum()),
                "F1_raw_lomn": f1["raw_lomn"]["F1"],
                "F1_lomn_xgb": f1["lomn_xgb"]["F1"],
                "TP_xgb": f1["lomn_xgb"]["TP"],
                "FP_xgb": f1["lomn_xgb"]["FP"],
            })
            LOG.info("pos_z=%.1f n_pos=%d n_neg=%d F1_raw=%.3f F1_xgb=%.3f",
                     pos_z, int((labeled["label"] == 1).sum()),
                     int((labeled["label"] == 0).sum()),
                     f1["raw_lomn"]["F1"], f1["lomn_xgb"]["F1"])
    finally:
        label_mod.POS_PERSIST_Z = orig_pos
        label_mod.NEG_PERSIST_Z = orig_neg
    return pd.DataFrame(rows)


def sweep_block_constant(book_dir: Path, trades_dir: Path,
                          aggregate_threshold: float = 2.0) -> pd.DataFrame:
    """Vary block-constant c for h_n = round(c * n^{1/3}). Re-runs LOMN."""
    from src.realdata.features import build_features_for_day

    rows = []
    # cache per-day book and trades (slow to load)
    books: dict[str, pd.DataFrame] = {}
    trades: dict[str, pd.DataFrame] = {}
    LOG.info("loading book and trade data for %d days", len(DAYS))
    for d in DAYS:
        bf = book_dir / f"resampled_1s_{d}.parquet"
        tf = trades_dir / f"futures_btcusdt_aggTrades_{d}.parquet"
        if bf.exists() and tf.exists():
            books[d] = pd.read_parquet(bf)
            trades[d] = pd.read_parquet(
                tf, columns=["transact_time", "quantity", "is_buyer_maker"]
            )

    for c in [0.5, 0.75, 1.0, 1.5, 2.0]:
        all_cands = []
        scales = {}
        n = 86_400
        h_n = optimal_block_size(n, c=c)
        for d, book in books.items():
            Y = book["log_ask"].values.astype(float)
            M = block_minima(Y, h_n)
            delta_M = np.diff(M)
            scale = robust_scale(delta_M)
            if scale <= 0:
                scale = float(np.std(delta_M, ddof=1)) or 1e-12
            standardized = delta_M / scale
            abs_std = np.abs(standardized)
            cand_block = np.where(abs_std >= aggregate_threshold)[0]
            cand_obs = np.clip((cand_block + 1) * h_n, 0, n - 1)
            cands = pd.DataFrame({
                "ts": book["ts"].values[cand_obs],
                "obs_idx": cand_obs,
                "block_idx": cand_block,
                "log_ask_at_boundary": Y[cand_obs],
                "delta_M": delta_M[cand_block],
                "signed_std": standardized[cand_block],
                "abs_std": abs_std[cand_block],
                "scale": scale,
                "h_n": h_n,
                "threshold": aggregate_threshold,
                "day": d,
            })
            scales[d] = scale
            all_cands.append(cands)
        cand_df = pd.concat(all_cands, ignore_index=True)

        feats = []
        for d, book in books.items():
            cands_day = cand_df[cand_df["day"] == d]
            if len(cands_day) == 0:
                continue
            f = build_features_for_day(cands_day, book, trades[d])
            feats.append(f)
        feats_df = pd.concat(feats, ignore_index=True)
        feats_df = feats_df.merge(
            cand_df[["day", "obs_idx", "delta_M"]], on=["day", "obs_idx"], how="left"
        )
        labeled = label_features(feats_df, scale_per_day=pd.Series(scales))
        try:
            model = _train_xgb(labeled)
        except RuntimeError as e:
            LOG.warning("c=%.2f: %s", c, e)
            continue
        f1 = _f1_on_test(labeled, model, raw_thr=RAW_LOMN_THR_DEFAULT)
        rows.append({
            "param": "block_constant_c", "value": c, "h_n": h_n,
            "n_candidates": len(cand_df),
            "F1_raw_lomn": f1["raw_lomn"]["F1"],
            "F1_lomn_xgb": f1["lomn_xgb"]["F1"],
            "TP_xgb": f1["lomn_xgb"]["TP"],
            "FP_xgb": f1["lomn_xgb"]["FP"],
        })
        LOG.info("c=%.2f h_n=%d cands=%d F1_raw=%.3f F1_xgb=%.3f",
                 c, h_n, len(cand_df), f1["raw_lomn"]["F1"], f1["lomn_xgb"]["F1"])
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--trades-dir", type=Path, default=Path("data/historical"))
    p.add_argument("--summary", type=Path, default=Path("data/interim/lomn_candidates_summary.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase6"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats_full = pd.read_parquet(args.features)
    summary = pd.read_csv(args.summary).set_index("day")["scale"]

    LOG.info("=== Sweep 1: candidate threshold ===")
    s1 = sweep_candidate_threshold(feats_full, summary)
    LOG.info("=== Sweep 2: persistence positive z ===")
    s2 = sweep_persistence_threshold(feats_full, summary)
    LOG.info("=== Sweep 3: block constant c ===")
    s3 = sweep_block_constant(args.book_dir, args.trades_dir)

    full = pd.concat([s1, s2, s3], ignore_index=True)
    full.to_csv(args.out_dir / "sensitivity.csv", index=False)
    print("\n=== Sensitivity sweep ===")
    print(full.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, p_name, title in [
        (axes[0], "candidate_threshold", "LOMN candidate threshold"),
        (axes[1], "persistence_pos_z",   "Persistence positive z-threshold"),
        (axes[2], "block_constant_c",    "Block constant c (h_n = c * n^(1/3))"),
    ]:
        sub = full[full["param"] == p_name]
        ax.plot(sub["value"], sub["F1_raw_lomn"], "o-", lw=1.5, ms=6,
                color="#264653", label="raw LOMN")
        ax.plot(sub["value"], sub["F1_lomn_xgb"], "s-", lw=1.5, ms=6,
                color="#2a9d8f", label="LOMN+XGB")
        ax.set_xlabel(p_name)
        ax.set_ylabel("F1 on test days")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Sensitivity of headline F1 to design choices")
    fig.tight_layout()
    fig.savefig(args.out_dir / "sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
