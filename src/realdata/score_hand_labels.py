"""Compare hand labels against the persistence proxy + each detector.

Reads `data/handlabel/labels_handlabeled.csv` produced by the human
review of the PNGs. Computes:

    1. Agreement rate between persistence-proxy and human labels
    2. Confusion matrix (persistence x hand)
    3. Per-method F1 against HUMAN gold (drop the 'ambig' rows first)

Outputs to results/phase_handlabel/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.realdata.train_xgb import FEATURE_COLS

LOG = logging.getLogger("score-hand")

LABEL_TO_INT = {"real": 1, "noise": 0, "ambig": -1}


def load_hand(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hand_label"] = df["hand_label"].astype(str).str.strip().str.lower()
    df["hand_int"] = df["hand_label"].map(LABEL_TO_INT)
    n_unfilled = int(df["hand_label"].isin(["", "nan"]).sum())
    if n_unfilled:
        LOG.warning("%d rows have no hand_label and will be dropped", n_unfilled)
    df = df[df["hand_int"].notna()]
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--hand", type=Path,
                   default=Path("data/handlabel/labels_handlabeled.csv"))
    p.add_argument("--features", type=Path,
                   default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--xgb-model", type=Path,
                   default=Path("results/phase3/xgb_lomn_refiner.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/phase_handlabel"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.hand.exists():
        raise SystemExit(f"missing {args.hand} — fill in labels_template.csv first")

    hand = load_hand(args.hand)
    feats = pd.read_parquet(args.features)
    feats = feats.merge(
        hand[["cand_id", "day", "obs_idx", "hand_label", "hand_int"]],
        on=["day", "obs_idx"], how="inner",
    )
    LOG.info("matched %d hand-labeled rows to feature set", len(feats))

    # ----- persistence vs hand agreement -----
    keep = feats[feats["hand_int"] != -1].copy()
    keep["persist_int"] = keep["label"].astype(int)
    agree = (keep["persist_int"] == keep["hand_int"]).mean()
    LOG.info("persistence vs hand agreement (excluding 'ambig'): %.1f%% (n=%d)",
             100.0 * agree, len(keep))

    # confusion
    cm = pd.crosstab(
        keep["persist_int"].map({1: "persist=pos", 0: "persist=neg", -1: "persist=drop"}),
        keep["hand_int"].map({1: "hand=real", 0: "hand=noise"}),
        margins=True,
    )
    cm.to_csv(args.out_dir / "confusion_persistence_vs_hand.csv")
    print("\n=== Persistence vs Human confusion ===")
    print(cm.to_string())

    # ----- F1 of each detector against HUMAN gold -----
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_model)
    proba = xgb_model.predict_proba(keep[FEATURE_COLS].values)[:, 1]

    methods = {
        "raw_lomn":   (keep["f_lomn_abs_std"] >= 4.0).astype(int).values,
        "lomn_xgb":   (proba >= 0.5).astype(int).values,
    }
    rows = []
    y_true = keep["hand_int"].astype(int).values
    for name, yhat in methods.items():
        tp = int(((yhat == 1) & (y_true == 1)).sum())
        fp = int(((yhat == 1) & (y_true == 0)).sum())
        fn = int(((yhat == 0) & (y_true == 1)).sum())
        tn = int(((yhat == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"method": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": prec, "recall": rec, "F1": f1})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "f1_vs_hand.csv", index=False)
    print("\n=== F1 vs HUMAN gold (real/noise; ambig dropped) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_json = {
        "n_hand_labeled": int(len(hand)),
        "n_real": int((hand["hand_int"] == 1).sum()),
        "n_noise": int((hand["hand_int"] == 0).sum()),
        "n_ambig": int((hand["hand_int"] == -1).sum()),
        "persistence_vs_hand_agreement": float(agree),
        "matched_to_features": int(len(feats)),
        "methods_vs_hand": rows,
    }
    (args.out_dir / "hand_validation_summary.json").write_text(json.dumps(out_json, indent=2))
    LOG.info("wrote summary -> %s", args.out_dir)


if __name__ == "__main__":
    main()
