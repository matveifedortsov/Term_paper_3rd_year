"""Event validation: do detectors flag known shock periods?

Hand-curated events within the Mar 15-29 2024 window:
    2024-03-15 ~ATH retest near $72,500 - intraday swings
    2024-03-19 ~12:00-15:00 UTC - first leg down toward $63k
    2024-03-19 ~22:00 UTC - sharp drop near $61k
    2024-03-20 ~early hours - selloff continued, low ~$60.8k
    2024-03-25 - rally back toward $71k

For each event window, count detections from each method and check
that >=1 detection lands inside it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.benchmarks.f1_evaluation import (
    detections_lee_mykland,
    detections_lomn_xgb,
    detections_pure_ml,
    detections_raw_lomn,
    load_book,
    train_pure_ml,
)
from src.realdata.train_xgb import FEATURE_COLS

LOG = logging.getLogger("events")

EVENTS = [
    {"name": "ATH retest spikes",       "day": "2024-03-15", "start_h": 12, "end_h": 18},
    {"name": "First leg down to $63k",  "day": "2024-03-19", "start_h": 12, "end_h": 16},
    {"name": "Late-night drop to $61k", "day": "2024-03-19", "start_h": 21, "end_h": 24},
    {"name": "Selloff continues",       "day": "2024-03-20", "start_h":  0, "end_h":  6},
    {"name": "Rally toward $71k",       "day": "2024-03-25", "start_h": 13, "end_h": 19},
]


def detections_in_window(det_obs_idx: np.ndarray, h_start: int, h_end: int) -> int:
    s = h_start * 3600
    e = h_end * 3600
    return int(((det_obs_idx >= s) & (det_obs_idx < e)).sum())


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
    pure_model, pure_cols = train_pure_ml(features)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_model)

    rows = []
    for ev in EVENTS:
        book = load_book(args.book_dir, ev["day"])
        det_lomn = detections_raw_lomn(features, ev["day"])
        det_lm = detections_lee_mykland(book)
        det_xgb = detections_lomn_xgb(features, ev["day"], xgb_model)
        det_pure = detections_pure_ml(features, ev["day"], pure_model, pure_cols)

        rows.append({
            "event": ev["name"], "day": ev["day"],
            "window_h": f"{ev['start_h']:02d}:00-{ev['end_h']:02d}:00",
            "raw_lomn":     detections_in_window(det_lomn, ev["start_h"], ev["end_h"]),
            "lee_mykland":  detections_in_window(det_lm,   ev["start_h"], ev["end_h"]),
            "lomn_xgb":     detections_in_window(det_xgb,  ev["start_h"], ev["end_h"]),
            "pure_ml":      detections_in_window(det_pure, ev["start_h"], ev["end_h"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "event_detections.csv", index=False)
    print("\n=== Detections per event window ===")
    print(df.to_string(index=False))

    # Hit rate: 1 if >= 1 detection inside the window
    hits = df.copy()
    for col in ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]:
        hits[col] = (df[col] >= 1).astype(int)
    print("\n=== Hit-rate per method (1 = >=1 detection in window) ===")
    print(hits.to_string(index=False))

    summary = {
        "events_total": len(EVENTS),
        "hits_per_method": {
            col: int(hits[col].sum())
            for col in ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]
        },
    }
    (args.out_dir / "event_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Hit summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
