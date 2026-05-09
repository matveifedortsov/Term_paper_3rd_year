"""Persistence-based labeling for LOMN candidates.

CRITICAL DESIGN CHOICE: gold-truth labels use a FORWARD-LOOKING signal
(log_mid 30s after tau minus 30s before tau) which is never given to
the XGBoost model as a feature. This keeps the H1 comparison honest:
both raw LOMN and ML are trying to predict an external label, neither
sees it during training.

Logic — a real jump should produce a sustained price-level shift over
a 1-minute window centered at tau, several times the local diffusion
scale. A noise spike reverts.

    persistence_z = |log_mid(tau+30) - log_mid(tau-30)| / scale
        where `scale` is the per-day MAD-based delta_M scale used by the
        LOMN detector.

    POSITIVE: persistence_z >= POS_PERSIST_Z  (sustained level shift)
    NEGATIVE: persistence_z <= NEG_PERSIST_Z  (mean-reverting blip)
    DROP   : in between

Optional secondary filters (require both persistence AND trade-flow
agreement / disagreement) tighten precision further.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("label")

POS_PERSIST_Z = 5.0
NEG_PERSIST_Z = 2.0


def label_features(df: pd.DataFrame, scale_per_day: pd.Series | None = None) -> pd.DataFrame:
    out = df.copy()
    if scale_per_day is not None:
        out["_scale"] = out["day"].map(scale_per_day)
    else:
        # Fallback: per-day MAD of f_realvar_60s as a robust scale proxy
        out["_scale"] = out.groupby("day")["f_realvar_60s"].transform(
            lambda s: max(float(np.sqrt(s.median())), 1e-9)
        )

    persist_abs = out["label_persistence_30s"].abs()
    persist_z = persist_abs / out["_scale"].clip(lower=1e-9)
    out["persist_z"] = persist_z

    is_pos = persist_z >= POS_PERSIST_Z
    is_neg = persist_z <= NEG_PERSIST_Z

    out["label"] = np.where(is_pos, 1, np.where(is_neg, 0, -1))
    out["label_reason"] = np.where(
        is_pos, "pos",
        np.where(is_neg, "neg", "drop"),
    )
    n_pos = int(is_pos.sum())
    n_neg = int(is_neg.sum())
    n_drop = int((~is_pos & ~is_neg).sum())
    LOG.info(
        "labels: pos=%d  neg=%d  drop=%d  (pos rate among labeled = %.1f%%)",
        n_pos, n_neg, n_drop,
        100.0 * n_pos / max(1, (n_pos + n_neg)),
    )
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/interim/features_all.parquet"))
    p.add_argument("--dst", type=Path, default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--summary", type=Path, default=Path("data/interim/lomn_candidates_summary.csv"))
    args = p.parse_args()

    df = pd.read_parquet(args.src)
    scale_per_day = None
    if args.summary.exists():
        s = pd.read_csv(args.summary).set_index("day")["scale"]
        scale_per_day = s
        LOG.info("loaded scale per day from %s", args.summary)
    out = label_features(df, scale_per_day=scale_per_day)
    out.to_parquet(args.dst, compression="snappy")
    LOG.info("wrote %d rows -> %s", len(out), args.dst)

    print("\n=== label breakdown by day ===")
    print(out.groupby(["day", "label_reason"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
