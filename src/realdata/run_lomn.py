"""Apply the LOMN detector to resampled real BTC data.

For each day, runs the block-minimum detector on log_ask, returns
per-block test statistics, and emits all block boundaries whose
|standardized stat| exceeds a threshold (default 3.0) as candidates.

A low threshold is used here to generate a CANDIDATE SET; downstream
Stage 2 (XGBoost) refines them. The block constant matches Phase 1's
Bibinger-rate convention h_n = round(c * n^{1/3}).

Output: data/interim/lomn_candidates_<date>.parquet with columns:
    ts, obs_idx, log_ask_at_min_left, log_ask_at_min_right,
    delta_M, abs_std, signed_std, scale, h_n, threshold, day
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.lomn.detector import block_minima, robust_scale, optimal_block_size

LOG = logging.getLogger("lomn-real")


def detect_one_day(
    df: pd.DataFrame,
    h_n: int,
    candidate_threshold: float,
) -> pd.DataFrame:
    Y = df["log_ask"].values.astype(float)
    n = len(Y)
    M = block_minima(Y, h_n)
    delta_M = np.diff(M)
    scale = robust_scale(delta_M)
    if scale <= 0:
        scale = float(np.std(delta_M, ddof=1)) or 1e-12
    standardized = delta_M / scale
    abs_std = np.abs(standardized)

    cand_block = np.where(abs_std > candidate_threshold)[0]
    cand_obs = (cand_block + 1) * h_n
    cand_obs = np.clip(cand_obs, 0, n - 1)

    out = pd.DataFrame({
        "ts": df["ts"].values[cand_obs],
        "obs_idx": cand_obs,
        "block_idx": cand_block,
        "log_ask_at_boundary": Y[cand_obs],
        "delta_M": delta_M[cand_block],
        "signed_std": standardized[cand_block],
        "abs_std": abs_std[cand_block],
        "scale": scale,
        "h_n": h_n,
        "threshold": candidate_threshold,
    })
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--dst-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--pattern", default="resampled_1s_*.parquet")
    p.add_argument("--threshold", type=float, default=3.0,
                   help="emit candidates with |std stat| > this")
    p.add_argument("--block-c", type=float, default=1.0)
    args = p.parse_args()

    files = sorted(args.src_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {args.src_dir}")

    all_cands = []
    summary_rows = []
    for f in files:
        date_str = f.stem.split("_")[-1]
        df = pd.read_parquet(f)
        n = len(df)
        h_n = optimal_block_size(n, c=args.block_c)
        cands = detect_one_day(df, h_n=h_n, candidate_threshold=args.threshold)
        cands["day"] = date_str

        n_blocks = (n // h_n) - 1
        n_cands = len(cands)
        scale = float(cands["scale"].iloc[0]) if n_cands else float("nan")
        max_stat = float(cands["abs_std"].max()) if n_cands else float("nan")
        LOG.info(
            "%s  n=%d  h_n=%d  blocks=%d  cands=%d  max|stat|=%.2f  scale=%.2e",
            date_str, n, h_n, n_blocks, n_cands, max_stat, scale,
        )

        dst = args.dst_dir / f"lomn_candidates_{date_str}.parquet"
        cands.to_parquet(dst, compression="snappy")
        all_cands.append(cands)
        summary_rows.append({
            "day": date_str,
            "n_obs": n, "h_n": h_n, "n_blocks": n_blocks,
            "n_candidates": n_cands,
            "max_abs_std": max_stat,
            "scale": scale,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.dst_dir / "lomn_candidates_summary.csv", index=False)
    print("\n=== Candidate counts per day ===")
    print(summary.to_string(index=False))
    total_cands = sum(len(c) for c in all_cands)
    print(f"\nTotal candidates across {len(files)} days: {total_cands}")


if __name__ == "__main__":
    main()
