"""Inspect captured Binance depth files.

Run after capture has been going for a while. Reports per-file row
counts, message rate, gap detection, and a sanity check that bids
are below asks.

    python scripts/verify_capture.py --dir data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_file(path: Path) -> dict:
    df = pd.read_parquet(path)
    n = len(df)
    if n == 0:
        return {"file": path.name, "rows": 0}
    ts = pd.to_datetime(df["ts_local_ns"], unit="ns", utc=True)
    span_s = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    rate = n / span_s if span_s > 0 else float("nan")
    deltas_ms = np.diff(df["ts_local_ns"].values) / 1e6
    spread_violations = int((df["ask_p1"] <= df["bid_p1"]).sum())
    return {
        "file": path.name,
        "rows": n,
        "span_s": round(span_s, 1),
        "rate_hz": round(rate, 1),
        "median_gap_ms": float(np.median(deltas_ms)),
        "p99_gap_ms": float(np.percentile(deltas_ms, 99)),
        "max_gap_ms": float(deltas_ms.max()),
        "spread_violations": spread_violations,
        "first_mid": round(0.5 * (df["bid_p1"].iloc[0] + df["ask_p1"].iloc[0]), 2),
        "last_mid": round(0.5 * (df["bid_p1"].iloc[-1] + df["ask_p1"].iloc[-1]), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("data/raw"))
    p.add_argument("--last", type=int, default=10, help="show only last N files")
    args = p.parse_args()

    files = sorted(args.dir.glob("*.parquet"))
    if not files:
        print(f"no parquet files under {args.dir}")
        sys.exit(1)
    print(f"found {len(files)} files; showing last {min(args.last, len(files))}")

    rows = [inspect_file(f) for f in files[-args.last :]]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    total_rows = sum(r["rows"] for r in rows)
    total_span = sum(r["span_s"] for r in rows)
    overall_rate = total_rows / total_span if total_span > 0 else float("nan")
    target_rate = 10.0  # depth20@100ms = 10 Hz
    print(
        f"\noverall: {total_rows} rows over {total_span:.0f}s "
        f"({overall_rate:.2f} Hz; target {target_rate:.0f} Hz)"
    )
    if overall_rate < 0.6 * target_rate:
        print("WARNING: rate well below target — check disconnects in capture.log")


if __name__ == "__main__":
    main()
