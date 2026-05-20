"""Convert Bybit L20 snapshot parquet -> 'resampled_1s' format expected by downstream.

Bybit pipeline output (from bybit_to_l20_snapshots.py):
    ts_local_ns, ts_ms,
    bid_p1..20, bid_q1..20, ask_p1..20, ask_q1..20

Downstream pipeline (run_lomn.py, features.py) expects:
    ts (datetime UTC), bid_p, bid_q, ask_p, ask_q, mid_p,
    log_ask, log_bid, log_mid, spread, [L20 columns kept as-is]

This adapter renames bid_p1->bid_p (top of book) and adds derived
columns (mid, logs, spread) while preserving the full L20 stack for
the new bucket-aggregation feature path.

Files land in the same data/interim directory as the original
resampled_1s_*.parquet from Path B so existing scripts find them
with the standard glob.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("bybit-adapter")


def adapt(src: Path, dst: Path) -> pd.DataFrame:
    df = pd.read_parquet(src)
    out = df.copy()

    # Datetime column from ts_ms
    out["ts"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True)

    # Top of book (L1) aliases for backwards compatibility
    out["bid_p"] = out["bid_p1"].astype(float)
    out["bid_q"] = out["bid_q1"].astype(float)
    out["ask_p"] = out["ask_p1"].astype(float)
    out["ask_q"] = out["ask_q1"].astype(float)
    out["mid_p"] = 0.5 * (out["bid_p"] + out["ask_p"])
    out["spread"] = out["ask_p"] - out["bid_p"]
    out["log_ask"] = np.log(out["ask_p"].clip(lower=1e-12))
    out["log_bid"] = np.log(out["bid_p"].clip(lower=1e-12))
    out["log_mid"] = np.log(out["mid_p"].clip(lower=1e-12))

    # Reorder so the legacy columns come first, then L20 stack
    legacy = ["ts", "bid_p", "bid_q", "ask_p", "ask_q",
              "mid_p", "log_ask", "log_bid", "log_mid", "spread"]
    l20 = [f"{side}_{kind}{i}"
           for i in range(1, 21)
           for side in ("bid", "ask")
           for kind in ("p", "q")]
    rest = [c for c in out.columns if c not in legacy and c not in l20]
    out = out[legacy + l20 + rest]

    out.to_parquet(dst, compression="snappy")
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=Path, default=Path("data/bybit_l20"))
    p.add_argument("--dst-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--symbol-suffix", default=True,
                   help="if true, dst files are named resampled_1s_<symbol>_<date>.parquet")
    args = p.parse_args()

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    ok = skipped = 0
    for src in sorted(args.src_dir.glob("*_l20_1000ms.parquet")):
        # Filename pattern: 2026-05-01_BTCUSDT_l20_1000ms.parquet
        parts = src.stem.split("_")
        date_str, symbol = parts[0], parts[1]
        if symbol not in symbols:
            continue
        # Per-symbol subdirectory so the runner can keep BTC/ETH/SOL pipelines separate
        symbol_dir = args.dst_dir / symbol.lower()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        dst_name = f"resampled_1s_{date_str}.parquet"
        dst = symbol_dir / dst_name
        if dst.exists():
            skipped += 1
            LOG.info("skip (exists): %s/%s", symbol.lower(), dst_name)
            continue
        df = adapt(src, dst)
        LOG.info("%s: %d rows -> %s/%s", src.name, len(df), symbol.lower(), dst_name)
        ok += 1
    LOG.info("done: written=%d skipped=%d", ok, skipped)


if __name__ == "__main__":
    main()
