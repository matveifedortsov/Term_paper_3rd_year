"""Resample raw bookTicker (event-driven) to a regular 1Hz grid.

Output columns (per second, UTC, last-update wins):
    ts, bid_p, bid_q, ask_p, ask_q, mid_p, log_ask, log_bid, log_mid, spread

Forward-fill is used for any gap second (rare on BTC futures: typical
gaps are sub-millisecond).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("resample")


def resample_one_day(src: Path, dst: Path, freq: str = "1s") -> pd.DataFrame:
    df = pd.read_parquet(
        src,
        columns=[
            "best_bid_price", "best_bid_qty",
            "best_ask_price", "best_ask_qty",
            "transaction_time",
        ],
    )
    df = df.sort_values("transaction_time", kind="mergesort")
    df["ts"] = pd.to_datetime(df["transaction_time"], unit="ms", utc=True)
    df = df.set_index("ts")

    # last update within each second
    grid = df.resample(freq).last()
    grid["best_bid_price"] = grid["best_bid_price"].ffill()
    grid["best_bid_qty"] = grid["best_bid_qty"].ffill()
    grid["best_ask_price"] = grid["best_ask_price"].ffill()
    grid["best_ask_qty"] = grid["best_ask_qty"].ffill()
    grid = grid.dropna(subset=["best_bid_price", "best_ask_price"])

    out = pd.DataFrame({
        "ts": grid.index,
        "bid_p": grid["best_bid_price"].values,
        "bid_q": grid["best_bid_qty"].values,
        "ask_p": grid["best_ask_price"].values,
        "ask_q": grid["best_ask_qty"].values,
    })
    out["mid_p"] = 0.5 * (out["bid_p"] + out["ask_p"])
    out["log_ask"] = np.log(out["ask_p"].values)
    out["log_bid"] = np.log(out["bid_p"].values)
    out["log_mid"] = np.log(out["mid_p"].values)
    out["spread"] = out["ask_p"] - out["bid_p"]

    out.to_parquet(dst, compression="snappy")
    LOG.info("%s -> %s  (%d rows)", src.name, dst.name, len(out))
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=Path, default=Path("data/historical"))
    p.add_argument("--dst-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--pattern", default="futures_btcusdt_bookTicker_*.parquet")
    p.add_argument("--freq", default="1s")
    args = p.parse_args()

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {args.src_dir}")

    LOG.info("resampling %d files at %s", len(files), args.freq)
    for f in files:
        date_str = f.stem.split("_")[-1]
        dst = args.dst_dir / f"resampled_{args.freq}_{date_str}.parquet"
        if dst.exists():
            LOG.info("skip (exists): %s", dst.name)
            continue
        resample_one_day(f, dst, freq=args.freq)


if __name__ == "__main__":
    main()
