"""Summarize downloaded Binance Vision historical data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def summarize(path: Path) -> dict:
    df = pd.read_parquet(path, columns=None)
    out = {"file": path.name, "rows": len(df)}
    if "best_bid_price" in df.columns:
        mid = 0.5 * (df["best_bid_price"] + df["best_ask_price"])
        spread = df["best_ask_price"] - df["best_bid_price"]
        out.update({
            "mid_min": float(mid.min()),
            "mid_max": float(mid.max()),
            "spread_med": float(spread.median()),
            "ts_min": pd.to_datetime(df["transaction_time"].min(), unit="ms", utc=True).isoformat(),
            "ts_max": pd.to_datetime(df["transaction_time"].max(), unit="ms", utc=True).isoformat(),
        })
    elif "price" in df.columns:
        out.update({
            "px_min": float(df["price"].min()),
            "px_max": float(df["price"].max()),
            "vol_total": float(df["quantity"].sum()),
            "buyer_maker_pct": float(df["is_buyer_maker"].mean() * 100),
            "ts_min": pd.to_datetime(df["transact_time"].min(), unit="ms", utc=True).isoformat(),
            "ts_max": pd.to_datetime(df["transact_time"].max(), unit="ms", utc=True).isoformat(),
        })
    return out


def main() -> None:
    root = Path("data/historical")
    files = sorted(root.glob("futures_btcusdt_*.parquet"))
    if not files:
        print("no files")
        sys.exit(1)
    book = [f for f in files if "bookTicker" in f.name]
    trades = [f for f in files if "aggTrades" in f.name]

    print(f"=== bookTicker ({len(book)} files) ===")
    rows_b = pd.DataFrame([summarize(f) for f in book])
    print(rows_b.to_string(index=False))
    print(f"  total rows : {rows_b['rows'].sum():,}")
    print(f"  total disk : {sum(f.stat().st_size for f in book) / 1e9:.2f} GB")

    print(f"\n=== aggTrades ({len(trades)} files) ===")
    rows_t = pd.DataFrame([summarize(f) for f in trades])
    print(rows_t.to_string(index=False))
    print(f"  total rows : {rows_t['rows'].sum():,}")
    print(f"  total disk : {sum(f.stat().st_size for f in trades) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
