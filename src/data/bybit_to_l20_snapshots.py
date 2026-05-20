"""End-to-end: Bybit raw archive -> 1Hz L20 snapshot parquet.

Combines the fetcher (downloading per-day .zip) and the reconstructor
(replaying snapshot + deltas) and writes parquet files with the schema
expected by the rest of the project's feature pipeline:

    ts_local_ns, ts_ms,
    bid_p1..20, bid_q1..20, ask_p1..20, ask_q1..20

These match the columns produced by `capture_binance_stream.py`, so
`src.realdata.resample` and downstream stages run unchanged.

Usage:
    python -m src.data.bybit_to_l20_snapshots \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
        --start 2025-05-01 --end 2025-05-05 \\
        --grid-ms 1000 \\
        --raw-dir data/bybit_raw \\
        --out-dir data/bybit_l20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.fetch_bybit_orderbook import (
    DEPTH as RAW_DEPTH,
    daterange,
    fetch_one,
)
from src.realdata.reconstruct_book import replay_to_l20_rows

LOG = logging.getLogger("bybit-l20")
L20_DEPTH = 20


def process_one(
    symbol: str,
    day: date,
    raw_dir: Path,
    out_dir: Path,
    grid_ms: int,
    overwrite: bool = False,
) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{day.isoformat()}_{symbol.upper()}_l20_{grid_ms}ms.parquet"
    if out_path.exists() and not overwrite:
        LOG.info("skip (exists): %s", out_path.name)
        return out_path

    raw_path = raw_dir / symbol.lower() / (
        f"{day.isoformat()}_{symbol.upper()}_ob{RAW_DEPTH}.data.zip"
    )
    if not raw_path.exists():
        LOG.info("downloading %s %s", symbol, day)
        result = fetch_one(symbol, day, raw_dir / symbol.lower())
        if result is None:
            return None
        raw_path = result

    t0 = time.perf_counter()
    rows = replay_to_l20_rows(raw_path, grid_ms=grid_ms, depth=L20_DEPTH)
    if not rows:
        LOG.warning("no rows after reconstruction for %s %s", symbol, day)
        return None
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, compression="snappy")
    elapsed = time.perf_counter() - t0
    LOG.info("%s %s: %d rows (grid=%dms) in %.1fs -> %s",
             symbol, day, len(df), grid_ms, elapsed, out_path.name)
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default="BTCUSDT")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--raw-dir", type=Path, default=Path("data/bybit_raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/bybit_l20"))
    p.add_argument("--grid-ms", type=int, default=1000,
                   help="snapshot grid in milliseconds (default 1000 = 1Hz)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    ok = skipped = fail = 0
    for symbol in symbols:
        for d in daterange(start, end):
            result = process_one(symbol, d, args.raw_dir, args.out_dir,
                                  args.grid_ms, overwrite=args.overwrite)
            if result is None:
                fail += 1
            else:
                ok += 1
    LOG.info("done: ok=%d fail=%d", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
