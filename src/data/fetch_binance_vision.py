"""Download Binance Vision daily data.

Free public dataset at https://data.binance.vision/. Two markets:
    - spot     : BTCUSDT (and others)
    - futures  : BTCUSDT perpetual (USD-M)

Daily files include aggTrades (all markets) and bookTicker (FUTURES
ONLY — spot bookTicker is not published historically).

For Phase 2 fallback the recommended combination is:
    market=futures, kind=bookTicker  -> L1 LOB history
    market=futures, kind=aggTrades   -> trade flow

No API key required; just HTTPS GETs to a CDN. Files arrive as zipped
CSVs; we decompress and write Parquet.

Example:
    python -m src.data.fetch_binance_vision --market futures \\
        --symbol BTCUSDT --kind bookTicker \\
        --start 2024-08-01 --end 2024-08-14 --out data/historical
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

LOG = logging.getLogger("fetch")

BASES = {
    "spot": "https://data.binance.vision/data/spot/daily",
    "futures": "https://data.binance.vision/data/futures/um/daily",
}

KIND_COLUMNS = {
    "bookTicker": [
        "update_id", "best_bid_price", "best_bid_qty",
        "best_ask_price", "best_ask_qty", "transaction_time", "event_time",
    ],
    "aggTrades": [
        "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
        "transact_time", "is_buyer_maker", "best_match",
    ],
    "trades": [
        "trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "best_match",
    ],
}


def daterange(start: date, end: date):
    n_days = (end - start).days + 1
    for i in range(n_days):
        yield start + timedelta(days=i)


def url_for(market: str, symbol: str, kind: str, day: date) -> str:
    base = BASES[market]
    return (
        f"{base}/{kind}/{symbol.upper()}/"
        f"{symbol.upper()}-{kind}-{day.isoformat()}.zip"
    )


def fetch_one(market: str, symbol: str, kind: str, day: date, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{market}_{symbol.lower()}_{kind}_{day.isoformat()}.parquet"
    if out_path.exists():
        LOG.info("skip (exists): %s", out_path.name)
        return out_path

    url = url_for(market, symbol, kind, day)
    try:
        with urlopen(url, timeout=60) as resp:
            blob = resp.read()
    except HTTPError as e:
        LOG.warning("HTTP %s for %s", e.code, url)
        return None

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        if not names:
            LOG.warning("empty zip for %s", url)
            return None
        with zf.open(names[0]) as fh:
            sniff = fh.read(256).decode("utf-8", errors="ignore")
        with zf.open(names[0]) as fh:
            first_token = sniff.split(",", 1)[0].strip()
            has_header = not first_token.lstrip("-").isdigit()
            if has_header:
                df = pd.read_csv(fh)
            else:
                df = pd.read_csv(fh, header=None, names=KIND_COLUMNS[kind])

    df.to_parquet(out_path, compression="snappy")
    LOG.info("%s rows -> %s", f"{len(df):>10,d}", out_path.name)
    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--market", choices=list(BASES), default="futures")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--kind", choices=list(KIND_COLUMNS), required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=Path, default=Path("data/historical"))
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    LOG.info("fetching %s %s %s [%s .. %s]",
             args.market, args.symbol, args.kind, start, end)

    ok = 0
    fail = 0
    for d in daterange(start, end):
        result = fetch_one(args.market, args.symbol, args.kind, d, args.out)
        if result is None:
            fail += 1
        else:
            ok += 1
    LOG.info("done: ok=%d fail=%d", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
