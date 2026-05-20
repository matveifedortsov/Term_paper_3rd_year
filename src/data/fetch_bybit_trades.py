"""Download Bybit USDT-perp daily aggregated trade archives.

Bybit publishes per-day perpetual-futures trade ticks at:
    https://public.bybit.com/trading/<SYMBOL>/<SYMBOL><YYYY-MM-DD>.csv.gz

Each row is one trade tick. Native columns:
    timestamp,symbol,side,size,price,tickDirection,trdMatchID,
    grossValue,homeNotional,foreignNotional

We convert to the parquet schema the rest of the project uses
(matches the Binance Vision aggTrades convention):
    transact_time   int64 ms epoch
    quantity        float64
    is_buyer_maker  bool  (True when the taker SOLD into the bid)

Bybit's `side` records the AGGRESSOR side:
    side="Buy"  -> taker bought (lifted ask)  -> is_buyer_maker=False
    side="Sell" -> taker sold (hit bid)       -> is_buyer_maker=True

Note: this is the FUTURES trade tape paired with our SPOT orderbook
(no spot daily archive exists yet for the May-2026 window). Spot and
perp prices move in lockstep within seconds via arbitrage, so the
5-second trade-flow features used by features.py are well-served by
either tape. Disclosed in the paper as a market-type detail.

Usage:
    python -m src.data.fetch_bybit_trades \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
        --start 2026-05-01 --end 2026-05-14 \\
        --out data/historical
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import io
import logging
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("fetch-bybit-trades")

BASE_URL = "https://public.bybit.com/trading"
USER_AGENT = "term-paper-fetcher/1.0 (+research)"
MAX_RETRIES = 40
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 15.0
CHUNK_SIZE = 64 * 1024


def url_for(symbol: str, day: date) -> str:
    return f"{BASE_URL}/{symbol.upper()}/{symbol.upper()}{day.isoformat()}.csv.gz"


def daterange(start: date, end: date):
    n = (end - start).days + 1
    for i in range(n):
        yield start + timedelta(days=i)


def _download_with_retry(url: str, tmp_path: Path) -> bool:
    """Resumable, retrying chunked download. Writes to tmp_path."""
    backoff = INITIAL_BACKOFF
    expected_total: int | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        start_byte = tmp_path.stat().st_size if tmp_path.exists() else 0
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "identity",
            })
            if start_byte > 0:
                req.add_header("Range", f"bytes={start_byte}-")
            resp = urllib.request.urlopen(req, timeout=180)
        except urllib.error.HTTPError as e:
            if e.code == 416 and expected_total is not None and start_byte >= expected_total:
                break
            LOG.warning("HTTP %s for %s (attempt %d/%d)", e.code, url, attempt, MAX_RETRIES)
            if e.code == 404:
                return False
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 1.5, MAX_BACKOFF)
            continue
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            LOG.warning("connection error %r (attempt %d/%d)", e, attempt, MAX_RETRIES)
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 1.5, MAX_BACKOFF)
            continue

        # Update expected size
        cr = resp.headers.get("Content-Range")
        if cr and "/" in cr:
            try:
                expected_total = int(cr.rsplit("/", 1)[1])
            except ValueError:
                pass
        if expected_total is None:
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                expected_total = int(cl) + start_byte

        try:
            mode = "ab" if start_byte > 0 else "wb"
            with resp, open(tmp_path, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
        except (http.client.IncompleteRead, socket.timeout, ConnectionError) as e:
            got = tmp_path.stat().st_size if tmp_path.exists() else 0
            LOG.warning("interrupted at %d bytes (%r); retry %d/%d", got, e, attempt + 1, MAX_RETRIES)
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 1.5, MAX_BACKOFF)
            continue

        got = tmp_path.stat().st_size
        if expected_total is None or got == expected_total:
            return True
        LOG.warning("short: %d/%d; resuming", got, expected_total)
        time.sleep(backoff + random.uniform(0, 0.5))
        backoff = min(backoff * 1.5, MAX_BACKOFF)
    LOG.error("giving up after %d retries: %s", MAX_RETRIES, url)
    return False


def convert_csv_to_parquet(csv_gz_path: Path, parquet_path: Path) -> int:
    """Read Bybit CSV.gz trades, rewrite as parquet in our schema."""
    with gzip.open(csv_gz_path, "rb") as f:
        df = pd.read_csv(f, usecols=["timestamp", "side", "size"])
    # Bybit "timestamp" is a float of seconds since epoch
    transact_time = (df["timestamp"].astype(float) * 1000.0).round().astype("int64")
    is_buyer_maker = (df["side"].astype(str).str.lower() == "sell")
    quantity = df["size"].astype(float)
    out = pd.DataFrame({
        "transact_time": transact_time,
        "quantity": quantity,
        "is_buyer_maker": is_buyer_maker,
    })
    out.to_parquet(parquet_path, compression="snappy")
    return len(out)


def fetch_and_convert(symbol: str, day: date, out_dir: Path) -> Path | None:
    day_str = day.isoformat()
    parquet_name = f"bybit_{symbol.lower()}_trades_{day_str}.parquet"
    parquet_path = out_dir / parquet_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists() and parquet_path.stat().st_size > 0:
        LOG.info("skip (exists): %s", parquet_name)
        return parquet_path

    csv_name = f"{symbol.upper()}{day_str}.csv.gz"
    csv_path = out_dir / csv_name
    tmp_path = csv_path.with_suffix(".csv.gz.tmp")

    if not csv_path.exists():
        url = url_for(symbol, day)
        if not _download_with_retry(url, tmp_path):
            return None
        tmp_path.replace(csv_path)
    LOG.info("converting %s -> parquet", csv_name)
    try:
        n = convert_csv_to_parquet(csv_path, parquet_path)
    except Exception:
        LOG.exception("conversion failed for %s", csv_name)
        return None
    LOG.info("%d trades -> %s", n, parquet_name)
    return parquet_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=Path, default=Path("data/historical"))
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    ok = fail = 0
    for sym in symbols:
        LOG.info("=== %s %s..%s ===", sym, start, end)
        for d in daterange(start, end):
            res = fetch_and_convert(sym, d, args.out)
            if res is None:
                fail += 1
            else:
                ok += 1
    LOG.info("done: ok=%d fail=%d", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
