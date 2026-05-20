"""Download Bybit historical spot orderbook archives.

Bybit publishes per-day L200 incremental orderbook archives at:
    https://quote-saver.bycsi.com/orderbook/spot/<SYMBOL>/<DATE>_<SYMBOL>_ob200.data.zip

Each .zip contains a single .data file: one JSON message per line, where
the first message is a full snapshot (200 bids + 200 asks) and every
subsequent message is a delta update. Cadence ~100-200 ms.

Per-day size: ~50-90 MB compressed, ~300-500 MB unzipped.
Levels: 200 per side (well beyond the 20 the paper specs).
No API key, no rate limits we hit at single-worker throughput.

Robust: retries with exponential backoff on IncompleteRead /
connection drops. Chunked download via Range when available.

Usage:
    python -m src.data.fetch_bybit_orderbook \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
        --start 2025-05-01 --end 2025-05-05 \\
        --out data/bybit_raw
"""

from __future__ import annotations

import argparse
import http.client
import logging
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

LOG = logging.getLogger("fetch-bybit")

BASE_URL = "https://quote-saver.bycsi.com/orderbook/spot"
DEPTH = 200  # Bybit publishes L200 only
USER_AGENT = "term-paper-fetcher/1.0 (+research)"

MAX_RETRIES = 40           # Bybit drops mid-stream often; many short retries are fine
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 15.0         # cap so we don't wait too long between attempts
CHUNK_SIZE = 64 * 1024     # 64 KB


def url_for(symbol: str, day: date) -> str:
    return f"{BASE_URL}/{symbol.upper()}/{day.isoformat()}_{symbol.upper()}_ob{DEPTH}.data.zip"


def daterange(start: date, end: date):
    n = (end - start).days + 1
    for i in range(n):
        yield start + timedelta(days=i)


def _open_with_range(url: str, start_byte: int, timeout: float) -> tuple[urllib.request.OpenerDirector, int]:
    """Open the URL with an optional Range header. Returns (response, total_size)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "identity",
    })
    if start_byte > 0:
        req.add_header("Range", f"bytes={start_byte}-")
    resp = urllib.request.urlopen(req, timeout=timeout)
    # Determine total size from Content-Length / Content-Range
    total = None
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        try:
            total = int(cr.rsplit("/", 1)[1])
        except ValueError:
            total = None
    if total is None:
        cl = resp.headers.get("Content-Length")
        if cl is not None:
            total = int(cl) + start_byte
    return resp, total


def _download_with_retry(url: str, out_path: Path) -> bool:
    """Resumable, retrying chunked download. Writes to .tmp then renames."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    backoff = INITIAL_BACKOFF
    expected_total: int | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        start_byte = tmp.stat().st_size if tmp.exists() else 0
        try:
            resp, total = _open_with_range(url, start_byte, timeout=180)
        except urllib.error.HTTPError as e:
            if e.code == 416 and expected_total is not None and start_byte >= expected_total:
                # We already have everything; treat as success
                break
            LOG.warning("HTTP %s for %s (attempt %d/%d)",
                        e.code, url, attempt, MAX_RETRIES)
            if e.code == 404:
                return False
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 1.5, MAX_BACKOFF)
            continue
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            LOG.warning("connection error %r on %s (attempt %d/%d)",
                        e, url, attempt, MAX_RETRIES)
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 1.5, MAX_BACKOFF)
            continue

        expected_total = total
        try:
            mode = "ab" if start_byte > 0 else "wb"
            with resp, open(tmp, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
        except (http.client.IncompleteRead, socket.timeout, ConnectionError) as e:
            got = tmp.stat().st_size if tmp.exists() else 0
            LOG.warning("transfer interrupted at %d bytes (%r). Retrying attempt %d/%d.",
                        got, e, attempt + 1, MAX_RETRIES)
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 1.5, MAX_BACKOFF)
            continue

        # Success check: file size matches expectation
        got = tmp.stat().st_size
        if expected_total is None or got == expected_total:
            tmp.replace(out_path)
            LOG.info("%6.1f MB -> %s (attempts=%d)",
                     got / 1e6, out_path.name, attempt)
            return True
        # Short file — retry to resume
        LOG.warning("short file %d/%d for %s; resuming", got, expected_total, url)
        time.sleep(backoff + random.uniform(0, 0.5)); backoff *= 2

    LOG.error("giving up after %d retries: %s", MAX_RETRIES, url)
    # Leave .tmp in place; next run will resume from it
    return False


def fetch_one(symbol: str, day: date, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{day.isoformat()}_{symbol.upper()}_ob{DEPTH}.data.zip"
    out_path = out_dir / fname
    if out_path.exists() and out_path.stat().st_size > 0:
        LOG.info("skip (exists): %s", fname)
        return out_path

    url = url_for(symbol, day)
    ok = _download_with_retry(url, out_path)
    return out_path if ok else None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default="BTCUSDT",
                   help="comma-separated; e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=Path, default=Path("data/bybit_raw"))
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    ok = fail = 0
    for symbol in symbols:
        symbol_dir = args.out / symbol.lower()
        LOG.info("=== %s   %s .. %s ===", symbol, start, end)
        for d in daterange(start, end):
            result = fetch_one(symbol, d, symbol_dir)
            if result is None:
                fail += 1
            else:
                ok += 1
    LOG.info("done: ok=%d fail=%d", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
