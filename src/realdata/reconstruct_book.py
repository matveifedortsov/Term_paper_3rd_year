"""Full-depth order-book reconstruction from incremental L2 streams.

ACTIVATED for Bybit historical archives. See
`src.data.fetch_bybit_orderbook` for the downloader.

Each Bybit per-day .data file is a sequence of JSON messages:
    line 1   : type="snapshot", full L200 book
    line 2+  : type="delta",    changes since previous message

A delta record contains only changed levels. qty="0" means "cancel
that price level". We replay every message into an OrderBook, then
emit snapshots at a requested time grid.

Public API:
    OrderBook                                      -- in-memory L2 book
    iterate_messages(path: Path)                   -- yields parsed dicts
    reconstruct_day(path: Path, grid_ms: int=1000) -- yields (ts_ms, OrderBook)
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator


# ----------------------------------------------------------------------
# In-memory book
# ----------------------------------------------------------------------

@dataclass
class OrderBook:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int = -1

    def apply_diff(self, side: str, price: float, qty: float) -> None:
        book = self.bids if side == "bid" else self.asks
        if qty == 0.0:
            book.pop(price, None)
        else:
            book[price] = qty

    def reset_from_snapshot(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        update_id: int,
    ) -> None:
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_update_id = int(update_id)

    def snapshot_top(self, depth: int = 20) -> tuple[list, list]:
        """Return top-`depth` bids (desc) and asks (asc) as list-of-pairs."""
        bids = sorted(self.bids.items(), key=lambda x: -x[0])[:depth]
        asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]
        return bids, asks

    def mid(self) -> float:
        if not self.bids or not self.asks:
            return float("nan")
        return 0.5 * (max(self.bids) + min(self.asks))

    def best_bid(self) -> float:
        return max(self.bids) if self.bids else float("nan")

    def best_ask(self) -> float:
        return min(self.asks) if self.asks else float("nan")


# ----------------------------------------------------------------------
# Bybit-specific parser
# ----------------------------------------------------------------------

def _open_archive(path: Path) -> io.TextIOBase:
    """Open the inner .data file from a Bybit zip; supports raw .data too."""
    p = Path(path)
    if p.suffix == ".zip":
        zf = zipfile.ZipFile(p, "r")
        names = zf.namelist()
        if not names:
            raise ValueError(f"empty zip: {p}")
        # Pick the .data member (there is usually only one)
        member = next((n for n in names if n.endswith(".data")), names[0])
        return io.TextIOWrapper(zf.open(member, "r"), encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def iterate_messages(path: Path) -> Iterator[dict]:
    """Yield parsed JSON messages from a Bybit orderbook archive.

    Each message has keys: ts (int, epoch ms), type ('snapshot'|'delta'),
    data.s (symbol), data.b (bid levels), data.a (ask levels),
    data.u (update id), data.seq (sequence number).
    """
    with _open_archive(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _apply_message(book: OrderBook, msg: dict) -> None:
    """Mutate `book` according to one Bybit message."""
    data = msg.get("data") or {}
    mtype = msg.get("type")
    if mtype == "snapshot":
        bids = data.get("b") or []
        asks = data.get("a") or []
        book.reset_from_snapshot(bids, asks, update_id=data.get("u", -1))
        return
    # delta: each [price, qty] string-pair adjusts the running book
    for price_s, qty_s in data.get("b", []):
        book.apply_diff("bid", float(price_s), float(qty_s))
    for price_s, qty_s in data.get("a", []):
        book.apply_diff("ask", float(price_s), float(qty_s))
    book.last_update_id = data.get("u", book.last_update_id)


def reconstruct_day(
    path: Path,
    grid_ms: int = 1000,
) -> Generator[tuple[int, OrderBook], None, None]:
    """Replay one Bybit day and emit (ts_ms, OrderBook) at the requested grid.

    The emitted ``ts_ms`` is the snapshot time of the most recent
    message ≤ a grid tick. Ticks where no message has yet arrived are
    skipped (no early-day padding).

    Note: yields the SAME OrderBook instance each tick; the caller
    should copy / extract values immediately (snapshot_top, mid, etc.).
    """
    book = OrderBook()
    last_emitted = -1
    for msg in iterate_messages(path):
        ts = int(msg.get("ts", 0))
        _apply_message(book, msg)
        # Emit on every grid tick that's been crossed since last emit
        if last_emitted < 0:
            # initialize the grid floor at the first message
            last_emitted = (ts // grid_ms) * grid_ms - grid_ms
        while last_emitted + grid_ms <= ts:
            last_emitted += grid_ms
            yield last_emitted, book


# ----------------------------------------------------------------------
# Convenience: per-day reconstruction into list of (price, qty) pairs
# ----------------------------------------------------------------------

def replay_to_l20_rows(path: Path, grid_ms: int = 1000, depth: int = 20) -> list[dict]:
    """Eager helper: reconstruct one day, return list of dict rows with
    bid_p1..N/bid_q1..N/ask_p1..N/ask_q1..N + ts.

    Each row matches the schema of `capture_binance_stream.py`'s depth
    output, so downstream feature engineering works unchanged.
    """
    rows: list[dict] = []
    for ts_ms, book in reconstruct_day(path, grid_ms=grid_ms):
        bids, asks = book.snapshot_top(depth=depth)
        if not bids or not asks:
            continue
        row = {"ts_local_ns": ts_ms * 1_000_000, "ts_ms": ts_ms}
        for i in range(depth):
            if i < len(bids):
                row[f"bid_p{i + 1}"] = bids[i][0]
                row[f"bid_q{i + 1}"] = bids[i][1]
            else:
                row[f"bid_p{i + 1}"] = float("nan")
                row[f"bid_q{i + 1}"] = float("nan")
            if i < len(asks):
                row[f"ask_p{i + 1}"] = asks[i][0]
                row[f"ask_q{i + 1}"] = asks[i][1]
            else:
                row[f"ask_p{i + 1}"] = float("nan")
                row[f"ask_q{i + 1}"] = float("nan")
        rows.append(row)
    return rows
