"""Generalized Binance public-stream capture.

Supports four (market, stream) combinations needed for the term paper's
L20 multi-asset extension:

    market   : spot | futures        # different websocket host
    stream   : depth | trades        # depth = depth20@<speed>ms
                                     # trades = aggTrade

For depth streams: emits one row per snapshot with bid/ask price+qty
for L1..L20 plus a local-clock timestamp.

For trade streams: emits one row per aggregated trade with price,
quantity, taker side (is_buyer_maker), exchange event-time, and
local-clock timestamp.

No API key required; all streams are public.

Example:
    python -m src.data.capture_binance_stream --market futures \\
        --symbol BTCUSDT --stream depth --levels 20 --speed-ms 100 \\
        --out data/raw_l20/futures_btcusdt_depth20

    python -m src.data.capture_binance_stream --market futures \\
        --symbol ETHUSDT --stream trades --out data/raw_l20/futures_ethusdt_trades
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websocket

LOG = logging.getLogger("capture")

HOSTS = {
    "spot":    "wss://stream.binance.com:9443/ws",
    "futures": "wss://fstream.binance.com/ws",
}


@dataclass
class CaptureConfig:
    market: str = "futures"           # spot | futures
    stream: str = "depth"             # depth | trades
    symbol: str = "BTCUSDT"
    depth_levels: int = 20            # 5 | 10 | 20  (depth stream only)
    update_speed_ms: int = 100        # 100 | 1000   (depth stream only)
    out_dir: Path = Path("data/raw")
    rotate_seconds: int = 3600
    ping_interval: int = 20
    ping_timeout: int = 10
    backoff_initial: float = 1.0
    backoff_max: float = 60.0


class StreamCapture:
    def __init__(self, cfg: CaptureConfig) -> None:
        if cfg.market not in HOSTS:
            raise ValueError(f"market must be one of {list(HOSTS)}")
        if cfg.stream not in {"depth", "trades"}:
            raise ValueError("stream must be 'depth' or 'trades'")
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._buffer_lock = threading.Lock()
        self._last_rotation = time.time()
        self._stop = threading.Event()
        self._backoff = cfg.backoff_initial
        self._n_msgs_total = 0
        self._n_msgs_last_log = 0
        self._last_log = time.time()
        self._ws: websocket.WebSocketApp | None = None

    @property
    def stream_url(self) -> str:
        sym = self.cfg.symbol.lower()
        host = HOSTS[self.cfg.market]
        if self.cfg.stream == "depth":
            return (f"{host}/{sym}@depth{self.cfg.depth_levels}"
                    f"@{self.cfg.update_speed_ms}ms")
        else:
            return f"{host}/{sym}@aggTrade"

    @property
    def file_prefix(self) -> str:
        if self.cfg.stream == "depth":
            return (f"{self.cfg.market}_{self.cfg.symbol.lower()}"
                    f"_depth{self.cfg.depth_levels}")
        else:
            return f"{self.cfg.market}_{self.cfg.symbol.lower()}_aggTrade"

    # ---------- handlers ----------

    def _record_depth(self, msg: dict, ts_ns: int) -> dict:
        rec: dict = {
            "ts_local_ns": ts_ns,
            "last_update_id": msg.get("lastUpdateId") or msg.get("u"),
        }
        bids = msg.get("bids") or msg.get("b") or []
        asks = msg.get("asks") or msg.get("a") or []
        for i in range(self.cfg.depth_levels):
            if i < len(bids):
                rec[f"bid_p{i+1}"] = float(bids[i][0])
                rec[f"bid_q{i+1}"] = float(bids[i][1])
            else:
                rec[f"bid_p{i+1}"] = float("nan")
                rec[f"bid_q{i+1}"] = float("nan")
            if i < len(asks):
                rec[f"ask_p{i+1}"] = float(asks[i][0])
                rec[f"ask_q{i+1}"] = float(asks[i][1])
            else:
                rec[f"ask_p{i+1}"] = float("nan")
                rec[f"ask_q{i+1}"] = float("nan")
        return rec

    def _record_trade(self, msg: dict, ts_ns: int) -> dict:
        # aggTrade message fields:
        #   e: event type, E: event time, s: symbol, a: agg trade id,
        #   p: price, q: quantity, f: first trade id, l: last trade id,
        #   T: trade time, m: is buyer market maker, M: ignore
        return {
            "ts_local_ns": ts_ns,
            "ts_event_ms": msg.get("E"),
            "ts_trade_ms": msg.get("T"),
            "agg_trade_id": msg.get("a"),
            "price": float(msg["p"]) if "p" in msg else float("nan"),
            "quantity": float(msg["q"]) if "q" in msg else float("nan"),
            "first_trade_id": msg.get("f"),
            "last_trade_id": msg.get("l"),
            "is_buyer_maker": bool(msg.get("m", False)),
        }

    def _on_message(self, _ws, message: str) -> None:
        try:
            msg = json.loads(message)
            ts_ns = time.time_ns()
            if self.cfg.stream == "depth":
                rec = self._record_depth(msg, ts_ns)
            else:
                rec = self._record_trade(msg, ts_ns)

            with self._buffer_lock:
                self._buffer.append(rec)
                self._n_msgs_total += 1

            now = time.time()
            if now - self._last_rotation > self.cfg.rotate_seconds:
                self._flush()
            if now - self._last_log > 60.0:
                with self._buffer_lock:
                    delta = self._n_msgs_total - self._n_msgs_last_log
                    self._n_msgs_last_log = self._n_msgs_total
                LOG.info(
                    "[%s] alive: total=%d  last_60s=%d  buf=%d",
                    self.file_prefix, self._n_msgs_total, delta, len(self._buffer),
                )
                self._last_log = now
        except Exception:
            LOG.exception("on_message failed")

    def _on_open(self, _ws) -> None:
        LOG.info("connected: %s", self.stream_url)
        self._backoff = self.cfg.backoff_initial

    def _on_error(self, _ws, error) -> None:
        LOG.warning("ws error: %s", error)

    def _on_close(self, _ws, code, msg) -> None:
        LOG.warning("ws closed: code=%s msg=%s", code, msg)

    def _flush(self) -> None:
        with self._buffer_lock:
            if not self._buffer:
                self._last_rotation = time.time()
                return
            buf = self._buffer
            self._buffer = []
        ts = datetime.now(timezone.utc)
        fname = self.cfg.out_dir / f"{self.file_prefix}_{ts:%Y%m%dT%H%M%S}Z.parquet"
        df = pd.DataFrame(buf)
        df.to_parquet(fname, compression="snappy")
        LOG.info("[%s] wrote %d rows -> %s", self.file_prefix, len(df), fname.name)
        self._last_rotation = time.time()

    def run(self) -> None:
        websocket.enableTrace(False)
        while not self._stop.is_set():
            self._ws = websocket.WebSocketApp(
                self.stream_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            try:
                self._ws.run_forever(
                    ping_interval=self.cfg.ping_interval,
                    ping_timeout=self.cfg.ping_timeout,
                )
            except KeyboardInterrupt:
                self._stop.set()
                break
            except Exception:
                LOG.exception("run_forever crashed")
            if self._stop.is_set():
                break
            sleep_s = self._backoff
            LOG.info("reconnecting in %.1fs", sleep_s)
            time.sleep(sleep_s)
            self._backoff = min(self._backoff * 2.0, self.cfg.backoff_max)
        self._flush()
        LOG.info("[%s] stopped (total=%d msgs)", self.file_prefix, self._n_msgs_total)

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass


def parse_args(argv: list[str]) -> CaptureConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", default="futures", choices=list(HOSTS))
    p.add_argument("--stream", default="depth", choices=["depth", "trades"])
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--levels", type=int, default=20, choices=[5, 10, 20])
    p.add_argument("--speed-ms", type=int, default=100, choices=[100, 1000])
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--rotate-seconds", type=int, default=3600)
    args = p.parse_args(argv)
    return CaptureConfig(
        market=args.market,
        stream=args.stream,
        symbol=args.symbol,
        depth_levels=args.levels,
        update_speed_ms=args.speed_ms,
        out_dir=args.out,
        rotate_seconds=args.rotate_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    capture = StreamCapture(cfg)

    def _handle_sig(signum, _frame):
        LOG.info("signal %s -> shutting down", signum)
        capture.stop()

    signal.signal(signal.SIGINT, _handle_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sig)

    LOG.info("config: %s", cfg)
    capture.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
