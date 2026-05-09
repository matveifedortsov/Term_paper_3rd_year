"""Capture Binance spot L2 order book to hourly Parquet files.

Stream: <symbol>@depth20@100ms (top 20 bid/ask levels every 100 ms).
No API key required; the public stream is unauthenticated.

Robust to disconnects: reconnects with exponential backoff. Flushes
to disk every ROTATE_SECONDS so a crash loses at most one rotation.

Example:
    python -m src.data.capture_binance_depth --symbol BTCUSDT --out data/raw
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


@dataclass
class CaptureConfig:
    symbol: str = "BTCUSDT"
    depth_levels: int = 20
    update_speed_ms: int = 100
    out_dir: Path = Path("data/raw")
    rotate_seconds: int = 3600
    ping_interval: int = 20
    ping_timeout: int = 10
    backoff_initial: float = 1.0
    backoff_max: float = 60.0


class DepthCapture:
    def __init__(self, cfg: CaptureConfig) -> None:
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
        return (
            f"wss://stream.binance.com:9443/ws/"
            f"{sym}@depth{self.cfg.depth_levels}@{self.cfg.update_speed_ms}ms"
        )

    def _on_message(self, _ws, message: str) -> None:
        try:
            msg = json.loads(message)
            ts_ns = time.time_ns()
            rec: dict = {
                "ts_local_ns": ts_ns,
                "last_update_id": msg.get("lastUpdateId"),
            }
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])
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
                    "alive: total=%d  last_60s=%d  buf=%d",
                    self._n_msgs_total,
                    delta,
                    len(self._buffer),
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
        fname = self.cfg.out_dir / (
            f"{self.cfg.symbol.lower()}_depth{self.cfg.depth_levels}_"
            f"{ts:%Y%m%dT%H%M%S}Z.parquet"
        )
        df = pd.DataFrame(buf)
        df.to_parquet(fname, compression="snappy")
        LOG.info("wrote %d rows -> %s", len(df), fname.name)
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
        LOG.info("stopped (total=%d msgs)", self._n_msgs_total)

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass


def parse_args(argv: list[str]) -> CaptureConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--levels", type=int, default=20, choices=[5, 10, 20])
    p.add_argument("--speed-ms", type=int, default=100, choices=[100, 1000])
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--rotate-seconds", type=int, default=3600)
    args = p.parse_args(argv)
    return CaptureConfig(
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
    capture = DepthCapture(cfg)

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
