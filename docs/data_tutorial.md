# Free BTC-USDT data — tutorial

Phase 2 of the term paper needs high-frequency BTC-USDT order book data. This document walks through two free options and tells you exactly what commands to run.

## TL;DR — pick one

| | Path A: live websocket capture | Path B: historical futures L1 |
|---|---|---|
| Cost | $0 | $0 |
| Time to start having data | None — grows as you wait | A few hours of downloads |
| Wall-clock to 14 days of data | 14 calendar days | 14 days of human time → ~1 hour download |
| LOB depth | **L20** (top 20 levels) | **L1** (best bid/ask only) |
| Update frequency | Every 100 ms | Every order-book update (often <10 ms) |
| Includes shock events? | Whatever happens during capture | Pick from 2023-05-16 to 2024-03-30 |
| Storage | ~2 GB compressed for 14 days | ~6 GB compressed for 14 days |

**Run both.** Start Path A *today* so a 14-day L20 dataset is ready in two weeks. While that runs, prototype the Phase 3 ML pipeline on Path B's L1 data.

---

## Path A: live L20 websocket capture

Connects to Binance's free public stream `btcusdt@depth20@100ms`. No API key, no rate limits to worry about, just a single websocket. The capture script auto-reconnects with exponential backoff and rotates to a new Parquet file every hour, so a crash loses at most 1 hour.

### A.1 Install dependencies

```powershell
python -m pip install websocket-client pyarrow pandas
```

### A.2 Start the capture

```powershell
cd "C:\Users\adm\Desktop\Term Paper"
python -m src.data.capture_binance_depth --symbol BTCUSDT --levels 20 --speed-ms 100 --out data/raw --rotate-seconds 3600
```

You should see:
```
2026-05-08 ... INFO connected: wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms
2026-05-08 ... INFO alive: total=600  last_60s=600  buf=600
2026-05-08 ... INFO wrote 36000 rows -> btcusdt_depth20_20260508T160000Z.parquet
```

Each hourly file holds ~36,000 rows (10 Hz × 3600 s). Filename format:

```
data/raw/btcusdt_depth20_<YYYYMMDDTHHMMSS>Z.parquet
```

### A.3 Run it for 14 days unattended

The hard part isn't the capture — it's keeping your laptop alive for 14 days. Three options:

**(i) Run on your desktop with sleep disabled.** From an admin PowerShell:
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```
Then start the capture in a regular terminal. If you reboot, just relaunch the same command — files are timestamped so nothing collides.

**(ii) Run as a Windows Scheduled Task** so it auto-restarts after reboots:
- Task Scheduler → Create Task
- Trigger: At log on (or At startup if running as SYSTEM)
- Action: `python.exe` with arguments `-m src.data.capture_binance_depth --out C:\Users\adm\Desktop\Term Paper\data\raw`
- Settings → "If the task fails, restart every: 1 minute"

**(iii) Free always-on cloud VM.** Oracle Cloud's "Always Free" tier gives you a 1-vCPU 1-GB ARM VM forever. Plenty for this. Setup:
1. Create account at cloud.oracle.com (no credit card required for Always Free tier)
2. Spin up an Ampere A1 VM (Ubuntu 22.04 ARM)
3. SSH in, `git clone` your repo, `pip install -r requirements.txt`
4. Run capture under `tmux` or as a `systemd` service

I'd start with (i) and only move to (iii) if you find your laptop is too unreliable.

### A.4 Verify the capture is healthy

After ~1 hour:
```powershell
python scripts/verify_capture.py --dir data/raw --last 5
```

What "healthy" looks like:
- `rate_hz` ≈ 10 (matches the 100 ms cadence)
- `median_gap_ms` ≈ 100
- `p99_gap_ms` < 500
- `spread_violations` = 0 (bid never above ask)

If `rate_hz` is consistently below 6, your network is dropping messages — investigate before letting it run for two weeks.

### A.5 Estimated outputs

| After | Files | Rows | Disk |
|---|---|---|---|
| 1 hour | 1 | ~36,000 | ~5 MB |
| 1 day | 24 | ~864,000 | ~120 MB |
| 14 days | 336 | ~12.1M | ~1.7 GB |

---

## Path B: historical futures L1 (immediate)

Binance Vision publishes daily zipped CSVs for **USD-M perpetual futures BTCUSDT**. The `bookTicker` files give every best-bid/ask change with millisecond timestamps. Spot bookTicker is not published, so we use futures, which is also more liquid (~$30B/day vs ~$2B/day for spot).

### B.1 Coverage and recommended window

- **bookTicker availability:** 2023-05-16 to 2024-03-30 (publication discontinued after that)
- **aggTrades:** continuously published, both spot and futures

**Recommended window: 2024-03-15 to 2024-03-29.** Rationale:
- Covers BTC's then-all-time-high above $73,800 on 2024-03-14
- Followed by a ~15% drawdown and high-vol consolidation — many real jumps to detect
- Final window of bookTicker availability before discontinuation

### B.2 Download

```powershell
cd "C:\Users\adm\Desktop\Term Paper"

# L1 book updates (~25M rows/day, ~400 MB Parquet/day)
python -m src.data.fetch_binance_vision --market futures --symbol BTCUSDT --kind bookTicker --start 2024-03-15 --end 2024-03-29 --out data/historical

# Trade flow (~2M rows/day, ~50 MB Parquet/day)
python -m src.data.fetch_binance_vision --market futures --symbol BTCUSDT --kind aggTrades --start 2024-03-15 --end 2024-03-29 --out data/historical
```

The script auto-skips files already on disk, so you can rerun safely if it disconnects partway through.

### B.3 What you get

```
data/historical/
  futures_btcusdt_bookTicker_2024-03-15.parquet   (26.7M rows, 380 MB)
  futures_btcusdt_bookTicker_2024-03-16.parquet   (...)
  ...
  futures_btcusdt_aggTrades_2024-03-15.parquet    (2.0M rows, 60 MB)
  ...
```

Schemas:

**bookTicker** (one row per L1 update):
| col | meaning |
|---|---|
| `update_id` | monotone Binance sequence id |
| `best_bid_price`, `best_bid_qty` | top of book bid |
| `best_ask_price`, `best_ask_qty` | top of book ask |
| `transaction_time` | ms since epoch — when the order matched |
| `event_time` | ms since epoch — when the message left Binance |

**aggTrades** (one row per aggregated trade):
| col | meaning |
|---|---|
| `agg_trade_id` | aggregated id |
| `price`, `quantity` | trade price and size |
| `first_trade_id`, `last_trade_id` | range of underlying trades |
| `transact_time` | ms since epoch |
| `is_buyer_maker` | True if taker was the seller (sells aggressing into bid) |
| `best_match` | True if the trade was a best match |

### B.4 Sanity check

```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/historical/futures_btcusdt_bookTicker_2024-03-15.parquet'); print(df.shape); print(df.head(2))"
```

Expect 25–28 million rows for a normal day, more for high-volatility days.

---

## Which features will Phase 3 use, by data path?

The paper specifies 15 LOB features. Here's what's computable from each data source:

| # | Feature | Path A (L20) | Path B (L1 + trades) |
|---|---|---|---|
| 1 | Bid-ask spread | ✓ | ✓ |
| 2 | Δ spread (pre vs at τ) | ✓ | ✓ |
| 3 | Order book imbalance (L1) | ✓ | ✓ |
| 4 | Order book imbalance (L1–L5) | ✓ | ✗ |
| 5–9 | Depth at levels 1–5 | ✓ | partial (L1 only) |
| 10 | Realized variance | ✓ | ✓ |
| 11 | Bipower variation | ✓ | ✓ |
| 12 | Realized kurtosis | ✓ | ✓ |
| 13 | Trade volume | (need separate trade stream) | ✓ |
| 14 | Signed order flow | (need separate trade stream) | ✓ |
| 15 | Time since last LOMN candidate | ✓ | ✓ |

Path A gives the deep-book features but doesn't include trade flow unless you also subscribe to the trade stream. Path B has trade flow but only L1 book.

**Best of both worlds**: capture both `btcusdt@depth20@100ms` AND `btcusdt@aggTrade` simultaneously. The capture script can be extended easily; ask when you're ready.

---

## Storage budget

If you run both paths for the suggested durations, plan for:

| Source | Rough disk |
|---|---|
| Path A live capture, 14 days | ~1.7 GB |
| Path B futures bookTicker, 15 days | ~5.8 GB |
| Path B futures aggTrades, 15 days | ~0.9 GB |
| **Total** | **~8.4 GB** |

Default location is `C:\Users\adm\Desktop\Term Paper\data\`. If your desktop is space-constrained, move the `data/` directory elsewhere and pass `--out` accordingly.

---

## What to do once data is flowing

1. **Confirm Path A is healthy** after 1 hour: `python scripts/verify_capture.py --dir data/raw`
2. **Download Path B** (~1 hour wall-clock for 15 days)
3. Tell me data is ready and I'll start Phase 3 (feature engineering + XGBoost)
