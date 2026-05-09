# Task brief: multi-asset L20 + trade-stream capture for the term paper

**Hi! Thank you for helping with this.** Read this whole document end-to-end before starting — it's ~15 minutes and you'll save hours of confusion. The work itself is straightforward: set up a few small Python processes, let them run for 14 days, and hand back the data. No live monitoring needed beyond a daily 60-second check.

---

## 1. Context (why we're doing this)

The term paper is on jump detection in BTC futures order books. Phase 3 of the implementation already trained an XGBoost classifier on **L1 (best bid/ask only) historical data** from Binance Vision. Two follow-up experiments need fresh data:

1. **Upgrade L1 → L20.** Capture the top 20 levels of the order book live. The paper hypothesizes that deeper book features (depth at levels 1–5, multi-level OBI) improve jump-detection F1. Without L20 data we can't test this.

2. **Multi-asset validation.** Repeat for ETH and SOL. The paper currently claims results "for crypto markets" but only has BTC. Adding ETH and SOL turns this into a real generalization claim.

This requires a free public Binance websocket stream. No API key, no money. Just bandwidth and a machine that stays online for ~14 days.

---

## 2. Deliverables

By end of capture (target = ~14 days from start):

| File pattern (in your output directory) | What it is | Approx size for 14 days |
|---|---|---|
| `futures_btcusdt_depth20_*.parquet` | BTC L20 snapshots @ 100 ms | ~1.7 GB |
| `futures_ethusdt_depth20_*.parquet` | ETH L20 snapshots @ 100 ms | ~1.5 GB |
| `futures_solusdt_depth20_*.parquet` | SOL L20 snapshots @ 100 ms | ~1.4 GB |
| `futures_btcusdt_aggTrade_*.parquet`* | BTC aggregated trades | ~0.7 GB |
| `futures_ethusdt_aggTrade_*.parquet`* | ETH aggregated trades | ~0.5 GB |
| `futures_solusdt_aggTrade_*.parquet`* | SOL aggregated trades | ~0.4 GB |
| `capture.log` (one per process) | rotated to disk every hour | small |

*For the trade streams, see §5 — there's a network gotcha that determines whether you capture from `futures` host or fall back to `spot`. The trade-stream files will have a `spot_` prefix instead if you used the fallback.

**Total disk needed: ~10 GB. Have 25 GB free to be safe.**

Hand-back format: a single tar.gz or zip of the output directory. We'll mount it as `data/raw_l20/` in the project.

---

## 3. The two big decisions you need to make first

### Decision 1: where will the captures run?

**Option A — recommended: Oracle Cloud Always Free VM in Tokyo or Singapore.**
Free forever (1 vCPU, 1 GB RAM ARM Ampere), 24/7 uptime, and crucially — Binance futures streams work reliably from those regions. From St. Petersburg / Russia networks, the futures `aggTrade` and `trade` streams sometimes don't push events; we verified this is regional. ~30 min of one-time setup. **Use this if at all possible.**

**Option B — fallback: your laptop or desktop running locally.**
If you can keep a machine on continuously for 14 days, this works. Disable sleep, disable lid-close action, plug into power. You'll need to use Binance **spot** streams (not futures) for trades — see §5.

Pick A if you can spare 30 min on cloud setup. Pick B if you can't.

### Decision 2: 14 days, or shorter?

The paper's analysis budget assumes 14 days. If you can only run for 7 days, that's still useful (we'll get a noisy version of the same results). If you can run for 21+ days, even better — more data = tighter statistics. **Start the capture immediately and keep it running as long as feasible.** Whatever you give us, we'll use.

---

## 4. Setup

### 4a. Get the project code

```bash
# On the machine that will run the captures
git clone <repo-url>     # ask Matvei for the URL or have him zip the project
cd term-paper
python3 -m venv .venv
source .venv/bin/activate                 # Linux/Mac
# OR on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install websocket-client pyarrow      # the capture's two extra deps
```

If you don't have `requirements.txt`, the minimal set is:
```
numpy pandas pyarrow scipy websocket-client
```

### 4b. (Option A only) Spin up Oracle Cloud Always Free VM

1. Sign up at https://signup.oraclecloud.com/. Free tier requires identity verification but **no credit-card hold for the always-free resources**.
2. After signup, in the cloud console: **Compute → Instances → Create Instance**.
3. Configuration:
   - **Name:** `binance-capture`
   - **Image:** Ubuntu 22.04 (the default Canonical image is fine)
   - **Shape:** "Always Free Eligible" → select `VM.Standard.A1.Flex`, 1 OCPU, 1 GB memory (or 2 OCPU, 6 GB if available — capacity varies)
   - **Region:** pick **ap-tokyo-1** or **ap-singapore-1** (these regions have working Binance streams)
   - **VCN:** create a new one with default settings
   - **SSH key:** generate a new key pair, **download both public and private keys** to your laptop, keep them safe
4. Wait ~2 min for provisioning, then SSH in:
   ```bash
   ssh -i path/to/private-key.pem ubuntu@<public-ip>
   ```
5. Install Python and tools:
   ```bash
   sudo apt update && sudo apt install -y python3.11-venv git tmux
   git clone <repo-url> termpaper && cd termpaper
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt websocket-client pyarrow
   ```
6. Verify the streams reach the VM:
   ```bash
   python -c "
   import websocket, threading, time
   got = []
   ws = websocket.WebSocketApp(
       'wss://fstream.binance.com/ws/btcusdt@aggTrade',
       on_message=lambda w, m: got.append(m))
   t = threading.Thread(target=ws.run_forever, daemon=True); t.start()
   time.sleep(8); ws.close()
   print('msgs in 8s:', len(got))
   "
   ```
   You should see 100+ messages. If you see 0, you picked the wrong region or there's a transient issue — try again or move to ap-singapore-1.

### 4c. (Option B only) Local-machine prep

- **Windows:** disable sleep & hibernate from an admin PowerShell:
  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change standby-timeout-dc 0
  powercfg /change hibernate-timeout-ac 0
  powercfg /change hibernate-timeout-dc 0
  ```
  Settings → System → Power → "When I close the lid: Do nothing".
- **macOS:** System Settings → Energy → "Prevent automatic sleeping when display is off" ✓.
- **Linux:** `systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`

---

## 5. The streams

The repo has a generalized capture script: `src/data/capture_binance_stream.py`. Each invocation captures one stream. We need 6 streams running in parallel.

### Quick smoke test (do this first, takes ~1 minute)

```bash
cd termpaper
python -m src.data.capture_binance_stream \
    --market futures --stream depth --symbol BTCUSDT \
    --levels 20 --speed-ms 100 \
    --out data/raw_l20/test_btc_depth \
    --rotate-seconds 60
```

You should see messages like:
```
2026-01-15 ... INFO connected: wss://fstream.binance.com/ws/btcusdt@depth20@100ms
2026-01-15 ... INFO [futures_btcusdt_depth20] alive: total=600 last_60s=600 buf=600
```

After ~70 seconds, hit `Ctrl+C`. Inspect:
```bash
ls data/raw_l20/test_btc_depth/
# expect: futures_btcusdt_depth20_<TIMESTAMP>.parquet

python scripts/verify_capture.py --dir data/raw_l20/test_btc_depth
```
Healthy: `rate_hz` near 10, `median_gap_ms` ≈ 100, zero `spread_violations`.

### The six production captures

These run in parallel. **From Option A (Oracle VM)** use these:

| # | symbol | market | stream | levels | output dir |
|---|---|---|---|---|---|
| 1 | BTCUSDT | futures | depth   | 20 | `data/raw_l20/btc_depth/`  |
| 2 | BTCUSDT | futures | trades  | —  | `data/raw_l20/btc_trades/` |
| 3 | ETHUSDT | futures | depth   | 20 | `data/raw_l20/eth_depth/`  |
| 4 | ETHUSDT | futures | trades  | —  | `data/raw_l20/eth_trades/` |
| 5 | SOLUSDT | futures | depth   | 20 | `data/raw_l20/sol_depth/`  |
| 6 | SOLUSDT | futures | trades  | —  | `data/raw_l20/sol_trades/` |

**From Option B (local machine in Russia/restricted region)**, replace `futures` with `spot` for the **trades** streams only — captures 2, 4, 6. Depth streams 1, 3, 5 stay on `futures`. The reason: futures depth and bookTicker push reliably everywhere, but futures aggTrade pushes inconsistently from some IPs. Spot aggTrade always works.

### Starting all six (Option A, recommended)

In one SSH session, use `tmux` to run all six in detachable windows:

```bash
tmux new-session -d -s capture
for sym in BTCUSDT ETHUSDT SOLUSDT; do
  short=${sym,,}; short=${short%usdt}
  tmux new-window -t capture -n "${short}-depth" \
    "source ~/termpaper/.venv/bin/activate && \
     python -m src.data.capture_binance_stream \
       --market futures --stream depth --symbol $sym --levels 20 \
       --out ~/termpaper/data/raw_l20/${short}_depth/ 2>&1 | tee -a ~/termpaper/${short}_depth.log"
  tmux new-window -t capture -n "${short}-trades" \
    "source ~/termpaper/.venv/bin/activate && \
     python -m src.data.capture_binance_stream \
       --market futures --stream trades --symbol $sym \
       --out ~/termpaper/data/raw_l20/${short}_trades/ 2>&1 | tee -a ~/termpaper/${short}_trades.log"
done
tmux attach -t capture
```

Detach with `Ctrl+B then D`. The processes keep running. Reattach later with `tmux attach -t capture`.

### Starting all six (Option B, local Windows fallback)

Open six PowerShell windows. In each, paste one of the following (after `cd C:\path\to\term-paper`):

```powershell
# Window 1
python -m src.data.capture_binance_stream --market futures --stream depth --symbol BTCUSDT --levels 20 --out data/raw_l20/btc_depth/

# Window 2  (note: spot, not futures)
python -m src.data.capture_binance_stream --market spot --stream trades --symbol BTCUSDT --out data/raw_l20/btc_trades/

# Window 3
python -m src.data.capture_binance_stream --market futures --stream depth --symbol ETHUSDT --levels 20 --out data/raw_l20/eth_depth/

# Window 4  (spot)
python -m src.data.capture_binance_stream --market spot --stream trades --symbol ETHUSDT --out data/raw_l20/eth_trades/

# Window 5
python -m src.data.capture_binance_stream --market futures --stream depth --symbol SOLUSDT --levels 20 --out data/raw_l20/sol_depth/

# Window 6  (spot)
python -m src.data.capture_binance_stream --market spot --stream trades --symbol SOLUSDT --out data/raw_l20/sol_trades/
```

Leave all six windows open.

---

## 6. Daily verification (60 seconds)

Once a day, run this from the project root:

```bash
for d in data/raw_l20/*/; do
  python scripts/verify_capture.py --dir "$d" --last 5
  echo "----"
done
```

For each directory you should see:
- `rate_hz` close to 10 for depth streams (sometimes 8–12 is fine)
- `rate_hz` for trade streams varies — BTC ~10–50 trades/sec, SOL ~1–5 trades/sec
- `median_gap_ms` ≈ 100 for depth, very variable for trades
- `spread_violations` should be **0** for depth streams (bid is never above ask)

If any directory has zero new files in the last 24 hours or `rate_hz` is consistently <2, that capture has died. Look at its log file and restart it.

---

## 7. What to do if something breaks

### A capture process dies
Just restart the same command. Files are timestamped so nothing collides; the analysis tolerates gaps.

### Network drops for a few minutes
The script auto-reconnects with exponential backoff. You'll see warnings in the log. Nothing to do.

### The VM reboots / your laptop reboots
Captures stop. Restart them. Lost data = whatever didn't get flushed in the current hour (max 1 hour per process per crash).

### Disk fills up
Each depth20 stream produces ~120 MB/day. Six streams ≈ 700 MB/day. After 14 days that's ~10 GB. If you're tight on space, compress completed files: `gzip data/raw_l20/btc_depth/*.parquet` (Parquet's snappy compression is already on, so gzip on top adds maybe 10% more).

### One stream gets 0 messages
For futures aggTrade specifically: this is the regional issue. Switch that one stream to `spot` instead. All other failures: just look at the log.

---

## 8. Handover when done

After ~14 days (or whenever Matvei tells you to stop):

1. Stop all six captures: `tmux kill-session -t capture` (Option A) or close all six windows (Option B). The script flushes its buffer on shutdown.
2. Run the verification one last time and save the output:
   ```bash
   for d in data/raw_l20/*/; do
     python scripts/verify_capture.py --dir "$d"
   done > data/raw_l20/verification_final.txt
   ```
3. Compress and upload:
   ```bash
   tar czf raw_l20_handover.tar.gz data/raw_l20/
   ```
   (Result will be ~7 GB.) Upload to Google Drive / Dropbox / wherever Matvei prefers, share the link.
4. Note in your handover message:
   - Which option you used (A or B)
   - Total wall-clock duration captured
   - Any issues you noticed (gaps, restarts)
   - Total file size

---

## 9. Reference: stream message formats

If Matvei needs to debug something post-hoc, this is what each stream's Parquet rows look like:

**Depth (`*_depth20_*.parquet`)**
- `ts_local_ns`: int64 — local clock at message receipt
- `last_update_id`: int64 — Binance sequence number
- `bid_p1` … `bid_p20`, `bid_q1` … `bid_q20`: float — top-20 bids
- `ask_p1` … `ask_p20`, `ask_q1` … `ask_q20`: float — top-20 asks

**Trades (`*_aggTrade_*.parquet`)**
- `ts_local_ns`: local clock at receipt
- `ts_event_ms`, `ts_trade_ms`: Binance event / trade times in ms
- `agg_trade_id`: aggregated trade id
- `price`, `quantity`: trade price and size
- `is_buyer_maker`: True if taker was the seller (passively filled an existing bid)

---

## 10. Troubleshooting checklist

| Symptom | First thing to check |
|---|---|
| Zero messages on any depth stream | Network — try `ping fstream.binance.com` from the machine |
| Zero messages on futures aggTrade only | Regional issue; switch that stream to `--market spot` |
| "Connection refused" on websocket | Likely a corporate firewall blocking ws:// on 9443; need a different network |
| Process exits immediately | Usually a typo in the command. Re-run the smoke test from §5 |
| Disk full | `df -h`; gzip the older files or move them off-machine |
| Files have suspicious size 0 | Process died during write. Delete the 0-byte files and check the log |

---

## 11. Final ground rules

- **No API keys.** Everything in this brief uses public anonymous streams.
- **No trading.** This is read-only data collection. Don't connect any account.
- **Respect the project layout.** Output goes under `data/raw_l20/`. Don't move things around.
- **If anything weird happens, message Matvei before improvising.** Better to pause than to lose a week of data.

When in doubt, the quickest answer is: copy the relevant log lines into a message. Don't try to interpret them yourself.

Thanks again — this unblocks two paper sections.
