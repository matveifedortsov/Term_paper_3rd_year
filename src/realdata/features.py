"""Feature engineering for LOMN candidates on Path B (L1 + trades).

For each candidate timestamp tau we compute:

L1 / book features (from 1Hz resampled grid):
    f_spread        : ask - bid at tau
    f_dspread_60s   : spread(tau) - spread(tau - 60s)
    f_obi_l1        : (bid_q - ask_q) / (bid_q + ask_q) at tau
    f_log_mid       : log mid price at tau

Volatility / return features (60 s pre-tau on log_mid):
    f_realvar_60s   : sum of squared 1s log returns
    f_bipower_60s   : (pi/2) * sum |r_i| |r_{i-1}|  (jump-robust vol)
    f_realkurt_60s  : kurtosis of 1s log returns
    f_jump_ratio    : realvar / bipower (>1 indicates jump in window)

Trade-flow features (5 s window centered at tau, from aggTrades):
    f_volume_pm5s   : total quantity traded
    f_signed_flow_pm5s : net taker flow (buy_qty - sell_qty)
    f_n_trades_pm5s : number of aggTrades

Derived:
    f_lomn_abs_std  : the LOMN test statistic itself
    f_lomn_signed   : signed standardized statistic (carries direction)
    f_dt_prev_cand  : seconds since previous candidate (cap 3600)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("features")

REALVAR_WINDOW_S = 60
TRADE_WINDOW_S = 5  # +/- this many seconds around tau
PERSISTENCE_WINDOW_S = 30  # +/- around tau, used for LABELING ONLY


def _vol_features(log_mid: np.ndarray, idx: int, window_s: int) -> dict:
    lo = max(0, idx - window_s)
    seg = log_mid[lo: idx + 1]
    if len(seg) < 3:
        return {
            "f_realvar_60s": np.nan,
            "f_bipower_60s": np.nan,
            "f_realkurt_60s": np.nan,
            "f_jump_ratio": np.nan,
        }
    r = np.diff(seg)
    realvar = float(np.sum(r * r))
    if len(r) >= 2:
        abs_r = np.abs(r)
        bipower = float((np.pi / 2.0) * np.sum(abs_r[1:] * abs_r[:-1]))
    else:
        bipower = np.nan
    if len(r) >= 4:
        m = r.mean()
        s2 = r.var()
        kurt = float(np.mean((r - m) ** 4) / (s2 * s2)) if s2 > 0 else 3.0
    else:
        kurt = np.nan
    jump_ratio = realvar / bipower if bipower and bipower > 0 else np.nan
    return {
        "f_realvar_60s": realvar,
        "f_bipower_60s": bipower,
        "f_realkurt_60s": kurt,
        "f_jump_ratio": jump_ratio,
    }


def _trade_features(trades: pd.DataFrame, ts_ns: int, window_ns: int) -> dict:
    lo = ts_ns - window_ns
    hi = ts_ns + window_ns
    seg = trades.iloc[
        np.searchsorted(trades["ts_ns"].values, lo):
        np.searchsorted(trades["ts_ns"].values, hi)
    ]
    if len(seg) == 0:
        return {
            "f_volume_pm5s": 0.0,
            "f_signed_flow_pm5s": 0.0,
            "f_n_trades_pm5s": 0,
        }
    qty = seg["quantity"].values
    is_buyer_maker = seg["is_buyer_maker"].values
    # is_buyer_maker = True  -> taker was the seller (aggressive sell, signed -)
    # is_buyer_maker = False -> taker was the buyer  (aggressive buy,  signed +)
    sign = np.where(is_buyer_maker, -1.0, 1.0)
    return {
        "f_volume_pm5s": float(qty.sum()),
        "f_signed_flow_pm5s": float((sign * qty).sum()),
        "f_n_trades_pm5s": int(len(seg)),
    }


def build_features_for_day(
    cands: pd.DataFrame,
    book: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    book = book.reset_index(drop=True)
    # Convert tz-aware ms-resolution timestamps to integer ns since epoch
    book_ts_ns = book["ts"].astype("int64") * 1_000_000  # ms -> ns
    trades = trades.sort_values("transact_time", kind="mergesort").reset_index(drop=True)
    trades_ts_ns = trades["transact_time"].astype("int64") * 1_000_000  # ms -> ns
    trades = trades.assign(ts_ns=trades_ts_ns.values)

    log_mid = book["log_mid"].values
    spread = book["spread"].values
    bid_q = book["bid_q"].values
    ask_q = book["ask_q"].values

    book_ts0_ns = int(book_ts_ns.iloc[0])

    rows = []
    cand_ts_ns = cands["ts"].astype("int64").values * 1_000_000  # ms -> ns
    prev_ts_ns = np.concatenate([[book_ts0_ns], cand_ts_ns[:-1]])
    dt_prev_s = np.minimum((cand_ts_ns - prev_ts_ns) / 1e9, 3600.0)

    trade_window_ns = TRADE_WINDOW_S * 1_000_000_000
    sec_ns = 1_000_000_000

    for k in range(len(cands)):
        ts_ns = int(cand_ts_ns[k])
        idx = int((ts_ns - book_ts0_ns) // sec_ns)
        idx = max(0, min(idx, len(book) - 1))

        spread_now = float(spread[idx])
        idx_60 = max(0, idx - 60)
        spread_60 = float(spread[idx_60])
        bq = float(bid_q[idx])
        aq = float(ask_q[idx])
        obi = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0

        # Forward-looking persistence: log_mid 30s AFTER tau - 30s BEFORE.
        # Used ONLY for labeling; never added to FEATURE_COLS in train_xgb.
        idx_pre = max(0, idx - PERSISTENCE_WINDOW_S)
        idx_post = min(len(book) - 1, idx + PERSISTENCE_WINDOW_S)
        persistence_30s = float(log_mid[idx_post] - log_mid[idx_pre])

        feat = {
            "ts": cands["ts"].iloc[k],
            "day": cands["day"].iloc[k],
            "obs_idx": int(cands["obs_idx"].iloc[k]),
            "f_spread": spread_now,
            "f_dspread_60s": spread_now - spread_60,
            "f_obi_l1": obi,
            "f_log_mid": float(log_mid[idx]),
            "f_lomn_abs_std": float(cands["abs_std"].iloc[k]),
            "f_lomn_signed": float(cands["signed_std"].iloc[k]),
            "f_dt_prev_cand": float(dt_prev_s[k]),
            "label_persistence_30s": persistence_30s,
        }
        feat.update(_vol_features(log_mid, idx, REALVAR_WINDOW_S))
        feat.update(_trade_features(trades, ts_ns, trade_window_ns))
        rows.append(feat)

    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--cands-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--trades-dir", type=Path, default=Path("data/historical"))
    p.add_argument("--out", type=Path, default=Path("data/interim/features_all.parquet"))
    args = p.parse_args()

    cand_files = sorted(args.cands_dir.glob("lomn_candidates_2024-*.parquet"))
    if not cand_files:
        raise SystemExit("no candidate files")

    all_feats = []
    for cf in cand_files:
        date_str = cf.stem.split("_")[-1]
        bf = args.book_dir / f"resampled_1s_{date_str}.parquet"
        tf = args.trades_dir / f"futures_btcusdt_aggTrades_{date_str}.parquet"
        if not bf.exists() or not tf.exists():
            LOG.warning("skip %s (missing %s or %s)", date_str, bf.name, tf.name)
            continue
        cands = pd.read_parquet(cf)
        book = pd.read_parquet(bf)
        trades = pd.read_parquet(
            tf, columns=["transact_time", "quantity", "is_buyer_maker"]
        )
        feats = build_features_for_day(cands, book, trades)
        LOG.info("%s: %d candidates -> %d feature rows", date_str, len(cands), len(feats))
        all_feats.append(feats)

    df = pd.concat(all_feats, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, compression="snappy")
    LOG.info("wrote %d rows -> %s", len(df), args.out)
    print("\n=== feature summary ===")
    print(df.describe(include="all").T.to_string())


if __name__ == "__main__":
    main()
