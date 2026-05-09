"""Generate a stratified set of LOMN candidates for hand-labeling.

For each selected candidate, produce a clear PNG showing:
    - log mid-price over [tau-60s, tau+60s]
    - bid / ask traces
    - aggTrade scatter colored by aggressor side
    - vertical line at tau
    - text panel with LOMN stat, persistence z, and the persistence label

Also write a CSV template the user fills in with columns:
    cand_id, day, persistence_label, hand_label, notes

Usage:
    python -m src.realdata.build_label_set
        # default: 100 positives + 50 negatives + 50 ambiguous, seed 42

The user opens each PNG, decides real/noise/ambig, and types their
verdict into the hand_label column. Then run:
    python -m src.realdata.score_hand_labels
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import config

LOG = logging.getLogger("build-labels")

WINDOW_S = 60          # window around tau for the chart
PALETTE = {
    "real_ask":   "#264653",
    "real_bid":   "#2a9d8f",
    "buyer_ag":   "#e63946",   # aggressive buy
    "seller_ag":  "#1d3557",   # aggressive sell
    "tau":        "#ff8800",
    "ambig":      "#888888",
}


def stratified_sample(
    features: pd.DataFrame,
    n_pos: int,
    n_neg: int,
    n_amb: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pick candidates from each label bucket with replacement-free sampling."""
    pos = features[features["label"] == 1]
    neg = features[features["label"] == 0]
    amb = features[features["label"] == -1]
    take = lambda df, n: df.sample(n=min(n, len(df)), random_state=int(rng.integers(0, 1 << 30)))
    return pd.concat([
        take(pos, n_pos).assign(_bucket="pos"),
        take(neg, n_neg).assign(_bucket="neg"),
        take(amb, n_amb).assign(_bucket="amb"),
    ], ignore_index=True)


def plot_candidate(
    cand_id: str,
    cand: pd.Series,
    book: pd.DataFrame,
    trades: pd.DataFrame,
    out_path: Path,
    window_s: int = WINDOW_S,
) -> None:
    """One multi-panel PNG per candidate."""
    obs_idx = int(cand["obs_idx"])
    n = len(book)
    lo = max(0, obs_idx - window_s)
    hi = min(n, obs_idx + window_s + 1)
    book_seg = book.iloc[lo:hi].copy()

    tau_ts = book["ts"].iloc[obs_idx]
    win_lo_ts = book_seg["ts"].iloc[0]
    win_hi_ts = book_seg["ts"].iloc[-1]

    # trade segment
    win_lo_ms = pd.Timestamp(win_lo_ts).value // 1_000_000
    win_hi_ms = pd.Timestamp(win_hi_ts).value // 1_000_000
    tr = trades[(trades["transact_time"] >= win_lo_ms) &
                (trades["transact_time"] <= win_hi_ms)].copy()
    if len(tr):
        tr["ts"] = pd.to_datetime(tr["transact_time"], unit="ms", utc=True)

    persist_z = float(cand.get("persist_z", float("nan")))
    persist_label = int(cand["label"])
    label_name = {1: "PERSIST: positive (real?)", 0: "PERSIST: negative (noise?)", -1: "PERSIST: ambiguous"}[persist_label]

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 2, 1]}, sharex=False,
    )
    ax_p, ax_t, ax_meta = axes

    ax_p.plot(book_seg["ts"], book_seg["ask_p"], lw=1.0,
              color=PALETTE["real_ask"], label="ask")
    ax_p.plot(book_seg["ts"], book_seg["bid_p"], lw=1.0,
              color=PALETTE["real_bid"], label="bid", alpha=0.85)
    ax_p.axvline(tau_ts, color=PALETTE["tau"], lw=1.5,
                 label=f"tau ({obs_idx})")
    ax_p.set_ylabel("price (USD)")
    ax_p.set_title(f"{cand_id}   day={cand['day']}   "
                   f"|stat|={cand['f_lomn_abs_std']:.2f}   "
                   f"persist_z={persist_z:.2f}   "
                   f"{label_name}")
    ax_p.legend(loc="best", fontsize=8)
    ax_p.grid(alpha=0.3)

    if len(tr):
        sign_color = np.where(tr["is_buyer_maker"].values,
                              PALETTE["seller_ag"], PALETTE["buyer_ag"])
        ax_t.scatter(tr["ts"], tr["quantity"], c=sign_color, s=14, alpha=0.7)
        ax_t.axvline(tau_ts, color=PALETTE["tau"], lw=1.0)
        ax_t.set_ylabel("trade size")
        ax_t.set_yscale("log")
    else:
        ax_t.text(0.5, 0.5, "(no trades in window)", ha="center", va="center",
                  transform=ax_t.transAxes, color="gray")
    ax_t.grid(alpha=0.3)

    ax_meta.axis("off")
    txt = (
        f"vol $\\pm$5s = {cand['f_volume_pm5s']:.2f} BTC          "
        f"signed flow = {cand['f_signed_flow_pm5s']:+.2f}          "
        f"n_trades = {int(cand['f_n_trades_pm5s'])}\n"
        f"jump_ratio = {cand['f_jump_ratio']:.2f}          "
        f"OBI L1 = {cand['f_obi_l1']:+.2f}          "
        f"spread = {cand['f_spread']:.2f}          "
        f"realkurt = {cand['f_realkurt_60s']:.1f}\n"
        f"\n"
        f"hand label?   r = real jump   n = noise spike   a = ambiguous"
    )
    ax_meta.text(0.0, 0.5, txt, transform=ax_meta.transAxes, family="monospace",
                 fontsize=10, va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path,
                   default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--book-dir", type=Path,
                   default=Path("data/interim"))
    p.add_argument("--trades-dir", type=Path,
                   default=Path("data/historical"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("data/handlabel"))
    p.add_argument("--n-pos", type=int, default=100)
    p.add_argument("--n-neg", type=int, default=50)
    p.add_argument("--n-amb", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(args.features)
    LOG.info("loaded features: %d rows", len(feats))

    rng = np.random.default_rng(args.seed)
    sample = stratified_sample(feats, args.n_pos, args.n_neg, args.n_amb, rng)
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    LOG.info("sampled: pos=%d  neg=%d  amb=%d  total=%d",
             int((sample["_bucket"] == "pos").sum()),
             int((sample["_bucket"] == "neg").sum()),
             int((sample["_bucket"] == "amb").sum()),
             len(sample))

    cache_book: dict[str, pd.DataFrame] = {}
    cache_trades: dict[str, pd.DataFrame] = {}

    for i, row in sample.iterrows():
        day = row["day"]
        if day not in cache_book:
            bf = args.book_dir / f"resampled_1s_{day}.parquet"
            cache_book[day] = pd.read_parquet(bf)
        if day not in cache_trades:
            tf = args.trades_dir / f"futures_btcusdt_aggTrades_{day}.parquet"
            cache_trades[day] = pd.read_parquet(
                tf, columns=["transact_time", "quantity", "is_buyer_maker"]
            )
        cand_id = f"cand_{i+1:03d}_{row['_bucket']}"
        out_png = args.out_dir / f"{cand_id}.png"
        plot_candidate(cand_id, row, cache_book[day], cache_trades[day], out_png)
        if (i + 1) % 25 == 0:
            LOG.info("plotted %d / %d", i + 1, len(sample))

    # Build the CSV template
    template = pd.DataFrame({
        "cand_id":           [f"cand_{i+1:03d}_{row['_bucket']}" for i, row in sample.iterrows()],
        "day":               sample["day"].values,
        "obs_idx":           sample["obs_idx"].astype(int).values,
        "persistence_label": sample["label"].astype(int).values,
        "lomn_abs_std":      sample["f_lomn_abs_std"].astype(float).values,
        "persist_z":         sample.get("persist_z", pd.Series([float("nan")] * len(sample))).values,
        "hand_label":        ["" for _ in range(len(sample))],
        "notes":             ["" for _ in range(len(sample))],
    })
    template.to_csv(args.out_dir / "labels_template.csv", index=False)
    LOG.info("wrote template -> %s", args.out_dir / "labels_template.csv")

    instr = args.out_dir / "INSTRUCTIONS.md"
    instr.write_text(
        "# Hand-labeling protocol\n\n"
        f"You have {len(sample)} candidates to label, in `labels_template.csv`.\n\n"
        "For each row:\n"
        "1. Open the corresponding PNG (`<cand_id>.png` in this folder).\n"
        "2. Decide whether it's a **real jump** (a sustained level shift driven by trade flow), "
        "a **noise spike** (a momentary blip with no real economic content), or **ambiguous**.\n"
        "3. Type one of:\n"
        "   - `real`  — clear sustained move with directional volume\n"
        "   - `noise` — flutter without sustained move\n"
        "   - `ambig` — genuinely unclear, even after looking carefully\n"
        "4. Optionally add a short `notes` (e.g., \"news at 13:30 UTC\", \"thin book\").\n\n"
        "Take ~5-second pauses between candidates. ~1 minute per candidate is normal. "
        "You don't have to do them all in one sitting — save the CSV between sessions.\n\n"
        "Save as `labels_handlabeled.csv` in this folder when done. Then run:\n"
        "```\npython -m src.realdata.score_hand_labels\n```\n",
        encoding="utf-8",
    )

    print(f"\n>>> {len(sample)} candidates plotted to {args.out_dir}/")
    print(f">>> Edit {args.out_dir}/labels_template.csv, save as labels_handlabeled.csv")
    print(f">>> See {args.out_dir}/INSTRUCTIONS.md for the protocol")


if __name__ == "__main__":
    main()
