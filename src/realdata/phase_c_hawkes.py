"""Phase C: Hawkes self-exciting intensity per asset.

Replicates Phase A7 (Bibinger-LOMN paper Section 4) per-day Hawkes fit
on persistence-labeled jump times, across BTC / ETH / SOL Bybit spot.
Tests whether the cross-asset branching ratio is stable and whether
the LR rejection of Poisson holds independently of the asset.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.calibration.hawkes import fit_hawkes, poisson_loglik
from src.realdata.phase_c_runner import DEFAULT_SYMBOLS, build_symbol_dataset

LOG = logging.getLogger("phase-c-hawkes")
T_DAY = 86400.0
N_STARTS = 10


def hawkes_per_day(times: np.ndarray) -> dict:
    if len(times) < 5:
        return {"ok": False, "n_jumps": int(len(times))}
    fit = fit_hawkes(times, T_DAY, n_starts=N_STARTS)
    ll_p, mu_p = poisson_loglik(times, T_DAY)
    lr = 2.0 * (fit.log_lik - ll_p)
    p = 1.0 - chi2.cdf(lr, df=2)
    return {
        "ok": True, "n_jumps": int(fit.n_events),
        "mu": fit.mu, "alpha": fit.alpha, "beta": fit.beta,
        "branching": fit.branching, "logL_hawkes": fit.log_lik,
        "mu_poisson": mu_p, "logL_poisson": ll_p,
        "LR": lr, "p_value": float(p),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--trades-dir", type=Path, default=Path("data/historical"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase_c_ext"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_rows = []
    for sym in DEFAULT_SYMBOLS:
        LOG.info("====== %s ======", sym)
        labeled, _ = build_symbol_dataset(sym, args.book_dir, args.trades_dir)
        pos = labeled[labeled["label"] == 1].copy()
        if len(pos) == 0:
            continue
        days = sorted(pos["day"].unique())
        for d in days:
            t = np.sort(pos[pos["day"] == d]["obs_idx"].values.astype(float))
            r = hawkes_per_day(t)
            r["symbol"] = sym; r["day"] = d
            all_rows.append(r)
        df = pd.DataFrame([r for r in all_rows if r["symbol"] == sym])
        if df.empty:
            continue
        valid = df[df["ok"]]
        n_sig = int((valid["p_value"] < 0.05).sum())
        summary_rows.append({
            "symbol": sym,
            "n_days_fit": int(len(valid)),
            "n_lr_significant_5pct": n_sig,
            "median_branching": float(valid["branching"].median()),
            "mean_branching": float(valid["branching"].mean()),
            "median_logL_uplift": float(
                (valid["logL_hawkes"] - valid["logL_poisson"]).median()
            ),
            "median_p_value": float(valid["p_value"].median()),
        })

    pd.DataFrame(all_rows).to_csv(args.out_dir / "hawkes_per_asset_per_day.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "hawkes_per_asset_summary.csv", index=False)
    (args.out_dir / "hawkes_per_asset_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2)
    )
    print("\n=== Hawkes summary per asset ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Branching boxplot per asset
    df_all = pd.DataFrame(all_rows)
    df_ok = df_all[df_all["ok"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    data = [df_ok[df_ok["symbol"] == s]["branching"].values for s in DEFAULT_SYMBOLS]
    colors = {"BTCUSDT": "#264653", "ETHUSDT": "#2a9d8f", "SOLUSDT": "#e76f51"}
    bp = ax.boxplot(data, labels=DEFAULT_SYMBOLS, patch_artist=True, widths=0.55)
    for patch, sym in zip(bp["boxes"], DEFAULT_SYMBOLS):
        patch.set_facecolor(colors[sym]); patch.set_alpha(0.6)
    ax.axhline(1.0, color="red", ls="--", lw=1, label="criticality (α=β)")
    ax.set_ylabel("Hawkes branching ratio α / β")
    ax.set_title(f"Per-day branching ratio across assets (n_days = {len(df_ok)//3})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "hawkes_branching_per_asset.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
