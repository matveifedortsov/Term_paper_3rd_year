"""Generate Phase C visualizations.

Produces:
    results/phase_c/roc_overlay.png            ROC for XGB vs raw_LOMN per asset
    results/phase_c/metrics_bar.png            grouped AUC + F1 + FPR@90
    results/phase_c/transfer_test.png          BTC -> ETH / SOL transfer bars
    results/phase_c/feature_importance.png     top-15 features per asset (3 panels)
    results/phase_c/feature_groups.png         contribution by family (LOMN/L1/L20/trade)
    results/phase_c/candidates_per_day.png     per-day candidate count 3 assets
    results/phase_c/significance_panel.png     F1 CI + DeLong p values
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.realdata.phase_c_runner import build_symbol_dataset, train_eval_split

LOG = logging.getLogger("plot-phase-c")
OUT = Path("results/phase_c")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
COLORS = {"BTCUSDT": "#264653", "ETHUSDT": "#2a9d8f", "SOLUSDT": "#e76f51"}


# ----------------------------------------------------------------------
# Recompute test-set scores once and cache for the plots
# ----------------------------------------------------------------------

def gather_test_scores() -> dict[str, dict]:
    book_dir = Path("data/interim")
    trades_dir = Path("data/historical")
    cache: dict[str, dict] = {}
    for sym in SYMBOLS:
        LOG.info("Refitting %s for plot data...", sym)
        labeled, _ = build_symbol_dataset(sym, book_dir, trades_dir)
        res = train_eval_split(labeled, n_test_days=2)
        if "error" in res:
            continue
        model = res["model"]
        Xte = res["test_X"]
        yte = res["test_y"]
        feat_cols = res["feat_cols"]
        p_xgb = model.predict_proba(Xte)[:, 1]
        idx_lomn = feat_cols.index("f_lomn_abs_std")
        raw_lomn = Xte[:, idx_lomn]
        cache[sym] = {
            "model": model, "Xte": Xte, "yte": yte,
            "feat_cols": feat_cols,
            "p_xgb": p_xgb, "raw_lomn": raw_lomn,
            "auc_xgb": res["auc_xgb"],
            "auc_raw": res["auc_raw_lomn"],
            "fpr90_xgb": res["fpr_at_recall_90_xgb"]["fpr"],
            "fpr90_raw": res["fpr_at_recall_90_raw_lomn"]["fpr"],
            "f1_xgb": res["F1_xgb@0.5"],
        }
    return cache


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_roc(cache: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, sym in zip(axes, SYMBOLS):
        c = cache[sym]
        fpr_x, tpr_x, _ = roc_curve(c["yte"], c["p_xgb"])
        fpr_l, tpr_l, _ = roc_curve(c["yte"], c["raw_lomn"])
        ax.plot(fpr_x, tpr_x, lw=2, color=COLORS[sym],
                label=f"XGB (AUC={c['auc_xgb']:.3f})")
        ax.plot(fpr_l, tpr_l, lw=1.5, ls="--", color=COLORS[sym], alpha=0.5,
                label=f"raw LOMN (AUC={c['auc_raw']:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.5)
        ax.axhline(0.9, color="red", lw=0.5, ls=":", label="recall 0.90")
        ax.set_xlabel("False positive rate")
        ax.set_title(sym)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("True positive rate")
    fig.suptitle("Phase C — ROC: XGBoost(L20+buckets) vs raw LOMN, per asset")
    fig.tight_layout()
    fig.savefig(OUT / "roc_overlay.png", dpi=150)
    plt.close(fig)


def plot_metrics_bar(cache: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    width = 0.35
    x = np.arange(len(SYMBOLS))

    # AUC
    ax = axes[0]
    xgb_aucs = [cache[s]["auc_xgb"] for s in SYMBOLS]
    raw_aucs = [cache[s]["auc_raw"] for s in SYMBOLS]
    ax.bar(x - width / 2, raw_aucs, width, label="raw LOMN", color="#888")
    ax.bar(x + width / 2, xgb_aucs, width, label="XGB (L20)",
           color=[COLORS[s] for s in SYMBOLS])
    ax.set_xticks(x); ax.set_xticklabels(SYMBOLS)
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("ROC AUC")
    ax.set_title("ROC AUC")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)

    # F1
    ax = axes[1]
    f1s = [cache[s]["f1_xgb"] for s in SYMBOLS]
    bars = ax.bar(x, f1s, width=0.6, color=[COLORS[s] for s in SYMBOLS])
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(SYMBOLS)
    ax.set_ylim(0.65, 0.85)
    ax.set_ylabel("F1 @ p=0.5")
    ax.set_title("XGB F1")
    ax.grid(axis="y", alpha=0.3)

    # FPR@90
    ax = axes[2]
    xgb_f = [cache[s]["fpr90_xgb"] for s in SYMBOLS]
    raw_f = [cache[s]["fpr90_raw"] for s in SYMBOLS]
    ax.bar(x - width / 2, raw_f, width, label="raw LOMN", color="#888")
    ax.bar(x + width / 2, xgb_f, width, label="XGB",
           color=[COLORS[s] for s in SYMBOLS])
    for i, (r, g) in enumerate(zip(raw_f, xgb_f)):
        red = (r - g) / r * 100 if r > 0 else 0
        if red > 5:
            ax.text(x[i] + width / 2, g + 0.01,
                    f"-{red:.0f}%", ha="center", color="red", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(SYMBOLS)
    ax.set_ylabel("FPR at recall ≥ 0.90")
    ax.set_title("FPR@90 (lower = better)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase C headline metrics across BTC/ETH/SOL (L20 + bucket features)")
    fig.tight_layout()
    fig.savefig(OUT / "metrics_bar.png", dpi=150)
    plt.close(fig)


def plot_transfer(cache: dict) -> None:
    with open(OUT / "per_asset_metrics.json") as f:
        meta = json.load(f)
    transfer = meta["transfer_from_BTCUSDT"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    targets = ["ETHUSDT", "SOLUSDT"]
    same = [cache[t]["auc_xgb"] for t in targets]
    cross = [transfer[t]["auc_btc_to_target"] for t in targets]

    x = np.arange(len(targets))
    w = 0.35
    ax.bar(x - w / 2, same, w, label="trained on target",
           color=[COLORS[t] for t in targets])
    ax.bar(x + w / 2, cross, w, label="trained on BTC (transfer)",
           color=[COLORS[t] for t in targets], alpha=0.5,
           edgecolor="black", linewidth=1.5)
    for i, (s, c) in enumerate(zip(same, cross)):
        ax.text(x[i] - w / 2, s + 0.005, f"{s:.3f}",
                ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + w / 2, c + 0.005, f"{c:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(targets)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("BTC -> ETH / SOL transfer test (XGBoost trained on BTCUSDT only)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "transfer_test.png", dpi=150)
    plt.close(fig)


def plot_feature_importance() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for ax, sym in zip(axes, SYMBOLS):
        df = pd.read_csv(OUT / f"feature_importance_{sym}.csv").head(15).iloc[::-1]
        colors = ["#264653" if f.startswith("f_lomn") else
                  "#2a9d8f" if any(f.startswith(p) for p in ("bid_", "ask_")) else
                  "#e76f51" if any(p in f for p in ("imb", "depth", "slope", "skew", "inner")) else
                  "#f4a261" if any(p in f for p in ("volume", "signed_flow", "n_trades")) else
                  "#888"
                  for f in df["feature"]]
        ax.barh(df["feature"], df["gain"], color=colors)
        ax.set_xlabel("XGBoost gain")
        ax.set_title(sym)
        ax.grid(axis="x", alpha=0.3)
    # legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#264653", label="LOMN test stat"),
        plt.Rectangle((0, 0), 1, 1, color="#2a9d8f", label="L20 raw buckets"),
        plt.Rectangle((0, 0), 1, 1, color="#e76f51", label="L20 derived (imb/slope/skew)"),
        plt.Rectangle((0, 0), 1, 1, color="#f4a261", label="trade flow"),
        plt.Rectangle((0, 0), 1, 1, color="#888", label="other (vol/L1/timing)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.suptitle("Top-15 feature importance per asset (Phase C)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_feature_groups() -> None:
    """Group total gain by feature family."""
    rows = []
    for sym in SYMBOLS:
        df = pd.read_csv(OUT / f"feature_importance_{sym}.csv")

        def group(f: str) -> str:
            if f.startswith("f_lomn"):
                return "LOMN stat"
            if any(p in f for p in ("volume", "signed_flow", "n_trades")):
                return "trade flow"
            if f.startswith("bid_") or f.startswith("ask_"):
                return "L20 raw"
            if any(p in f for p in ("imb", "depth", "slope", "skew", "inner")):
                return "L20 derived"
            if f in {"f_spread", "f_dspread_60s", "f_obi_l1", "f_log_mid"}:
                return "L1 features"
            if f in {"f_realvar_60s", "f_bipower_60s", "f_realkurt_60s", "f_jump_ratio"}:
                return "vol moments"
            return "timing/other"

        df["family"] = df["feature"].map(group)
        agg = df.groupby("family")["gain"].sum().reset_index()
        agg["symbol"] = sym
        rows.append(agg)
    full = pd.concat(rows, ignore_index=True)
    pivot = full.pivot(index="family", columns="symbol", values="gain").fillna(0)
    order = ["LOMN stat", "L20 raw", "L20 derived", "trade flow",
             "L1 features", "vol moments", "timing/other"]
    pivot = pivot.reindex(order).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.index))
    w = 0.27
    for i, sym in enumerate(SYMBOLS):
        ax.bar(x + (i - 1) * w, pivot[sym], w, label=sym, color=COLORS[sym])
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("total XGBoost gain (sum across features)")
    ax.set_title("Phase C — total feature-family gain by asset")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "feature_groups.png", dpi=150)
    plt.close(fig)


def plot_candidates_per_day() -> None:
    book_dir = Path("data/interim")
    rows = []
    for sym in SYMBOLS:
        for f in sorted((book_dir / sym.lower()).glob("resampled_1s_*.parquet")):
            date_str = f.stem.split("_")[-1]
            if not date_str.startswith("2026-"):
                continue
            book = pd.read_parquet(f, columns=["log_ask"])
            from src.realdata.phase_c_runner import run_lomn_for_day
            cands = run_lomn_for_day(book.assign(ts=pd.NaT))  # ts irrelevant for count
            rows.append({"symbol": sym, "day": date_str, "n": len(cands)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for sym in SYMBOLS:
        sub = df[df["symbol"] == sym].sort_values("day")
        ax.plot(sub["day"], sub["n"], "o-", color=COLORS[sym], label=sym, lw=1.7, ms=6)
    ax.set_xlabel("day")
    ax.set_ylabel("LOMN candidates (threshold = 2.0)")
    ax.set_title("Phase C — candidates per day, 2026-05-01 to 2026-05-14")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "candidates_per_day.png", dpi=150)
    plt.close(fig)


def plot_significance_panel() -> None:
    df = pd.read_csv(OUT / "significance_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # F1 with CIs
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.35
    for i, row in df.iterrows():
        ax.errorbar(i - w / 2, row["F1_raw_mean"],
                    yerr=[[row["F1_raw_mean"] - row["F1_raw_ci_lo"]],
                          [row["F1_raw_ci_hi"] - row["F1_raw_mean"]]],
                    fmt="s", color="#888", markersize=8, capsize=4)
        ax.errorbar(i + w / 2, row["F1_xgb_mean"],
                    yerr=[[row["F1_xgb_mean"] - row["F1_xgb_ci_lo"]],
                          [row["F1_xgb_ci_hi"] - row["F1_xgb_mean"]]],
                    fmt="o", color=COLORS[row["symbol"]], markersize=8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(df["symbol"])
    ax.set_ylabel("F1 (95% bootstrap CI)")
    ax.set_title("F1: raw LOMN (gray) vs XGB (colored), with 95% bootstrap CI")
    ax.set_ylim(0.55, 0.92)
    ax.grid(axis="y", alpha=0.3)

    # p-values for each test
    ax = axes[1]
    methods = ["bootstrap (F1 diff)", "McNemar", "DeLong (AUC)"]
    p_values = df[["F1_diff_p", "mcnemar_p", "delong_p"]].values
    width = 0.27
    x = np.arange(len(methods))
    for i, sym in enumerate(SYMBOLS):
        row = df[df["symbol"] == sym].iloc[0]
        vals = [row["F1_diff_p"], row["mcnemar_p"], row["delong_p"]]
        ax.bar(x + (i - 1) * width, vals, width, color=COLORS[sym], label=sym)
    ax.axhline(0.05, color="red", lw=1, ls="--", label="α = 0.05")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=8)
    ax.set_ylabel("p-value")
    ax.set_yscale("log")
    ax.set_title("Statistical significance — XGB vs raw LOMN, by test")
    ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "significance_panel.png", dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.info("gathering test-set scores (re-fits XGB per asset)")
    cache = gather_test_scores()
    LOG.info("plotting...")
    plot_roc(cache)
    plot_metrics_bar(cache)
    plot_transfer(cache)
    plot_feature_importance()
    plot_feature_groups()
    plot_candidates_per_day()
    plot_significance_panel()
    LOG.info("plots saved to %s", OUT)


if __name__ == "__main__":
    main()
