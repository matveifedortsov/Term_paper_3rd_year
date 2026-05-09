"""Phase 5 master runner: BNS per-day test + summary visualization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.benchmarks.bns_bipower import bns_per_day

OUT = Path("results/phase5")
BOOK_DIR = Path("data/interim")
DAYS = [
    "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18", "2024-03-19",
    "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24",
    "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29",
]


def run_bns() -> pd.DataFrame:
    books = {d: pd.read_parquet(BOOK_DIR / f"resampled_1s_{d}.parquet") for d in DAYS}
    df = bns_per_day(books)
    df.to_csv(OUT / "bns_per_day.csv", index=False)
    print("\n=== BNS bipower jump test, per day ===")
    print(df[["day", "Z", "RV", "BV", "reject"]].to_string(
        index=False, float_format=lambda x: f"{x:.4g}"
    ))
    print(f"\nDays flagged as containing >=1 jump: {int(df['reject'].sum())}/{len(df)}")
    return df


def plot_f1_summary() -> None:
    summary = pd.read_csv(OUT / "f1_summary.csv")
    summary = summary.set_index("method").reindex(
        ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]
    )

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    # F1 bars
    bars = ax[0].bar(
        summary.index, summary["F1"],
        color=["#264653", "#e76f51", "#2a9d8f", "#f4a261"],
    )
    ax[0].set_ylabel("F1 score (test days, +/-60s match)")
    ax[0].set_title("F1: jump-detection methods on persistence ground truth")
    ax[0].set_ylim(0, max(summary["F1"]) * 1.2)
    for b, v in zip(bars, summary["F1"]):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                   ha="center", va="bottom", fontsize=10)
    ax[0].grid(axis="y", alpha=0.3)

    # Detections vs truth bars
    width = 0.4
    x = range(len(summary.index))
    ax[1].bar([i - width / 2 for i in x], summary["TP"] + summary["FN"],
              width=width, label="ground truth", color="#264653")
    ax[1].bar([i + width / 2 for i in x], summary["n_detected_total"],
              width=width, label="total detections", color="#e76f51")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels(summary.index)
    ax[1].set_ylabel("count")
    ax[1].set_yscale("log")
    ax[1].set_title("Detection counts vs ground truth (log scale)")
    ax[1].legend()
    ax[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "f1_summary.png", dpi=150)
    plt.close(fig)


def plot_event_detections() -> None:
    df = pd.read_csv(OUT / "event_detections.csv")
    methods = ["raw_lomn", "lee_mykland", "lomn_xgb", "pure_ml"]
    colors = ["#264653", "#e76f51", "#2a9d8f", "#f4a261"]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.2
    x = range(len(df))
    for i, (m, c) in enumerate(zip(methods, colors)):
        ax.bar([j + (i - 1.5) * width for j in x], df[m], width=width, label=m, color=c)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{r['day']}\n{r['event']}" for _, r in df.iterrows()],
        rotation=15, ha="right", fontsize=8,
    )
    ax.set_ylabel("Detections inside event window")
    ax.set_title("Per-event detection counts (lower = less noisy at known events)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "event_detections.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bns = run_bns()
    plot_f1_summary()
    plot_event_detections()
    print(f"\nphase 5 plots saved to {OUT}")


if __name__ == "__main__":
    main()
