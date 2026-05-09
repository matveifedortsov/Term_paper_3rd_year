"""Generate Phase 4 figures from saved CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/phase4")

PARAM_NAMES = ["mu", "sigma", "lambda", "mu_J", "sigma_J"]
PARAM_LABELS = {
    "mu": r"$\mu$ (daily drift)",
    "sigma": r"$\sigma$ (daily diffusion vol)",
    "lambda": r"$\lambda$ (jumps/day)",
    "mu_J": r"$\mu_J$ (mean jump)",
    "sigma_J": r"$\sigma_J$ (jump std)",
}


def plot_synthetic_recovery() -> None:
    df = pd.read_csv(OUT / "neural_synthetic_preds.csv")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, name in zip(axes, PARAM_NAMES):
        true = df[f"true_{name}"].values
        pred = df[f"pred_{name}"].values
        lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
        ax.scatter(true, pred, s=3, alpha=0.35, color="#264653")
        ax.plot([lo, hi], [lo, hi], color="#e76f51", lw=1.2)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"pred {name}")
        ax.set_title(PARAM_LABELS[name], fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle("Neural calibrator: predicted vs true on synthetic test set")
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_recovery.png", dpi=150)
    plt.close(fig)


def plot_real_per_day() -> None:
    df = pd.read_csv(OUT / "neural_vs_mle_real.csv")
    days = pd.to_datetime(df["day"]).dt.strftime("%m-%d")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    axes = axes.flatten()
    for i, name in enumerate(PARAM_NAMES):
        ax = axes[i]
        ax.plot(days, df[f"neural_{name}"], lw=1.4, marker="o", ms=4,
                label="Neural", color="#2a9d8f")
        ax.plot(days, df[f"mle_{name}"], lw=1.4, marker="s", ms=4,
                label="MLE (4-sigma threshold)", color="#e76f51")
        ax.set_title(PARAM_LABELS[name], fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Per-day calibration on real BTC futures (Mar 15-29 2024)")
    fig.tight_layout()
    fig.savefig(OUT / "real_per_day.png", dpi=150)
    plt.close(fig)


def plot_train_history() -> None:
    df = pd.read_csv(OUT / "neural_train_history.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["epoch"] + 1, df["train_loss"], lw=1.5, marker="o", ms=4,
            label="train", color="#264653")
    ax.plot(df["epoch"] + 1, df["val_loss"], lw=1.5, marker="s", ms=4,
            label="val", color="#e76f51")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss (standardized targets)")
    ax.set_title("Neural calibrator training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "neural_train_history.png", dpi=150)
    plt.close(fig)


def main() -> None:
    plot_synthetic_recovery()
    plot_real_per_day()
    plot_train_history()
    print("plots saved to", OUT)


if __name__ == "__main__":
    main()
