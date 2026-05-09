"""Deep ensemble of MertonCNN calibrators for uncertainty quantification.

Trains N independently-seeded copies of the 1D-CNN from
`src.calibration.neural` on independently-sampled synthetic Merton
batches. At inference returns mean and std across the ensemble per
parameter, giving an (admittedly approximate) credibility band that
quantifies the prior-bias caveat from Phase 4.

Quick rationale: the deep-ensemble approach (Lakshminarayanan, Pritzel
& Blundell 2017) is the simplest reliable uncertainty method for
neural regression and is cited as such in Ovadia et al. (2019).

Outputs (results/phase4/):
    ensemble_predictions_synthetic.csv : (true, mean, std) per param
    ensemble_predictions_real.csv      : per-day mean +/- std for BTC
    ensemble_train_history.csv         : per-model val loss
    ensemble_uncertainty.png           : 5-panel scatter w/ error bars
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.calibration.neural import (
    L,
    PARAM_NAMES,
    MertonCNN,
    TrainConfig,
    make_dataset,
    predict_params,
    returns_per_day_from_book,
    sample_priors,
    simulate_returns_batch,
    train_cnn,
)
from src.config import config

LOG = logging.getLogger("ensemble")


def train_one_member(seed: int, n_train: int, epochs: int, batch_size: int) -> dict:
    cfg = TrainConfig(
        n_train=n_train, n_val=4_000, n_test=4_000,
        batch_size=batch_size, epochs=epochs, seed=seed,
    )
    out = train_cnn(cfg, device="cpu")
    return {"model": out["model"], "y_mean": out["y_mean"], "y_std": out["y_std"],
            "history": out["history"], "best_val_loss": out["best_val_loss"],
            "seed": seed}


def predict_ensemble(
    members: list[dict], R: np.ndarray, device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and std (n, 5) of natural-scale parameter predictions."""
    preds = np.stack([
        predict_params(m["model"], R, m["y_mean"], m["y_std"], device=device)
        for m in members
    ], axis=0)  # (M, n, 5)
    return preds.mean(axis=0), preds.std(axis=0)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = config()
    p = argparse.ArgumentParser()
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase4"))
    p.add_argument("--n-train", type=int,
                   default=int(cfg["neural"].get("n_train", 20000)) // 2)
    p.add_argument("--epochs", type=int,
                   default=int(cfg["neural"].get("epochs", 20)))
    p.add_argument("--batch-size", type=int,
                   default=int(cfg["neural"].get("batch_size", 256)))
    p.add_argument("--ensemble-size", type=int,
                   default=int(cfg["neural"].get("ensemble_size", 5)))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg["neural"].get("seed", 20260508))

    # ----- Train members -----
    LOG.info("training ensemble of %d MertonCNN members (n_train=%d, epochs=%d)",
             args.ensemble_size, args.n_train, args.epochs)
    members = []
    histories = []
    t0 = time.perf_counter()
    for k in range(args.ensemble_size):
        seed = base_seed + k * 1000
        LOG.info("--- member %d/%d (seed=%d) ---", k + 1, args.ensemble_size, seed)
        m = train_one_member(seed, args.n_train, args.epochs, args.batch_size)
        members.append(m)
        for h in m["history"]:
            histories.append({**h, "member": k, "seed": seed})
    elapsed = time.perf_counter() - t0
    LOG.info("ensemble trained in %.1fs (avg %.1fs/member)",
             elapsed, elapsed / args.ensemble_size)

    pd.DataFrame(histories).to_csv(args.out_dir / "ensemble_train_history.csv", index=False)

    # ----- Synthetic test recovery -----
    rng = np.random.default_rng(base_seed + 99_999)
    n_test = 2000
    true_params = sample_priors(n_test, rng)
    R = simulate_returns_batch(true_params, rng).astype(np.float32)
    mean, std = predict_ensemble(members, R)

    rec_rows = []
    for i, name in enumerate(PARAM_NAMES):
        rec_rows.append({
            "param": name,
            "rmse_ensemble_mean": float(np.sqrt(np.mean((mean[:, i] - true_params[:, i]) ** 2))),
            "mean_predictive_std": float(std[:, i].mean()),
            "rel_rmse": float(np.sqrt(np.mean((mean[:, i] - true_params[:, i]) ** 2)) /
                              (true_params[:, i].std() + 1e-9)),
            "calibration_z_var": float(((mean[:, i] - true_params[:, i]) / (std[:, i] + 1e-12)).var()),
        })
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(args.out_dir / "ensemble_synthetic_recovery.csv", index=False)
    print("\n=== Ensemble synthetic recovery ===")
    print(rec.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    syn_pred = pd.DataFrame(
        {**{f"true_{n}": true_params[:, i] for i, n in enumerate(PARAM_NAMES)},
         **{f"mean_{n}": mean[:, i] for i, n in enumerate(PARAM_NAMES)},
         **{f"std_{n}":  std[:, i]  for i, n in enumerate(PARAM_NAMES)}}
    )
    syn_pred.to_csv(args.out_dir / "ensemble_predictions_synthetic.csv", index=False)

    # ----- Real BTC inference -----
    days = cfg["split"]["train_days"] + cfg["split"]["test_days"]
    used_days, R_real = returns_per_day_from_book(args.book_dir, days)
    mean_real, std_real = predict_ensemble(members, R_real)
    real_df = pd.DataFrame({"day": used_days})
    for i, n in enumerate(PARAM_NAMES):
        real_df[f"mean_{n}"] = mean_real[:, i]
        real_df[f"std_{n}"]  = std_real[:, i]
    real_df.to_csv(args.out_dir / "ensemble_predictions_real.csv", index=False)
    print("\n=== Ensemble per-day BTC (mean +/- std) ===")
    print(real_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ----- Plot: scatter with error bars -----
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, name, i in zip(axes, PARAM_NAMES, range(5)):
        true_v = true_params[:, i]
        m = mean[:, i]
        s = std[:, i]
        order = np.argsort(true_v)
        ax.errorbar(true_v[order], m[order], yerr=s[order],
                    fmt="o", ms=2.5, alpha=0.35, color="#264653",
                    elinewidth=0.4, ecolor="#999", label="pred ± 1σ")
        lo, hi = float(min(true_v.min(), m.min())), float(max(true_v.max(), m.max()))
        ax.plot([lo, hi], [lo, hi], color="#e76f51", lw=1.2)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"pred {name}")
        ax.set_title(f"{name}")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Ensemble (M={args.ensemble_size}) predictive uncertainty on synthetic")
    fig.tight_layout()
    fig.savefig(args.out_dir / "ensemble_uncertainty.png", dpi=150)
    plt.close(fig)

    # Save ensemble: list of state dicts
    torch.save({
        "members": [
            {"model_state": m["model"].state_dict(),
             "y_mean": m["y_mean"], "y_std": m["y_std"],
             "seed": m["seed"]}
            for m in members
        ],
    }, args.out_dir / "merton_cnn_ensemble.pt")

    summary = {
        "ensemble_size": args.ensemble_size,
        "n_train_per_member": args.n_train,
        "epochs": args.epochs,
        "total_train_seconds": elapsed,
        "synthetic_recovery": rec_rows,
    }
    (args.out_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("done")


if __name__ == "__main__":
    main()
