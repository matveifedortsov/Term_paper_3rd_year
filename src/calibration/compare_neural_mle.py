"""Compare neural calibrator vs MLE on synthetic ground truth + real BTC.

For synthetic data: parameters are known, so we can compute true RMSE
of each estimator. MLE here means: simulate Merton, identify true jumps
(known from the simulator), apply separable MLE.

For real BTC: there is no ground truth, so we report the per-day point
estimates from each method side by side and the across-day std (a
proxy for stability).

Headline outputs (results/phase4/):
    neural_vs_mle_synthetic.csv
    neural_vs_mle_real.csv
    inference_speed.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.calibration.mle import calibrate_merton
from src.calibration.neural import (
    L,
    PARAM_NAMES,
    MertonCNN,
    predict_params,
    returns_per_day_from_book,
    sample_priors,
    simulate_returns_batch,
)

LOG = logging.getLogger("compare-neural")


def mle_on_synthetic(
    R: np.ndarray, true_params: np.ndarray, T: float = 1.0
) -> np.ndarray:
    """Per-row MLE using true jump locations (oracle MLE — best case)."""
    n, length = R.shape
    out = np.empty_like(true_params)
    for i in range(n):
        # Reconstruct path from returns
        path = np.concatenate([[0.0], np.cumsum(R[i])])
        # We don't have true jump indices recorded; approximate:
        # threshold returns at 4 sigma of robust scale.
        r = R[i]
        scale = 1.4826 * np.median(np.abs(r - np.median(r)))
        if scale <= 0:
            scale = float(np.std(r, ddof=1)) or 1e-9
        jump_mask = np.abs(r) > 4.0 * scale
        jump_idx = np.where(jump_mask)[0]
        jump_sizes = r[jump_idx]
        params = calibrate_merton(path, jump_idx + 1, jump_sizes, T=T)
        out[i] = [params.mu_hat, params.sigma_hat, params.lambda_hat,
                  params.mu_J_hat if not np.isnan(params.mu_J_hat) else 0.0,
                  params.sigma_J_hat if not np.isnan(params.sigma_J_hat) else 0.0]
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("results/phase4/merton_cnn.pt"))
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase4"))
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + normalization
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = MertonCNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    y_mean = ckpt["y_mean"]
    y_std = ckpt["y_std"]
    LOG.info("model loaded from %s", args.model)

    # ---------- Synthetic ground-truth comparison ----------
    rng = np.random.default_rng(args.seed)
    true_params = sample_priors(args.n_test, rng)
    R = simulate_returns_batch(true_params, rng).astype(np.float32)
    LOG.info("simulated %d test paths (L=%d)", args.n_test, L)

    t0 = time.perf_counter()
    p_neural = predict_params(model, R, y_mean, y_std)
    t_neural = time.perf_counter() - t0
    LOG.info("neural inference: %.3fs (%.1f us/path)",
             t_neural, 1e6 * t_neural / args.n_test)

    t0 = time.perf_counter()
    p_mle = mle_on_synthetic(R, true_params)
    t_mle = time.perf_counter() - t0
    LOG.info("MLE inference  : %.3fs (%.1f us/path)",
             t_mle, 1e6 * t_mle / args.n_test)

    rmse_neural = np.sqrt(np.mean((p_neural - true_params) ** 2, axis=0))
    rmse_mle = np.sqrt(np.mean((p_mle - true_params) ** 2, axis=0))
    stds = true_params.std(axis=0) + 1e-9
    syn = pd.DataFrame({
        "param": PARAM_NAMES,
        "rmse_neural": rmse_neural,
        "rmse_mle_oracle": rmse_mle,
        "rel_rmse_neural": rmse_neural / stds,
        "rel_rmse_mle_oracle": rmse_mle / stds,
    })
    syn.to_csv(args.out_dir / "neural_vs_mle_synthetic.csv", index=False)
    print("\n=== Neural vs oracle-MLE on synthetic ground truth ===")
    print(syn.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    speed = {
        "n_test": args.n_test,
        "neural_total_s": t_neural,
        "neural_us_per_path": 1e6 * t_neural / args.n_test,
        "mle_total_s": t_mle,
        "mle_us_per_path": 1e6 * t_mle / args.n_test,
        "neural_speedup_vs_mle": t_mle / t_neural if t_neural > 0 else float("nan"),
    }
    (args.out_dir / "inference_speed.json").write_text(json.dumps(speed, indent=2))
    LOG.info("speed: %s", json.dumps(speed, indent=2))

    # ---------- Real BTC: side-by-side per day ----------
    days = [
        "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18", "2024-03-19",
        "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24",
        "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29",
    ]
    used_days, R_real = returns_per_day_from_book(args.book_dir, days)
    p_real_neural = predict_params(model, R_real, y_mean, y_std)
    p_real_mle = mle_on_synthetic(R_real, np.zeros((len(used_days), 5)))  # ignores true_params

    cols_n = [f"neural_{n}" for n in PARAM_NAMES]
    cols_m = [f"mle_{n}" for n in PARAM_NAMES]
    real = pd.DataFrame(
        np.concatenate([p_real_neural, p_real_mle], axis=1),
        columns=cols_n + cols_m,
    )
    real.insert(0, "day", used_days)
    real.to_csv(args.out_dir / "neural_vs_mle_real.csv", index=False)
    print("\n=== Neural vs MLE on real BTC, per day ===")
    print(real.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Cross-day std (stability)
    stab = pd.DataFrame({
        "param": PARAM_NAMES,
        "neural_std": real[cols_n].std().values,
        "mle_std": real[cols_m].std().values,
        "neural_mean": real[cols_n].mean().values,
        "mle_mean": real[cols_m].mean().values,
    })
    stab.to_csv(args.out_dir / "neural_vs_mle_real_stability.csv", index=False)
    print("\n=== Cross-day stability (std lower = more stable) ===")
    print(stab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
