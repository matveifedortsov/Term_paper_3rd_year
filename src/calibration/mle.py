"""Separable maximum-likelihood calibration of the Merton jump-diffusion.

Model:
    d log P(t) = mu * dt + sigma * dW(t) + dJ(t)
    J(t) = sum_{k=1..N(t)} Z_k,   N(t) ~ Poisson(lambda * t),
    Z_k ~ N(mu_J, sigma_J^2),  iid, independent of W.

Given:
    - log_mid : 1Hz log-price grid for one day, length n
    - jump_set : DataFrame with columns ['obs_idx', 'jump_size'] giving
                 the index of each detected jump and its estimated size

Returns Merton parameters in DAILY units (to align with sigma=0.03/day
used elsewhere in the project):
    lambda_hat   : jumps per day  (= |jump_set| / 1)
    mu_J_hat     : mean detected jump size
    sigma_J_hat  : std of detected jump sizes (NaN if <2 jumps)
    mu_hat       : daily drift, from continuous returns of cleaned path
    sigma_hat    : daily diffusion vol

Cleaning: returns at jump indices are dropped before computing (mu, sigma).
This is the Aït-Sahalia-Jacod two-step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MertonParams:
    n_obs: int
    n_jumps: int
    lambda_hat: float
    mu_J_hat: float
    sigma_J_hat: float
    mu_hat: float
    sigma_hat: float
    T: float


def calibrate_merton(
    log_mid: np.ndarray,
    jump_idx: np.ndarray,
    jump_sizes: np.ndarray,
    T: float = 1.0,
) -> MertonParams:
    """Estimate (lambda, mu_J, sigma_J, mu, sigma) for a single window.

    Parameters
    ----------
    log_mid : array of log mid-prices on a regular grid.
    jump_idx : observation indices flagged as jumps (will be excluded
        when computing diffusion stats).
    jump_sizes : signed magnitudes of those jumps.
    T : window length in days (default 1).
    """
    n = len(log_mid)
    n_jumps = len(jump_idx)
    lambda_hat = n_jumps / T

    if n_jumps >= 1:
        mu_J_hat = float(np.mean(jump_sizes))
    else:
        mu_J_hat = float("nan")
    if n_jumps >= 2:
        sigma_J_hat = float(np.std(jump_sizes, ddof=1))
    else:
        sigma_J_hat = float("nan")

    # Returns of the path; drop those that span a detected jump
    r = np.diff(log_mid)
    if n_jumps > 0:
        # Each jump at obs_idx tau corresponds to return r[tau-1] (= log_mid[tau] - log_mid[tau-1])
        # Drop indices tau-1 (clipped to valid range)
        bad = np.clip(jump_idx - 1, 0, len(r) - 1)
        keep = np.ones(len(r), dtype=bool)
        keep[bad] = False
        r_clean = r[keep]
    else:
        r_clean = r

    if len(r_clean) < 5:
        mu_hat = float("nan")
        sigma_hat = float("nan")
    else:
        # n returns over T days  =>  per-step dt = T/n
        dt = T / n
        mu_hat = float(r_clean.mean() / dt)
        sigma_hat = float(r_clean.std(ddof=1) / np.sqrt(dt))

    return MertonParams(
        n_obs=n,
        n_jumps=n_jumps,
        lambda_hat=lambda_hat,
        mu_J_hat=mu_J_hat,
        sigma_J_hat=sigma_J_hat,
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        T=T,
    )


def calibrate_per_day(
    days_book: dict[str, pd.DataFrame],
    jumps_per_day: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply calibrate_merton to each day's data.

    days_book : {day_str -> resampled book DataFrame with 'log_mid'}
    jumps_per_day : {day_str -> DataFrame with 'obs_idx' and 'jump_size'}
        Days with no jumps may be passed as empty DataFrames.
    """
    rows = []
    for day, book in days_book.items():
        log_mid = book["log_mid"].values.astype(float)
        j = jumps_per_day.get(day, pd.DataFrame(columns=["obs_idx", "jump_size"]))
        params = calibrate_merton(
            log_mid,
            jump_idx=j["obs_idx"].values.astype(int) if len(j) else np.empty(0, dtype=int),
            jump_sizes=j["jump_size"].values.astype(float) if len(j) else np.empty(0, dtype=float),
        )
        rows.append({
            "day": day,
            "n_obs": params.n_obs,
            "n_jumps": params.n_jumps,
            "lambda_hat": params.lambda_hat,
            "mu_J_hat": params.mu_J_hat,
            "sigma_J_hat": params.sigma_J_hat,
            "mu_hat": params.mu_hat,
            "sigma_hat": params.sigma_hat,
        })
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
