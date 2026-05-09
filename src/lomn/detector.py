"""LOMN block-minimum jump detector.

Algorithm (Bibinger, Hautsch & Ristig, 2024):

1. Partition [0, T] into m = floor(n/h_n) non-overlapping blocks of size h_n.
2. Compute block minima M_k = min Y_i over block k.
3. Form first differences DeltaM_k = M_{k+1} - M_k as candidate jump
   estimators. Under H0 (no jumps), DeltaM_k is dominated by the
   diffusion increment of X across the block boundary plus a small
   contribution from the noise minima (mean q/h_n each).
4. Standardize DeltaM_k by a robust scale estimate s_hat (MAD-based,
   jump-robust by construction).
5. Reject H0 if max_k |DeltaM_k| / s_hat exceeds the Gumbel critical
   value for the maximum of m-1 standard normals.

The optimal block size under one-sided Exp(q/h_n) noise is
    h_n* = round(c * n^{1/3})
which balances the block-min noise bias O(q/h_n) against the diffusion
bias O(sigma * sqrt(h_n / n)). The constant c is set to 1 by default
and can be tuned via Monte Carlo size calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorResult:
    block_mins: np.ndarray
    delta_M: np.ndarray
    standardized: np.ndarray
    scale: float
    T_max: float
    critical_value: float
    reject: bool
    candidate_block_idx: np.ndarray
    candidate_obs_idx: np.ndarray
    candidate_jump_sizes: np.ndarray


def optimal_block_size(n: int, c: float = 1.0) -> int:
    """h_n = max(2, round(c * n^{1/3}))."""
    return int(max(2, round(c * n ** (1.0 / 3.0))))


def gumbel_critical_value(m: int, alpha: float = 0.05, two_sided: bool = True) -> float:
    """Critical value for max_k |Z_k|, k=1..m, with Z_k ~ N(0,1) i.i.d.

    Uses the standard extreme-value approximation:
        (T - a_m) / b_m -> Gumbel(0, 1),
    where a_m = sqrt(2 log m), b_m = 1 / a_m.
    """
    if m <= 1:
        return float("inf")
    a_m = np.sqrt(2.0 * np.log(m))
    b_m = 1.0 / a_m
    p = 1.0 - (alpha / 2.0 if two_sided else alpha)
    g = -np.log(-np.log(p))
    return float(a_m + b_m * g)


def block_minima(Y: np.ndarray, h_n: int) -> np.ndarray:
    n_obs = len(Y)
    m = n_obs // h_n
    if m < 2:
        raise ValueError(f"need at least 2 blocks, got n={n_obs}, h_n={h_n}")
    truncated = Y[: m * h_n].reshape(m, h_n)
    return truncated.min(axis=1)


def robust_scale(delta_M: np.ndarray) -> float:
    """MAD-based scale estimate, Gaussian-consistent (factor 1.4826)."""
    med = np.median(delta_M)
    mad = np.median(np.abs(delta_M - med))
    return float(1.4826 * mad)


def lomn_detector(
    Y: np.ndarray,
    h_n: int | None = None,
    alpha: float = 0.05,
    critical_value: float | None = None,
) -> DetectorResult:
    """Run the LOMN block-minimum detector on observed prices Y.

    Parameters
    ----------
    Y : observed log-prices, length n_obs.
    h_n : block size. If None, set to optimal_block_size(n_obs).
    alpha : nominal test level.
    critical_value : if provided, overrides the Gumbel theoretical value
        (use for empirically calibrated thresholds).
    """
    Y = np.asarray(Y, dtype=float)
    if h_n is None:
        h_n = optimal_block_size(len(Y))

    M = block_minima(Y, h_n)
    delta_M = np.diff(M)

    s_hat = robust_scale(delta_M)
    if s_hat <= 0.0:
        s_hat = float(np.std(delta_M, ddof=1)) or 1e-12

    standardized = delta_M / s_hat
    abs_std = np.abs(standardized)
    T_max = float(abs_std.max())

    m_eff = len(delta_M)
    crit = (
        critical_value
        if critical_value is not None
        else gumbel_critical_value(m_eff, alpha=alpha, two_sided=True)
    )

    cand_block = np.where(abs_std > crit)[0]
    cand_obs = (cand_block + 1) * h_n
    cand_jump = delta_M[cand_block]

    return DetectorResult(
        block_mins=M,
        delta_M=delta_M,
        standardized=standardized,
        scale=s_hat,
        T_max=T_max,
        critical_value=crit,
        reject=bool(T_max > crit),
        candidate_block_idx=cand_block,
        candidate_obs_idx=cand_obs,
        candidate_jump_sizes=cand_jump,
    )
