"""Lee-Mykland (2008) nonparametric jump test.

For a price series Y on a regular grid with returns r_i = Y_i - Y_{i-1},
LM defines local volatility via bipower variation over a window of K
prior returns:

    sigma_hat_i^2 = (1 / (K - 2)) * sum_{j=i-K+2..i-1} |r_j| |r_{j-1}|

(K is set to ~sqrt(n) per the paper; we use 270 for n=86400, ~4.5 min).
The standardized statistic is

    L_i = |r_i| / sigma_hat_i,

and under the null of no jumps, max_i L_i has a Gumbel limit:

    (max_i L_i - C_n) / S_n  -> Gumbel(0, 1),

where, with c = sqrt(2/pi),

    C_n = (sqrt(2 ln n)) / c  -  (ln(pi) + ln(ln n)) / (2 c sqrt(2 ln n))
    S_n = 1 / (c sqrt(2 ln n))

A return is flagged as a jump candidate if its standardized stat
exceeds the Gumbel critical value at level alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

C_PI = np.sqrt(2.0 / np.pi)


def lee_mykland_constants(n: int) -> tuple[float, float]:
    ln_n = np.log(n)
    sqrt_2lnn = np.sqrt(2.0 * ln_n)
    C_n = sqrt_2lnn / C_PI - (np.log(np.pi) + np.log(ln_n)) / (2.0 * C_PI * sqrt_2lnn)
    S_n = 1.0 / (C_PI * sqrt_2lnn)
    return float(C_n), float(S_n)


def gumbel_quantile(alpha: float) -> float:
    """1-alpha quantile of Gumbel(0,1) (one-sided)."""
    return float(-np.log(-np.log(1.0 - alpha)))


def detect_jumps(
    Y: np.ndarray, K: int = 270, alpha: float = 0.05
) -> dict:
    """Run Lee-Mykland on a 1D price series Y. Returns dict with stats."""
    Y = np.asarray(Y, dtype=float)
    r = np.diff(Y)
    n = len(r)
    abs_r = np.abs(r)

    # Bipower local vol over rolling window of K returns
    bp = abs_r[:-1] * abs_r[1:]
    cumsum_bp = np.concatenate([[0.0], np.cumsum(bp)])
    # var estimate at index i (for return r_i, i >= K) uses returns i-K+1..i-1
    # So bp window: indices i-K+1 to i-2 (inclusive), each bp[j] = |r_j| |r_{j+1}|
    # This is a length-(K-2) window in bp
    sigma2 = np.full(n, np.nan)
    win = K - 2
    if n > K:
        # bp window for return index i: bp indices [i-K+1, i-2]
        # Equivalent: for i = K..n-1, use bp[i-K+1:i-1] of length K-2
        starts = np.arange(K - 1, n - 1)  # i goes from K-1 to n-2
        # Actually let's match LM: variance for r_i uses returns r_{i-K+2}..r_{i-1}
        # In bp space: bp[j] for j in [i-K+1, i-2], length K-2
        for idx, i in enumerate(starts):
            lo = i - K + 1
            hi = i - 1  # exclusive
            if lo < 0:
                continue
            sigma2[i] = (cumsum_bp[hi] - cumsum_bp[lo]) / win
    sigma_hat = np.sqrt(np.maximum(sigma2, 1e-30))
    L = abs_r / np.where(sigma_hat > 0, sigma_hat, np.inf)

    Cn, Sn = lee_mykland_constants(n)
    g_q = gumbel_quantile(alpha)
    crit = Cn + Sn * g_q  # critical value on the L scale

    detected = np.where(L > crit)[0]
    return {
        "L": L,
        "sigma_hat": sigma_hat,
        "K": K,
        "C_n": Cn,
        "S_n": Sn,
        "critical_value": float(crit),
        "alpha": alpha,
        "detected_return_idx": detected,
        "detected_obs_idx": detected + 1,  # the second point of return r_i is Y[i+1]
    }


def detect_jumps_df(book: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Wrapper returning a DataFrame with one row per detection."""
    Y = book["log_mid"].values
    out = detect_jumps(Y, **kwargs)
    rows = pd.DataFrame({
        "obs_idx": out["detected_obs_idx"],
        "L": out["L"][out["detected_return_idx"]],
        "sigma_hat": out["sigma_hat"][out["detected_return_idx"]],
        "delta_logmid": np.diff(Y)[out["detected_return_idx"]],
    })
    rows["ts"] = book["ts"].iloc[out["detected_obs_idx"]].values
    rows["critical_value"] = out["critical_value"]
    return rows
