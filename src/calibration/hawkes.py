"""Hawkes self-exciting jump intensity for the persistence-labeled jumps.

Replaces the Merton Poisson assumption with

    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

(exponential kernel). The closed-form log-likelihood (Ozaki 1979) is

    log L(theta; t_1..t_n) = sum_i log(lambda(t_i))
                            - mu * T
                            - (alpha / beta) * sum_i (1 - exp(-beta * (T - t_i)))

with stationarity requiring alpha < beta.

We fit by L-BFGS-B with positivity reparameterization
(theta = log[mu, alpha, beta]) and 10 multistart restarts. Compare
against the Poisson MLE (mu_hat = n / T, alpha = 0) by likelihood-
ratio: under H0 (Poisson), 2 * (logL_Hawkes - logL_Poisson) is
asymptotically chi^2 with df = 2.

Inputs (default): per-day persistence-labeled positives from
    `data/interim/features_labeled.parquet`.

Outputs (results/phase_a7/):
    hawkes_per_day.csv        : per-day mu, alpha, beta, branching, log-L
    hawkes_summary.json
    hawkes_intensity.png      : sample plot of fitted intensity for one day
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from src.config import config

LOG = logging.getLogger("hawkes")


# ----------------------------------------------------------------------
# Likelihoods (exponential kernel)
# ----------------------------------------------------------------------

def hawkes_loglik(times: np.ndarray, T: float, mu: float, alpha: float, beta: float) -> float:
    """Closed-form Hawkes log-likelihood with exponential kernel.

    Implements the recursion of Ozaki (1979): R_i = exp(-beta * dt) * (1 + R_{i-1}),
    so lambda(t_i) = mu + alpha * R_i. O(n) evaluation.
    """
    if mu <= 0 or alpha < 0 or beta <= 0:
        return -np.inf
    n = len(times)
    if n == 0:
        return -mu * T
    # Recursive R_i
    R = np.empty(n)
    R[0] = 0.0
    for i in range(1, n):
        R[i] = np.exp(-beta * (times[i] - times[i - 1])) * (1.0 + R[i - 1])
    lam = mu + alpha * R
    if np.any(lam <= 0):
        return -np.inf
    log_term = np.sum(np.log(lam))
    int_self = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))
    return float(log_term - mu * T - int_self)


def poisson_loglik(times: np.ndarray, T: float, mu: float | None = None) -> tuple[float, float]:
    """Homogeneous Poisson MLE log-likelihood.  Returns (logL, mu_hat)."""
    n = len(times)
    if mu is None:
        mu = n / T if T > 0 else 0.0
    if mu <= 0:
        return -np.inf, mu
    return float(n * np.log(mu) - mu * T), float(mu)


@dataclass(frozen=True)
class HawkesFit:
    mu: float
    alpha: float
    beta: float
    log_lik: float
    branching: float       # alpha / beta = expected children per event
    n_events: int
    T: float
    converged: bool


def fit_hawkes(
    times: np.ndarray, T: float, n_starts: int = 10, seed: int = 42,
) -> HawkesFit:
    """L-BFGS-B with multistart in log-parameters."""
    n = len(times)
    if n < 5:
        return HawkesFit(mu=n / T if T > 0 else 0.0, alpha=0.0, beta=1.0,
                          log_lik=poisson_loglik(times, T)[0], branching=0.0,
                          n_events=n, T=T, converged=False)
    rng = np.random.default_rng(seed)

    # Initial guesses spanning a few decades; first start is the Poisson MLE
    starts = [(n / T, 1e-3, 1e-2)]
    for _ in range(n_starts - 1):
        mu0 = (n / T) * 10 ** rng.uniform(-1.5, 0.5)
        beta0 = 10 ** rng.uniform(-4, -1)
        alpha0 = beta0 * rng.uniform(0.05, 0.85)
        starts.append((mu0, alpha0, beta0))

    def neg_logL(theta_log: np.ndarray) -> float:
        mu, alpha, beta = np.exp(theta_log)
        if alpha >= beta:  # enforce sub-criticality (stationary process)
            return -hawkes_loglik(times, T, mu, alpha * 0.99, beta) + 1.0
        return -hawkes_loglik(times, T, mu, alpha, beta)

    best = None
    for s in starts:
        x0 = np.log(np.array(s, dtype=float))
        try:
            res = minimize(neg_logL, x0, method="L-BFGS-B",
                           options={"maxiter": 200, "gtol": 1e-7})
        except Exception:
            continue
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        ll, mu_p = poisson_loglik(times, T)
        return HawkesFit(mu=mu_p, alpha=0.0, beta=1.0, log_lik=ll,
                          branching=0.0, n_events=n, T=T, converged=False)
    mu_h, alpha_h, beta_h = np.exp(best.x)
    return HawkesFit(
        mu=float(mu_h), alpha=float(alpha_h), beta=float(beta_h),
        log_lik=-float(best.fun), branching=float(alpha_h / beta_h),
        n_events=n, T=T, converged=bool(best.success),
    )


# ----------------------------------------------------------------------
# Per-day driver
# ----------------------------------------------------------------------

def jump_times_per_day(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """For each day, gather positive-labeled jump times (in seconds within the day)."""
    pos = features[features["label"] == 1]
    out: dict[str, np.ndarray] = {}
    for d, g in pos.groupby("day"):
        out[d] = np.sort(g["obs_idx"].values.astype(float))
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = config()
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path,
                   default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/phase_a7"))
    p.add_argument("--n-starts", type=int,
                   default=int(cfg["hawkes"].get("random_starts", 10)))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    by_day = jump_times_per_day(feats)
    LOG.info("days with positive jumps: %d", len(by_day))

    T_day = 86400.0  # one day in seconds (1Hz grid)

    rows = []
    fit_cache: dict[str, HawkesFit] = {}
    for d, t in by_day.items():
        if len(t) < 5:
            LOG.info("%s: %d jumps; skipping (too few)", d, len(t))
            continue
        fit = fit_hawkes(t, T=T_day, n_starts=args.n_starts)
        ll_pois, mu_pois = poisson_loglik(t, T_day)
        lr = 2.0 * (fit.log_lik - ll_pois)
        p_value = 1.0 - chi2.cdf(lr, df=2)
        fit_cache[d] = fit
        rows.append({
            "day": d, "n_jumps": int(fit.n_events),
            "mu_hawkes": fit.mu, "alpha": fit.alpha, "beta": fit.beta,
            "branching": fit.branching, "logL_hawkes": fit.log_lik,
            "mu_poisson": mu_pois, "logL_poisson": ll_pois,
            "LR": lr, "lr_p_value_df2": float(p_value),
            "converged": fit.converged,
        })
        LOG.info("%s  n=%-3d  mu=%.2e  branching=%.3f  LR=%.2f (p=%.4f)",
                 d, fit.n_events, fit.mu, fit.branching, lr, p_value)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "hawkes_per_day.csv", index=False)

    n_significant = int((df["lr_p_value_df2"] < 0.05).sum())
    summary = {
        "n_days_fit": int(len(df)),
        "n_lr_significant_at_5pct": n_significant,
        "median_branching": float(df["branching"].median()),
        "mean_branching": float(df["branching"].mean()),
        "median_logL_uplift_per_day": float((df["logL_hawkes"] - df["logL_poisson"]).median()),
        "median_p_value_df2": float(df["lr_p_value_df2"].median()),
    }
    (args.out_dir / "hawkes_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Hawkes per-day fits ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    # ---- Plot intensity for the most-jumpy day ----
    if len(df):
        most_jumpy_day = df.loc[df["n_jumps"].idxmax(), "day"]
        fit = fit_cache[most_jumpy_day]
        t_eval = by_day[most_jumpy_day]
        grid = np.linspace(0, T_day, 5000)
        # Vectorized evaluation of intensity on the grid
        lam = np.full_like(grid, fit.mu)
        for ti in t_eval:
            mask = grid >= ti
            lam[mask] += fit.alpha * np.exp(-fit.beta * (grid[mask] - ti))
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(grid / 3600.0, lam, color="#264653", lw=1, label="lambda(t)")
        ax.axhline(fit.mu, color="#e76f51", ls="--", lw=1,
                   label=f"mu = {fit.mu:.3e}")
        ax.scatter(t_eval / 3600.0, np.full_like(t_eval, fit.mu * 0.5),
                   color="#2a9d8f", s=18, marker="x",
                   label=f"jumps (n={len(t_eval)})")
        ax.set_xlabel("hour of day")
        ax.set_ylabel("intensity (events / second)")
        ax.set_title(f"Hawkes intensity, {most_jumpy_day}   "
                     f"branching = {fit.branching:.2f}   "
                     f"LR p (vs Poisson) = "
                     f"{df.set_index('day').loc[most_jumpy_day, 'lr_p_value_df2']:.3f}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "hawkes_intensity.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
