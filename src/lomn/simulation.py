"""Data-generating process for LOMN Monte Carlo experiments.

Efficient log-price X follows a Merton jump-diffusion:
    dX_t = mu * dt + sigma * dW_t + dJ_t,
where J is a compound Poisson process with intensity lambda and jump
sizes drawn from N(mu_J, sigma_J^2).

Observed best-ask log-price Y is contaminated by one-sided exponential
microstructure noise:
    Y_i = X_{t_i} + q * E_i,    E_i ~ Exp(1), independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class JumpDiffusionParams:
    mu: float = 0.0
    sigma: float = 0.4
    lam: float = 0.0
    mu_jump: float = 0.0
    sigma_jump: float = 0.0
    fixed_jump_times: tuple[float, ...] = field(default_factory=tuple)
    fixed_jump_sizes: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimulationResult:
    times: np.ndarray
    efficient: np.ndarray
    observed: np.ndarray
    noise: np.ndarray
    jump_times: np.ndarray
    jump_sizes: np.ndarray


def simulate_path(
    n: int,
    T: float,
    params: JumpDiffusionParams,
    noise_scale: float,
    rng: np.random.Generator,
) -> SimulationResult:
    """Simulate one Merton + LOMN-noise sample path on [0, T].

    Parameters
    ----------
    n : number of observation points (excluding t=0).
    T : time horizon (in days; sigma and lam are interpreted accordingly).
    params : drift, diffusion, jump parameters.
    noise_scale : q in Y = X + q * Exp(1). Set to 0 for noise-free X.
    rng : numpy Generator.
    """
    dt = T / n
    times = np.linspace(0.0, T, n + 1)

    # Brownian increments
    z = rng.standard_normal(n)
    dW = z * np.sqrt(dt)
    drift = params.mu * dt

    # Jumps: either fixed (deterministic) or Poisson-driven
    if params.fixed_jump_times:
        jump_times = np.asarray(params.fixed_jump_times, dtype=float)
        jump_sizes = np.asarray(params.fixed_jump_sizes, dtype=float)
    elif params.lam > 0.0:
        n_jumps = rng.poisson(params.lam * T)
        jump_times = np.sort(rng.uniform(0.0, T, size=n_jumps))
        jump_sizes = rng.normal(params.mu_jump, params.sigma_jump, size=n_jumps)
    else:
        jump_times = np.empty(0, dtype=float)
        jump_sizes = np.empty(0, dtype=float)

    # Map jump times to nearest observation index in (0, n]
    jump_at_step = np.zeros(n)
    if jump_times.size > 0:
        idx = np.clip(np.searchsorted(times, jump_times, side="right") - 1, 0, n - 1)
        np.add.at(jump_at_step, idx, jump_sizes)

    increments = drift + params.sigma * dW + jump_at_step
    efficient = np.empty(n + 1)
    efficient[0] = 0.0
    np.cumsum(increments, out=efficient[1:])

    # One-sided exponential noise on observed grid (t_0, t_1, ..., t_n)
    if noise_scale > 0.0:
        noise = noise_scale * rng.exponential(scale=1.0, size=n + 1)
    else:
        noise = np.zeros(n + 1)
    observed = efficient + noise

    return SimulationResult(
        times=times,
        efficient=efficient,
        observed=observed,
        noise=noise,
        jump_times=jump_times,
        jump_sizes=jump_sizes,
    )
