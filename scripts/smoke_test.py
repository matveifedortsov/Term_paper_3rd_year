"""Quick sanity check on a single simulated path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lomn.detector import lomn_detector, optimal_block_size, gumbel_critical_value
from src.lomn.simulation import JumpDiffusionParams, simulate_path


def main() -> None:
    n = 23_400
    T = 1.0
    sigma = 0.03  # per-day log-return std (~30% / sqrt(252) range)
    h_n = optimal_block_size(n)
    print(f"n = {n},  T = {T},  sigma = {sigma},  h_n = {h_n},  m = ~{n // h_n} blocks")
    print(f"per-block diffusion std = sigma*sqrt(h_n/n) = {sigma*np.sqrt(h_n/n):.5f}")

    rng = np.random.default_rng(42)

    # Case 1: H0 (no jump)
    params_null = JumpDiffusionParams(mu=0.0, sigma=sigma)
    sim0 = simulate_path(n, T, params_null, noise_scale=0.001, rng=rng)
    res0 = lomn_detector(sim0.observed, h_n=h_n)
    print("\n[H0: no jump]")
    print(f"  T_max     = {res0.T_max:.3f}")
    print(f"  Gumbel CV = {res0.critical_value:.3f}")
    print(f"  reject?   = {res0.reject}")
    print(f"  cands     = {len(res0.candidate_block_idx)}")

    # Case 2: H1 (one mid-sample jump)
    rng = np.random.default_rng(43)
    jump = 0.01
    params_alt = JumpDiffusionParams(
        mu=0.0,
        sigma=sigma,
        fixed_jump_times=(T / 2.0,),
        fixed_jump_sizes=(jump,),
    )
    sim1 = simulate_path(n, T, params_alt, noise_scale=0.001, rng=rng)
    res1 = lomn_detector(sim1.observed, h_n=h_n)
    print(f"\n[H1: one jump of {jump} at t=T/2]")
    print(f"  T_max     = {res1.T_max:.3f}")
    print(f"  Gumbel CV = {res1.critical_value:.3f}")
    print(f"  reject?   = {res1.reject}")
    print(f"  cands     = {len(res1.candidate_block_idx)}")
    if len(res1.candidate_block_idx):
        true_obs = n // 2
        nearest = res1.candidate_obs_idx[
            np.argmin(np.abs(res1.candidate_obs_idx - true_obs))
        ]
        print(f"  jump @ obs idx {true_obs}; nearest detected at {nearest} "
              f"(gap = {abs(int(nearest) - true_obs)})")

    # Case 3: validate noise distribution
    print("\n[Noise sanity]")
    print(f"  noise mean (theory q*1)   = {0.001:.4f}, sample = {sim0.noise.mean():.4f}")
    print(f"  noise min  (theory 0)     = 0.0000, sample = {sim0.noise.min():.4f}")
    print(f"  noise frac >= 0           = {(sim0.noise >= 0).mean():.4f}")

    # Returns kurtosis sanity (should be > 3 with jumps)
    print("\n[Return kurtosis]")
    r0 = np.diff(sim0.observed)
    r1 = np.diff(sim1.observed)
    from scipy.stats import kurtosis
    print(f"  kurtosis no jump  = {kurtosis(r0, fisher=False):.2f}")
    print(f"  kurtosis w/ jump  = {kurtosis(r1, fisher=False):.2f}")


if __name__ == "__main__":
    main()
