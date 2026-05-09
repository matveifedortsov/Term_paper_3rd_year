"""End-to-end empirical size test for the LOMN detector.

Runs a small Monte Carlo (50 reps) under H0 with the calibrated CV
threshold and checks that the rejection rate is roughly consistent
with the nominal 5% level. Tolerant bounds because n_reps is small.
"""

from __future__ import annotations

import numpy as np

from src.lomn.detector import lomn_detector
from src.lomn.simulation import JumpDiffusionParams, simulate_path


def test_empirical_size_in_envelope():
    """Rejection rate under H0 should land in [0%, 25%] with this small MC."""
    rng = np.random.default_rng(1729)
    params = JumpDiffusionParams(mu=0.0, sigma=0.03)
    n = 4_000
    n_reps = 50
    rejects = 0
    for _ in range(n_reps):
        sim = simulate_path(n, 1.0, params, noise_scale=0.001, rng=rng)
        res = lomn_detector(sim.observed)
        rejects += int(res.reject)
    rate = rejects / n_reps
    # Loose envelope. The full Phase 1 MC at 500 reps gives ~5%; with 50 reps
    # the SE of a Bernoulli proportion at p=0.05 is ~3pp.
    assert 0.0 <= rate <= 0.25, f"empirical size {rate:.2f} outside envelope"


def test_h1_power_majority():
    """With a clear 0.02 jump, most replications should reject."""
    rng = np.random.default_rng(99)
    n = 4_000
    n_reps = 30
    rejects = 0
    for _ in range(n_reps):
        params = JumpDiffusionParams(
            mu=0.0, sigma=0.03,
            fixed_jump_times=(0.5,),
            fixed_jump_sizes=(0.02,),
        )
        sim = simulate_path(n, 1.0, params, noise_scale=0.001, rng=rng)
        res = lomn_detector(sim.observed)
        rejects += int(res.reject)
    rate = rejects / n_reps
    assert rate >= 0.5, f"power {rate:.2f} too low for delta=0.02"
