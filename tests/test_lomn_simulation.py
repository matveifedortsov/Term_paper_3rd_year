"""Tests for the synthetic Merton + LOMN-noise data generator."""

from __future__ import annotations

import numpy as np

from src.lomn.simulation import JumpDiffusionParams, simulate_path


def test_seed_reproducibility():
    """Same seed -> byte-exact noise array."""
    p = JumpDiffusionParams(mu=0.0, sigma=0.03)
    a = simulate_path(1000, 1.0, p, 0.001, np.random.default_rng(42))
    b = simulate_path(1000, 1.0, p, 0.001, np.random.default_rng(42))
    np.testing.assert_array_equal(a.observed, b.observed)
    np.testing.assert_array_equal(a.efficient, b.efficient)


def test_noise_one_sided_exponential():
    """Noise is non-negative and exponentially distributed (mean ~ scale)."""
    p = JumpDiffusionParams(mu=0.0, sigma=0.03)
    sim = simulate_path(50_000, 1.0, p, 0.001, np.random.default_rng(7))
    assert (sim.noise >= 0.0).all(), "LOMN noise must be one-sided non-negative"
    assert sim.noise.mean() == \
        __import__("pytest").approx(0.001, rel=0.05), "noise mean ~ scale"


def test_efficient_path_unaffected_by_noise():
    """The efficient path X must not depend on the noise scale q."""
    p = JumpDiffusionParams(mu=0.0, sigma=0.03)
    a = simulate_path(2000, 1.0, p, 0.001, np.random.default_rng(123))
    b = simulate_path(2000, 1.0, p, 0.005, np.random.default_rng(123))
    np.testing.assert_allclose(a.efficient, b.efficient)


def test_fixed_jumps_show_up_in_efficient():
    """A deterministic jump must create exactly that increment in X."""
    p = JumpDiffusionParams(
        mu=0.0, sigma=0.0,
        fixed_jump_times=(0.5,),
        fixed_jump_sizes=(0.05,),
    )
    sim = simulate_path(1000, 1.0, p, 0.0, np.random.default_rng(0))
    assert sim.efficient[-1] - sim.efficient[0] == \
        __import__("pytest").approx(0.05, abs=1e-12)


def test_observed_minus_efficient_is_noise():
    p = JumpDiffusionParams(mu=0.0, sigma=0.03)
    sim = simulate_path(2000, 1.0, p, 0.002, np.random.default_rng(0))
    np.testing.assert_allclose(sim.observed - sim.efficient, sim.noise)
