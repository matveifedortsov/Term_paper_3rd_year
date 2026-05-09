"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.lomn.simulation import JumpDiffusionParams, simulate_path  # noqa: E402


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(2026_05_09)


@pytest.fixture(scope="session")
def small_h0_path():
    """A short path under H0 with no jumps and one-sided exponential noise."""
    rng = np.random.default_rng(0)
    params = JumpDiffusionParams(mu=0.0, sigma=0.03)
    return simulate_path(n=2880, T=1.0, params=params, noise_scale=0.001, rng=rng)


@pytest.fixture(scope="session")
def small_h1_path():
    """Short path with one deterministic jump at t=T/2."""
    rng = np.random.default_rng(1)
    params = JumpDiffusionParams(
        mu=0.0, sigma=0.03,
        fixed_jump_times=(0.5,),
        fixed_jump_sizes=(0.01,),
    )
    return simulate_path(n=2880, T=1.0, params=params, noise_scale=0.001, rng=rng)


@pytest.fixture(scope="session")
def labeled_features_fixture():
    """Tiny synthetic feature DataFrame with mixed labels."""
    n = 60
    df = pd.DataFrame({
        "ts": pd.date_range("2024-03-15", periods=n, freq="1h", tz="UTC"),
        "day": ["2024-03-15"] * 30 + ["2024-03-27"] * 30,
        "obs_idx": list(range(n)),
        "label": ([1] * 12 + [0] * 12 + [-1] * 6) * 2,
        "f_lomn_abs_std":      np.linspace(2.0, 8.0, n),
        "f_lomn_signed":       np.linspace(-5.0, 5.0, n),
        "f_volume_pm5s":       np.linspace(0.0, 200.0, n),
        "f_signed_flow_pm5s":  np.linspace(-50.0, 50.0, n),
        "f_n_trades_pm5s":     np.linspace(10, 1000, n).astype(int),
        "f_spread":            np.full(n, 0.1),
        "f_dspread_60s":       np.zeros(n),
        "f_obi_l1":            np.linspace(-1, 1, n),
        "f_log_mid":           np.full(n, 11.1),
        "f_dt_prev_cand":      np.full(n, 600.0),
        "f_realvar_60s":       np.full(n, 1e-6),
        "f_bipower_60s":       np.full(n, 1e-6),
        "f_realkurt_60s":      np.full(n, 4.0),
        "f_jump_ratio":        np.full(n, 1.2),
        "label_persistence_30s": np.linspace(-0.01, 0.01, n),
    })
    return df
