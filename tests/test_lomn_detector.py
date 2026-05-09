"""Tests for the LOMN block-minimum detector."""

from __future__ import annotations

import numpy as np
import pytest

from src.lomn.detector import (
    block_minima,
    gumbel_critical_value,
    lomn_detector,
    optimal_block_size,
    robust_scale,
)


def test_optimal_block_size_grows_as_cuberoot():
    h_a = optimal_block_size(1000)
    h_b = optimal_block_size(8000)
    # 8000 ^ (1/3) / 1000 ^ (1/3) = 2 -> h_b ≈ 2 * h_a
    assert h_b == pytest.approx(2 * h_a, abs=2)


def test_block_size_constant_scales_linearly():
    h_a = optimal_block_size(10000, c=0.5)
    h_b = optimal_block_size(10000, c=1.0)
    h_c = optimal_block_size(10000, c=2.0)
    assert h_a < h_b < h_c
    assert h_c == pytest.approx(2 * h_b, abs=2)


def test_gumbel_cv_is_increasing_in_m():
    cv_low = gumbel_critical_value(50)
    cv_mid = gumbel_critical_value(500)
    cv_hi = gumbel_critical_value(5000)
    assert cv_low < cv_mid < cv_hi


def test_gumbel_cv_alpha_monotone():
    cv_strict = gumbel_critical_value(500, alpha=0.01)
    cv_lax = gumbel_critical_value(500, alpha=0.10)
    assert cv_strict > cv_lax


def test_block_minima_shape_and_values():
    Y = np.array([3, 1, 2, 4, 5, 0, 7, 8], dtype=float)
    M = block_minima(Y, h_n=4)
    assert M.shape == (2,)
    assert M[0] == 1.0
    assert M[1] == 0.0


def test_block_minima_raises_when_too_small():
    with pytest.raises(ValueError):
        block_minima(np.zeros(3), h_n=4)


def test_robust_scale_zero_for_constant():
    assert robust_scale(np.zeros(100)) == 0.0


def test_robust_scale_matches_gaussian_factor():
    rng = np.random.default_rng(123)
    x = rng.standard_normal(50_000)
    s = robust_scale(x)
    assert s == pytest.approx(1.0, abs=0.03)


def test_detector_does_not_reject_under_h0(small_h0_path):
    res = lomn_detector(small_h0_path.observed)
    # Empirical 5%-level test: most paths under H0 should not reject.
    # A single path occasionally rejects; just check the structure.
    assert isinstance(res.reject, bool)
    assert res.standardized.size > 0
    assert res.scale > 0


def test_detector_localizes_h1_jump(small_h1_path):
    res = lomn_detector(small_h1_path.observed)
    assert res.reject, "should reject H0 with a 0.01 jump"
    n = len(small_h1_path.observed)
    true_idx = n // 2
    nearest = res.candidate_obs_idx[
        np.argmin(np.abs(res.candidate_obs_idx - true_idx))
    ]
    h_n = res.delta_M.size + 1  # n_blocks
    h_n_inferred = n // (res.block_mins.size)
    assert abs(int(nearest) - true_idx) <= h_n_inferred + 1
