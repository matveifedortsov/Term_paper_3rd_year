"""Tests for the persistence-based labeling rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.realdata.label import label_features


def _df(persist_30s: list[float], scale: float = 1e-3) -> pd.DataFrame:
    """Build a minimal labeling input DataFrame with one-day fake data."""
    return pd.DataFrame({
        "day":                  ["2024-03-15"] * len(persist_30s),
        "obs_idx":              np.arange(len(persist_30s)),
        "label_persistence_30s": persist_30s,
        "f_realvar_60s":        np.full(len(persist_30s), scale ** 2),
    })


def test_label_assignment_basic():
    """High |persistence|/scale -> 1; low -> 0; in-between -> -1."""
    scale = 1e-3
    persist = [
        scale * 6.0,    # |z| = 6 >= 5  -> POS
        -scale * 5.5,   # |z| = 5.5 -> POS
        scale * 1.0,    # |z| = 1 <= 2  -> NEG
        scale * 1.5,    # |z| = 1.5 <= 2 -> NEG
        scale * 3.0,    # |z| = 3, in (2, 5) -> DROP
        -scale * 4.0,   # |z| = 4 -> DROP
    ]
    df = _df(persist, scale=scale)
    scale_per_day = pd.Series({"2024-03-15": scale})
    out = label_features(df, scale_per_day=scale_per_day)
    assert list(out["label"].values) == [1, 1, 0, 0, -1, -1]


def test_label_drop_when_scale_zero_uses_fallback():
    """No scale_per_day argument: fallback uses sqrt(median realvar)."""
    df = _df([0.01, 0.001, 0.0001], scale=1e-3)
    out = label_features(df, scale_per_day=None)
    assert "label" in out.columns
    assert len(out) == 3


def test_label_includes_persist_z_column():
    df = _df([0.005, -0.001], scale=1e-3)
    out = label_features(df, scale_per_day=pd.Series({"2024-03-15": 1e-3}))
    assert "persist_z" in out.columns
    np.testing.assert_allclose(
        out["persist_z"].values, [5.0, 1.0], rtol=1e-9
    )


def test_label_threshold_change_via_module_constants():
    """Labels respect module-level POS/NEG thresholds."""
    from src.realdata import label as label_mod
    orig_pos, orig_neg = label_mod.POS_PERSIST_Z, label_mod.NEG_PERSIST_Z
    try:
        label_mod.POS_PERSIST_Z = 3.0
        label_mod.NEG_PERSIST_Z = 1.0
        df = _df([0.0035, 0.0005, 0.002], scale=1e-3)
        out = label_mod.label_features(df, scale_per_day=pd.Series({"2024-03-15": 1e-3}))
        # z = 3.5, 0.5, 2.0 -> POS, NEG, DROP
        assert list(out["label"]) == [1, 0, -1]
    finally:
        label_mod.POS_PERSIST_Z = orig_pos
        label_mod.NEG_PERSIST_Z = orig_neg
