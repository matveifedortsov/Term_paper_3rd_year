"""Smoke test that the feature pipeline has stable column names."""

from __future__ import annotations

from src.realdata.train_xgb import FEATURE_COLS


def test_feature_column_names_stable():
    expected = {
        "f_spread",
        "f_dspread_60s",
        "f_obi_l1",
        "f_log_mid",
        "f_lomn_abs_std",
        "f_lomn_signed",
        "f_dt_prev_cand",
        "f_realvar_60s",
        "f_bipower_60s",
        "f_realkurt_60s",
        "f_jump_ratio",
        "f_volume_pm5s",
        "f_signed_flow_pm5s",
        "f_n_trades_pm5s",
    }
    assert set(FEATURE_COLS) == expected, \
        "Feature columns changed; update labels and train code together"


def test_feature_count_is_14():
    assert len(FEATURE_COLS) == 14


def test_no_duplicate_features():
    assert len(set(FEATURE_COLS)) == len(FEATURE_COLS)
