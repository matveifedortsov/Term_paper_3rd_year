"""Tests for percentage-distance bucket aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.realdata.bucket_aggregate import (
    DEFAULT_BP_EDGES,
    aggregate,
    aggregate_from_snapshot_row,
    bucket_features,
)


def test_aggregate_basic_bucket_assignment():
    """Two orders at 0.5 bp and 7 bp land in the right buckets."""
    mid = 80_000.0
    # Bid 0.5 bp below mid -> bucket [0, 1)bp ; bid 7 bp below mid -> [5, 10)bp
    bids = [(80_000.0 * (1 - 0.5 / 10000), 1.0),
            (80_000.0 * (1 - 7.0 / 10000), 2.0)]
    asks = []
    out = aggregate(bids, asks, mid)
    assert out["bid_0_1bp"] == pytest.approx(1.0)
    assert out["bid_5_10bp"] == pytest.approx(2.0)
    # Other buckets must be zero
    for k, v in out.items():
        if k not in {"bid_0_1bp", "bid_5_10bp"}:
            assert v == 0.0


def test_aggregate_symmetry_bids_vs_asks():
    """Symmetric orders -> mirror buckets. Use 7 bp to avoid sitting on edge=5."""
    mid = 100.0
    bids = [(100.0 * (1 - 7 / 10000), 3.0)]
    asks = [(100.0 * (1 + 7 / 10000), 3.0)]
    out = aggregate(bids, asks, mid)
    assert out["bid_5_10bp"] == out["ask_5_10bp"] == pytest.approx(3.0)


def test_aggregate_boundary_goes_to_upper_bucket():
    """Edges are [lo, hi); a value exactly on the edge ends up in the upper bucket."""
    mid = 1_000_000.0   # exact, so bp arithmetic stays exact
    # bid 5 bp below mid: 1_000_000 - 500 = 999_500. Distance = 5 bp exactly.
    bids = [(999_500.0, 1.0)]
    out = aggregate(bids, [], mid)
    # Goes to [5,10)bp under side="right" semantics
    assert out["bid_5_10bp"] == pytest.approx(1.0)
    assert out["bid_2_5bp"] == 0.0


def test_aggregate_drops_far_orders():
    """Orders beyond the max edge are dropped."""
    mid = 100.0
    far_bid = (100.0 * (1 - 1000 / 10000), 999.0)
    bids = [far_bid]
    out = aggregate(bids, [], mid)
    assert all(v == 0.0 for k, v in out.items() if k.startswith("bid_"))


def test_aggregate_zero_qty_skipped():
    """Empty levels (qty=0) contribute nothing; second order at 7 bp lands."""
    mid = 100.0
    bids = [(99.99, 0.0), (100.0 * (1 - 7 / 10000), 1.5)]
    out = aggregate(bids, [], mid)
    assert out["bid_5_10bp"] == pytest.approx(1.5)


def test_aggregate_custom_edges():
    edges = (0, 10, 100)
    mid = 1000.0
    bids = [(999.5, 1.0),    # 5 bp -> bucket [0, 10)
            (995.0, 2.0)]    # 50 bp -> bucket [10, 100)
    out = aggregate(bids, [], mid, bp_edges=edges)
    assert set(out) == {"bid_0_10bp", "bid_10_100bp", "ask_0_10bp", "ask_10_100bp"}
    assert out["bid_0_10bp"] == pytest.approx(1.0)
    assert out["bid_10_100bp"] == pytest.approx(2.0)


def test_aggregate_rejects_bad_mid():
    with pytest.raises(ValueError):
        aggregate([], [], mid=0.0)


def test_aggregate_rejects_non_monotone_edges():
    with pytest.raises(ValueError):
        aggregate([], [], mid=100.0, bp_edges=(0, 5, 3, 10))


def test_aggregate_from_snapshot_row():
    row = pd.Series({
        "bid_p1": 80_000.0, "bid_q1": 1.0,
        "bid_p2": 79_999.9, "bid_q2": 0.5,
        "bid_p3": 79_999.5, "bid_q3": 0.2,
        "ask_p1": 80_000.1, "ask_q1": 0.7,
        "ask_p2": 80_000.2, "ask_q2": 1.2,
        "ask_p3": 80_000.5, "ask_q3": 0.9,
    })
    out = aggregate_from_snapshot_row(row, n_levels=3)
    # bid_p1 80000 < mid 80000.05; distance = (mid - bid_p1)/mid * 1e4
    # ~ 0.05 / 80000 * 1e4 = 0.0625 bp -> bucket [0, 1)bp
    assert out["bid_0_1bp"] >= 1.0
    assert out["ask_0_1bp"] >= 0.7


def test_bucket_features_symmetric_book_has_zero_imbalance():
    """Mirror-symmetric depth -> all imbalance features ~ 0."""
    mid = 100.0
    bids = [(99.99, 1.0), (99.95, 2.0), (99.50, 5.0)]
    asks = [(100.01, 1.0), (100.05, 2.0), (100.50, 5.0)]
    bucket = aggregate(bids, asks, mid)
    feats = bucket_features(bucket)
    imb_keys = [k for k in feats if k.startswith("f_imb_") or k.startswith("f_cumimb_")]
    for k in imb_keys:
        assert abs(feats[k]) < 1e-9, f"{k} should be 0 for symmetric book"
    # Log-skew of total cumulative depth = 0
    assert abs(feats["f_book_skew"]) < 1e-6


def test_bucket_features_bid_heavy_has_positive_skew():
    mid = 100.0
    bids = [(99.99, 10.0), (99.90, 20.0)]
    asks = [(100.01, 1.0),  (100.10, 2.0)]
    bucket = aggregate(bids, asks, mid)
    feats = bucket_features(bucket)
    assert feats["f_book_skew"] > 0.0
    assert feats["f_cumimb_100bp"] > 0.0


def test_default_edges_have_expected_buckets():
    out = aggregate([], [], mid=100.0)
    expected_bid = {f"bid_{int(DEFAULT_BP_EDGES[i])}_{int(DEFAULT_BP_EDGES[i+1])}bp"
                    for i in range(len(DEFAULT_BP_EDGES) - 1)}
    assert expected_bid.issubset(out.keys())
    assert len(out) == 2 * (len(DEFAULT_BP_EDGES) - 1)
