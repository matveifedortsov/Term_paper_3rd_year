"""Percentage-distance (basis-points) book aggregation.

Schema 2 from the design discussion: instead of feeding raw L1-L20
levels into the classifier, aggregate resting depth into log-spaced
basis-point buckets relative to the mid price.

Advantages over raw levels:
    1. Tick-size invariant — same buckets for BTC, ETH, SOL.
    2. Robust to spoofing / empty levels — a missing tick contributes 0.
    3. Lower feature dimensionality (8 buckets per side instead of 20).
    4. Cross-asset training becomes a one-line change.

Limitations:
    - With L20 partial-book streams, only the inner ~5 buckets (up to
      ~25 bp from mid) are populated. The tail buckets light up when
      reconstruct_book.py wires in full-depth incremental L2 data.

Two-layer API:
    aggregate(bids, asks, mid, bp_edges) -> dict of 2 * (K-1) buckets
        Pure-numpy, works on any (price, quantity) list-of-pairs.
    aggregate_from_snapshot_row(row, n_levels, bp_edges) -> dict
        Convenience: pulls bid_p1..N / bid_q1..N from a DataFrame row.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


DEFAULT_BP_EDGES = (0, 1, 2, 5, 10, 25, 50, 100, 500)


def aggregate(
    bids: Iterable[tuple[float, float]],
    asks: Iterable[tuple[float, float]],
    mid: float,
    bp_edges: Sequence[float] = DEFAULT_BP_EDGES,
) -> dict[str, float]:
    """Bin resting depth by bp distance from mid.

    Parameters
    ----------
    bids : iterable of (price, quantity) tuples. Order is not required;
        prices below ``mid * (1 - max(bp_edges)/10000)`` are dropped.
    asks : iterable of (price, quantity) tuples on the ask side.
    mid : reference mid-price.
    bp_edges : monotonically-increasing distance edges in basis points.
        Bucket i covers [bp_edges[i], bp_edges[i+1]).  Defaults to
        DEFAULT_BP_EDGES = (0, 1, 2, 5, 10, 25, 50, 100, 500).

    Returns
    -------
    dict with 2 * (len(bp_edges) - 1) entries:
        "bid_{lo}_{hi}bp" : float
        "ask_{lo}_{hi}bp" : float
    """
    if mid <= 0:
        raise ValueError("mid must be positive")
    bp_edges = np.asarray(bp_edges, dtype=float)
    if not np.all(np.diff(bp_edges) > 0):
        raise ValueError("bp_edges must be strictly increasing")
    n_buckets = len(bp_edges) - 1

    bid_buckets = np.zeros(n_buckets)
    ask_buckets = np.zeros(n_buckets)
    bp_max = float(bp_edges[-1])

    inv_mid = 10000.0 / mid

    for price, qty in bids:
        if price <= 0 or qty <= 0 or not np.isfinite(qty):
            continue
        bp_dist = (mid - price) * inv_mid
        if bp_dist < bp_edges[0] or bp_dist >= bp_max:
            continue
        idx = int(np.searchsorted(bp_edges, bp_dist, side="right") - 1)
        if 0 <= idx < n_buckets:
            bid_buckets[idx] += qty

    for price, qty in asks:
        if price <= 0 or qty <= 0 or not np.isfinite(qty):
            continue
        bp_dist = (price - mid) * inv_mid
        if bp_dist < bp_edges[0] or bp_dist >= bp_max:
            continue
        idx = int(np.searchsorted(bp_edges, bp_dist, side="right") - 1)
        if 0 <= idx < n_buckets:
            ask_buckets[idx] += qty

    out: dict[str, float] = {}
    for i in range(n_buckets):
        lo = int(bp_edges[i])
        hi = int(bp_edges[i + 1])
        out[f"bid_{lo}_{hi}bp"] = float(bid_buckets[i])
        out[f"ask_{lo}_{hi}bp"] = float(ask_buckets[i])
    return out


def aggregate_from_snapshot_row(
    row,
    n_levels: int = 20,
    bp_edges: Sequence[float] = DEFAULT_BP_EDGES,
) -> dict[str, float]:
    """Aggregate one row of an L<N> snapshot DataFrame into bp-buckets.

    Expects columns: ``bid_p1`` .. ``bid_p<n_levels>``,
                     ``bid_q1`` .. ``bid_q<n_levels>``,
                     ``ask_p1`` .. ``ask_p<n_levels>``,
                     ``ask_q1`` .. ``ask_q<n_levels>``.
    Computes mid as 0.5*(bid_p1+ask_p1).
    """
    bid_p1 = float(row["bid_p1"])
    ask_p1 = float(row["ask_p1"])
    mid = 0.5 * (bid_p1 + ask_p1)

    bids = [(float(row[f"bid_p{i}"]), float(row[f"bid_q{i}"])) for i in range(1, n_levels + 1)]
    asks = [(float(row[f"ask_p{i}"]), float(row[f"ask_q{i}"])) for i in range(1, n_levels + 1)]
    return aggregate(bids, asks, mid, bp_edges)


# ----------------------------------------------------------------------
# Derived features from a bucketed snapshot
# ----------------------------------------------------------------------

def bucket_features(
    bucket_row: dict[str, float] | "object",
    bp_edges: Sequence[float] = DEFAULT_BP_EDGES,
) -> dict[str, float]:
    """Compute imbalance / cumulative-depth / slope / skew features.

    Input: dict (or DataFrame row) keyed by ``bid_{lo}_{hi}bp`` /
    ``ask_{lo}_{hi}bp`` as produced by ``aggregate(...)``.
    """
    edges = np.asarray(bp_edges, dtype=float)
    n = len(edges) - 1

    bid = np.array([float(bucket_row[f"bid_{int(edges[i])}_{int(edges[i+1])}bp"])
                    for i in range(n)])
    ask = np.array([float(bucket_row[f"ask_{int(edges[i])}_{int(edges[i+1])}bp"])
                    for i in range(n)])
    cum_bid = np.cumsum(bid)
    cum_ask = np.cumsum(ask)

    feat: dict[str, float] = {}

    # Per-bucket imbalance
    for i in range(n):
        lo = int(edges[i]); hi = int(edges[i + 1])
        b, a = bid[i], ask[i]
        feat[f"f_imb_{lo}_{hi}bp"] = float((b - a) / (b + a)) if (b + a) > 0 else 0.0

    # Cumulative imbalance up to each upper edge
    for i in range(n):
        hi = int(edges[i + 1])
        b, a = cum_bid[i], cum_ask[i]
        feat[f"f_cumimb_{hi}bp"] = float((b - a) / (b + a)) if (b + a) > 0 else 0.0

    # Total depth up to a wide cone (e.g. 100 bp) — useful as a regime marker
    feat["f_total_depth_bid_100bp"] = float(
        cum_bid[np.searchsorted(edges, 100.0, side="right") - 1]
    ) if 100.0 <= edges[-1] else float(cum_bid[-1])
    feat["f_total_depth_ask_100bp"] = float(
        cum_ask[np.searchsorted(edges, 100.0, side="right") - 1]
    ) if 100.0 <= edges[-1] else float(cum_ask[-1])

    # Book slope: how fast cumulative depth grows from inner to outer
    # (centres of edges[1] and edges[-1] in bp).
    inner_bp = max(float(edges[1]), 1e-6)
    outer_bp = float(edges[-1])
    span = max(outer_bp - inner_bp, 1e-6)
    feat["f_book_slope_bid"] = float((cum_bid[-1] - cum_bid[0]) / span)
    feat["f_book_slope_ask"] = float((cum_ask[-1] - cum_ask[0]) / span)

    # Log skew of total depth (positive = thicker bid side)
    feat["f_book_skew"] = float(
        np.log((cum_bid[-1] + 1e-9) / (cum_ask[-1] + 1e-9))
    )

    # Concentration: share of total depth within 5 bp of mid (early bucket)
    inner_idx = max(0, np.searchsorted(edges, 5.0, side="right") - 2)
    inner_b = cum_bid[inner_idx] if inner_idx < n else cum_bid[-1]
    inner_a = cum_ask[inner_idx] if inner_idx < n else cum_ask[-1]
    feat["f_inner5_share_bid"] = float(inner_b / (cum_bid[-1] + 1e-9))
    feat["f_inner5_share_ask"] = float(inner_a / (cum_ask[-1] + 1e-9))

    return feat
