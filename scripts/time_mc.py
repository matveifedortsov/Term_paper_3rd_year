"""Estimate time per Monte Carlo cell."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lomn.detector import lomn_detector, optimal_block_size
from src.lomn.simulation import JumpDiffusionParams, simulate_path


def main() -> None:
    n = 23_400
    T = 1.0
    sigma = 0.03
    h_n = optimal_block_size(n)
    rng = np.random.default_rng(0)
    params = JumpDiffusionParams(mu=0.0, sigma=sigma)

    n_warm = 5
    n_meas = 50
    for _ in range(n_warm):
        sim = simulate_path(n, T, params, 0.001, rng)
        lomn_detector(sim.observed, h_n=h_n)

    t0 = time.perf_counter()
    for _ in range(n_meas):
        sim = simulate_path(n, T, params, 0.001, rng)
        lomn_detector(sim.observed, h_n=h_n)
    elapsed = time.perf_counter() - t0
    per_rep = elapsed / n_meas
    print(f"per-replication: {per_rep*1000:.2f} ms")
    print(f"500 reps per cell: {per_rep*500:.2f} s")
    n_cells = 4 * 5
    print(f"{n_cells} cells x 500 reps + calib: {per_rep*(500*n_cells + 500*4):.1f} s")


if __name__ == "__main__":
    main()
