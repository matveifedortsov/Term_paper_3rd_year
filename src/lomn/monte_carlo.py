"""Monte Carlo size and power experiments for the LOMN detector.

For each (noise_scale q, jump_size delta) cell we simulate `n_reps`
independent paths and record:
    - whether the detector rejects H0 (raw Gumbel CV)
    - whether the detector rejects under an empirically calibrated CV
    - the magnitude of the largest standardized statistic

Under jump_size=0 the rejection rate is the test SIZE; under
jump_size>0 it is the POWER against a single deterministic jump
inserted at the midpoint of the sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .detector import gumbel_critical_value, lomn_detector, optimal_block_size
from .simulation import JumpDiffusionParams, simulate_path


@dataclass(frozen=True)
class Cell:
    noise_scale: float
    jump_size: float


@dataclass(frozen=True)
class MCConfig:
    n: int = 23_400
    T: float = 1.0
    sigma: float = 0.4
    mu: float = 0.0
    n_reps: int = 500
    alpha: float = 0.05
    block_constant: float = 1.0
    base_seed: int = 20260508


def _simulate_one(
    cfg: MCConfig,
    cell: Cell,
    h_n: int,
    rng: np.random.Generator,
    crit_calibrated: float | None,
) -> dict:
    if cell.jump_size == 0.0:
        params = JumpDiffusionParams(mu=cfg.mu, sigma=cfg.sigma)
    else:
        params = JumpDiffusionParams(
            mu=cfg.mu,
            sigma=cfg.sigma,
            fixed_jump_times=(cfg.T / 2.0,),
            fixed_jump_sizes=(cell.jump_size,),
        )
    sim = simulate_path(cfg.n, cfg.T, params, cell.noise_scale, rng)
    res = lomn_detector(sim.observed, h_n=h_n, alpha=cfg.alpha)
    res_cal = (
        lomn_detector(sim.observed, h_n=h_n, critical_value=crit_calibrated)
        if crit_calibrated is not None
        else None
    )
    return {
        "T_max": res.T_max,
        "reject_gumbel": res.reject,
        "reject_calibrated": (res_cal.reject if res_cal is not None else None),
        "n_candidates": len(res.candidate_block_idx),
    }


def run_monte_carlo(
    cfg: MCConfig,
    cells: Iterable[Cell],
    progress: bool = False,
) -> pd.DataFrame:
    """Run Monte Carlo over (q, delta) cells.

    Two-pass design:
      Pass 1: under H0 (jump_size=0) only, collect T_max distribution
              per noise scale to compute empirical 1-alpha quantile.
      Pass 2: full grid, applying both raw Gumbel CV and the calibrated
              CV from Pass 1.
    """
    cells = list(cells)
    h_n = optimal_block_size(cfg.n, c=cfg.block_constant)
    m_eff = (cfg.n + 1) // h_n - 1
    crit_gumbel = gumbel_critical_value(m_eff, alpha=cfg.alpha, two_sided=True)

    # ---------- Pass 1: calibrate per-noise threshold under H0 ----------
    noise_scales = sorted({c.noise_scale for c in cells})
    calibrated: dict[float, float] = {}
    for i, q in enumerate(noise_scales):
        if progress:
            print(f"[calib {i+1}/{len(noise_scales)}] q={q}")
        rng = np.random.default_rng(cfg.base_seed + i)
        tmax_h0 = np.empty(cfg.n_reps)
        params = JumpDiffusionParams(mu=cfg.mu, sigma=cfg.sigma)
        for r in range(cfg.n_reps):
            sim = simulate_path(cfg.n, cfg.T, params, q, rng)
            tmax_h0[r] = lomn_detector(sim.observed, h_n=h_n, alpha=cfg.alpha).T_max
        calibrated[q] = float(np.quantile(tmax_h0, 1.0 - cfg.alpha))

    # ---------- Pass 2: full grid using calibrated thresholds ----------
    rows = []
    for ci, cell in enumerate(cells):
        if progress:
            print(f"[cell {ci+1}/{len(cells)}] q={cell.noise_scale}, delta={cell.jump_size}")
        rng = np.random.default_rng(cfg.base_seed + 1_000 + ci)
        rejects_g = 0
        rejects_c = 0
        tmax_acc = np.empty(cfg.n_reps)
        for r in range(cfg.n_reps):
            out = _simulate_one(cfg, cell, h_n, rng, calibrated[cell.noise_scale])
            tmax_acc[r] = out["T_max"]
            rejects_g += int(out["reject_gumbel"])
            rejects_c += int(out["reject_calibrated"])
        rows.append({
            "noise_scale": cell.noise_scale,
            "jump_size": cell.jump_size,
            "n_reps": cfg.n_reps,
            "rejection_rate_gumbel": rejects_g / cfg.n_reps,
            "rejection_rate_calibrated": rejects_c / cfg.n_reps,
            "tmax_mean": float(tmax_acc.mean()),
            "tmax_q95": float(np.quantile(tmax_acc, 0.95)),
            "calibrated_cv": calibrated[cell.noise_scale],
            "gumbel_cv": crit_gumbel,
            "h_n": h_n,
            "m_blocks_eff": m_eff,
        })
    return pd.DataFrame(rows)
