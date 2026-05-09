"""Phase 1 deliverable: LOMN size/power Monte Carlo.

Runs the full Monte Carlo grid, writes CSV + LaTeX tables, and produces
the three diagnostic figures referenced in the term paper:
    - figures/noise_distribution.png
    - figures/return_distribution.png
    - figures/sample_path.png
plus
    - figures/size_calibration.png  (T_max distribution under H0)
    - figures/power_curves.png      (rejection rate vs jump size)

Outputs land under results/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lomn.detector import lomn_detector, optimal_block_size
from src.lomn.monte_carlo import Cell, MCConfig, run_monte_carlo
from src.lomn.simulation import JumpDiffusionParams, simulate_path

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

NOISE_SCALES = [0.0005, 0.001, 0.002, 0.005]
JUMP_SIZES = [0.0, 0.0025, 0.005, 0.01, 0.02]


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def write_pivot_tables(df: pd.DataFrame) -> None:
    df.to_csv(TABLES / "phase1_full_grid.csv", index=False)

    for col, fname in [
        ("rejection_rate_gumbel", "size_power_raw_gumbel"),
        ("rejection_rate_calibrated", "size_power_calibrated"),
    ]:
        pivot = df.pivot(index="noise_scale", columns="jump_size", values=col)
        pivot.to_csv(TABLES / f"{fname}.csv")
        caption_text = (
            "Size (jump size = 0) and power (jump size > 0) of the LOMN "
            "detector at nominal level $\\alpha = 0.05$ "
            + (
                "under the asymptotic Gumbel critical value."
                if col.endswith("gumbel")
                else "under empirically calibrated critical values."
            )
        )
        latex = pivot.to_latex(
            float_format=lambda x: f"{x:.3f}",
            caption=caption_text,
            label=f"tab:{fname}",
        )
        (TABLES / f"{fname}.tex").write_text(latex, encoding="utf-8")


# ----------------------------------------------------------------------
# Diagnostic figures
# ----------------------------------------------------------------------

def make_diagnostic_figures(cfg: MCConfig) -> None:
    rng = np.random.default_rng(cfg.base_seed + 99_999)
    h_n = optimal_block_size(cfg.n, c=cfg.block_constant)

    # Sample path with multiple jumps for visualization
    params_demo = JumpDiffusionParams(
        mu=0.0,
        sigma=cfg.sigma,
        fixed_jump_times=(0.2, 0.45, 0.72),
        fixed_jump_sizes=(0.012, -0.015, 0.008),
    )
    sim = simulate_path(cfg.n, cfg.T, params_demo, noise_scale=0.001, rng=rng)
    res = lomn_detector(sim.observed, h_n=h_n)

    # ---- Noise distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sim.noise, bins=60, color="#2a9d8f", edgecolor="white")
    ax.axvline(0.0, ls="--", color="black", label="lower bound (0)")
    ax.set_xlabel("noise value $\\varepsilon$")
    ax.set_ylabel("frequency")
    ax.set_title(f"Microstructure noise (one-sided exponential), $q={0.001}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "noise_distribution.png", dpi=150)
    plt.close(fig)

    # ---- Return distribution
    rets = np.diff(sim.observed)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rets, bins=80, color="#9b5de5", edgecolor="white")
    ax.set_xlabel("log return")
    ax.set_ylabel("frequency")
    ax.set_title("Observed log returns (jumps + LOMN noise)")
    fig.tight_layout()
    fig.savefig(FIGURES / "return_distribution.png", dpi=150)
    plt.close(fig)

    # ---- Sample path
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sim.times, sim.efficient, color="#264653", lw=1.0, label="efficient $X_t$")
    ax.plot(sim.times, sim.observed, color="#e76f51", lw=0.4, alpha=0.8,
            label="observed $Y_t$")
    if len(res.candidate_obs_idx):
        idx = res.candidate_obs_idx
        idx = idx[idx < len(sim.times)]
        ax.scatter(sim.times[idx], sim.observed[idx],
                   color="red", s=20, zorder=5, label="LOMN candidates")
    for jt in sim.jump_times:
        ax.axvline(jt, color="#588157", ls=":", alpha=0.5)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("log price")
    ax.set_title("Simulated path with LOMN noise and detected jump candidates")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "sample_path.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Result figures
# ----------------------------------------------------------------------

def make_result_figures(df: pd.DataFrame) -> None:
    # Power curves: rejection rate vs jump size, one line per noise level
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, col, title in [
        (axes[0], "rejection_rate_gumbel", "Asymptotic Gumbel CV"),
        (axes[1], "rejection_rate_calibrated", "Empirically calibrated CV"),
    ]:
        for q, sub in df.groupby("noise_scale"):
            sub = sub.sort_values("jump_size")
            ax.plot(sub["jump_size"], sub[col], marker="o", label=f"q={q}")
        ax.axhline(0.05, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("jump size $\\delta$")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rejection rate")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("LOMN detector size/power curves (n=23,400, $\\sigma$=0.03, T=1)")
    fig.tight_layout()
    fig.savefig(FIGURES / "power_curves.png", dpi=150)
    plt.close(fig)


def make_size_calibration_figure(cfg: MCConfig) -> None:
    """Distribution of T_max under H0 vs Gumbel approximation."""
    h_n = optimal_block_size(cfg.n, c=cfg.block_constant)
    m_eff = (cfg.n + 1) // h_n - 1

    rng = np.random.default_rng(cfg.base_seed + 7777)
    params = JumpDiffusionParams(mu=0.0, sigma=cfg.sigma)
    n_paths = max(cfg.n_reps, 500)
    tmax = np.empty(n_paths)
    q = 0.001
    for r in range(n_paths):
        sim = simulate_path(cfg.n, cfg.T, params, q, rng)
        tmax[r] = lomn_detector(sim.observed, h_n=h_n).T_max

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(tmax, bins=40, density=True, color="#457b9d", edgecolor="white", alpha=0.85,
            label=f"empirical T_max (n_reps={n_paths})")
    a_m = np.sqrt(2 * np.log(m_eff))
    b_m = 1 / a_m
    xs = np.linspace(tmax.min(), tmax.max(), 400)
    z = (xs - a_m) / b_m
    pdf_gumbel = np.exp(-z - np.exp(-z)) / b_m
    ax.plot(xs, pdf_gumbel, color="black", lw=1.5, label="Gumbel approx (max |Z|)")
    cv = a_m + b_m * (-np.log(-np.log(0.975)))
    ax.axvline(cv, color="red", ls="--", label=f"Gumbel CV ({cv:.2f})")
    cv_cal = float(np.quantile(tmax, 0.95))
    ax.axvline(cv_cal, color="green", ls="--", label=f"calibrated CV ({cv_cal:.2f})")
    ax.set_xlabel("T_max under H0")
    ax.set_ylabel("density")
    ax.set_title(f"Calibration of LOMN test statistic under H0  (q={q}, m={m_eff})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "size_calibration.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    cfg = MCConfig(
        n=23_400,
        T=1.0,
        sigma=0.03,
        mu=0.0,
        n_reps=500,
        alpha=0.05,
        block_constant=1.0,
        base_seed=20_260_508,
    )
    print(
        f"Phase 1 config: n={cfg.n}, T={cfg.T}, sigma={cfg.sigma}, "
        f"reps={cfg.n_reps}, block_c={cfg.block_constant}"
    )
    cells = [Cell(noise_scale=q, jump_size=d) for q in NOISE_SCALES for d in JUMP_SIZES]
    print(f"Total cells: {len(cells)}")

    t0 = time.perf_counter()
    df = run_monte_carlo(cfg, cells, progress=True)
    elapsed = time.perf_counter() - t0
    print(f"\nMonte Carlo done in {elapsed:.1f} s")

    write_pivot_tables(df)
    make_diagnostic_figures(cfg)
    make_size_calibration_figure(cfg)
    make_result_figures(df)

    # Console summary
    print("\n=== Size table (rejection rate at jump_size = 0) ===")
    sizes = df[df.jump_size == 0.0][
        ["noise_scale", "rejection_rate_gumbel", "rejection_rate_calibrated", "calibrated_cv"]
    ].set_index("noise_scale")
    print(sizes.to_string())

    print("\n=== Power table (calibrated, rows=q, cols=delta) ===")
    pwr = df.pivot(index="noise_scale", columns="jump_size", values="rejection_rate_calibrated")
    print(pwr.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
