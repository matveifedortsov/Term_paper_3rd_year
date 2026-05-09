"""Barndorff-Nielsen & Shephard (2006) bipower-variation jump test.

Per-day binary test:

    RV  = sum r_i^2                                 (realized variance)
    BV  = mu_1^{-2} * sum |r_i| |r_{i-1}|           (bipower variation)
    Z   = sqrt(n) * (1 - BV/RV) / sqrt(theta * (TQ / BV^2))
    where mu_1 = sqrt(2/pi),
          theta = (pi^2 / 4) + pi - 5,
          TQ = mu_4^{-3} * n * sum |r_{i-2}| |r_{i-1}| |r_i|^{4/3}_{...}
                  (tripower quarticity; we approximate via realized
                   quadpower variation for stability)

Z under H0 (no jumps) ~ N(0, 1). Reject if Z > z_{1-alpha}.

This is a per-DAY indicator; compare against per-day persistence-based
"any large move" ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MU_1 = np.sqrt(2.0 / np.pi)
MU_43 = 2.0 ** (2.0 / 3.0) * 0.8856  # E[|N|^{4/3}], approximate constant


def bns_jump_test(Y: np.ndarray) -> dict:
    Y = np.asarray(Y, dtype=float)
    r = np.diff(Y)
    n = len(r)
    if n < 5:
        return {"Z": float("nan"), "RV": float("nan"), "BV": float("nan"),
                "n": n, "reject": False}

    abs_r = np.abs(r)
    RV = float(np.sum(r * r))
    BV = float(MU_1 ** -2 * np.sum(abs_r[:-1] * abs_r[1:]))

    # Realized quad-power for variance of BV (more stable than tripower)
    # QP = mu_1^{-4} * n * sum |r_{i-3}| |r_{i-2}| |r_{i-1}| |r_i|
    if n >= 4:
        qp = (MU_1 ** -4) * n * np.sum(
            abs_r[:-3] * abs_r[1:-2] * abs_r[2:-1] * abs_r[3:]
        )
    else:
        qp = float("nan")

    theta = (np.pi ** 2) / 4.0 + np.pi - 5.0
    if BV <= 0 or qp <= 0:
        return {"Z": float("nan"), "RV": RV, "BV": BV, "QP": qp,
                "n": n, "reject": False}

    Z = np.sqrt(n) * (1.0 - BV / RV) / np.sqrt(theta * qp / (BV * BV))
    return {"Z": float(Z), "RV": RV, "BV": BV, "QP": qp, "n": n,
            "reject": bool(Z > 1.96)}


def bns_per_day(books: dict[str, pd.DataFrame], alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    z_crit = 1.6449 if alpha == 0.05 else 1.96
    for d, b in books.items():
        out = bns_jump_test(b["log_mid"].values)
        out["day"] = d
        out["reject"] = out["Z"] > z_crit if not np.isnan(out["Z"]) else False
        rows.append(out)
    return pd.DataFrame(rows)
