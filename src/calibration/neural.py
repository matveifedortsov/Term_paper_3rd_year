"""1D-CNN amortized neural calibrator for the Merton jump-diffusion.

Architecture: 4 conv blocks (channels 32-64-128-128, kernel 5,
stride 2, ReLU + LayerNorm) -> global average pool -> 2 dense layers
-> 5 outputs.

Targets: (mu, log_sigma, log1p_lambda, mu_J, log_sigma_J), so all
positive parameters are predicted in log space and inverted at
inference time. Loss: HuberLoss on transformed targets.

Training data: Merton paths simulated at 1-minute frequency for one
day (1440 returns). Parameters drawn from priors:
    mu      ~ Uniform(-0.05, 0.05)        daily drift in log space
    sigma   ~ Uniform(0.01, 0.06)         daily diffusion vol
    lambda  ~ Uniform(5, 100)             jumps per day
    mu_J    ~ Uniform(-0.005, 0.005)      mean jump size
    sigma_J ~ Uniform(0.0005, 0.005)      jump-size std

For inference on real BTC, the 1Hz log-mid grid is downsampled to
1-minute returns (period=60) before being fed to the network.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

LOG = logging.getLogger("neural-cal")

L = 1440  # 1-minute returns per day
SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class TrainConfig:
    n_train: int = 40_000
    n_val: int = 4_000
    n_test: int = 4_000
    batch_size: int = 256
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 20260508


# ----------------------------------------------------------------------
# Synthetic data generator
# ----------------------------------------------------------------------

def sample_priors(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([
        rng.uniform(-0.05, 0.05, n),       # mu
        rng.uniform(0.01, 0.06, n),        # sigma
        rng.uniform(5.0, 100.0, n),        # lambda
        rng.uniform(-0.005, 0.005, n),     # mu_J
        rng.uniform(0.0005, 0.005, n),     # sigma_J
    ])


def simulate_returns_batch(
    params: np.ndarray, rng: np.random.Generator, T: float = 1.0,
) -> np.ndarray:
    """Vectorized simulation of Merton 1-minute return paths.

    Returns shape (n, L). For each row, total time T (in days) is
    discretized into L steps; jump times are sampled as i.i.d. Uniform
    over [0,T), and the corresponding step in [0, L) gets the jump.
    """
    n = params.shape[0]
    mu, sigma, lam, mu_J, sigma_J = params.T
    dt = T / L

    drift = mu[:, None] * dt
    bm = sigma[:, None] * np.sqrt(dt) * rng.standard_normal((n, L))
    R = drift + bm

    # Jumps: count per row ~ Poisson(lam*T), placed at random steps
    counts = rng.poisson(lam * T)
    for i in range(n):
        k = int(counts[i])
        if k == 0:
            continue
        steps = rng.integers(0, L, size=k)
        sizes = rng.normal(mu_J[i], sigma_J[i], size=k)
        np.add.at(R[i], steps, sizes)
    return R


def transform_params(params: np.ndarray) -> np.ndarray:
    """Map (mu, sigma, lam, mu_J, sigma_J) -> CNN target space."""
    out = np.empty_like(params)
    out[:, 0] = params[:, 0]                       # mu (raw)
    out[:, 1] = np.log(params[:, 1])               # log sigma
    out[:, 2] = np.log1p(params[:, 2])             # log(1 + lambda)
    out[:, 3] = params[:, 3]                       # mu_J (raw)
    out[:, 4] = np.log(params[:, 4])               # log sigma_J
    return out


def invert_params(targets: np.ndarray) -> np.ndarray:
    out = np.empty_like(targets)
    out[:, 0] = targets[:, 0]
    out[:, 1] = np.exp(targets[:, 1])
    out[:, 2] = np.expm1(targets[:, 2])
    out[:, 3] = targets[:, 3]
    out[:, 4] = np.exp(targets[:, 4])
    return out


PARAM_NAMES = ["mu", "sigma", "lambda", "mu_J", "sigma_J"]


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 5, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=stride,
                              padding=kernel // 2)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=c_out)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class MertonCNN(nn.Module):
    def __init__(self, n_outputs: int = 5):
        super().__init__()
        self.b1 = ConvBlock(1, 32)
        self.b2 = ConvBlock(32, 64)
        self.b3 = ConvBlock(64, 128)
        self.b4 = ConvBlock(128, 128)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        # x: (B, L) -> (B, 1, L)
        x = x.unsqueeze(1)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = x.mean(dim=-1)  # global avg pool
        return self.head(x)


# ----------------------------------------------------------------------
# Training & evaluation
# ----------------------------------------------------------------------

def make_dataset(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = sample_priors(n, rng)
    R = simulate_returns_batch(params, rng)
    targets = transform_params(params)
    return R.astype(np.float32), targets.astype(np.float32), params


def standardize_returns(R: np.ndarray) -> np.ndarray:
    """Per-path standardization to make CNN scale-invariant.

    The model still has to recover sigma from the SCALE removed here, so
    we feed both standardized returns AND the per-path log std as a
    duplicated channel. To keep the architecture simple we just append
    the per-path log std as a constant scalar feature inside the path.
    """
    s = R.std(axis=1, keepdims=True) + 1e-9
    return R / s


def train_cnn(cfg: TrainConfig, device: str = "cpu") -> dict:
    rng = np.random.default_rng(cfg.seed)
    LOG.info("simulating %d train + %d val paths (L=%d minutes)",
             cfg.n_train, cfg.n_val, L)
    Xtr, Ytr, ptr = make_dataset(cfg.n_train, rng)
    Xva, Yva, pva = make_dataset(cfg.n_val, rng)

    # Per-path standardization + append log_std as constant feature
    Xtr_s = standardize_returns(Xtr)
    Xva_s = standardize_returns(Xva)
    log_std_tr = np.log(Xtr.std(axis=1, keepdims=True) + 1e-9).astype(np.float32)
    log_std_va = np.log(Xva.std(axis=1, keepdims=True) + 1e-9).astype(np.float32)
    Xtr_in = np.concatenate([Xtr_s, np.broadcast_to(log_std_tr, (cfg.n_train, 1))], axis=1)
    Xva_in = np.concatenate([Xva_s, np.broadcast_to(log_std_va, (cfg.n_val, 1))], axis=1)

    # Standardize targets to zero mean / unit std per dimension
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = Ytr.std(axis=0, keepdims=True) + 1e-9
    Ytr_z = ((Ytr - y_mean) / y_std).astype(np.float32)
    Yva_z = ((Yva - y_mean) / y_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(Xtr_in), torch.from_numpy(Ytr_z))
    val_ds = TensorDataset(torch.from_numpy(Xva_in), torch.from_numpy(Yva_z))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0)

    model = MertonCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)

    history = []
    best_val = float("inf")
    best_state = None
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        train_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            train_loss += float(loss) * xb.size(0)
            n += xb.size(0)
        train_loss /= max(1, n)

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                yhat = model(xb)
                loss = loss_fn(yhat, yb)
                val_loss += float(loss) * xb.size(0)
                n += xb.size(0)
        val_loss /= max(1, n)
        epoch_dt = time.perf_counter() - t0

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "time_s": epoch_dt})
        LOG.info("epoch %2d  train=%.4f  val=%.4f  (%.1fs)",
                 epoch, train_loss, val_loss, epoch_dt)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_loss": best_val,
            "y_mean": y_mean, "y_std": y_std}


def predict_params(
    model: MertonCNN,
    R: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """R: (n, L) returns -> (n, 5) params on the natural scale."""
    R = np.asarray(R, dtype=np.float32)
    R_s = standardize_returns(R)
    log_std = np.log(R.std(axis=1, keepdims=True) + 1e-9).astype(np.float32)
    X = np.concatenate([R_s, np.broadcast_to(log_std, (R.shape[0], 1))], axis=1)
    model.eval()
    with torch.no_grad():
        yhat_z = model(torch.from_numpy(X).to(device)).cpu().numpy()
    yhat = yhat_z * y_std + y_mean
    return invert_params(yhat)


# ----------------------------------------------------------------------
# Real-data inference
# ----------------------------------------------------------------------

def returns_per_day_from_book(
    book_dir: Path,
    days: list[str],
    minute_freq: int = 60,
) -> tuple[list[str], np.ndarray]:
    """Build (n_days, L) array of 1-minute returns by downsampling 1Hz log_mid."""
    rows = []
    used_days = []
    for d in days:
        f = book_dir / f"resampled_1s_{d}.parquet"
        if not f.exists():
            continue
        log_mid = pd.read_parquet(f, columns=["log_mid"])["log_mid"].values
        # Take every `minute_freq`-th sample
        sub = log_mid[::minute_freq]
        # Trim or pad to length L+1
        if len(sub) >= L + 1:
            sub = sub[: L + 1]
        else:
            pad = np.full(L + 1 - len(sub), sub[-1])
            sub = np.concatenate([sub, pad])
        r = np.diff(sub)
        rows.append(r)
        used_days.append(d)
    return used_days, np.asarray(rows, dtype=np.float32)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase4"))
    p.add_argument("--n-train", type=int, default=40_000)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(n_train=args.n_train, epochs=args.epochs, batch_size=args.batch_size)

    device = "cpu"
    out = train_cnn(cfg, device=device)
    model: MertonCNN = out["model"]
    history = out["history"]

    pd.DataFrame(history).to_csv(args.out_dir / "neural_train_history.csv", index=False)
    torch.save({
        "model_state": model.state_dict(),
        "y_mean": out["y_mean"], "y_std": out["y_std"],
    }, args.out_dir / "merton_cnn.pt")
    LOG.info("trained model saved (%d epochs, best val=%.4f)",
             cfg.epochs, out["best_val_loss"])

    # ---------- Synthetic test recovery ----------
    rng = np.random.default_rng(cfg.seed + 1)
    Xte, Yte, pte = make_dataset(cfg.n_test, rng)
    t0 = time.perf_counter()
    pte_hat = predict_params(model, Xte, out["y_mean"], out["y_std"], device=device)
    elapsed_pred = time.perf_counter() - t0

    rmse = np.sqrt(np.mean((pte_hat - pte) ** 2, axis=0))
    rel_rmse = rmse / (np.std(pte, axis=0) + 1e-9)
    summary = {
        "n_test": cfg.n_test,
        "rmse_per_param": dict(zip(PARAM_NAMES, [float(v) for v in rmse])),
        "rel_rmse_per_param": dict(zip(PARAM_NAMES, [float(v) for v in rel_rmse])),
        "time_inference_s": elapsed_pred,
        "time_per_path_us": elapsed_pred / cfg.n_test * 1e6,
    }
    LOG.info("=== synthetic-test recovery ===")
    LOG.info(json.dumps(summary, indent=2))
    (args.out_dir / "neural_synthetic_eval.json").write_text(json.dumps(summary, indent=2))

    # Save synthetic predictions for later plots
    pd.DataFrame({
        **{f"true_{n}": pte[:, i] for i, n in enumerate(PARAM_NAMES)},
        **{f"pred_{n}": pte_hat[:, i] for i, n in enumerate(PARAM_NAMES)},
    }).to_csv(args.out_dir / "neural_synthetic_preds.csv", index=False)

    # ---------- Real BTC inference ----------
    days = [
        "2024-03-15", "2024-03-16", "2024-03-17", "2024-03-18", "2024-03-19",
        "2024-03-20", "2024-03-21", "2024-03-22", "2024-03-23", "2024-03-24",
        "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-03-29",
    ]
    used_days, R_real = returns_per_day_from_book(args.book_dir, days)
    LOG.info("real-data inference on %d days, returns shape %s", len(used_days), R_real.shape)
    p_real = predict_params(model, R_real, out["y_mean"], out["y_std"], device=device)
    df_real = pd.DataFrame(p_real, columns=PARAM_NAMES)
    df_real.insert(0, "day", used_days)
    df_real.to_csv(args.out_dir / "neural_real_per_day.csv", index=False)
    print("\n=== Neural calibration on real BTC (per day) ===")
    print(df_real.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
