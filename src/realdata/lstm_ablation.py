"""LSTM baseline for Stage 2 — sequence-model ablation.

Trains a small LSTM on the raw log-return sequence centered at each
candidate, instead of on engineered features. The point is to test
whether a deep sequence model can match XGBoost on this sample size.

The literature consensus (Shwartz-Ziv & Armon 2022; Grinsztajn et al.
2022) is that GBDTs beat deep sequence models on tabular regimes
under ~10k samples. Our train set is ~1k. We expect LSTM to lose by a
small but measurable margin and want to demonstrate that empirically.

Inputs per candidate tau:
    - log-return sequence over [tau-W, tau+W], length 2*W+1, W from cfg
    - the LOMN test stat (concatenated as a static feature at the tail
      of the LSTM hidden state for the head)

Output: probability of "real jump" (binary).

Outputs:
    results/phase_a8/lstm_metrics.json      training + test metrics
    results/phase_a8/lstm_history.csv
    results/phase_a8/roc_lstm_vs_xgb.png    LSTM vs trained XGBoost on test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from src.config import config
from src.realdata.train_xgb import FEATURE_COLS, TEST_DAYS, TRAIN_DAYS, fpr_at_recall

LOG = logging.getLogger("lstm-abl")


@dataclass(frozen=True)
class LSTMAblationConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    window_seconds: int = 120  # +/- 60s
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    seed: int = 42


def load_book_log_returns(book_dir: Path, day: str) -> np.ndarray:
    """Return log-mid 1Hz log-returns for one day; length 86399."""
    f = book_dir / f"resampled_1s_{day}.parquet"
    book = pd.read_parquet(f, columns=["log_mid"])
    return np.diff(book["log_mid"].values).astype(np.float32)


def build_sequence_dataset(
    features: pd.DataFrame, book_dir: Path, half_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each labeled candidate, build a (2*half_window) return slice.

    Edges are zero-padded to keep shape uniform. Also returns the
    LOMN abs_std value as a 1-D static feature.
    """
    by_day_returns: dict[str, np.ndarray] = {}
    seq_list, lomn_list, y_list, day_list = [], [], [], []
    L_seq = 2 * half_window
    for _, row in features.iterrows():
        d = row["day"]
        if d not in by_day_returns:
            by_day_returns[d] = load_book_log_returns(book_dir, d)
        r = by_day_returns[d]
        idx = int(row["obs_idx"])
        lo = idx - half_window
        hi = idx + half_window
        if lo < 0:
            seg = np.concatenate([np.zeros(-lo, dtype=np.float32), r[:hi]])
        elif hi > len(r):
            seg = np.concatenate([r[lo:], np.zeros(hi - len(r), dtype=np.float32)])
        else:
            seg = r[lo:hi]
        if len(seg) != L_seq:
            seg = np.zeros(L_seq, dtype=np.float32)
        seq_list.append(seg)
        lomn_list.append(float(row["f_lomn_abs_std"]))
        y_list.append(int(row["label"]))
        day_list.append(d)
    return (
        np.asarray(seq_list, dtype=np.float32),
        np.asarray(lomn_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(day_list, dtype=object),
    )


class LSTMRefiner(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float,
                 use_lomn_static: bool = True):
        super().__init__()
        self.use_lomn_static = use_lomn_static
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        head_in = hidden_size + (1 if use_lomn_static else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x_seq = x_seq.unsqueeze(-1)  # (B, L, 1)
        out, (h_n, _) = self.lstm(x_seq)
        last = h_n[-1]  # (B, hidden)
        if self.use_lomn_static:
            z = torch.cat([last, x_static.unsqueeze(-1)], dim=-1)
        else:
            z = last
        return self.head(z).squeeze(-1)


def train_lstm(
    Xtr_seq: np.ndarray, Xtr_stat: np.ndarray, ytr: np.ndarray,
    Xva_seq: np.ndarray, Xva_stat: np.ndarray, yva: np.ndarray,
    cfg: LSTMAblationConfig,
    use_lomn_static: bool = True,
) -> tuple[LSTMRefiner, list[dict]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = LSTMRefiner(cfg.hidden_size, cfg.num_layers, cfg.dropout,
                        use_lomn_static=use_lomn_static)
    pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    train_ds = TensorDataset(
        torch.from_numpy(Xtr_seq), torch.from_numpy(Xtr_stat),
        torch.from_numpy(ytr.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(Xva_seq), torch.from_numpy(Xva_stat),
        torch.from_numpy(yva.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    history = []
    best_val = float("inf")
    best_state = None
    patience = 0
    for ep in range(cfg.epochs):
        model.train()
        tr_loss, n = 0.0, 0
        for xs, xt, y in train_loader:
            opt.zero_grad()
            logits = model(xs, xt)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * xs.size(0); n += xs.size(0)
        tr_loss /= max(1, n)

        model.eval()
        va_loss, n = 0.0, 0
        scores = []
        ys_v = []
        with torch.no_grad():
            for xs, xt, y in val_loader:
                logits = model(xs, xt)
                loss = loss_fn(logits, y)
                va_loss += float(loss) * xs.size(0); n += xs.size(0)
                scores.append(torch.sigmoid(logits).numpy())
                ys_v.append(y.numpy())
        va_loss /= max(1, n)
        scores = np.concatenate(scores); ys_v = np.concatenate(ys_v)
        try:
            va_auc = roc_auc_score(ys_v, scores)
        except ValueError:
            va_auc = float("nan")
        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_auc": va_auc})
        LOG.info("ep %2d  tr=%.4f  va=%.4f  auc=%.4f", ep, tr_loss, va_loss, va_auc)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                LOG.info("early stop at ep %d", ep)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    raw_cfg = config()
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path,
                   default=Path("data/interim/features_labeled.parquet"))
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--xgb-model", type=Path,
                   default=Path("results/phase3/xgb_lomn_refiner.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results/phase_a8"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = LSTMAblationConfig(
        hidden_size=int(raw_cfg["lstm"]["hidden_size"]),
        num_layers=int(raw_cfg["lstm"]["num_layers"]),
        dropout=float(raw_cfg["lstm"]["dropout"]),
        window_seconds=int(raw_cfg["lstm"]["window_seconds"]),
        batch_size=int(raw_cfg["lstm"]["batch_size"]),
        epochs=int(raw_cfg["lstm"]["epochs"]),
        lr=float(raw_cfg["lstm"]["lr"]),
        seed=int(raw_cfg["lstm"]["seed"]),
    )
    half_window = cfg.window_seconds // 2
    LOG.info("config: %s; sequence length = %d", cfg, 2 * half_window)

    feats = pd.read_parquet(args.features)
    labeled = feats[feats["label"] != -1].copy()

    train_mask = labeled["day"].isin(TRAIN_DAYS)
    test_mask = labeled["day"].isin(TEST_DAYS)
    LOG.info("train candidates: %d  test: %d",
             int(train_mask.sum()), int(test_mask.sum()))

    seq, lomn, y, day = build_sequence_dataset(labeled, args.book_dir, half_window)
    is_train = np.isin(day, TRAIN_DAYS)
    is_test = np.isin(day, TEST_DAYS)

    Xtr_seq, Xtr_stat, ytr = seq[is_train], lomn[is_train], y[is_train]
    Xte_seq, Xte_stat, yte = seq[is_test],  lomn[is_test],  y[is_test]

    val_split = int(0.85 * len(ytr))
    perm = np.random.default_rng(cfg.seed).permutation(len(ytr))
    Xtr_seq, Xtr_stat, ytr = Xtr_seq[perm], Xtr_stat[perm], ytr[perm]
    Xtr2_seq, Xva_seq = Xtr_seq[:val_split], Xtr_seq[val_split:]
    Xtr2_stat, Xva_stat = Xtr_stat[:val_split], Xtr_stat[val_split:]
    ytr2, yva = ytr[:val_split], ytr[val_split:]

    # Train two LSTMs: with and without the static LOMN feature.
    LOG.info("=== variant 1: LSTM(seq) + static LOMN ===")
    t0 = time.perf_counter()
    model_with, hist_with = train_lstm(
        Xtr2_seq, Xtr2_stat, ytr2, Xva_seq, Xva_stat, yva, cfg,
        use_lomn_static=True,
    )
    t_with = time.perf_counter() - t0

    LOG.info("=== variant 2: LSTM(seq) ALONE (no LOMN feature) ===")
    t0 = time.perf_counter()
    model_solo, hist_solo = train_lstm(
        Xtr2_seq, Xtr2_stat, ytr2, Xva_seq, Xva_stat, yva, cfg,
        use_lomn_static=False,
    )
    t_solo = time.perf_counter() - t0
    elapsed_train = t_with + t_solo
    LOG.info("LSTM training: %.1fs (with-LOMN: %.1fs, sequence-only: %.1fs)",
             elapsed_train, t_with, t_solo)

    pd.DataFrame(hist_with).to_csv(args.out_dir / "lstm_history_with_lomn.csv", index=False)
    pd.DataFrame(hist_solo).to_csv(args.out_dir / "lstm_history_seq_only.csv", index=False)
    torch.save(model_with.state_dict(), args.out_dir / "lstm_model_with_lomn.pt")
    torch.save(model_solo.state_dict(), args.out_dir / "lstm_model_seq_only.pt")

    # ----- Test evaluation: both LSTMs vs trained XGBoost vs raw LOMN -----
    model_with.eval(); model_solo.eval()
    model = model_with  # keep for back-compat below
    with torch.no_grad():
        logits = model_with(torch.from_numpy(Xte_seq), torch.from_numpy(Xte_stat))
        p_lstm = torch.sigmoid(logits).numpy()
        logits_solo = model_solo(torch.from_numpy(Xte_seq), torch.from_numpy(Xte_stat))
        p_lstm_solo = torch.sigmoid(logits_solo).numpy()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(args.xgb_model)
    Xte_feats = labeled[test_mask][FEATURE_COLS].values
    p_xgb = xgb_model.predict_proba(Xte_feats)[:, 1]
    raw_lomn = labeled[test_mask]["f_lomn_abs_std"].values

    metrics = {}
    for name, score in [
        ("lstm_with_lomn", p_lstm),
        ("lstm_seq_only",  p_lstm_solo),
        ("xgb",            p_xgb),
        ("raw_lomn",       raw_lomn),
    ]:
        try:
            auc = float(roc_auc_score(yte, score))
            ap = float(average_precision_score(yte, score))
            fr = fpr_at_recall(yte, score, 0.90)
        except ValueError:
            auc = ap = float("nan"); fr = {"fpr": float("nan")}
        metrics[name] = {"roc_auc": auc, "pr_auc": ap, "fpr_at_recall_90": fr}

    for key, score in [("lstm_with_lomn", p_lstm), ("lstm_seq_only", p_lstm_solo)]:
        yhat = (score >= 0.5).astype(int)
        tp = int(((yhat == 1) & (yte == 1)).sum())
        fp = int(((yhat == 1) & (yte == 0)).sum())
        fn = int(((yhat == 0) & (yte == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[key]["F1@0.5"] = f1
        metrics[key]["precision@0.5"] = prec
        metrics[key]["recall@0.5"] = rec
    metrics["lstm_train_time_s"] = elapsed_train
    metrics["n_train"] = int(len(ytr))
    metrics["n_test"] = int(len(yte))

    out = {"config": cfg.__dict__, "metrics": metrics}
    (args.out_dir / "lstm_metrics.json").write_text(json.dumps(out, indent=2))
    print("\n=== LSTM ablation summary ===")
    print(json.dumps(out, indent=2))

    # ----- Plot ROC overlay -----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, score, ls, color in [
        ("LSTM(seq) + LOMN static", p_lstm, "-", "#9b5de5"),
        ("LSTM(seq) only",          p_lstm_solo, "-", "#f4a261"),
        ("XGBoost (engineered features)", p_xgb, "-", "#2a9d8f"),
        ("Raw LOMN stat",           raw_lomn, "--", "#264653"),
    ]:
        fpr, tpr, _ = roc_curve(yte, score)
        try:
            auc_ = roc_auc_score(yte, score)
        except ValueError:
            auc_ = float("nan")
        ax.plot(fpr, tpr, ls=ls, lw=1.7, color=color,
                label=f"{name} (AUC={auc_:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.5)
    ax.axhline(0.9, color="red", lw=0.5, ls=":", label="recall 0.90")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("LSTM vs XGBoost (Stage-2 ablation, test = Mar 27-29)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "roc_lstm_vs_xgb.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
