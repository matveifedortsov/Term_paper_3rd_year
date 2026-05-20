"""Phase C orchestrator: end-to-end for BTC/ETH/SOL on Bybit L20.

For each symbol in {BTCUSDT, ETHUSDT, SOLUSDT}:
    1. Run LOMN on `data/interim/<symbol>/resampled_1s_*.parquet`
    2. Build features (auto-detects L20 -> adds 39 bucket features)
    3. Label via persistence-z (forward-looking 30s)
    4. Time-split train/test (12 train / 2 test by default)
    5. Train XGBoost on FEATURE_COLS_L20
    6. Evaluate ROC AUC, F1, FPR-at-90-recall

Then transfer test: train on BTC -> apply to ETH and SOL test slices.

Outputs (results/phase_c/):
    per_asset_metrics.json   per-symbol AUC / F1 / FPR@90
    per_asset_summary.csv    one row per (symbol, method)
    transfer_test.json       BTC-trained model on ETH/SOL
    feature_importance_<symbol>.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.lomn.detector import block_minima, optimal_block_size, robust_scale
from src.realdata.features import build_features_for_day
from src.realdata.label import label_features
from src.realdata.train_xgb import FEATURE_COLS, FEATURE_COLS_L20, fpr_at_recall, select_feature_cols

LOG = logging.getLogger("phase-c")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TRADES_DIR_DEFAULT = Path("data/historical")  # placeholder; multi-asset trades may be absent
TOLERANCE_S = 60
CAND_THRESHOLD = 2.0
ML_THRESHOLD = 0.5


def _trades_for(symbol: str, day_str: str, trades_dir: Path) -> pd.DataFrame:
    """Try to load aggTrades for this asset+day. Returns empty if absent.

    Falls back gracefully: if there's no trades file, features.py just
    fills the trade-flow columns with zeros.
    """
    candidates = [
        trades_dir / f"futures_{symbol.lower()}_aggTrades_{day_str}.parquet",
        trades_dir / f"bybit_{symbol.lower()}_trades_{day_str}.parquet",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_parquet(
                c, columns=["transact_time", "quantity", "is_buyer_maker"]
            )
    return pd.DataFrame({
        "transact_time": pd.Series(dtype="int64"),
        "quantity": pd.Series(dtype="float64"),
        "is_buyer_maker": pd.Series(dtype="bool"),
    })


def run_lomn_for_day(book: pd.DataFrame, threshold: float = CAND_THRESHOLD) -> pd.DataFrame:
    Y = book["log_ask"].values.astype(float)
    n = len(Y)
    h_n = optimal_block_size(n)
    M = block_minima(Y, h_n)
    delta_M = np.diff(M)
    scale = robust_scale(delta_M)
    if scale <= 0:
        scale = float(np.std(delta_M, ddof=1)) or 1e-12
    standardized = delta_M / scale
    abs_std = np.abs(standardized)
    cand_block = np.where(abs_std >= threshold)[0]
    cand_obs = np.clip((cand_block + 1) * h_n, 0, n - 1)
    return pd.DataFrame({
        "ts": book["ts"].values[cand_obs],
        "obs_idx": cand_obs,
        "block_idx": cand_block,
        "log_ask_at_boundary": Y[cand_obs],
        "delta_M": delta_M[cand_block],
        "signed_std": standardized[cand_block],
        "abs_std": abs_std[cand_block],
        "scale": scale,
        "h_n": h_n,
        "threshold": threshold,
    })


def build_symbol_dataset(
    symbol: str,
    book_dir: Path,
    trades_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run LOMN + features + persistence labels for one symbol."""
    book_files = sorted((book_dir / symbol.lower()).glob("resampled_1s_*.parquet"))
    if not book_files:
        raise SystemExit(f"no resampled files in {book_dir / symbol.lower()}")

    all_feats = []
    scale_per_day: dict[str, float] = {}
    for bf in book_files:
        date_str = bf.stem.split("_")[-1]
        book = pd.read_parquet(bf)
        cands = run_lomn_for_day(book)
        cands["day"] = date_str
        scale_per_day[date_str] = float(cands["scale"].iloc[0]) if len(cands) else float("nan")

        trades = _trades_for(symbol, date_str, trades_dir)
        if len(cands) == 0:
            LOG.warning("%s %s: 0 candidates", symbol, date_str)
            continue
        feats = build_features_for_day(cands, book, trades)
        feats = feats.merge(cands[["day", "obs_idx", "delta_M"]],
                            on=["day", "obs_idx"], how="left")
        all_feats.append(feats)
        LOG.info("%s %s: %d candidates -> %d feature rows",
                 symbol, date_str, len(cands), len(feats))

    if not all_feats:
        raise SystemExit(f"no candidates for {symbol}")
    feats_full = pd.concat(all_feats, ignore_index=True)
    scale_series = pd.Series(scale_per_day)
    labeled = label_features(feats_full, scale_per_day=scale_series)
    return labeled, scale_series


def train_eval_split(
    labeled: pd.DataFrame, n_test_days: int = 2,
) -> dict:
    feat_cols = select_feature_cols(labeled)
    LOG.info("using %d feature columns (L20 buckets %s)", len(feat_cols),
             "ON" if len(feat_cols) > len(FEATURE_COLS) else "OFF")

    labeled_valid = labeled[labeled["label"] != -1].copy()
    days = sorted(labeled_valid["day"].unique())
    if len(days) < 3:
        raise SystemExit(f"need >=3 days for train/test split, got {len(days)}")
    test_days = days[-n_test_days:]
    train_days = days[:-n_test_days]
    train = labeled_valid[labeled_valid["day"].isin(train_days)]
    test = labeled_valid[labeled_valid["day"].isin(test_days)]

    Xtr = train[feat_cols].values
    ytr = train["label"].values.astype(int)
    Xte = test[feat_cols].values
    yte = test["label"].values.astype(int)

    if int(ytr.sum()) == 0 or int((1 - ytr).sum()) == 0:
        return {"error": "degenerate train set",
                "n_train": int(len(ytr)), "n_test": int(len(yte))}
    pos_w = (1 - ytr).sum() / max(1, int(ytr.sum()))
    val = int(0.85 * len(ytr))
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=pos_w, eval_metric="logloss",
        random_state=42, n_jobs=-1, early_stopping_rounds=20,
    )
    model.fit(Xtr[:val], ytr[:val], eval_set=[(Xtr[val:], ytr[val:])], verbose=False)

    p_xgb = model.predict_proba(Xte)[:, 1]
    raw_lomn = test["f_lomn_abs_std"].values
    auc_xgb = float(roc_auc_score(yte, p_xgb))
    auc_lomn = float(roc_auc_score(yte, raw_lomn))
    fr_xgb = fpr_at_recall(yte, p_xgb, 0.9)
    fr_lomn = fpr_at_recall(yte, raw_lomn, 0.9)
    yhat = (p_xgb >= ML_THRESHOLD).astype(int)
    tp = int(((yhat == 1) & (yte == 1)).sum())
    fp = int(((yhat == 1) & (yte == 0)).sum())
    fn = int(((yhat == 0) & (yte == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "n_features": len(feat_cols),
        "train_days": train_days, "test_days": test_days,
        "auc_xgb": auc_xgb, "auc_raw_lomn": auc_lomn,
        "fpr_at_recall_90_xgb": fr_xgb,
        "fpr_at_recall_90_raw_lomn": fr_lomn,
        "F1_xgb@0.5": f1,
        "precision_xgb@0.5": prec,
        "recall_xgb@0.5": rec,
        "model": model,
        "feat_cols": feat_cols,
        "test_X": Xte, "test_y": yte,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--book-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--trades-dir", type=Path, default=TRADES_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=Path("results/phase_c"))
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--n-test-days", type=int, default=2)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    per_asset = {}
    models = {}
    test_blocks = {}

    for symbol in symbols:
        LOG.info("====== %s ======", symbol)
        labeled, _ = build_symbol_dataset(symbol, args.book_dir, args.trades_dir)
        n_pos = int((labeled["label"] == 1).sum())
        n_neg = int((labeled["label"] == 0).sum())
        n_drop = int((labeled["label"] == -1).sum())
        LOG.info("%s labels: pos=%d neg=%d drop=%d", symbol, n_pos, n_neg, n_drop)

        res = train_eval_split(labeled, n_test_days=args.n_test_days)
        if "error" in res:
            LOG.warning("%s: %s (skipping)", symbol, res["error"])
            per_asset[symbol] = res
            continue
        models[symbol] = res["model"]
        test_blocks[symbol] = (res["test_X"], res["test_y"], res["feat_cols"])
        # Strip non-serializable items before saving
        per_asset[symbol] = {k: v for k, v in res.items()
                             if k not in {"model", "test_X", "test_y", "feat_cols"}}
        per_asset[symbol]["labels"] = {"pos": n_pos, "neg": n_neg, "drop": n_drop}

        # Feature importance dump
        booster = res["model"].get_booster()
        booster.feature_names = list(res["feat_cols"])
        gain = booster.get_score(importance_type="gain")
        fi = pd.DataFrame(
            [(name, gain.get(name, 0.0)) for name in res["feat_cols"]],
            columns=["feature", "gain"],
        ).sort_values("gain", ascending=False)
        fi.to_csv(args.out_dir / f"feature_importance_{symbol}.csv", index=False)

    # Transfer test: train on BTCUSDT, apply to ETHUSDT and SOLUSDT test slices
    transfer = {}
    if "BTCUSDT" in models:
        btc_feats = test_blocks["BTCUSDT"][2]
        btc_feat_set = set(btc_feats)
        for tgt in ("ETHUSDT", "SOLUSDT"):
            if tgt not in test_blocks:
                continue
            Xte, yte, tgt_feats = test_blocks[tgt]
            # Reorder columns to match BTC ordering (drop any missing)
            common = [c for c in btc_feats if c in tgt_feats]
            if not common:
                transfer[tgt] = {"error": "no overlapping features"}
                continue
            idx = [tgt_feats.index(c) for c in common]
            Xte_aligned = Xte[:, idx]
            p_btc = models["BTCUSDT"].predict_proba(Xte_aligned)[:, 1]
            try:
                transfer_auc = float(roc_auc_score(yte, p_btc))
            except ValueError:
                transfer_auc = float("nan")
            transfer[tgt] = {
                "auc_btc_to_target": transfer_auc,
                "n_common_features": len(common),
            }

    out = {"per_asset": per_asset, "transfer_from_BTCUSDT": transfer}
    (args.out_dir / "per_asset_metrics.json").write_text(json.dumps(out, indent=2))

    # Tabular summary
    rows = []
    for sym, r in per_asset.items():
        if "error" in r:
            rows.append({"symbol": sym, "error": r["error"]})
            continue
        rows.append({
            "symbol": sym,
            "n_train": r["n_train"], "n_test": r["n_test"],
            "n_features": r["n_features"],
            "AUC_xgb": r["auc_xgb"], "AUC_raw_lomn": r["auc_raw_lomn"],
            "FPR90_xgb": r["fpr_at_recall_90_xgb"]["fpr"],
            "FPR90_raw_lomn": r["fpr_at_recall_90_raw_lomn"]["fpr"],
            "F1_xgb": r["F1_xgb@0.5"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "per_asset_summary.csv", index=False)
    print("\n=== Phase C per-asset summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Transfer test from BTCUSDT ===")
    print(json.dumps(transfer, indent=2))


if __name__ == "__main__":
    main()
