"""End-to-end pipeline runner: every stage of every phase, one command.

    python run_all.py [--force] [--skip-data] [--skip-neural]
                      [--skip-handlabel] [--from STAGE]

By default each stage skips itself if its primary output already exists,
which makes this idempotent. Use --force to re-run everything from
scratch.

Stages
------
    download   : Path B historical data (Binance Vision)
    resample   : 1Hz log-mid grid per day
    lomn       : LOMN candidate detection
    features   : Feature engineering at candidates
    label      : Persistence-based labeling
    train      : XGBoost LOMN-refinement classifier
    calibrate  : MLE comparison across jump-set definitions
    neural     : Neural calibrator (CNN ensemble) — slow, optional
    benchmarks : Lee-Mykland, BNS, F1 vs persistence, event windows
    sig        : Bootstrap CIs + McNemar + DeLong
    sens       : Sensitivity sweep (candidate threshold, persistence z, c)
    regime     : Regime stratification by hourly RV
    plots      : Phase 5/6 summary plots
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("run-all")

ROOT = Path(__file__).resolve().parent


@dataclass
class Stage:
    name: str
    description: str
    cmd: list[str]
    output_path: Path     # primary artefact; if it exists we skip
    skip_in_dryrun: bool = False


def stages() -> list[Stage]:
    return [
        Stage(
            "download",
            "Path B futures bookTicker + aggTrades (Mar 15-29 2024)",
            [sys.executable, "-m", "src.data.fetch_binance_vision",
             "--market", "futures", "--symbol", "BTCUSDT", "--kind", "bookTicker",
             "--start", "2024-03-15", "--end", "2024-03-29",
             "--out", "data/historical"],
            ROOT / "data/historical/futures_btcusdt_bookTicker_2024-03-29.parquet",
        ),
        Stage(
            "download_trades",
            "Path B aggTrades download",
            [sys.executable, "-m", "src.data.fetch_binance_vision",
             "--market", "futures", "--symbol", "BTCUSDT", "--kind", "aggTrades",
             "--start", "2024-03-15", "--end", "2024-03-29",
             "--out", "data/historical"],
            ROOT / "data/historical/futures_btcusdt_aggTrades_2024-03-29.parquet",
        ),
        Stage(
            "resample",
            "Resample bookTicker to 1Hz log-mid grid",
            [sys.executable, "-m", "src.realdata.resample"],
            ROOT / "data/interim/resampled_1s_2024-03-29.parquet",
        ),
        Stage(
            "lomn",
            "LOMN candidate detection (threshold 2.0)",
            [sys.executable, "-m", "src.realdata.run_lomn", "--threshold", "2.0"],
            ROOT / "data/interim/lomn_candidates_2024-03-29.parquet",
        ),
        Stage(
            "features",
            "14-feature engineering at candidate times",
            [sys.executable, "-m", "src.realdata.features"],
            ROOT / "data/interim/features_all.parquet",
        ),
        Stage(
            "label",
            "Persistence-based labels (forward-looking gold)",
            [sys.executable, "-m", "src.realdata.label"],
            ROOT / "data/interim/features_labeled.parquet",
        ),
        Stage(
            "train",
            "XGBoost LOMN-refinement classifier",
            [sys.executable, "-m", "src.realdata.train_xgb"],
            ROOT / "results/phase3/xgb_lomn_refiner.json",
        ),
        Stage(
            "calibrate",
            "Phase 4 MLE — per-day Merton params under three jump sets",
            [sys.executable, "-m", "src.calibration.compare_jump_sets"],
            ROOT / "results/phase4/per_day_params.csv",
        ),
        Stage(
            "neural",
            "Phase 4 neural calibrator (CNN, ~16 min)",
            [sys.executable, "-m", "src.calibration.neural"],
            ROOT / "results/phase4/merton_cnn.pt",
            skip_in_dryrun=True,
        ),
        Stage(
            "neural_compare",
            "Neural vs MLE on synthetic + real",
            [sys.executable, "-m", "src.calibration.compare_neural_mle"],
            ROOT / "results/phase4/neural_vs_mle_synthetic.csv",
        ),
        Stage(
            "benchmarks",
            "Phase 5 F1 evaluation: Lee-Mykland + BNS + pure-ML + raw + xgb",
            [sys.executable, "-m", "src.benchmarks.f1_evaluation"],
            ROOT / "results/phase5/f1_summary.csv",
        ),
        Stage(
            "events",
            "Event-window detection counts",
            [sys.executable, "-m", "src.benchmarks.event_validation"],
            ROOT / "results/phase5/event_detections.csv",
        ),
        Stage(
            "sig",
            "Phase 6 Item 1: bootstrap CIs + McNemar + DeLong",
            [sys.executable, "-m", "src.benchmarks.significance"],
            ROOT / "results/phase6/significance.json",
        ),
        Stage(
            "sens",
            "Phase 6 Item 2: sensitivity sweep",
            [sys.executable, "-m", "src.benchmarks.sensitivity"],
            ROOT / "results/phase6/sensitivity.csv",
        ),
        Stage(
            "regime",
            "Phase 6 Item 4: regime stratification",
            [sys.executable, "-m", "src.benchmarks.regime_analysis"],
            ROOT / "results/phase6/regime_summary.csv",
        ),
        Stage(
            "plots_phase4",
            "Phase 4 plots",
            [sys.executable, "scripts/plot_phase4.py"],
            ROOT / "results/phase4/synthetic_recovery.png",
        ),
        Stage(
            "plots_phase5",
            "Phase 5 summary plots",
            [sys.executable, "scripts/run_phase5.py"],
            ROOT / "results/phase5/f1_summary.png",
        ),
    ]


def run_stage(s: Stage, force: bool, dry_run: bool) -> dict:
    if dry_run and s.skip_in_dryrun:
        LOG.info("[DRY] would run %s — skipped (heavy)", s.name)
        return {"name": s.name, "status": "dry_skipped", "elapsed_s": 0.0}

    if not force and s.output_path.exists():
        LOG.info("[skip] %-16s — already at %s", s.name, s.output_path.name)
        return {"name": s.name, "status": "cached", "elapsed_s": 0.0}

    LOG.info("[run]  %-16s — %s", s.name, s.description)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            s.cmd, cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        elapsed = time.perf_counter() - t0
        LOG.info("       %-16s ✓ (%.1fs)", s.name, elapsed)
        return {"name": s.name, "status": "ok", "elapsed_s": elapsed}
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        LOG.error("       %s FAILED (%.1fs)", s.name, elapsed)
        LOG.error("       last output:\n%s", "\n".join(e.stdout.splitlines()[-15:]))
        return {"name": s.name, "status": "fail", "elapsed_s": elapsed,
                "stdout_tail": e.stdout[-2000:] if e.stdout else ""}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="rerun every stage; ignore cached outputs")
    p.add_argument("--skip-neural", action="store_true",
                   help="skip the slow neural calibrator stages")
    p.add_argument("--skip-data", action="store_true",
                   help="skip download stages (use existing data/)")
    p.add_argument("--from", dest="start_from",
                   help="start from this stage name (skip earlier stages)")
    p.add_argument("--only", help="run only this single stage")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be run without doing slow stages")
    args = p.parse_args()

    all_stages = stages()

    if args.only:
        all_stages = [s for s in all_stages if s.name == args.only]
        if not all_stages:
            LOG.error("unknown stage: %s", args.only)
            return 2

    if args.start_from:
        idx = next((i for i, s in enumerate(all_stages) if s.name == args.start_from), None)
        if idx is None:
            LOG.error("unknown stage: %s", args.start_from)
            return 2
        all_stages = all_stages[idx:]

    if args.skip_neural:
        all_stages = [s for s in all_stages if "neural" not in s.name]
    if args.skip_data:
        all_stages = [s for s in all_stages if not s.name.startswith("download")]

    LOG.info("Running %d stages%s", len(all_stages),
             " (forced)" if args.force else " (cached where possible)")

    results = []
    for s in all_stages:
        r = run_stage(s, force=args.force, dry_run=args.dry_run)
        results.append(r)
        if r["status"] == "fail":
            LOG.error("stopping pipeline after %s failure", s.name)
            break

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_cached = sum(1 for r in results if r["status"] == "cached")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    total_runtime = sum(r["elapsed_s"] for r in results)
    LOG.info("=" * 60)
    LOG.info("done: ok=%d cached=%d fail=%d   total runtime=%.1fs",
             n_ok, n_cached, n_fail, total_runtime)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
