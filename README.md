# Jump Detection and Calibration of Jump-Diffusion Models Using ML

Term paper code (HSE, BMB-2301). Implements the LOMN (Limit Order Microstructure Noise) detector of Bibinger, Hautsch & Ristig (2024) and extends it with ML-based candidate refinement and jump-diffusion calibration.

## Project layout

```
src/lomn/
  simulation.py      Merton jump-diffusion + one-sided exponential noise DGP
  detector.py        Block-minimum detector + Gumbel test
  monte_carlo.py     Two-pass MC: H0 calibration + size/power grid
scripts/
  smoke_test.py      Single-path sanity check
  time_mc.py         Per-replication timing
  run_phase1.py      Phase 1 deliverable orchestration
results/
  tables/            CSV + LaTeX tables
  figures/           PNG figures
```

## Phase 1 — Stage 1 LOMN size/power validation (DONE)

### Configuration
- n = 23,400 observations per day, T = 1 day
- σ = 0.03 (per-day log-return std, BTC-realistic)
- block size h_n = ⌊n^(1/3)⌋ = 29 (Bibinger rate-optimal)
- 500 replications per cell
- noise scales q ∈ {0.0005, 0.001, 0.002, 0.005}
- jump sizes δ ∈ {0, 0.0025, 0.005, 0.01, 0.02}
- nominal level α = 0.05

### Run
```
python scripts/run_phase1.py
```
Total runtime: ~16 s on a single core.

### Key results

**Size (rejection rate at δ=0):**

| q       | Gumbel CV | Calibrated CV |
|---------|-----------|---------------|
| 0.0005  | 0.066     | 0.060         |
| 0.001   | 0.056     | 0.042         |
| 0.002   | 0.042     | 0.044         |
| 0.005   | 0.044     | 0.056         |

Empirical size sits in [4.2%, 6.6%] vs. nominal 5%. The asymptotic Gumbel critical value is well-calibrated; the empirically calibrated CV (per noise level) tightens this slightly.

**Power (calibrated CV, two-sided test):**

| q \\ δ  | 0.0025 | 0.005 | 0.01  | 0.02  |
|---------|--------|-------|-------|-------|
| 0.0005  | 0.090  | 0.768 | 1.000 | 1.000 |
| 0.001   | 0.084  | 0.704 | 1.000 | 1.000 |
| 0.002   | 0.094  | 0.702 | 1.000 | 1.000 |
| 0.005   | 0.112  | 0.602 | 1.000 | 1.000 |

Clear power transition. The detector saturates at δ ≥ 0.01. The δ=0.0025 row is below the detection floor at this n: per-block diffusion std under σ=0.03 is ~0.00106, so a 0.0025 jump produces a ~2.4σ standardized statistic against an extreme-value benchmark of ~4.7.

**Correction to the term paper draft.** The draft claims "Power exceeds 70% for jump sizes > 0.0025 with q=0.001." The Monte Carlo finds the 70%-power threshold at δ ≈ 0.005, not 0.0025. Update Section 1.5 of the paper accordingly.

### Deliverables

| File | Content |
|------|---------|
| `results/tables/phase1_full_grid.csv`            | Full 20-cell grid with both CV variants |
| `results/tables/size_power_calibrated.{csv,tex}` | Calibrated-CV pivot |
| `results/tables/size_power_raw_gumbel.{csv,tex}` | Asymptotic-CV pivot |
| `results/figures/size_calibration.png`           | T_max distribution under H0 vs. Gumbel approximation |
| `results/figures/power_curves.png`               | Rejection rate vs. δ, by q, both CVs |
| `results/figures/sample_path.png`                | Efficient path + LOMN noise + detected candidates |
| `results/figures/noise_distribution.png`         | One-sided exponential noise histogram |
| `results/figures/return_distribution.png`        | Observed log-return histogram |

## Reproducibility
- All seeds derived from `MCConfig.base_seed = 20260508`; pass-1 calibration and pass-2 grid use disjoint seed offsets.
- Pure NumPy/SciPy/Pandas. No GPU. No external data.

## Phase 2 — Path B historical data (DONE)

Binance Vision USD-M futures BTCUSDT, 2024-03-15 to 2024-03-29 (15 days).

| Stream | Files | Rows | Disk |
|---|---|---|---|
| bookTicker (L1)    | 15 | 267.3 M | 3.74 GB |
| aggTrades          | 15 |  34.3 M | 0.66 GB |
| **Total**          | 30 | **301.6 M** | **4.40 GB** |

Coverage:
- BTC range: $60,819 (Mar-20 selloff low) → $72,500 (Mar-15 ATH attempt)
- Median L1 spread: $0.10 (one tick — extremely liquid)
- Notable events: Mar-15 ATH retest, Mar-19/20 ~12% drawdown, Mar-25/26 recovery rally
- Timestamps complete from 00:00:00 to 23:59:59 UTC each day, no gaps

Live capture (Path A) still pending user start. See [docs/data_tutorial.md](docs/data_tutorial.md).

## Phase 3 — ML refinement on real BTC futures (DONE, Path B)

End-to-end pipeline: bookTicker → 1Hz log-price grid → LOMN candidate set → 14-feature engineering → persistence-based labels → XGBoost classifier → evaluation against raw-LOMN baseline.

### Pipeline modules

| Module | Purpose |
|---|---|
| `src/realdata/resample.py`   | bookTicker → 1Hz forward-filled grid (1.30M rows total) |
| `src/realdata/run_lomn.py`   | LOMN block-min detector at threshold 2.0 → 3,601 candidates over 15 days |
| `src/realdata/features.py`   | 14 features + persistence proxy at each candidate |
| `src/realdata/label.py`      | Persistence-based gold labeling (forward-looking; not a feature) |
| `src/realdata/train_xgb.py`  | Time-split XGBoost training and FPR-at-recall comparison |

### Critical design choice — labels avoid leakage

Gold labels use only `label_persistence_30s = log_mid(τ+30s) - log_mid(τ-30s)`, which is **never given to XGBoost as a feature**. A candidate is POSITIVE if `|persistence| / scale ≥ 5σ` (sustained level shift), NEGATIVE if `≤ 2σ` (mean-reverting blip), else dropped. This makes the comparison "raw LOMN stat vs. XGBoost on book+trade features" honest — both score against an external label.

| | Train (Mar 15–26) | Test (Mar 27–29) |
|---|---:|---:|
| Total labeled | 1,013 | 293 |
| Positive | 381 | 128 |
| Negative | 632 | 165 |

### Headline results (test set, 293 samples)

| Metric | Raw LOMN stat | XGBoost (14 features) |
|---|---:|---:|
| ROC AUC | 0.886 | **0.902** |
| PR AUC (AP) | 0.875 | 0.862 |
| FPR at recall ≥ 90% | 33.3% | **27.9%** |

**FPR reduction at matched recall: 16.4%.** Below the paper's H1 target of ≥30%, but a real, honest improvement on a non-leaky label.

### Feature importance (XGBoost gain)

1. `f_lomn_abs_std` — 32 (the LOMN stat itself, dominant)
2. `f_n_trades_pm5s` — 18
3. `f_lomn_signed` — 13
4. `f_volume_pm5s` — 7
5. `f_dt_prev_cand` — 6
6. `f_bipower_60s`, `f_realvar_60s` — 5, 4.5
7. `f_signed_flow_pm5s`, `f_obi_l1`, `f_jump_ratio`, `f_log_mid` — 3–4
8. `f_realkurt_60s`, `f_dspread_60s` — 2–3
9. `f_spread` — 0.3 (essentially useless: 1-tick spread 99% of the time)

The LOMN stat carries most of the signal; trade-activity features (`n_trades`, `volume`) provide the meaningful lift; L1 book features add little — consistent with my pre-Phase-3 prediction that **trade-flow > deep-book features for jump validation**.

### Deliverables

```
results/phase3/
  metrics.json
  roc.png                 ROC: XGBoost vs raw LOMN
  pr.png                  Precision-Recall comparison
  feature_importance.png  bar chart of XGBoost gain
  feature_importance.csv
  xgb_lomn_refiner.json   trained model
```

### Candid assessment

- The H1 target (≥30% FPR reduction) was set before having real data; on Path B with L1-only book features, 16% is the truth.
- The dominant predictor is the LOMN stat itself — XGBoost mostly fine-tunes its decision boundary using trade activity. The win is real but small.
- Path A's L20 features (depth at levels 1–5, multi-level OBI) are expected to push FPR reduction higher, but based on feature-importance evidence the headline gain will likely come from continuing to lean on trade flow, not depth.
- Persistence-based labeling is a meaningful but imperfect proxy. Hand-validation of a few hundred candidates would tighten the analysis but adds substantial human cost.

### Hyperparameter tuning ablation (Optuna, 50 trials)

To verify the default hyperparameters were not leaving substantial performance on the table, I ran a 50-trial Bayesian search using Optuna (TPE sampler + median pruner) with 5-fold expanding-window TimeSeriesSplit CV on the training period only. Search space: `max_depth ∈ [3,8]`, `learning_rate ∈ [0.01, 0.2]` (log), `min_child_weight ∈ [1,20]`, `gamma`, `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`. `n_estimators` left to early stopping. Total search time: 100 s.

| Model | ROC AUC | PR AUC | FPR @ recall ≥ 90% |
|---|---:|---:|---:|
| Raw LOMN | 0.886 | 0.875 | 33.3% |
| XGBoost default | 0.902 | 0.862 | **27.9%** |
| XGBoost tuned (Optuna) | **0.906** | 0.864 | 29.7% |

**Result: Optuna improved ROC AUC by 0.4 pp but made FPR-at-90%-recall *worse* by 1.8 pp.** The tuned model and default model occupy different ROC operating regimes that have similar total area but allocate errors differently. The Optuna search history (`results/phase3/optuna_history.png`) shows the best CV AUC (0.915) was found on trial 2 and never beaten across the remaining 48 trials — all sat between 0.89 and 0.91, well within the standard error of an AUC estimate at n≈200 per fold (≈0.018).

This is consistent with classical results on small-sample tuning (Cawley & Talbot 2010): when the sampling variance of the validation metric exceeds the typical between-config improvement, hyperparameter search becomes a noise-fitting exercise. For the paper, the methodologically defensible position is to keep the default configuration and disclose this ablation. The Optuna script is preserved at `src/realdata/tune_xgb.py` for reproducibility.

## Phase 4 — Calibration of Merton parameters (DONE)

Two calibration methods built and evaluated against each other and against jump-set definitions:

| Module | Purpose |
|---|---|
| `src/calibration/mle.py`                  | Separable MLE (Aït-Sahalia-Jacod two-step): jump params from detected jump set, diffusion params from cleaned path |
| `src/calibration/compare_jump_sets.py`    | Per-day calibration under raw-LOMN, ML-refined, persistence-truth jump sets — **H2 test** |
| `src/calibration/neural.py`               | 1D-CNN amortized calibrator (4 conv blocks, 1440-min returns, 5 outputs in transformed space) |
| `src/calibration/compare_neural_mle.py`   | Synthetic ground-truth recovery, real-BTC stability, inference speed |

### H2: across-day variance of parameter estimates

For each day, calibrate Merton params on three jump sets and compare std across the 15 days:

| Parameter | Raw LOMN std | ML-refined std | **Reduction** | Persist-truth std |
|---|---:|---:|---:|---:|
| λ (jumps/day)  | 11.94 | 16.10 | **−35%** (ML worse) | 11.71 |
| μ_J            | 3.97e-4 | 2.77e-4 | **+30%** | 5.17e-4 |
| σ_J            | 9.40e-4 | 6.64e-4 | **+29%** | 8.58e-4 |
| σ (daily)      | 0.0113 | 0.0113 | +0.1% (tie) | 0.0113 |
| μ (daily)      | 0.0461 | 0.0453 | +2% | 0.0456 |

**Mixed verdict on H2.** ML refinement *wins* on jump-size distribution stability (μ_J, σ_J each ~30% lower std) but *loses* on jump-intensity stability (λ has ~35% higher std because the ML threshold flags ~2× more candidates than the raw-LOMN threshold). Diffusion (μ, σ) is unaffected — separable MLE estimates these from the cleaned path, which barely changes between definitions.

The qualitative result is **partial support for H2**: the parameters describing what jumps look like are more stable, the parameter counting how many there are is less stable. For the paper, soften H2 to *jump-size parameters* rather than *all parameters*.

### Neural calibrator: 1D-CNN trained on synthetic Merton

Architecture: 4 conv blocks (32-64-128-128 channels, kernel 5, stride 2, GroupNorm + ReLU) → global average pool → 2 dense → 5 outputs in transformed space `(μ, log σ, log(1+λ), μ_J, log σ_J)`. Targets standardized to unit variance during training. Trained on 20k synthetic Merton paths (1440 minute-frequency returns), 20 epochs, Adam, Huber loss. Total training: ~16 minutes on CPU.

**Synthetic ground-truth recovery (relative RMSE = RMSE / true-param-std, lower is better):**

| Param | Neural CNN | Threshold-MLE |
|---|---:|---:|
| μ        | 0.82 | 2.40 |
| **σ**    | **0.21** | **0.17** |
| λ        | 0.63 | 1.43 |
| μ_J      | 0.34 | 1.00 |
| σ_J      | 0.50 | 1.26 |

The neural calibrator dominates the threshold-based MLE on 4 of 5 parameters; MLE narrowly wins on diffusion vol σ. This is mostly because threshold-MLE depends on a 4σ heuristic for jump detection that misclassifies — the CNN learns to combine the full distributional shape rather than committing to a single threshold.

**Real BTC futures cross-day stability (std lower = more stable):**

| Param | Neural | MLE | Winner |
|---|---:|---:|:---:|
| μ      | 0.0113  | 0.0420  | Neural |
| σ      | 0.00874 | 0.01084 | Neural |
| λ      | 12.73   | 7.86    | MLE |
| μ_J    | 6.4e-4  | 1.5e-3  | Neural |
| σ_J    | 6.1e-4  | 1.5e-3  | Neural |

Neural is more stable on 4 of 5 parameters. MLE is more stable on λ for the same reason it lost on synthetic accuracy: it uses a fixed-threshold detection rule that produces a deterministic count per day with low variance but high bias. Neural's λ varies more because the model is responsive to actual return-distribution shape.

**Important caveat — prior bias.** The neural λ on real BTC averages 62 jumps/day, while threshold-MLE finds 15. The training prior `λ ~ Uniform(5, 100)` has mean 52, so the neural estimate is suspiciously close to the prior mean — likely a sign of partial regression-to-prior under uncertainty. For the paper this should be disclosed: the neural estimator's *stability* is a real finding, but its *level* is conditioned on the prior.

**Speed comparison:**

| | Neural CNN | Threshold-MLE | Notes |
|---|---:|---:|---|
| Inference per path | 1.32 ms | 0.54 ms | MLE faster on small data (just numpy) |
| Training cost | ~16 min one-off | 0 (no training) | |
| Scaling | Constant per path | Linear in path length | Neural wins for large/many paths |

The paper draft hypothesized "100× faster inference" for neural calibration. On this scale that is **wrong** — at 1440 returns per path, a vectorized threshold detector beats a CNN forward pass by ~2.5×. Neural calibration's real advantage isn't speed, it's accuracy under uncertainty. Update Section 2.4 of the paper.

### Phase 4 deliverables

```
results/phase4/
  per_day_params.csv               per-day MLE params under each jump-set definition
  h2_variance_comparison.csv       across-day std summary
  h2_summary.json                  numeric H2 reduction percentages
  lambda_per_day.png               per-day lambda by jump-set definition
  sigma_per_day.png                per-day sigma
  sigma_J_per_day.png              per-day sigma_J
  merton_cnn.pt                    trained 1D-CNN weights + target normalization
  neural_train_history.{csv,png}   training curve
  neural_synthetic_eval.json       synthetic recovery RMSE
  neural_synthetic_preds.csv       4000 (true, pred) pairs
  synthetic_recovery.png           predicted-vs-true scatter, all 5 params
  neural_real_per_day.csv          neural calibrator on real BTC, per day
  neural_vs_mle_synthetic.csv      head-to-head RMSE
  neural_vs_mle_real.csv           per-day comparison on real BTC
  neural_vs_mle_real_stability.csv cross-day std comparison
  real_per_day.png                 per-day param trajectories: neural vs MLE
  inference_speed.json             per-path timing
```

### Honest assessment

- **H2 is partially supported, not cleanly.** ML refinement helps for jump-size distribution; doesn't help for jump intensity. The paper should report this nuance.
- **Neural calibration is more accurate on synthetic than threshold-MLE,** by 2-3× relative RMSE on most params. This is a paper-worthy result.
- **Neural calibration is more stable on real BTC** for 4 of 5 parameters, but with a clear prior-bias caveat for absolute levels of λ.
- **The "100× faster" speed claim from the paper draft is empirically false** at this scale. Update or remove.

## Phase 5 — Benchmarks and event validation (DONE)

Four detection methods compared on test days (Mar 27-29) against persistence-labeled ground truth (128 positives), with ±60 s matching tolerance:

| Method | TP | FP | FN | Precision | Recall | **F1** |
|---|---:|---:|---:|---:|---:|---:|
| Raw LOMN (\|stat\| ≥ 4.0)        |  80 |   56 | 48 | 0.588 | 0.625 | **0.606** |
| Lee-Mykland (α=0.05, K=270)    |  71 | 4320 | 57 | 0.016 | 0.555 | **0.031** |
| LOMN + XGBoost (proba ≥ 0.5)   |  99 |   91 | 29 | 0.521 | 0.773 | **0.623** |
| Pure ML (no LOMN features)     |  47 |   44 | 81 | 0.516 | 0.367 | **0.429** |

### Key findings

**1. Lee-Mykland confirms the paper's pulverization motivation.** 4,320 false positives across 3 test days — F1 = 0.03, essentially useless as a detector. The classical nonparametric test fires on every microstructure-noise outlier because the bipower local-vol estimate cannot disentangle noise from genuine returns at second-level frequency. This empirically validates Section 1.1 of the draft and the central premise of switching to LOMN's one-sided-noise model.

**2. The hybrid framework (H3) is supported.** LOMN+XGBoost achieves the highest F1 (0.623), beating raw LOMN (0.606), pure ML (0.429), and Lee-Mykland (0.031). The improvement over raw LOMN is small (+0.017 F1) but consistent with Phase 3's 16% FPR reduction at matched recall. The improvement over pure ML and Lee-Mykland is large.

**3. Pure ML without LOMN as anchor is significantly worse.** Removing `f_lomn_abs_std` and `f_lomn_signed` from the feature set drops F1 from 0.623 to 0.429. The LOMN test statistic is the single most important feature — book and trade features alone cannot recover comparable detection quality. The paper's framing of "LOMN as a strong prior, ML as refinement" is empirically correct.

**4. BNS bipower is uninformative for crypto.** The per-day BNS test rejects the no-jump null on 15/15 days with Z-scores 18 to 71. Crypto futures have significant jump activity on every day in the sample, so the binary "is there a jump today" answer is always yes. BNS is included for completeness but not useful for per-event detection.

### Event validation

Detections inside five hand-curated event windows (BTC March-2024 swings):

| Event | Day | Window UTC | Raw LOMN | Lee-Mykland | LOMN+XGB | Pure ML |
|---|---|---|---:|---:|---:|---:|
| ATH retest spikes | 03-15 | 12:00–18:00 |  3 |  93 |  5 |  4 |
| First leg down to $63k | 03-19 | 12:00–16:00 |  2 |  32 | 13 | 21 |
| Late-night drop to $61k | 03-19 | 21:00–24:00 |  3 |  52 |  7 |  8 |
| Selloff continues | 03-20 | 00:00–06:00 | 11 |  45 | 26 | 38 |
| Rally toward $71k | 03-25 | 13:00–19:00 | 25 | 128 | 51 | 43 |
| **Hit rate (≥1 detection)** | | | **5/5** | **5/5** | **5/5** | **5/5** |

All four methods recognize all five known events. They differentiate by detection density: Lee-Mykland fires 7-30× more often than raw LOMN inside the same window, consistent with the global noise-pulverization problem.

### Phase 5 deliverables

```
results/phase5/
  f1_per_day.csv               per-day F1 by method
  f1_summary.csv               headline F1 comparison
  f1_summary.png               F1 bars + detection counts (log)
  all_detections.csv           full list of detections per method
  bns_per_day.csv              BNS Z-statistics per day
  event_detections.csv         per-event detection counts
  event_summary.json           hit-rate summary
  event_detections.png         per-event grouped bar chart
```

### Verdict on the paper's H3

The paper hypothesizes:

> H3: The hybrid framework outperforms both pure statistical (Lee-Mykland) and pure ML (Rao et al.) baselines on F1-score and calibration stability metrics.

**SUPPORTED on F1.** Ordering: LOMN+XGB (0.623) > raw LOMN (0.606) > pure ML (0.429) > Lee-Mykland (0.031). On calibration stability (H2 from Phase 4), partially supported — wins on jump-size distribution stability, loses on jump-intensity stability.

## Final summary

All five phases complete. End-to-end findings:

| Phase | Hypothesis | Verdict |
|---|---|---|
| 1 | Stage 1 LOMN size correctly calibrated | ✓ size 4-7%, power saturates above δ ≈ 0.005 |
| 3 | H1: ML refinement reduces FPR by ≥30% at TPR ≥90% | Partial — 16% reduction at 91% recall |
| 4 | H2: ML-refined calibration is more stable than raw LOMN | Mixed — wins on μ_J/σ_J (-30%), loses on λ (+35%) |
| 5 | H3: Hybrid beats pure statistical and pure ML | ✓ on F1 (0.623 > 0.606 > 0.429 > 0.031) |

The paper's central thesis stands. Suggested revisions to the draft:
- Soften H1 from "≥30%" to "statistically significant FPR reduction"
- Soften H2 from "all parameters" to "jump-size distribution parameters"
- Remove the "100× faster" claim for neural calibration (false at this scale)
- Update §1.5 power claim from "70% at δ > 0.0025" to "70% at δ > 0.005"
- Fix the broken `\cite` keys (still pending from initial review)

## What's still open
- Phase 2 Path A live L20 capture — start when ready
- Re-run Phase 3 + Phase 5 on Path A data when it accumulates; expect modest improvement from deep-book features but not a step change
