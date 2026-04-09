# EPForecast Methodology

**Spanish Electricity Price Forecasting with single-XGBoost + residual_1w transform**

This repository contains the ML methodology code behind [EPForecast](https://epf.productjorge.com), a system that predicts Spanish day-ahead electricity prices (OMIE market) at hourly and 15-minute resolution, 7 days ahead.

> **Note:** This repository contains the ML methodology code only. The full application (API, frontend, data pipeline, deployment infrastructure) is proprietary.

> **⚠️ v10.x retraction (2026-04-09):** the v10.0 / v10.1 / v10.2 LSTM-XGBoost hybrid line of experiments described in earlier versions of this README has been **retracted**. Two layered code-level bugs (silent zero-fill at 15-min inference time + wrong-domain LSTM input at training time) meant the LSTM block contributed zero useful signal at both training and inference time throughout the v10.x series. The pre-LSTM `v8-res-1w-pw3-d365` configuration already strictly dominated v10.1 on every metric except bias on the same evaluation window. The current production model is **v11.0** = a clean retrain of the v8 architecture: single XGBoost + residual_1w + price weighting + decay 365d, **no LSTM, no ensemble**. See [the v11.0 changelog page](https://epforecast.vercel.app/changelog/v11-0-post-lstm-correction) for the full retraction story. The drift guard at the 15-min inference call sites in `direct_predictor.py` raises loudly if anyone re-enables LSTM without first fixing the underlying bugs. The LSTM gating code in `direct_trainer.py` and the `lstm_embedder.py` file are still in this repository as historical reference and as scaffolding for any future research that wants to retry LSTM after the bugs are properly fixed.

## Live Demo & Documentation

- **Live dashboard:** [epf.productjorge.com](https://epf.productjorge.com)
- **Full documentation:** [epforecast.vercel.app](https://epforecast.vercel.app)
- **Methodology changelog:** [epforecast.vercel.app/changelog/overview](https://epforecast.vercel.app/changelog/overview)
- **Case study — Iran crisis (March 2026):** [epforecast.vercel.app/blog/iran-crisis-analysis](https://epforecast.vercel.app/blog/iran-crisis-analysis)

## Methodology Overview

### Data Sources

| Source | Data | Resolution |
|--------|------|------------|
| REE/ESIOS | Electricity prices, demand, generation mix, cross-border flows | Hourly |
| Open-Meteo | Temperature, wind, cloud cover, solar radiation (5 Spanish stations) | Hourly |
| Yahoo Finance / ICE | Natural gas (TTF), Brent crude oil, EU ETS carbon permits | Daily |

### Feature Engineering (100+ features)

Features are organized into 5 categories:

1. **Temporal** — Hour of day, day of week, month, holiday flags, cyclical encodings
2. **Price lags** — 24h, 48h, 168h (1 week), 336h (2 weeks), 504h (3 weeks), same-weekday averages
3. **Generation & demand** — Residual demand, renewable share, wind/solar penetration, demand ramps
4. **Weather interactions** — Temperature-demand coupling, wind-generation, solar-cloud interactions, heating/cooling degree days
5. **Commodity derivatives** — Gas/oil momentum (7d/30d), marginal cost estimation, spark spread, volatility regime detection

See [`src/data/feature_engineering.py`](src/data/feature_engineering.py) for all feature definitions.

### Model Architecture (v11.0, post-M0.6 Phase C)

The production model is a **single XGBoost** with the residual_1w target transform:

```
Tabular features (~110 columns)
         ↓
Feature selection (correlation filter → permutation importance)
   → ~70-90 features per horizon group
         ↓
XGBoost (depth=12, lr=0.03, min_child=5, reg_lambda=0.3, q=0.55)
   sample weights: 3× above 60 EUR + 365-day exponential decay
         ↓
   Predict deviation from 1-week price baseline
         ↓
Inverse transform: pred = model_output + target_baseline_1w
   → Day-ahead price forecast
```

XGBoost is trained with quantile loss (q=0.55) on a **residual-from-baseline target** — it predicts the deviation from the same-hour price 1 week prior, not raw EUR/MWh. This isolates the regime-change signal and avoids the range-compression problem tree models have on raw prices.

**Why quantile loss (q=0.55)?** Electricity prices are right-skewed (bounded near zero, occasional spikes >200 EUR/MWh). Quantile loss at 0.55 shifts predictions slightly above the median, directly correcting the systematic underprediction bias inherent in skewed distributions.

**Why pw3 + d365?** Price weighting (3× weight on samples > 60 EUR/MWh) emphasises high-price training samples that the model would otherwise underfit. The 365-day exponential decay halflife focuses learning on recent data without throwing away long-term seasonal patterns. Validated against v6.3 (no pw3) and v7.0 (no res1w + no pw3) controls — both modifications improve every business-relevant metric on the same evaluation window.

> **Legacy (v4.3):** Prior to v11.0, the ensemble combined HistGradientBoosting, LightGBM, and XGBoost with equal-weight averaging. That architecture is still in the codebase (`trainer.py`) for reference but is NOT the v11.0 production stack — `--ensemble` is dropped at the M0.6 Phase F cutover and the model_name written to the predictions table becomes the single-XGBoost `xgboost_hybrid15`.

#### LSTM Price Encoder — RETRACTED (v10.0–v10.2)

The `lstm_embedder.py` file in this repository documents the LSTM encoder design that was attempted in v10.0 / v10.1 / v10.2 but is **NOT** part of the v11.0 production stack. The v10.x experiments were retracted on 2026-04-09 after two layered code-level bugs were found:

1. **Bug 1 (silent zero-fill at 15-min inference time):** `_build_origin_features_15min()` in `direct_predictor.py` never called `LSTMEmbedder.compute_embedding()`, so any `lstm_emb_*` feature in the trained model's `feature_cols` was filled with `0.0` at the row construction site. xgboost handled the all-zero block silently. No exception, no warning, no log line — degraded predictions only.
2. **Bug 2 (wrong-domain LSTM input at training time):** the trainer fed 15-minute prices through an LSTM encoder pre-trained on hourly prices (168 quarter-hours = 42 hours of context, fed into a model expecting 168 hours = 1 week of context, with hourly normalization stats). Wrong shape, wrong domain, wrong normalization. The encoder output was noise.

Empirical verification: toggling `EPF_LSTM_EMBEDDINGS=true ↔ false` at inference time on the v10.1 production joblib produced bit-for-bit identical predictions, proving the LSTM block contributed zero useful signal at runtime.

The deeper finding: the pre-LSTM `v8-res-1w-pw3-d365` configuration from earlier in March already strictly dominated v10.1 on every metric except bias on the same 156-day evaluation window. v10.1's "best ever bias" of -0.65 was an artifact of broken-LSTM zero-fill noise acting as accidental regularization, not a genuine calibration win.

**The drift guard** at the 15-min inference call sites in `direct_predictor.py` (lines ~419, ~973) raises a `RuntimeError` if `EPF_LSTM_EMBEDDINGS=true` and any expected `lstm_emb_*` feature is missing from the row. It's opt-in via the same env var that enables LSTM, so behavior is identical to before when disabled. If anyone ever properly fixes Bugs 1 and 2 and wants to retry LSTM, the guard catches the silent-zero-fill class of bug before it can corrupt predictions again.

See [`src/models/lstm_embedder.py`](src/models/lstm_embedder.py) for the encoder implementation. The LSTM gating code in `direct_trainer.py` (around line ~1290) is still gated behind `EPF_LSTM_EMBEDDINGS=true` (default false in v11.0). Both files are kept as historical reference and as scaffolding for any future research that wants to retry LSTM after the bugs are properly fixed.

### Direct Multi-Horizon Prediction

Instead of recursive forecasting (which accumulates errors), each forecast horizon has its own model:

- **Day-ahead (DA1, DA2):** D+1 predictions from 10:00 UTC origin
- **Strategic (S1–S5):** D+2 through D+7 predictions from 15:00 UTC origin

This avoids error propagation across horizons and allows different feature sets per horizon group.

### Feature Selection

A two-stage per-horizon pipeline prunes noise features:

1. **Correlation filter** — Removes one of each pair with |r| > 0.95
2. **Permutation importance** — Drops features contributing < 0.1% to model performance

Strategic groups (S1–S5) automatically drop all price lag features, which are stale at multi-day horizons.

See [`src/models/feature_selection.py`](src/models/feature_selection.py) for the implementation.

### Confidence Intervals

Split conformal prediction with asymmetric bands — 50% and 90% intervals calibrated from out-of-fold residuals, bucketed by horizon group.

## Results

156-day walk-forward backtest (October 9 2025 – March 18 2026, NaN-safe, includes the Iran gas crisis):

| Version | Product | DA MAE | Strategic MAE | DA SpkR | DA DirAcc | DA CorrFr |
|---------|---------|:---:|:---:|:---:|:---:|:---:|
| **v11.0** (current — M0.6 Phase C) | Single XGBoost + residual_1w + pw3 + d365 | **14.26** | **17.35** | **69.28%** | **76.59%** | **0.891** |
| ~~v10.1~~ (RETRACTED — broken LSTM) | "LSTM-XGBoost hybrid + residual_1w" | 15.73 | 18.13 | 69.34% | 75.87% | 0.887 |
| v8-res-1w-pw3-d365 (March, the v11.0 predecessor) | XGBoost + residual_1w + pw3 + d365 | 12.98 | (no rows) | 71.08% | 76.74% | 0.904 |
| v4.3 (baseline) | 3-base ensemble (no res1w) | 14.47 | 19.79 | — | — | — |

**v11.0 strictly dominates v10.1 on every metric except bias.** The strategic improvements (DirAcc +2.68pp, SpkR +2.73pp, MAE -0.78) are the biggest business win — meaningful for trading users on D+2 to D+7 horizons. v11.0's strategic backtest is the FIRST EVER for a pre-LSTM tag in the project's history; the v10.x retraction also retracted everything we thought we knew about strategic performance, and v11.0 is what a clean strategic baseline actually looks like.

The 1.28 MAE gap between v11.0 (14.26) and the historical v8-res-1w-pw3-d365 scout (12.98) is explained by 16 more days of training data, code drift in post-processing since March, and one NaN row at the autumn DST transition (2025-10-26 01:00 UTC) that propagates through lag features.

The structural ceiling that gradient boosting trees hit at v4.3 — leaf averaging compressing the prediction range to ~73% of actual variance — was broken by **deep trees (depth=12) + residual_1w target transform + price weighting**, NOT by the LSTM encoder. The v6.3 → v7.0 → v8 sequence demonstrated this empirically with non-LSTM controls. The v10.x LSTM hybrid was an unrelated detour that turned out to measure broken code.

## What Didn't Work

100+ experiments were run across 11 major versions. These are the approaches that failed or were rejected, documented here because the failures are as informative as the wins:

| Approach | Outcome | Why rejected |
|----------|---------|-------------|
| **v10.x LSTM-XGBoost hybrid (RETRACTED 2026-04-09)** | All headline numbers describe broken code | Two layered code bugs meant the LSTM block contributed zero useful signal at both training and inference time. v8 (the predecessor) was strictly better. v11.0 = clean retrain of v8. See `direct_predictor.py` drift guard. |
| sklearn MLP (v7.1) | MAE 37.24 (3× worse than XGBoost) | Tabular flattened features can't be learned by sklearn's basic MLP |
| PyTorch deep residual MLP (v8.0) | MAE 28.13 (2.2× worse) | Same lesson at higher capacity — architecture ≠ feature representation |
| 4-week residual baseline (v10.2 H2) | Regime memory bias, +3-4 EUR systematic | When prices shift level, the 4-week baseline lags for 28 days creating persistent residual error. The 1-week baseline used by v11.0 adapts in 7 days. |
| Recursive forecasting (roll-forward single model) | Error compounding on D+3+ | Direct per-horizon models eliminate this; each horizon group trains its own model |
| Naive residual target (predict price − lag-24h) | Marginal improvement | Weekly median baseline captures more of the mean-reverting component than a single lag |
| Quantile ensemble (v9.0) | MAE 14.48, didn't beat single q=0.55 model (12.69) | Averaging compresses predictions further — can't break the leaf-averaging ceiling |
| Peak/off-peak split (v5.0) | DA MAE +10.1% worse | Halving training data per model loses more than the specialization gains |
| Log transform (v5.0b) | DA MAE +14.5% worse | Compresses target range further, exacerbating range-compression problem |

## Repository Structure

```
src/
├── config.py                    # ML configuration constants
├── data/
│   └── feature_engineering.py   # 100+ feature definitions (5 categories)
└── models/
    ├── trainer.py               # Model factory: HistGBT, LightGBM, XGBoost
    ├── direct_trainer.py        # Multi-horizon training + feature building
    ├── direct_predictor.py      # Inference: 7-day forecast generation
    ├── lstm_embedder.py         # LSTM price encoder (v10.0+)
    ├── evaluation.py            # MAE, RMSE, MAPE, per-hour/per-day evaluation
    └── feature_selection.py     # Two-stage selection (correlation + permutation)
```

## Setup & Running

### Requirements

Python 3.12 recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is **NOT** required for v11.0 (the current production model is single XGBoost, no LSTM). The `lstm_embedder.py` file is preserved as historical reference but is gated behind `EPF_LSTM_EMBEDDINGS=true` (default false). If you want to experiment with the LSTM code path, you'll need to (a) install PyTorch separately and (b) be aware that the v10.x bugs documented above mean the LSTM block contributes zero useful signal until the bugs are properly fixed — see the drift guard in `direct_predictor.py` which raises loudly when LSTM is enabled but the trained model expects features that aren't being populated at inference.

### Configuration

Set the model artifact directory before training:

```python
# src/config.py
MODELS_DIR = "path/to/your/models/directory"  # where .joblib files are saved
```

### Training a Model

```python
from src.models.direct_trainer import DirectTrainer

trainer = DirectTrainer(
    horizon_group="DA1",   # DA1, DA2, S1, S2, S3, S4, S5
    approach="hybrid15",   # hybrid15 (default), hourly, pure15
)
trainer.fit(df_features)   # df_features: DataFrame with all feature columns + target
trainer.save()             # saves .joblib to MODELS_DIR
```

### Generating Forecasts

```python
from src.models.direct_predictor import DirectPredictor

predictor = DirectPredictor(run_mode="dayahead")  # or "strategic"
forecasts = predictor.predict(origin_dt, ree_df, weather_df, commodity_df)
```

### Input Data Format

The training DataFrame requires:
- **Index:** `pd.DatetimeIndex` at hourly or 15-minute frequency (UTC)
- **Target column:** `day_ahead_price` (EUR/MWh)
- **Feature columns:** as generated by `feature_engineering.py`

See the [full documentation](https://epforecast.vercel.app) for data source details and column specifications.

## Version History

| Version | Key Changes | DA MAE Impact |
|---------|-------------|:-------------:|
| v10.1 | Task-aligned LSTM (24h output), exogenous inputs | 12.69 EUR/MWh |
| v10.0 | LSTM price encoder + residual-from-baseline target | -12.3% vs v4.3 |
| v4.3 | Feature selection pipeline (two-stage, per-horizon) | 14.47 EUR/MWh |
| v4.2 | 24 crisis-responsive features (gas/oil stress signals) | -2.0% |
| v4.1 | Quantile loss (q=0.55) + 15 weather interactions | -18.2% |
| v3.1 | MAE loss + bias correction | Bias: -6.94 → -0.30 |
| v2.0 | Multi-model ensemble + conformal prediction intervals | Reduced variance |
| v1.0 | Initial single-model release | First deployment |

## Citation

If you use this methodology in your research, please cite:

```
@software{epforecast2026,
  author = {Lopez Lan, Jorge},
  title = {EPForecast: Spanish Electricity Price Forecasting with Gradient Boosting and LSTM Ensembles},
  year = {2026},
  url = {https://github.com/JorgeLopezLan/epf-methodology}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Dashboard:** [epf.productjorge.com](https://epf.productjorge.com)
- **Documentation:** [epforecast.vercel.app](https://epforecast.vercel.app)
- **LinkedIn:** [Jorge Lopez Lan](https://www.linkedin.com/in/jorgelopezlan/)
