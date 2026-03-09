# EPForecast Methodology

**Spanish Electricity Price Forecasting with Gradient Boosting Ensembles**

This repository contains the ML methodology code behind [EPForecast](https://epf.productjorge.com), a system that predicts Spanish day-ahead electricity prices (OMIE market) at hourly and 15-minute resolution, 7 days ahead.

> **Note:** This repository contains the ML methodology code only. The full application (API, frontend, data pipeline, deployment infrastructure) is proprietary.

## Live Demo & Documentation

- **Live dashboard:** [epf.productjorge.com](https://epf.productjorge.com)
- **Full documentation:** [epforecast.vercel.app](https://epforecast.vercel.app)
- **Methodology changelog:** [epforecast.vercel.app/changelog/overview](https://epforecast.vercel.app/changelog/overview)

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

### Model Architecture

Three gradient boosting implementations trained independently and combined via equal-weight averaging:

| Model | Library | Loss | Key Config |
|-------|---------|------|------------|
| HistGradientBoosting | scikit-learn | `quantile` (q=0.55) | depth-wise, native NaN support |
| LightGBM | Microsoft | `quantile` (alpha=0.55) | leaf-wise, GPU support |
| XGBoost | Distributed ML | `reg:quantileerror` (alpha=0.55) | histogram binning, CUDA support |

**Why quantile loss (q=0.55)?** Electricity prices are right-skewed (bounded near zero, occasional spikes >200 EUR/MWh). Standard MAE targets the median; quantile loss at 0.55 shifts predictions slightly above the median, directly correcting the systematic underprediction bias inherent in skewed distributions.

### Direct Multi-Horizon Prediction

Instead of recursive forecasting (which accumulates errors), each forecast horizon has its own model:

- **Day-ahead (DA1, DA2):** D+1 predictions from 10:00 UTC origin
- **Strategic (S1–S5):** D+2 through D+7 predictions from 15:00 UTC origin

This avoids error propagation across horizons and allows different feature sets per horizon group.

### Feature Selection (v4.3)

A two-stage per-horizon pipeline prunes noise features:

1. **Correlation filter** — Removes one of each pair with |r| > 0.95
2. **Permutation importance** — Drops features contributing < 0.1% to model performance

Strategic groups (S1–S5) automatically drop all price lag features, which are stale at multi-day horizons.

See [`src/models/feature_selection.py`](src/models/feature_selection.py) for the implementation.

### Confidence Intervals

Split conformal prediction with asymmetric bands — 50% and 90% intervals calibrated from out-of-fold residuals, bucketed by horizon group.

## Results (v4.3 Backtest)

149-day walk-forward backtest (October 2025 – February 2026):

| Product | Ensemble MAE | Best Single Model |
|---------|:-----------:|:-----------------:|
| D+1 Day-Ahead | **14.47 EUR/MWh** | XGBoost (13.95) |
| D+2–D+7 Strategic | **19.79 EUR/MWh** | HistGBT (21.42) |

February 2026 (high-volatility period driven by gas price spikes):
- D+1 MAE: **7.97 EUR/MWh** — best single-month result

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
    ├── evaluation.py            # MAE, RMSE, MAPE, per-hour/per-day evaluation
    └── feature_selection.py     # Two-stage selection (correlation + permutation)
```

## Version History

| Version | Date | Key Changes | Impact |
|---------|------|-------------|--------|
| v4.3 | 2026-03-05 | Feature selection pipeline | Strategic MAE 19.79 (-2.8%) |
| v4.2 | 2026-03-04 | 24 crisis-responsive features | Strategic MAE 20.36 (-2.0%) |
| v4.1 | 2026-03-02 | Quantile loss + 15 weather interactions | DA MAE 13.42 (-18.2%) |
| v3.1 | 2026-02-26 | MAE loss + bias correction | Bias: -6.94 to -0.30 |
| v2.0 | 2026-02-17 | Multi-model ensemble + conformal | Reduced variance |
| v1.0 | 2026-02-12 | Initial single-model release | First deployment |

## Citation

If you use this methodology in your research, please cite:

```
@software{epforecast2026,
  author = {Lopez Lan, Jorge},
  title = {EPForecast: Spanish Electricity Price Forecasting with Gradient Boosting Ensembles},
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
