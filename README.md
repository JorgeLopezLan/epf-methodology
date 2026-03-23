# EPForecast Methodology

**Spanish Electricity Price Forecasting with Gradient Boosting + LSTM Ensembles**

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

### Model Architecture (v10.1)

The production ensemble combines three gradient boosting models with a pre-trained LSTM encoder:

#### Gradient Boosting Ensemble

Three implementations trained independently and combined via equal-weight averaging:

| Model | Library | Loss | Key Config |
|-------|---------|------|------------|
| HistGradientBoosting | scikit-learn | `quantile` (q=0.55) | depth-wise, native NaN support |
| LightGBM | Microsoft | `quantile` (alpha=0.55) | leaf-wise, GPU support |
| XGBoost | Distributed ML | `reg:quantileerror` (alpha=0.55) | histogram binning, CUDA support |

**Why quantile loss (q=0.55)?** Electricity prices are right-skewed (bounded near zero, occasional spikes >200 EUR/MWh). Quantile loss at 0.55 shifts predictions slightly above the median, directly correcting the systematic underprediction bias inherent in skewed distributions.

#### LSTM Price Encoder (v10.0+)

A pre-trained LSTM converts the last 168 hours of prices into a 64-dimensional embedding, added as extra features to the gradient boosting models. This captures temporal patterns that flattened lag columns cannot represent — in particular, multi-day trend and volatility regime signals.

Key finding: LSTM embeddings combined with **residual-from-baseline targeting** (predict deviation from weekly median rather than raw price) yielded the largest single improvement in the project, breaking the structural ceiling that tree-based models hit at v4.3.

See [`src/models/lstm_embedder.py`](src/models/lstm_embedder.py) for the implementation.

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

150-day walk-forward backtest (October 2025 – March 2026):

| Version | Product | Ensemble MAE | Notes |
|---------|---------|:------------:|-------|
| **v10.1** (current) | D+1 Day-Ahead | **12.69 EUR/MWh** | LSTM + residual target |
| **v10.1** (current) | D+2–D+7 Strategic | **17.84 EUR/MWh** | LSTM + residual target |
| v4.3 (baseline) | D+1 Day-Ahead | 14.47 EUR/MWh | Gradient boosting only |
| v4.3 (baseline) | D+2–D+7 Strategic | 19.79 EUR/MWh | Gradient boosting only |

The LSTM encoder (v10.0) broke a structural plateau held since v4.3 — tree-based models were compressing the prediction range to ~73% of actual variance due to leaf averaging. LSTM embeddings provide temporal context that gradient boosting cannot learn from flattened features alone.

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

For LSTM embeddings (v10.0+), PyTorch is required (included in `requirements.txt`). If you only want to run the gradient boosting baseline, torch is optional.

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
