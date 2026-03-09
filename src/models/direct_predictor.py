# EPForecast Methodology — Direct Multi-Horizon Predictor
# Extracted from the EPForecast project (github.com/JorgeLopezLan/epf-methodology)
# Full application: epf.productjorge.com | Docs: epforecast.vercel.app

"""
Direct multi-horizon predictor for EPF.

Uses pre-trained horizon-group models to generate 168-hour forecasts
without recursive error propagation. Each horizon group's model directly
predicts prices using only features known at the forecast origin.

Phase 1.3: Integrates Open-Meteo 7-day hourly weather forecasts so that
predictions for days 2-7 have target-hour weather information instead of
being weather-blind.
"""
import logging
import datetime
import numpy as np
import pandas as pd

# DataPipeline: provides historical data access (not included in this extract)
# REECollector: fetches data from REE/ESIOS API (not included in this extract)
from src.models.direct_trainer import (
    DirectMultiHorizonTrainer,
    HORIZON_GROUPS,
    HORIZON_GROUPS_DAYAHEAD,
    HORIZON_GROUPS_STRATEGIC,
    HORIZON_GROUPS_15MIN,
    HORIZON_GROUPS_15MIN_DAYAHEAD,
    HORIZON_GROUPS_15MIN_STRATEGIC,
    build_direct_features,
    _extract_d1_price_features,
)

logger = logging.getLogger(__name__)


def _fetch_weather_forecast() -> pd.DataFrame | None:
    """Fetch 7-day hourly weather forecast from Open-Meteo for prediction."""
    try:
        # OpenMeteoCollector: fetches Open-Meteo 7-day hourly weather forecast
        # (population-weighted national average for Spain — not included in this extract)
        from src.data.openmeteo_collector import OpenMeteoCollector
        collector = OpenMeteoCollector()
        df = collector.fetch_weighted_national()
        if df is not None and not df.empty:
            df = df.set_index("datetime_utc") if "datetime_utc" in df.columns else df
            logger.info("Fetched %d-hour weather forecast for predictions", len(df))
            return df
    except Exception as e:
        logger.warning("Could not fetch weather forecast: %s", e)
    return None


class DirectPredictor:
    """Generates 7-day forecasts using direct multi-horizon models."""

    def __init__(self, trainer: DirectMultiHorizonTrainer | None = None,
                 pipeline=None):
        # pipeline and ree are infrastructure components for data access and
        # API communication — not included in this methodology extract.
        self.pipeline = pipeline
        self.trainer = trainer or DirectMultiHorizonTrainer()
        self.model_version = "unknown"

    def load_models(self, version: str = "latest", run_mode: str | None = None):
        """Load trained direct multi-horizon models.

        Parameters
        ----------
        version : Model version or "latest".
        run_mode : "dayahead", "strategic", or None (legacy).
        """
        artifact = self.trainer.load_models(version, run_mode=run_mode)
        self.model_version = artifact.get("version", "unknown")

    def predict_next_7_days(self,
                            weather_df: pd.DataFrame | None = None,
                            commodity_df: pd.DataFrame | None = None,
                            weather_hourly_df: pd.DataFrame | None = None,
                            run_mode: str | None = None) -> pd.DataFrame:
        """
        Generate hourly price predictions.

        No recursive loop — each horizon group's model predicts directly
        from the current state, eliminating error compounding.

        Parameters
        ----------
        weather_df : Optional daily weather (legacy AEMET fallback).
        commodity_df : Optional daily commodity prices.
        weather_hourly_df : Optional hourly weather from DB (recent historical).
                           Open-Meteo 7-day forecast is automatically fetched and
                           appended to cover the prediction horizon.
        run_mode : "dayahead" (D+1, 24h), "strategic" (D+2-D+7, 144h),
                   or None (legacy 168h).

        Returns DataFrame with columns:
            prediction_date, target_date, target_hour, predicted_price,
            actual_price, model_version
        """
        if not self.trainer.models:
            raise ValueError("No models loaded. Call load_models() first.")

        # Select horizon groups based on run_mode
        if run_mode == "dayahead":
            horizon_groups = HORIZON_GROUPS_DAYAHEAD
        elif run_mode == "strategic":
            horizon_groups = HORIZON_GROUPS_STRATEGIC
        else:
            horizon_groups = HORIZON_GROUPS

        # Get recent data (14 days for feature computation)
        recent = self.pipeline.get_recent_data(days=14)
        if recent.empty:
            raise ValueError("No recent data. Run pipeline update first.")

        now = datetime.datetime.now(datetime.timezone.utc)
        prediction_date = now.strftime("%Y-%m-%d")

        # Find the most recent complete hour as our forecast origin
        origin_dt = now.replace(minute=0, second=0, microsecond=0)
        if origin_dt not in recent.index:
            # Use the last available timestamp as origin
            origin_dt = recent.index[-1]

        origin_idx = recent.index.get_loc(origin_dt)
        if isinstance(origin_idx, slice):
            origin_idx = origin_idx.stop - 1

        # Build combined weather: historical hourly + 7-day forecast
        combined_weather = self._build_combined_weather(weather_hourly_df)

        predictions = []
        price = recent["day_ahead_price"]

        for group_name, hours in horizon_groups.items():
            if group_name not in self.trainer.models:
                logger.warning("No model for %s, skipping", group_name)
                continue

            model = self.trainer.models[group_name]
            feature_cols = self.trainer.feature_names[group_name]

            for h in hours:
                target_dt = origin_dt + pd.Timedelta(hours=h)

                # Build feature vector for this origin + horizon
                feat = self._build_origin_features(
                    recent, origin_idx, origin_dt, target_dt, h,
                    weather_df, commodity_df, combined_weather,
                    run_mode=run_mode,
                )

                if feat is not None:
                    # Align features with model's expected columns
                    row = pd.DataFrame([feat])
                    for f in feature_cols:
                        if f not in row.columns:
                            row[f] = 0.0

                    try:
                        pred_price = float(model.predict(row[feature_cols].values)[0])
                        if np.isnan(pred_price):
                            logger.warning("NaN prediction for %s h=%d, using fallback", group_name, h)
                            pred_price = float(price.tail(24).mean())
                    except Exception as e:
                        logger.warning("Prediction error for %s h=%d: %s", group_name, h, e)
                        pred_price = float(price.tail(24).mean())
                else:
                    pred_price = float(price.tail(24).mean())

                predictions.append({
                    "prediction_date": prediction_date,
                    "target_date": target_dt.strftime("%Y-%m-%d"),
                    "target_hour": target_dt.hour,
                    "predicted_price": round(pred_price, 2),
                    "actual_price": None,
                    "model_version": self.model_version,
                    "_hours_ahead": h,
                })

        result = pd.DataFrame(predictions)
        # Sort by target datetime
        result = result.sort_values(["target_date", "target_hour"]).reset_index(drop=True)

        # Add confidence intervals if conformal calibrator is available
        if self.trainer.conformal_calibrator is not None and len(result) > 0:
            intervals = self.trainer.conformal_calibrator.predict_intervals(
                result["predicted_price"].values,
                result["_hours_ahead"].values,
            )
            result["prediction_lower_90"] = np.round(intervals["lower_90"], 2)
            result["prediction_upper_90"] = np.round(intervals["upper_90"], 2)
            result["prediction_lower_50"] = np.round(intervals["lower_50"], 2)
            result["prediction_upper_50"] = np.round(intervals["upper_50"], 2)
            logger.info("Added 50%% and 90%% prediction intervals")
        else:
            result["prediction_lower_90"] = None
            result["prediction_upper_90"] = None
            result["prediction_lower_50"] = None
            result["prediction_upper_50"] = None

        result = result.drop(columns=["_hours_ahead"])
        logger.info("Generated %d direct predictions (run_mode=%s)", len(result), run_mode)
        return result

    def predict_from_origin(self, data: pd.DataFrame,
                            origin_dt: pd.Timestamp,
                            weather_hourly_df: pd.DataFrame | None = None,
                            weather_df: pd.DataFrame | None = None,
                            commodity_df: pd.DataFrame | None = None,
                            run_mode: str | None = None) -> pd.DataFrame:
        """Generate forecast from an explicit historical origin.

        Used for backtesting — takes pre-sliced data (no DB queries, no live
        weather API calls). The caller is responsible for ensuring `data` only
        contains information available at `origin_dt` (no future leakage).

        Parameters
        ----------
        data : DataFrame indexed by datetime_utc, sliced up to origin_dt.
        origin_dt : The forecast origin timestamp.
        weather_hourly_df : Historical hourly weather (covers both origin and
                           future target hours since this is a backtest).
        weather_df : Optional daily weather.
        commodity_df : Optional daily commodity prices.
        run_mode : "dayahead", "strategic", or None (legacy H1-H8).
        """
        if not self.trainer.models:
            raise ValueError("No models loaded. Call load_models() first.")

        # Select horizon groups based on run_mode
        if run_mode == "dayahead":
            horizon_groups = HORIZON_GROUPS_DAYAHEAD
        elif run_mode == "strategic":
            horizon_groups = HORIZON_GROUPS_STRATEGIC
        else:
            horizon_groups = HORIZON_GROUPS

        prediction_date = origin_dt.strftime("%Y-%m-%d")

        origin_idx = data.index.get_loc(origin_dt)
        if isinstance(origin_idx, slice):
            origin_idx = origin_idx.stop - 1

        predictions = []
        price = data["day_ahead_price"]

        for group_name, hours in horizon_groups.items():
            if group_name not in self.trainer.models:
                continue

            model = self.trainer.models[group_name]
            feature_cols = self.trainer.feature_names[group_name]

            for h in hours:
                target_dt = origin_dt + pd.Timedelta(hours=h)

                feat = self._build_origin_features(
                    data, origin_idx, origin_dt, target_dt, h,
                    weather_df, commodity_df, weather_hourly_df,
                    run_mode=run_mode,
                )

                if feat is not None:
                    row = pd.DataFrame([feat])
                    for f in feature_cols:
                        if f not in row.columns:
                            row[f] = 0.0
                    try:
                        pred_price = float(model.predict(row[feature_cols].values)[0])
                    except Exception:
                        pred_price = float(price.tail(24).mean())
                else:
                    pred_price = float(price.tail(24).mean())

                predictions.append({
                    "prediction_date": prediction_date,
                    "target_date": target_dt.strftime("%Y-%m-%d"),
                    "target_hour": target_dt.hour,
                    "predicted_price": round(pred_price, 2),
                    "actual_price": None,
                    "model_version": self.model_version,
                    "_hours_ahead": h,
                })

        result = pd.DataFrame(predictions)
        result = result.sort_values(["target_date", "target_hour"]).reset_index(drop=True)

        if self.trainer.conformal_calibrator is not None and len(result) > 0:
            intervals = self.trainer.conformal_calibrator.predict_intervals(
                result["predicted_price"].values,
                result["_hours_ahead"].values,
            )
            result["prediction_lower_90"] = np.round(intervals["lower_90"], 2)
            result["prediction_upper_90"] = np.round(intervals["upper_90"], 2)
            result["prediction_lower_50"] = np.round(intervals["lower_50"], 2)
            result["prediction_upper_50"] = np.round(intervals["upper_50"], 2)
        else:
            result["prediction_lower_90"] = None
            result["prediction_upper_90"] = None
            result["prediction_lower_50"] = None
            result["prediction_upper_50"] = None

        result = result.drop(columns=["_hours_ahead"])
        return result

    def predict_from_origin_15min(self, data: pd.DataFrame,
                                   origin_dt: pd.Timestamp,
                                   weather_hourly_df: pd.DataFrame | None = None,
                                   commodity_df: pd.DataFrame | None = None,
                                   run_mode: str | None = None) -> pd.DataFrame:
        """Generate 15-min forecast from an explicit historical origin.

        Used for backtesting 15-min approaches. Takes pre-sliced 15-min data
        (no DB queries, no live API calls). The caller is responsible for
        ensuring `data` only contains information available at `origin_dt`.

        Parameters
        ----------
        data : DataFrame indexed by datetime_utc (15-min resolution), sliced up to origin_dt.
        origin_dt : The forecast origin timestamp.
        weather_hourly_df : Historical hourly weather (covers both origin and future).
        commodity_df : Optional daily commodity prices.
        run_mode : "dayahead", "strategic", or None (legacy).
                   Selects horizon groups (DA1-DA2 / S1-S5 / D1-D7).
        """
        if not self.trainer.models:
            raise ValueError("No models loaded. Call load_models() first.")

        prediction_date = origin_dt.strftime("%Y-%m-%d")

        origin_idx = data.index.get_loc(origin_dt)
        if isinstance(origin_idx, slice):
            origin_idx = origin_idx.stop - 1

        # Broadcast hourly weather to 15-min if available
        weather_15min = None
        if weather_hourly_df is not None and not weather_hourly_df.empty:
            future_end = origin_dt + pd.Timedelta(days=8)
            full_15min_idx = pd.date_range(
                data.index.min(), future_end, freq="15min", tz="UTC"
            )
            weather_15min = weather_hourly_df.reindex(full_15min_idx, method="ffill")

        predictions = []
        price = data["day_ahead_price"]

        # Select horizon groups based on run_mode
        if run_mode == "dayahead":
            horizon_groups = HORIZON_GROUPS_15MIN_DAYAHEAD
        elif run_mode == "strategic":
            horizon_groups = HORIZON_GROUPS_15MIN_STRATEGIC
        else:
            horizon_groups = HORIZON_GROUPS_15MIN

        for group_name, quarters in horizon_groups.items():
            if group_name not in self.trainer.models:
                continue

            model = self.trainer.models[group_name]
            feature_cols = self.trainer.feature_names[group_name]

            for q in quarters:
                target_dt = origin_dt + pd.Timedelta(minutes=q * 15)

                feat = self._build_origin_features_15min(
                    data, origin_idx, origin_dt, target_dt, q,
                    commodity_df, weather_15min, run_mode=run_mode,
                )

                if feat is not None:
                    row = pd.DataFrame([feat])
                    for f in feature_cols:
                        if f not in row.columns:
                            row[f] = 0.0
                    try:
                        pred_price = float(model.predict(row[feature_cols].values)[0])
                    except Exception:
                        pred_price = float(price.tail(96).mean())
                else:
                    pred_price = float(price.tail(96).mean())

                predictions.append({
                    "prediction_date": prediction_date,
                    "target_date": target_dt.strftime("%Y-%m-%d"),
                    "target_hour": target_dt.hour,
                    "target_minute": target_dt.minute,
                    "predicted_price": round(pred_price, 2),
                    "actual_price": None,
                    "model_version": self.model_version,
                    "_quarters_ahead": q,
                })

        result = pd.DataFrame(predictions)
        result = result.sort_values(
            ["target_date", "target_hour", "target_minute"]
        ).reset_index(drop=True)

        if self.trainer.conformal_calibrator is not None and len(result) > 0:
            intervals = self.trainer.conformal_calibrator.predict_intervals(
                result["predicted_price"].values,
                result["_quarters_ahead"].values,
            )
            result["prediction_lower_90"] = np.round(intervals["lower_90"], 2)
            result["prediction_upper_90"] = np.round(intervals["upper_90"], 2)
            result["prediction_lower_50"] = np.round(intervals["lower_50"], 2)
            result["prediction_upper_50"] = np.round(intervals["upper_50"], 2)
        else:
            result["prediction_lower_90"] = None
            result["prediction_upper_90"] = None
            result["prediction_lower_50"] = None
            result["prediction_upper_50"] = None

        result = result.drop(columns=["_quarters_ahead"])
        return result

    def _build_combined_weather(self,
                                 weather_hourly_df: pd.DataFrame | None) -> pd.DataFrame | None:
        """
        Combine historical hourly weather with Open-Meteo 7-day forecast.

        The historical data covers the past (origin-time weather features).
        The forecast covers the future (target-hour weather features for days 1-7).
        """
        forecast_df = _fetch_weather_forecast()

        if weather_hourly_df is not None and forecast_df is not None:
            # Combine: historical for past, forecast for future
            # Forecast takes precedence for overlapping timestamps
            combined = pd.concat([weather_hourly_df, forecast_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            logger.info("Combined weather: %d historical + %d forecast = %d total hours",
                        len(weather_hourly_df), len(forecast_df), len(combined))
            return combined
        elif forecast_df is not None:
            return forecast_df
        elif weather_hourly_df is not None:
            return weather_hourly_df
        return None

    def _build_origin_features(self, data: pd.DataFrame,
                                origin_idx: int,
                                origin_dt: pd.Timestamp,
                                target_dt: pd.Timestamp,
                                hours_ahead: int,
                                weather_df: pd.DataFrame | None,
                                commodity_df: pd.DataFrame | None,
                                weather_hourly_df: pd.DataFrame | None = None,
                                run_mode: str | None = None) -> dict | None:
        """Build feature dict for a single origin+target pair."""
        try:
            price = data["day_ahead_price"]
            feat = {}

            # Price lags relative to origin
            feat["price_lag_1h"] = price.iloc[origin_idx - 1] if origin_idx >= 1 else np.nan
            feat["price_lag_2h"] = price.iloc[origin_idx - 2] if origin_idx >= 2 else np.nan
            feat["price_lag_3h"] = price.iloc[origin_idx - 3] if origin_idx >= 3 else np.nan
            feat["price_lag_24h"] = price.iloc[origin_idx - 24] if origin_idx >= 24 else np.nan
            feat["price_lag_48h"] = price.iloc[origin_idx - 48] if origin_idx >= 48 else np.nan
            feat["price_lag_72h"] = price.iloc[origin_idx - 72] if origin_idx >= 72 else np.nan
            feat["price_lag_168h"] = price.iloc[origin_idx - 168] if origin_idx >= 168 else np.nan
            feat["price_lag_336h"] = price.iloc[origin_idx - 336] if origin_idx >= 336 else np.nan
            feat["price_lag_504h"] = price.iloc[origin_idx - 504] if origin_idx >= 504 else np.nan

            # Same-weekday 4-week average and std
            same_wd_prices = []
            for w in range(1, 5):
                idx = origin_idx - (w * 168)
                if idx >= 0:
                    same_wd_prices.append(price.iloc[idx])
            feat["price_same_weekday_4w_avg"] = float(np.mean(same_wd_prices)) if same_wd_prices else np.nan
            feat["price_same_weekday_4w_std"] = float(np.std(same_wd_prices)) if len(same_wd_prices) >= 2 else np.nan

            # Rolling statistics
            recent_24 = price.iloc[max(0, origin_idx - 24):origin_idx]
            recent_168 = price.iloc[max(0, origin_idx - 168):origin_idx]
            feat["price_rolling_24h"] = recent_24.mean() if len(recent_24) > 0 else np.nan
            feat["price_rolling_168h"] = recent_168.mean() if len(recent_168) > 0 else np.nan
            feat["price_std_24h"] = recent_24.std() if len(recent_24) > 1 else 0.0
            feat["price_std_168h"] = recent_168.std() if len(recent_168) > 1 else 0.0
            feat["price_min_24h"] = recent_24.min() if len(recent_24) > 0 else np.nan
            feat["price_max_24h"] = recent_24.max() if len(recent_24) > 0 else np.nan
            feat["price_range_24h"] = feat["price_max_24h"] - feat["price_min_24h"]
            feat["price_change_24h"] = (
                price.iloc[origin_idx - 1] - price.iloc[origin_idx - 25]
                if origin_idx >= 25 else np.nan
            )

            # Volatility & regime indicators (3.3)
            q25 = recent_168.quantile(0.25) if len(recent_168) >= 4 else np.nan
            q75 = recent_168.quantile(0.75) if len(recent_168) >= 4 else np.nan
            feat["price_iqr_7d"] = q75 - q25 if not (np.isnan(q25) or np.isnan(q75)) else np.nan
            feat["price_skewness_7d"] = float(recent_168.skew()) if len(recent_168) >= 24 else np.nan
            recent_30d = price.iloc[max(0, origin_idx - 720):origin_idx]
            median_30d = recent_30d.median() if len(recent_30d) > 0 else np.nan
            if not np.isnan(median_30d) and median_30d > 0:
                feat["extreme_price_count_7d"] = float((recent_168 > median_30d * 2).sum())
            else:
                feat["extreme_price_count_7d"] = 0.0
            std_7d = feat["price_std_168h"]
            std_30d = recent_30d.std() if len(recent_30d) > 48 else np.nan
            feat["vol_regime_high"] = (
                1.0 if (not np.isnan(std_30d) and std_7d > 2.0 * std_30d) else 0.0
            )
            change_prev_24h = (
                price.iloc[origin_idx - 25] - price.iloc[origin_idx - 49]
                if origin_idx >= 49 else np.nan
            )
            feat["price_acceleration_24h"] = (
                feat["price_change_24h"] - change_prev_24h
                if not (np.isnan(feat.get("price_change_24h", np.nan)) or np.isnan(change_prev_24h))
                else np.nan
            )

            # Price curve shape features (3.1)
            if len(recent_24) >= 24:
                peak_prices = []
                offpeak_prices = []
                for j in range(len(recent_24)):
                    h = price.index[max(0, origin_idx - 24) + j].hour
                    if 8 <= h <= 21:
                        peak_prices.append(recent_24.iloc[j])
                    else:
                        offpeak_prices.append(recent_24.iloc[j])
                peak_avg = np.mean(peak_prices) if peak_prices else np.nan
                offpeak_avg = np.mean(offpeak_prices) if offpeak_prices else np.nan
                feat["price_peak_to_base_ratio"] = (
                    peak_avg / offpeak_avg if offpeak_avg and offpeak_avg > 0 else np.nan
                )
                morning = [recent_24.iloc[j] for j in range(len(recent_24))
                            if 6 <= price.index[max(0, origin_idx - 24) + j].hour <= 9]
                night = [recent_24.iloc[j] for j in range(len(recent_24))
                          if price.index[max(0, origin_idx - 24) + j].hour <= 5]
                feat["price_morning_slope"] = (
                    (np.mean(morning) - np.mean(night))
                    if morning and night else np.nan
                )
                evening = [recent_24.iloc[j] for j in range(len(recent_24))
                            if 17 <= price.index[max(0, origin_idx - 24) + j].hour <= 21]
                afternoon = [recent_24.iloc[j] for j in range(len(recent_24))
                              if 12 <= price.index[max(0, origin_idx - 24) + j].hour <= 16]
                feat["price_evening_slope"] = (
                    (np.mean(evening) - np.mean(afternoon))
                    if evening and afternoon else np.nan
                )
            else:
                feat["price_peak_to_base_ratio"] = np.nan
                feat["price_morning_slope"] = np.nan
                feat["price_evening_slope"] = np.nan

            # Origin time features
            feat["origin_hour_sin"] = np.sin(2 * np.pi * origin_dt.hour / 24)
            feat["origin_hour_cos"] = np.cos(2 * np.pi * origin_dt.hour / 24)
            feat["origin_dow_sin"] = np.sin(2 * np.pi * origin_dt.weekday() / 7)
            feat["origin_dow_cos"] = np.cos(2 * np.pi * origin_dt.weekday() / 7)
            feat["origin_month_sin"] = np.sin(2 * np.pi * origin_dt.month / 12)
            feat["origin_month_cos"] = np.cos(2 * np.pi * origin_dt.month / 12)
            feat["origin_is_weekend"] = 1.0 if origin_dt.weekday() >= 5 else 0.0

            # Target time features
            feat["target_hour_sin"] = np.sin(2 * np.pi * target_dt.hour / 24)
            feat["target_hour_cos"] = np.cos(2 * np.pi * target_dt.hour / 24)
            feat["target_dow_sin"] = np.sin(2 * np.pi * target_dt.weekday() / 7)
            feat["target_dow_cos"] = np.cos(2 * np.pi * target_dt.weekday() / 7)
            feat["target_is_weekend"] = 1.0 if target_dt.weekday() >= 5 else 0.0
            feat["hours_ahead"] = hours_ahead

            # Demand features
            if "real_demand" in data.columns:
                feat["demand_at_origin"] = data["real_demand"].iloc[origin_idx]
                feat["demand_lag_24h"] = (
                    data["real_demand"].iloc[origin_idx - 24] if origin_idx >= 24 else np.nan
                )
                # Demand ramp features (3.2)
                feat["demand_ramp_4h"] = (
                    data["real_demand"].iloc[origin_idx] - data["real_demand"].iloc[origin_idx - 4]
                    if origin_idx >= 4 else np.nan
                )
            if "demand_forecast" in data.columns:
                feat["demand_forecast_at_origin"] = data["demand_forecast"].iloc[origin_idx]
                # Use demand forecast for target hour if available
                target_idx = origin_idx + hours_ahead
                if target_idx < len(data):
                    feat["demand_forecast_target"] = data["demand_forecast"].iloc[target_idx]

            # Generation mix at origin
            demand_val = data.get("real_demand", pd.Series(1, index=data.index)).iloc[origin_idx]
            demand_safe = demand_val if demand_val != 0 else np.nan
            wind_val = data.get("wind_generation", pd.Series(0, index=data.index)).iloc[origin_idx]
            solar_pv_val = data.get("solar_pv_generation", pd.Series(0, index=data.index)).iloc[origin_idx]
            solar_th_val = data.get("solar_thermal_gen", pd.Series(0, index=data.index)).iloc[origin_idx]
            nuclear_val = data.get("nuclear_generation", pd.Series(0, index=data.index)).iloc[origin_idx]

            total_ren = wind_val + solar_pv_val + solar_th_val
            feat["renewable_share"] = total_ren / demand_safe if demand_safe else np.nan
            feat["wind_share"] = wind_val / demand_safe if demand_safe else np.nan
            feat["solar_share"] = (solar_pv_val + solar_th_val) / demand_safe if demand_safe else np.nan
            feat["nuclear_share"] = nuclear_val / demand_safe if demand_safe else np.nan
            feat["residual_demand"] = demand_val - total_ren - nuclear_val

            # Price at target hour from yesterday/last week (known at origin)
            same_hour_yesterday = origin_idx - 24 + (hours_ahead % 24)
            if 0 <= same_hour_yesterday < origin_idx:
                feat["target_hour_price_yesterday"] = price.iloc[same_hour_yesterday]
            else:
                feat["target_hour_price_yesterday"] = np.nan

            same_hour_last_week = origin_idx - 168 + (hours_ahead % 168)
            if 0 <= same_hour_last_week < origin_idx:
                feat["target_hour_price_last_week"] = price.iloc[same_hour_last_week]
            else:
                feat["target_hour_price_last_week"] = np.nan

            # Weather features at origin
            if weather_hourly_df is not None and not weather_hourly_df.empty:
                if origin_dt in weather_hourly_df.index:
                    w_row = weather_hourly_df.loc[origin_dt]
                    for col in weather_hourly_df.columns:
                        feat[f"weather_{col}"] = w_row[col]
                    # Weather interaction features (encode physical relationships)
                    _tc = w_row.get("temp_c", np.nan)
                    _ws = w_row.get("wind_speed_kmh", 0) or 0
                    _pr = w_row.get("precipitation_mm", 0) or 0
                    _cl = w_row.get("cloud_cover_pct", 0) or 0
                    _dr = w_row.get("direct_radiation_wm2", 0) or 0
                    _di = w_row.get("diffuse_radiation_wm2", 0) or 0
                    _sh = w_row.get("sunshine_hours", 0) or 0
                    if not np.isnan(_tc):
                        feat["weather_cold_x_demand"] = max(0, 15 - _tc) * demand_val
                        feat["weather_temp_deviation"] = _tc - 15.0
                        feat["weather_heating_degree_days"] = max(0, 18 - _tc)
                        feat["weather_cooling_degree_days"] = max(0, _tc - 24)
                        feat["weather_temp_deviation_sq"] = (_tc - 15.0) ** 2
                    else:
                        feat["weather_cold_x_demand"] = np.nan
                        feat["weather_temp_deviation"] = np.nan
                        feat["weather_heating_degree_days"] = np.nan
                        feat["weather_cooling_degree_days"] = np.nan
                        feat["weather_temp_deviation_sq"] = np.nan
                    _wind_share = feat.get("wind_share", 0) or 0
                    _solar_share = feat.get("solar_share", 0) or 0
                    feat["weather_wind_x_wind_share"] = _ws * _wind_share
                    hydro_val = data.get("hydro_generation", pd.Series(0, index=data.index)).iloc[origin_idx]
                    feat["weather_precip_x_hydro"] = _pr * hydro_val / demand_safe if demand_safe else np.nan
                    feat["weather_cloud_x_solar"] = _cl * _solar_share
                    feat["weather_sunshine_x_solar"] = _sh * _solar_share
                    feat["weather_ghi"] = _dr + _di
            elif weather_df is not None and not weather_df.empty:
                origin_date = origin_dt.date() if hasattr(origin_dt, 'date') else origin_dt
                if hasattr(weather_df.index, "date"):
                    weather_idx = weather_df.index.date
                else:
                    weather_idx = pd.to_datetime(weather_df.index).date
                if origin_date in weather_idx:
                    mask = weather_idx == origin_date
                    w_row = weather_df[mask].iloc[0]
                    for col in weather_df.columns:
                        feat[f"weather_{col}"] = w_row[col] if col in w_row.index else np.nan

            # Target-hour weather features (Phase 1.3)
            # Uses Open-Meteo forecast for future hours
            if weather_hourly_df is not None and not weather_hourly_df.empty:
                if target_dt in weather_hourly_df.index:
                    tw = weather_hourly_df.loc[target_dt]
                    feat["target_weather_temp_c"] = tw.get("temp_c", np.nan)
                    feat["target_weather_wind_kmh"] = tw.get("wind_speed_kmh", np.nan)
                    feat["target_weather_cloud_pct"] = tw.get("cloud_cover_pct", np.nan)
                    _tdr = (tw.get("direct_radiation_wm2", 0) or 0)
                    _tdi = (tw.get("diffuse_radiation_wm2", 0) or 0)
                    feat["target_weather_radiation"] = _tdr + _tdi
                    feat["target_weather_precip_mm"] = tw.get("precipitation_mm", np.nan)
                    # Target weather interaction features
                    _ttc = tw.get("temp_c", np.nan)
                    if _ttc is not None and not np.isnan(_ttc):
                        feat["target_weather_temp_deviation"] = _ttc - 15.0
                        feat["target_weather_heating_dd"] = max(0, 18 - _ttc)
                        feat["target_weather_cooling_dd"] = max(0, _ttc - 24)
                    else:
                        feat["target_weather_temp_deviation"] = np.nan
                        feat["target_weather_heating_dd"] = np.nan
                        feat["target_weather_cooling_dd"] = np.nan
                    _tws = feat.get("wind_share", 0) or 0
                    _tss = feat.get("solar_share", 0) or 0
                    feat["target_weather_wind_x_wind_share"] = (tw.get("wind_speed_kmh", 0) or 0) * _tws
                    feat["target_weather_cloud_x_solar"] = (tw.get("cloud_cover_pct", 0) or 0) * _tss
                    feat["target_weather_ghi_x_solar"] = (_tdr + _tdi) * _tss
                else:
                    feat["target_weather_temp_c"] = np.nan
                    feat["target_weather_wind_kmh"] = np.nan
                    feat["target_weather_cloud_pct"] = np.nan
                    feat["target_weather_radiation"] = np.nan
                    feat["target_weather_precip_mm"] = np.nan
                    feat["target_weather_temp_deviation"] = np.nan
                    feat["target_weather_heating_dd"] = np.nan
                    feat["target_weather_cooling_dd"] = np.nan
                    feat["target_weather_wind_x_wind_share"] = np.nan
                    feat["target_weather_cloud_x_solar"] = np.nan
                    feat["target_weather_ghi_x_solar"] = np.nan

            # Commodity features
            if commodity_df is not None and not commodity_df.empty:
                origin_date = origin_dt.date() if hasattr(origin_dt, 'date') else origin_dt
                if hasattr(commodity_df.index, "date"):
                    comm_idx = commodity_df.index.date
                else:
                    comm_idx = pd.to_datetime(commodity_df.index).date
                if origin_date in comm_idx:
                    mask = comm_idx == origin_date
                    c_row = commodity_df[mask].iloc[0]
                    for col in commodity_df.columns:
                        feat[f"commodity_{col}"] = c_row[col] if col in c_row.index else np.nan

            # Commodity derivatives (3.5): momentum, ratios, marginal cost, interactions
            from src.models.direct_trainer import _compute_commodity_derivatives
            comm_derivs = _compute_commodity_derivatives(
                commodity_df, origin_dt,
                feat.get("price_lag_24h", np.nan),
                feat.get("residual_demand", np.nan),
            )
            feat.update(comm_derivs)

            # Demand ramp x gas interaction (3.2)
            gas_val = feat.get("commodity_ttf_gas_eur_mwh", np.nan)
            demand_ramp = feat.get("demand_ramp_4h", np.nan)
            if gas_val and not np.isnan(gas_val) and not np.isnan(demand_ramp):
                feat["demand_ramp_x_gas"] = demand_ramp * gas_val
            else:
                feat["demand_ramp_x_gas"] = np.nan

            # D+1 published price features (strategic mode only)
            # At afternoon origin, OMIE has already published D+1 prices
            if run_mode == "strategic":
                d1_feats = _extract_d1_price_features(
                    price, origin_dt, target_dt, data.index,
                )
                feat.update(d1_feats)
            elif run_mode == "dayahead":
                # Explicitly set NaN — D+1 prices not yet published at morning origin
                feat["d1_mean_price"] = np.nan
                feat["d1_min_price"] = np.nan
                feat["d1_max_price"] = np.nan
                feat["d1_std_price"] = np.nan
                feat["d1_peak_spread"] = np.nan
                feat["d1_same_hour_price"] = np.nan

            return feat

        except Exception as e:
            logger.warning("Error building features for h=%d: %s", hours_ahead, e)
            return None

    def predict_next_7_days_15min(self,
                                  weather_hourly_df: pd.DataFrame | None = None,
                                  commodity_df: pd.DataFrame | None = None,
                                  run_mode: str | None = None) -> pd.DataFrame:
        """
        Generate 15-min price predictions.

        Uses 15-min data from ree_15min table. Weather (hourly) is broadcast
        to 15-min via forward-fill.

        Parameters
        ----------
        run_mode : "dayahead" (D+1, DA1-DA2), "strategic" (D+2-D+7, S1-S5),
                   or None (legacy D1-D7, 672 quarter-hours).

        Returns DataFrame with columns:
            prediction_date, target_date, target_hour, target_minute,
            predicted_price, actual_price, model_version
        """
        if not self.trainer.models:
            raise ValueError("No models loaded. Call load_models() first.")

        # Get recent 15-min data (14 days)
        recent = self.pipeline.get_15min_data(
            start_date=(
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(days=14)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        if recent.empty:
            raise ValueError("No recent 15-min data. Run 15-min backfill first.")

        now = datetime.datetime.now(datetime.timezone.utc)
        prediction_date = now.strftime("%Y-%m-%d")

        # Find the most recent 15-min step as origin
        origin_dt = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        if origin_dt not in recent.index:
            origin_dt = recent.index[-1]

        origin_idx = recent.index.get_loc(origin_dt)
        if isinstance(origin_idx, slice):
            origin_idx = origin_idx.stop - 1

        # Build combined weather
        combined_weather = self._build_combined_weather(weather_hourly_df)
        # Broadcast hourly weather to 15-min if needed
        weather_15min = None
        if combined_weather is not None and not combined_weather.empty:
            future_end = origin_dt + pd.Timedelta(days=8)
            full_15min_idx = pd.date_range(
                recent.index.min(), future_end, freq="15min", tz="UTC"
            )
            weather_15min = combined_weather.reindex(full_15min_idx, method="ffill")

        predictions = []
        price = recent["day_ahead_price"]

        # Select horizon groups based on run_mode
        if run_mode == "dayahead":
            horizon_groups = HORIZON_GROUPS_15MIN_DAYAHEAD
        elif run_mode == "strategic":
            horizon_groups = HORIZON_GROUPS_15MIN_STRATEGIC
        else:
            horizon_groups = HORIZON_GROUPS_15MIN

        for group_name, quarters in horizon_groups.items():
            if group_name not in self.trainer.models:
                logger.warning("No model for %s, skipping", group_name)
                continue

            model = self.trainer.models[group_name]
            feature_cols = self.trainer.feature_names[group_name]

            for q in quarters:
                target_dt = origin_dt + pd.Timedelta(minutes=q * 15)

                feat = self._build_origin_features_15min(
                    recent, origin_idx, origin_dt, target_dt, q,
                    commodity_df, weather_15min, run_mode=run_mode,
                )

                if feat is not None:
                    row = pd.DataFrame([feat])
                    for f in feature_cols:
                        if f not in row.columns:
                            row[f] = 0.0
                    try:
                        pred_price = float(model.predict(row[feature_cols].values)[0])
                        if np.isnan(pred_price):
                            logger.warning("NaN prediction for %s q=%d, using fallback", group_name, q)
                            pred_price = float(price.tail(96).mean())
                    except Exception as e:
                        logger.warning("15min prediction error %s q=%d: %s",
                                       group_name, q, e)
                        pred_price = float(price.tail(96).mean())
                else:
                    pred_price = float(price.tail(96).mean())

                predictions.append({
                    "prediction_date": prediction_date,
                    "target_date": target_dt.strftime("%Y-%m-%d"),
                    "target_hour": target_dt.hour,
                    "target_minute": target_dt.minute,
                    "predicted_price": round(pred_price, 2),
                    "actual_price": None,
                    "model_version": self.model_version,
                    "_quarters_ahead": q,
                })

        result = pd.DataFrame(predictions)
        if result.empty:
            logger.warning("No predictions generated (no matching horizon groups for loaded models)")
            return result
        result = result.sort_values(
            ["target_date", "target_hour", "target_minute"]
        ).reset_index(drop=True)

        # Add confidence intervals
        if self.trainer.conformal_calibrator is not None and len(result) > 0:
            intervals = self.trainer.conformal_calibrator.predict_intervals(
                result["predicted_price"].values,
                result["_quarters_ahead"].values,
            )
            result["prediction_lower_90"] = np.round(intervals["lower_90"], 2)
            result["prediction_upper_90"] = np.round(intervals["upper_90"], 2)
            result["prediction_lower_50"] = np.round(intervals["lower_50"], 2)
            result["prediction_upper_50"] = np.round(intervals["upper_50"], 2)
        else:
            result["prediction_lower_90"] = None
            result["prediction_upper_90"] = None
            result["prediction_lower_50"] = None
            result["prediction_upper_50"] = None

        result = result.drop(columns=["_quarters_ahead"])
        logger.info("Generated %d 15-min direct predictions", len(result))
        return result

    def _build_origin_features_15min(self, data: pd.DataFrame,
                                      origin_idx: int,
                                      origin_dt: pd.Timestamp,
                                      target_dt: pd.Timestamp,
                                      quarters_ahead: int,
                                      commodity_df: pd.DataFrame | None,
                                      weather_15min: pd.DataFrame | None,
                                      run_mode: str | None = None) -> dict | None:
        """Build feature dict for a single 15-min origin+target pair."""
        try:
            price = data["day_ahead_price"]
            feat = {}

            # Price lags scaled to 15-min
            feat["price_lag_15m"] = price.iloc[origin_idx - 1] if origin_idx >= 1 else np.nan
            feat["price_lag_30m"] = price.iloc[origin_idx - 2] if origin_idx >= 2 else np.nan
            feat["price_lag_45m"] = price.iloc[origin_idx - 3] if origin_idx >= 3 else np.nan
            feat["price_lag_1h"] = price.iloc[origin_idx - 4] if origin_idx >= 4 else np.nan
            feat["price_lag_2h"] = price.iloc[origin_idx - 8] if origin_idx >= 8 else np.nan
            feat["price_lag_3h"] = price.iloc[origin_idx - 12] if origin_idx >= 12 else np.nan
            feat["price_lag_24h"] = price.iloc[origin_idx - 96] if origin_idx >= 96 else np.nan
            feat["price_lag_48h"] = price.iloc[origin_idx - 192] if origin_idx >= 192 else np.nan
            feat["price_lag_168h"] = price.iloc[origin_idx - 672] if origin_idx >= 672 else np.nan
            feat["price_lag_336h"] = price.iloc[origin_idx - 1344] if origin_idx >= 1344 else np.nan
            feat["price_lag_504h"] = price.iloc[origin_idx - 2016] if origin_idx >= 2016 else np.nan

            # Same-weekday 4-week average and std (15-min: 672 steps = 1 week)
            same_wd_prices = []
            for w in range(1, 5):
                idx = origin_idx - (w * 672)
                if idx >= 0:
                    same_wd_prices.append(price.iloc[idx])
            feat["price_same_weekday_4w_avg"] = float(np.mean(same_wd_prices)) if same_wd_prices else np.nan
            feat["price_same_weekday_4w_std"] = float(np.std(same_wd_prices)) if len(same_wd_prices) >= 2 else np.nan

            # Rolling stats
            recent_24h = price.iloc[max(0, origin_idx - 96):origin_idx]
            recent_7d = price.iloc[max(0, origin_idx - 672):origin_idx]
            feat["price_rolling_24h"] = recent_24h.mean() if len(recent_24h) > 0 else np.nan
            feat["price_rolling_7d"] = recent_7d.mean() if len(recent_7d) > 0 else np.nan
            feat["price_std_24h"] = recent_24h.std() if len(recent_24h) > 1 else 0.0
            feat["price_std_7d"] = recent_7d.std() if len(recent_7d) > 1 else 0.0
            feat["price_min_24h"] = recent_24h.min() if len(recent_24h) > 0 else np.nan
            feat["price_max_24h"] = recent_24h.max() if len(recent_24h) > 0 else np.nan
            feat["price_range_24h"] = (feat["price_max_24h"] or 0) - (feat["price_min_24h"] or 0)
            feat["price_change_1h"] = (
                price.iloc[origin_idx - 1] - price.iloc[origin_idx - 5]
                if origin_idx >= 5 else 0.0
            )
            feat["price_change_24h"] = (
                price.iloc[origin_idx - 1] - price.iloc[origin_idx - 97]
                if origin_idx >= 97 else np.nan
            )

            # Volatility & regime indicators (3.3) — 15-min resolution
            q25 = recent_7d.quantile(0.25) if len(recent_7d) >= 4 else np.nan
            q75 = recent_7d.quantile(0.75) if len(recent_7d) >= 4 else np.nan
            feat["price_iqr_7d"] = q75 - q25 if not (np.isnan(q25) or np.isnan(q75)) else np.nan
            feat["price_skewness_7d"] = float(recent_7d.skew()) if len(recent_7d) >= 96 else np.nan
            recent_30d = price.iloc[max(0, origin_idx - 2880):origin_idx]
            median_30d = recent_30d.median() if len(recent_30d) > 0 else np.nan
            if not np.isnan(median_30d) and median_30d > 0:
                feat["extreme_price_count_7d"] = float((recent_7d > median_30d * 2).sum())
            else:
                feat["extreme_price_count_7d"] = 0.0
            std_7d = feat["price_std_7d"]
            std_30d = recent_30d.std() if len(recent_30d) > 192 else np.nan
            feat["vol_regime_high"] = (
                1.0 if (not np.isnan(std_30d) and std_7d > 2.0 * std_30d) else 0.0
            )
            change_prev_24h = (
                price.iloc[origin_idx - 97] - price.iloc[origin_idx - 193]
                if origin_idx >= 193 else np.nan
            )
            feat["price_acceleration_24h"] = (
                feat["price_change_24h"] - change_prev_24h
                if not (np.isnan(feat.get("price_change_24h", np.nan)) or np.isnan(change_prev_24h))
                else np.nan
            )

            # Price curve shape features (3.1) — 15-min resolution
            if len(recent_24h) >= 96:
                peak_prices = []
                offpeak_prices = []
                for j in range(len(recent_24h)):
                    h = price.index[max(0, origin_idx - 96) + j].hour
                    if 8 <= h <= 21:
                        peak_prices.append(recent_24h.iloc[j])
                    else:
                        offpeak_prices.append(recent_24h.iloc[j])
                peak_avg = np.mean(peak_prices) if peak_prices else np.nan
                offpeak_avg = np.mean(offpeak_prices) if offpeak_prices else np.nan
                feat["price_peak_to_base_ratio"] = (
                    peak_avg / offpeak_avg if offpeak_avg and offpeak_avg > 0 else np.nan
                )
                morning = [recent_24h.iloc[j] for j in range(len(recent_24h))
                            if 6 <= price.index[max(0, origin_idx - 96) + j].hour <= 9]
                night = [recent_24h.iloc[j] for j in range(len(recent_24h))
                          if price.index[max(0, origin_idx - 96) + j].hour <= 5]
                feat["price_morning_slope"] = (
                    (np.mean(morning) - np.mean(night))
                    if morning and night else np.nan
                )
                evening = [recent_24h.iloc[j] for j in range(len(recent_24h))
                            if 17 <= price.index[max(0, origin_idx - 96) + j].hour <= 21]
                afternoon = [recent_24h.iloc[j] for j in range(len(recent_24h))
                              if 12 <= price.index[max(0, origin_idx - 96) + j].hour <= 16]
                feat["price_evening_slope"] = (
                    (np.mean(evening) - np.mean(afternoon))
                    if evening and afternoon else np.nan
                )
            else:
                feat["price_peak_to_base_ratio"] = np.nan
                feat["price_morning_slope"] = np.nan
                feat["price_evening_slope"] = np.nan

            # Origin time features
            feat["origin_hour_sin"] = np.sin(2 * np.pi * origin_dt.hour / 24)
            feat["origin_hour_cos"] = np.cos(2 * np.pi * origin_dt.hour / 24)
            feat["origin_dow_sin"] = np.sin(2 * np.pi * origin_dt.weekday() / 7)
            feat["origin_dow_cos"] = np.cos(2 * np.pi * origin_dt.weekday() / 7)
            feat["origin_month_sin"] = np.sin(2 * np.pi * origin_dt.month / 12)
            feat["origin_month_cos"] = np.cos(2 * np.pi * origin_dt.month / 12)
            feat["origin_is_weekend"] = 1.0 if origin_dt.weekday() >= 5 else 0.0

            # Target time features (quarter-hour encoding)
            quarter_of_day = target_dt.hour * 4 + target_dt.minute // 15
            feat["target_quarter_sin"] = np.sin(2 * np.pi * quarter_of_day / 96)
            feat["target_quarter_cos"] = np.cos(2 * np.pi * quarter_of_day / 96)
            feat["target_hour_sin"] = np.sin(2 * np.pi * target_dt.hour / 24)
            feat["target_hour_cos"] = np.cos(2 * np.pi * target_dt.hour / 24)
            feat["target_dow_sin"] = np.sin(2 * np.pi * target_dt.weekday() / 7)
            feat["target_dow_cos"] = np.cos(2 * np.pi * target_dt.weekday() / 7)
            feat["target_is_weekend"] = 1.0 if target_dt.weekday() >= 5 else 0.0
            feat["target_minute"] = target_dt.minute
            feat["quarters_ahead"] = quarters_ahead

            # Demand features
            if "real_demand" in data.columns:
                feat["demand_at_origin"] = data["real_demand"].iloc[origin_idx]
                feat["demand_lag_24h"] = (
                    data["real_demand"].iloc[origin_idx - 96]
                    if origin_idx >= 96 else np.nan
                )
                feat["demand_ramp_15m"] = (
                    data["real_demand"].iloc[origin_idx] - data["real_demand"].iloc[origin_idx - 1]
                    if origin_idx >= 1 else 0.0
                )
                feat["demand_ramp_1h"] = (
                    data["real_demand"].iloc[origin_idx] - data["real_demand"].iloc[origin_idx - 4]
                    if origin_idx >= 4 else 0.0
                )
                # Demand ramp 4h (3.2) — 16 steps at 15-min
                feat["demand_ramp_4h"] = (
                    data["real_demand"].iloc[origin_idx] - data["real_demand"].iloc[origin_idx - 16]
                    if origin_idx >= 16 else np.nan
                )
            if "demand_forecast" in data.columns:
                feat["demand_forecast_at_origin"] = data["demand_forecast"].iloc[origin_idx]
                target_idx = origin_idx + quarters_ahead
                if target_idx < len(data):
                    feat["demand_forecast_target"] = data["demand_forecast"].iloc[target_idx]

            # Generation mix
            demand_val = data.get("real_demand", pd.Series(1, index=data.index)).iloc[origin_idx]
            demand_safe = demand_val if demand_val != 0 else np.nan
            wind_val = data.get("wind_generation", pd.Series(0, index=data.index)).iloc[origin_idx]
            solar_pv = data.get("solar_pv_generation", pd.Series(0, index=data.index)).iloc[origin_idx]
            solar_th = data.get("solar_thermal_gen", pd.Series(0, index=data.index)).iloc[origin_idx]
            nuclear = data.get("nuclear_generation", pd.Series(0, index=data.index)).iloc[origin_idx]

            total_ren = wind_val + solar_pv + solar_th
            feat["renewable_share"] = total_ren / demand_safe if demand_safe else np.nan
            feat["wind_share"] = wind_val / demand_safe if demand_safe else np.nan
            feat["solar_share"] = (solar_pv + solar_th) / demand_safe if demand_safe else np.nan
            feat["nuclear_share"] = nuclear / demand_safe if demand_safe else np.nan
            feat["residual_demand"] = demand_val - total_ren - nuclear

            # Same-quarter price yesterday and last week
            same_q_yesterday = origin_idx - 96 + (quarters_ahead % 96)
            if 0 <= same_q_yesterday < origin_idx:
                feat["target_quarter_price_yesterday"] = price.iloc[same_q_yesterday]
            else:
                feat["target_quarter_price_yesterday"] = np.nan

            same_q_last_week = origin_idx - 672 + (quarters_ahead % 672)
            if 0 <= same_q_last_week < origin_idx:
                feat["target_quarter_price_last_week"] = price.iloc[same_q_last_week]
            else:
                feat["target_quarter_price_last_week"] = np.nan

            # Weather at origin and target (15-min broadcast)
            if weather_15min is not None and not weather_15min.empty:
                if origin_dt in weather_15min.index:
                    w = weather_15min.loc[origin_dt]
                    for col in weather_15min.columns:
                        feat[f"weather_{col}"] = w[col]
                if target_dt in weather_15min.index:
                    tw = weather_15min.loc[target_dt]
                    feat["target_weather_temp_c"] = tw.get("temp_c", np.nan)
                    feat["target_weather_wind_kmh"] = tw.get("wind_speed_kmh", np.nan)
                    feat["target_weather_cloud_pct"] = tw.get("cloud_cover_pct", np.nan)
                    feat["target_weather_radiation"] = (
                        (tw.get("direct_radiation_wm2", 0) or 0)
                        + (tw.get("diffuse_radiation_wm2", 0) or 0)
                    )
                    feat["target_weather_precip_mm"] = tw.get("precipitation_mm", np.nan)

            # Commodity features (daily)
            if commodity_df is not None and not commodity_df.empty:
                origin_date = origin_dt.date() if hasattr(origin_dt, 'date') else origin_dt
                if hasattr(commodity_df.index, "date"):
                    comm_idx = commodity_df.index.date
                else:
                    comm_idx = pd.to_datetime(commodity_df.index).date
                if origin_date in comm_idx:
                    mask = comm_idx == origin_date
                    c_row = commodity_df[mask].iloc[0]
                    for col in commodity_df.columns:
                        feat[f"commodity_{col}"] = c_row[col] if col in c_row.index else np.nan

            # Commodity derivatives (3.5): momentum, ratios, marginal cost, interactions
            from src.models.direct_trainer import _compute_commodity_derivatives
            comm_derivs = _compute_commodity_derivatives(
                commodity_df, origin_dt,
                feat.get("price_lag_24h", np.nan),
                feat.get("residual_demand", np.nan),
            )
            feat.update(comm_derivs)

            # Demand ramp x gas interaction (3.2)
            gas_val = feat.get("commodity_ttf_gas_eur_mwh", np.nan)
            demand_ramp = feat.get("demand_ramp_4h", np.nan)
            if gas_val and not np.isnan(gas_val) and not np.isnan(demand_ramp):
                feat["demand_ramp_x_gas"] = demand_ramp * gas_val
            else:
                feat["demand_ramp_x_gas"] = np.nan

            # D+1 published price features (strategic mode)
            if run_mode == "strategic":
                # Resample 15-min prices to hourly for _extract_d1_price_features
                hourly_price = price.resample("h").mean().dropna()
                d1_feat = _extract_d1_price_features(
                    hourly_price, origin_dt, target_dt, hourly_price.index,
                )
                feat.update(d1_feat)

            return feat

        except Exception as e:
            logger.warning("Error building 15min features q=%d: %s", quarters_ahead, e)
            return None

    def run_and_store(self, weather_df=None, commodity_df=None,
                      weather_hourly_df=None, run_mode: str | None = None,
                      bias_correct: bool = True):
        """Generate predictions and store them in the database.

        # NOTE: This method stores predictions to the database and backfills
        # actual prices — infrastructure concerns not included in this extract.
        # The BiasCorrector and pipeline.store_predictions/backfill_actual_prices
        # are application-level components.

        Args:
            bias_correct: If True, apply rolling bias correction and negative
                price floor clipping before storing.
        """
        predictions_df = self.predict_next_7_days(
            weather_df, commodity_df, weather_hourly_df,
            run_mode=run_mode,
        )

        if bias_correct and predictions_df is not None and not predictions_df.empty:
            from src.models.bias_corrector import BiasCorrector
            corrector = BiasCorrector(self.pipeline)
            model_name = predictions_df["model_version"].iloc[0] if "model_version" in predictions_df.columns else "ensemble"
            predictions_df = corrector.apply(predictions_df, model_name, run_mode or "dayahead")

        self.pipeline.store_predictions(predictions_df, run_mode=run_mode)
        self.pipeline.backfill_actual_prices()
        logger.info("Direct predictions stored and actual prices backfilled (run_mode=%s)", run_mode)
        return predictions_df
