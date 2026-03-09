# EPForecast Methodology — Direct Multi-Horizon Training
# Extracted from the EPForecast project (github.com/JorgeLopezLan/epf-methodology)
# Full application: epf.productjorge.com | Docs: epforecast.vercel.app
"""
Direct multi-horizon model training for EPF.

Instead of recursive prediction (where each hour's prediction feeds into the next),
this trains separate models for different forecast horizons. Each model predicts
the price at a specific future time using only features available at the forecast
origin, eliminating error compounding.
"""
import logging
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS_DIR = None  # Set to your model artifacts directory (pathlib.Path)

logger = logging.getLogger(__name__)

# Legacy horizon groups (origin 23:00 UTC, kept for backward compatibility)
HORIZON_GROUPS = {
    "H1": list(range(1, 7)),       # Hours 1-6:   very short term
    "H2": list(range(7, 13)),      # Hours 7-12:  same day remaining
    "H3": list(range(13, 25)),     # Hours 13-24: next day
    "H4": list(range(25, 49)),     # Hours 25-48: day 2
    "H5": list(range(49, 73)),     # Hours 49-72: day 3
    "H6": list(range(73, 97)),     # Hours 73-96: day 4
    "H7": list(range(97, 121)),    # Hours 97-120: day 5
    "H8": list(range(121, 169)),   # Hours 121-168: days 6-7
}

# D+1 Day-Ahead: origin ~10:00 UTC, predict D+1 full day (00:00-23:00)
# D+1 00:00 = 14h ahead from 10:00, D+1 23:00 = 37h ahead
HORIZON_GROUPS_DAYAHEAD = {
    "DA1": list(range(14, 26)),    # D+1 00:00-11:00 (14-25h ahead)
    "DA2": list(range(26, 38)),    # D+1 12:00-23:00 (26-37h ahead)
}

# D+2-D+7 Strategic: origin ~15:00 UTC, predict D+2 through D+7
# D+2 00:00 = 33h ahead from 15:00, D+7 23:00 = 177h ahead
HORIZON_GROUPS_STRATEGIC = {
    "S1": list(range(33, 57)),     # D+2 full day (33-56h ahead)
    "S2": list(range(57, 81)),     # D+3 full day
    "S3": list(range(81, 105)),    # D+4 full day
    "S4": list(range(105, 129)),   # D+5 full day
    "S5": list(range(129, 177)),   # D+6-D+7 full days
}

# Origin hour ranges for training sample filtering
DAYAHEAD_ORIGIN_HOURS = range(8, 13)    # 08:00-12:00 UTC
STRATEGIC_ORIGIN_HOURS = range(13, 19)  # 13:00-18:00 UTC

# 15-min horizon groups: each group covers one forecast day (96 quarter-hours)
HORIZON_GROUPS_15MIN = {
    "D1": list(range(1, 97)),       # Quarter-hours 1-96:   day 1
    "D2": list(range(97, 193)),     # Quarter-hours 97-192: day 2
    "D3": list(range(193, 289)),    # Quarter-hours 193-288: day 3
    "D4": list(range(289, 385)),    # Quarter-hours 289-384: day 4
    "D5": list(range(385, 481)),    # Quarter-hours 385-480: day 5
    "D6": list(range(481, 577)),    # Quarter-hours 481-576: day 6
    "D7": list(range(577, 673)),    # Quarter-hours 577-672: day 7
}

# 15-min Day-Ahead: origin ~10:00 UTC, D+1 at 15-min resolution
# D+1 00:00 = 14h ahead = quarter 56, D+1 23:45 = quarter 151
HORIZON_GROUPS_15MIN_DAYAHEAD = {
    "DA1": list(range(56, 104)),    # D+1 00:00-11:45 (48 quarters)
    "DA2": list(range(104, 152)),   # D+1 12:00-23:45 (48 quarters)
}

# 15-min Strategic: origin ~15:00 UTC, D+2-D+7 at 15-min resolution
# D+2 00:00 = 33h ahead = quarter 132, D+7 23:45 = quarter 707
HORIZON_GROUPS_15MIN_STRATEGIC = {
    "S1": list(range(132, 228)),    # D+2 full day (96 quarters)
    "S2": list(range(228, 324)),    # D+3 full day
    "S3": list(range(324, 420)),    # D+4 full day
    "S4": list(range(420, 516)),    # D+5 full day
    "S5": list(range(516, 708)),    # D+6-D+7 (192 quarters)
}

# Default hyperparameters (can be overridden by Optuna)
DEFAULT_PARAMS = {
    "max_iter": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 42,
}


def _extract_d1_price_features(price: pd.Series, origin_dt, target_dt,
                                df_index) -> dict:
    """Extract D+1 published price features for strategic model training.

    At origin_dt (afternoon, after OMIE publication), D+1's 24 hourly prices
    are known. These are strong features for predicting D+2+ prices.

    Returns a dict of features, or all-NaN if D+1 prices aren't available.
    """
    feat = {}
    # D+1 is the next calendar day after origin
    d1_start = (pd.Timestamp(origin_dt) + pd.Timedelta(days=1)).normalize()  # D+1 00:00
    d1_end = d1_start + pd.Timedelta(hours=23)                  # D+1 23:00

    # Get D+1 prices from the DataFrame (they exist in historical data)
    d1_mask = (df_index >= d1_start) & (df_index <= d1_end)
    d1_prices = price.loc[d1_mask]

    if len(d1_prices) >= 20:  # need most of the 24 hours
        feat["d1_mean_price"] = d1_prices.mean()
        feat["d1_min_price"] = d1_prices.min()
        feat["d1_max_price"] = d1_prices.max()
        feat["d1_std_price"] = d1_prices.std()
        # Peak (8-21) vs off-peak spread
        peak_hours = [h for h in range(8, 22)]
        peak_mask = d1_prices.index.hour.isin(peak_hours)
        offpeak_mask = ~peak_mask
        if peak_mask.sum() > 0 and offpeak_mask.sum() > 0:
            feat["d1_peak_spread"] = d1_prices[peak_mask].mean() - d1_prices[offpeak_mask].mean()
        else:
            feat["d1_peak_spread"] = np.nan
        # D+1 price at same hour-of-day as target (strong seasonal signal)
        target_hour = target_dt.hour
        same_hour = d1_prices[d1_prices.index.hour == target_hour]
        feat["d1_same_hour_price"] = same_hour.iloc[0] if len(same_hour) > 0 else np.nan
    else:
        feat["d1_mean_price"] = np.nan
        feat["d1_min_price"] = np.nan
        feat["d1_max_price"] = np.nan
        feat["d1_std_price"] = np.nan
        feat["d1_peak_spread"] = np.nan
        feat["d1_same_hour_price"] = np.nan

    return feat


def _compute_commodity_derivatives(commodity_df: pd.DataFrame,
                                    origin_date,
                                    price_lag_24h: float,
                                    residual_demand: float) -> dict:
    """Compute derived commodity features for a single origin.

    Returns a dict of feature_name -> value with commodity momentum,
    cross-commodity ratios, marginal cost, and interaction features.
    """
    feat = {}
    if commodity_df is None or commodity_df.empty:
        return feat

    # Resolve origin_date to date object
    if hasattr(origin_date, 'date'):
        origin_date = origin_date.date()

    # Get the commodity row for this date
    if hasattr(commodity_df.index, "date"):
        comm_dates = commodity_df.index.date
    else:
        comm_dates = pd.to_datetime(commodity_df.index).date
    if origin_date not in comm_dates:
        return feat

    mask = comm_dates == origin_date
    c_row = commodity_df[mask].iloc[0]
    gas_val = c_row.get("ttf_gas_eur_mwh", np.nan) if "ttf_gas_eur_mwh" in c_row.index else np.nan
    oil_val = c_row.get("brent_oil_usd_bbl", np.nan) if "brent_oil_usd_bbl" in c_row.index else np.nan
    carbon_val = c_row.get("ets_carbon_eur", np.nan) if "ets_carbon_eur" in c_row.index else np.nan

    # --- Cross-commodity ratios ---
    if gas_val and not np.isnan(gas_val) and gas_val > 0:
        feat["commodity_oil_gas_ratio"] = oil_val / gas_val if oil_val and not np.isnan(oil_val) else np.nan
    else:
        feat["commodity_oil_gas_ratio"] = np.nan
    if carbon_val and not np.isnan(carbon_val) and carbon_val > 0:
        feat["commodity_gas_carbon_ratio"] = gas_val / carbon_val if gas_val and not np.isnan(gas_val) else np.nan
    else:
        feat["commodity_gas_carbon_ratio"] = np.nan

    # --- Marginal cost of gas generation ---
    gas_f = gas_val if gas_val and not np.isnan(gas_val) else 0
    carbon_f = carbon_val if carbon_val and not np.isnan(carbon_val) else 0
    marginal_cost = gas_f * 1.9 + carbon_f * 0.37
    feat["commodity_marginal_cost_gas"] = marginal_cost if (gas_f > 0 or carbon_f > 0) else np.nan

    # --- Spark spread ---
    if not np.isnan(price_lag_24h) and feat["commodity_marginal_cost_gas"] is not np.nan:
        feat["commodity_spark_spread"] = price_lag_24h - marginal_cost
    else:
        feat["commodity_spark_spread"] = np.nan

    # --- Gas x residual demand interaction ---
    if gas_val and not np.isnan(gas_val) and not np.isnan(residual_demand):
        feat["commodity_gas_x_residual"] = gas_val * residual_demand
    else:
        feat["commodity_gas_x_residual"] = np.nan

    # --- Momentum features (require time series lookback) ---
    gas_series = commodity_df.get("ttf_gas_eur_mwh")
    oil_series = commodity_df.get("brent_oil_usd_bbl")

    if gas_series is not None:
        gas_history = gas_series.loc[gas_series.index <= pd.Timestamp(origin_date)].dropna()
        if len(gas_history) >= 8:
            feat["commodity_gas_change_7d"] = float(gas_history.iloc[-1] - gas_history.iloc[-8])
            prev = float(gas_history.iloc[-8])
            feat["commodity_gas_pct_change_7d"] = (
                (float(gas_history.iloc[-1]) / prev - 1) * 100 if prev > 0 else np.nan
            )
        else:
            feat["commodity_gas_change_7d"] = np.nan
            feat["commodity_gas_pct_change_7d"] = np.nan
        if len(gas_history) >= 31:
            feat["commodity_gas_change_30d"] = float(gas_history.iloc[-1] - gas_history.iloc[-31])
        else:
            feat["commodity_gas_change_30d"] = np.nan
    else:
        feat["commodity_gas_change_7d"] = np.nan
        feat["commodity_gas_pct_change_7d"] = np.nan
        feat["commodity_gas_change_30d"] = np.nan

    if oil_series is not None:
        oil_history = oil_series.loc[oil_series.index <= pd.Timestamp(origin_date)].dropna()
        if len(oil_history) >= 8:
            feat["commodity_oil_change_7d"] = float(oil_history.iloc[-1] - oil_history.iloc[-8])
            prev = float(oil_history.iloc[-8])
            feat["commodity_oil_pct_change_7d"] = (
                (float(oil_history.iloc[-1]) / prev - 1) * 100 if prev > 0 else np.nan
            )
        else:
            feat["commodity_oil_change_7d"] = np.nan
            feat["commodity_oil_pct_change_7d"] = np.nan
    else:
        feat["commodity_oil_change_7d"] = np.nan
        feat["commodity_oil_pct_change_7d"] = np.nan

    return feat


def build_direct_features(ree_df: pd.DataFrame, horizon_hours: list[int],
                           weather_df: pd.DataFrame | None = None,
                           commodity_df: pd.DataFrame | None = None,
                           weather_hourly_df: pd.DataFrame | None = None,
                           run_mode: str | None = None) -> pd.DataFrame:
    """
    Build feature matrix for direct multi-horizon prediction.

    Unlike the recursive approach, lag features here are always relative to the
    forecast origin (time t), not relative to the target hour.

    For each row (origin time t) and each horizon h in horizon_hours, creates
    a training sample where:
    - Features = values known at time t (lags, rolling stats, demand forecast, etc.)
    - Target = price at time t + h
    - Additional features: cyclical encoding of the TARGET hour (not origin hour)
    - Target-hour weather: actual weather at t+h (during training) or forecast (prediction)

    Parameters
    ----------
    ree_df : Raw REE hourly data indexed by datetime_utc.
    horizon_hours : List of hours ahead to predict (e.g., [1,2,3,4,5,6] for H1).
    weather_df : Optional daily weather data (legacy AEMET).
    commodity_df : Optional daily commodity price data.
    weather_hourly_df : Optional hourly weather data indexed by datetime_utc.
                        Used for both origin-time and target-time weather features.
    run_mode : "dayahead", "strategic", or None (legacy).
               Filters origin hours and adds D+1 price features for strategic mode.

    Returns
    -------
    DataFrame with features and 'target_price' column.
    """
    df = ree_df.copy()
    if "day_ahead_price" not in df.columns:
        raise ValueError("ree_df must contain 'day_ahead_price'")

    price = df["day_ahead_price"]

    rows = []
    min_history = 504  # need 3 weeks of history for multi-week lag features

    # Determine origin hour filter based on run_mode
    if run_mode == "dayahead":
        allowed_hours = set(DAYAHEAD_ORIGIN_HOURS)
    elif run_mode == "strategic":
        allowed_hours = set(STRATEGIC_ORIGIN_HOURS)
    else:
        allowed_hours = None  # no filter (legacy mode)

    for i in range(min_history, len(df)):
        origin_idx = i
        origin_dt = df.index[origin_idx]

        # Skip origins outside the allowed hour range for this run_mode
        if allowed_hours is not None and origin_dt.hour not in allowed_hours:
            continue

        # Features known at origin time (lags relative to origin)
        feat = {}

        # Price lags (relative to origin)
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

        # Rolling statistics at origin
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
        # Price acceleration (2nd derivative of price momentum)
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
            # Split last 24h into peak (h8-21) and off-peak by position
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
            # Morning slope: avg h6-9 minus avg h0-5
            morning = [recent_24.iloc[j] for j in range(len(recent_24))
                        if 6 <= price.index[max(0, origin_idx - 24) + j].hour <= 9]
            night = [recent_24.iloc[j] for j in range(len(recent_24))
                      if price.index[max(0, origin_idx - 24) + j].hour <= 5]
            feat["price_morning_slope"] = (
                (np.mean(morning) - np.mean(night))
                if morning and night else np.nan
            )
            # Evening slope: avg h17-21 minus avg h12-16
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
        feat["origin_dow_sin"] = np.sin(2 * np.pi * origin_dt.dayofweek / 7)
        feat["origin_dow_cos"] = np.cos(2 * np.pi * origin_dt.dayofweek / 7)
        feat["origin_month_sin"] = np.sin(2 * np.pi * origin_dt.month / 12)
        feat["origin_month_cos"] = np.cos(2 * np.pi * origin_dt.month / 12)

        # Demand features at origin
        if "real_demand" in df.columns:
            feat["demand_at_origin"] = df["real_demand"].iloc[origin_idx]
            feat["demand_lag_24h"] = (
                df["real_demand"].iloc[origin_idx - 24] if origin_idx >= 24 else np.nan
            )
            # Demand ramp features (3.2)
            feat["demand_ramp_4h"] = (
                df["real_demand"].iloc[origin_idx] - df["real_demand"].iloc[origin_idx - 4]
                if origin_idx >= 4 else np.nan
            )
        if "demand_forecast" in df.columns:
            feat["demand_forecast_at_origin"] = df["demand_forecast"].iloc[origin_idx]

        # Generation mix at origin
        demand_val = df.get("real_demand", pd.Series(1, index=df.index)).iloc[origin_idx]
        demand_safe = demand_val if demand_val != 0 else np.nan
        wind_val = df.get("wind_generation", pd.Series(0, index=df.index)).iloc[origin_idx]
        solar_pv_val = df.get("solar_pv_generation", pd.Series(0, index=df.index)).iloc[origin_idx]
        solar_th_val = df.get("solar_thermal_gen", pd.Series(0, index=df.index)).iloc[origin_idx]
        nuclear_val = df.get("nuclear_generation", pd.Series(0, index=df.index)).iloc[origin_idx]

        total_ren = wind_val + solar_pv_val + solar_th_val
        feat["renewable_share"] = total_ren / demand_safe if demand_safe else np.nan
        feat["wind_share"] = wind_val / demand_safe if demand_safe else np.nan
        feat["solar_share"] = (solar_pv_val + solar_th_val) / demand_safe if demand_safe else np.nan
        feat["nuclear_share"] = nuclear_val / demand_safe if demand_safe else np.nan
        feat["residual_demand"] = demand_val - total_ren - nuclear_val

        # Weekend/holiday at origin
        feat["origin_is_weekend"] = 1.0 if origin_dt.dayofweek >= 5 else 0.0

        # Weather features at origin
        if weather_hourly_df is not None and not weather_hourly_df.empty:
            # Hourly weather: look up the origin hour directly
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
                hydro_val = df.get("hydro_generation", pd.Series(0, index=df.index)).iloc[origin_idx]
                feat["weather_precip_x_hydro"] = _pr * hydro_val / demand_safe if demand_safe else np.nan
                feat["weather_cloud_x_solar"] = _cl * _solar_share
                feat["weather_sunshine_x_solar"] = _sh * _solar_share
                feat["weather_ghi"] = _dr + _di
        elif weather_df is not None and not weather_df.empty:
            # Fallback: daily weather broadcast to origin
            origin_date = origin_dt.date()
            if hasattr(weather_df.index, "date"):
                weather_idx = weather_df.index.date
            else:
                weather_idx = pd.to_datetime(weather_df.index).date
            if origin_date in weather_idx:
                mask = weather_idx == origin_date
                w_row = weather_df[mask].iloc[0]
                for col in weather_df.columns:
                    feat[f"weather_{col}"] = w_row[col] if col in w_row.index else np.nan

        # Commodity features (daily)
        if commodity_df is not None and not commodity_df.empty:
            origin_date = origin_dt.date()
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
        comm_derivs = _compute_commodity_derivatives(
            commodity_df, origin_dt,
            feat.get("price_lag_24h", np.nan),
            feat.get("residual_demand", np.nan),
        )
        feat.update(comm_derivs)

        # Demand ramp x gas interaction (3.2) — gas cost matters when demand surging
        gas_val = feat.get("commodity_ttf_gas_eur_mwh", np.nan)
        demand_ramp = feat.get("demand_ramp_4h", np.nan)
        if gas_val and not np.isnan(gas_val) and not np.isnan(demand_ramp):
            feat["demand_ramp_x_gas"] = demand_ramp * gas_val
        else:
            feat["demand_ramp_x_gas"] = np.nan

        # Create one sample per horizon hour
        for h in horizon_hours:
            target_idx = origin_idx + h
            if target_idx >= len(df):
                continue

            target_dt = df.index[target_idx]
            sample = feat.copy()

            # Target time features (cyclical encoding of the future hour)
            sample["target_hour_sin"] = np.sin(2 * np.pi * target_dt.hour / 24)
            sample["target_hour_cos"] = np.cos(2 * np.pi * target_dt.hour / 24)
            sample["target_dow_sin"] = np.sin(2 * np.pi * target_dt.dayofweek / 7)
            sample["target_dow_cos"] = np.cos(2 * np.pi * target_dt.dayofweek / 7)
            sample["target_is_weekend"] = 1.0 if target_dt.dayofweek >= 5 else 0.0
            sample["hours_ahead"] = h

            # Price at same target hour yesterday and last week (known at origin)
            same_hour_yesterday = origin_idx - 24 + (h % 24)
            if 0 <= same_hour_yesterday < origin_idx:
                sample["target_hour_price_yesterday"] = price.iloc[same_hour_yesterday]
            else:
                sample["target_hour_price_yesterday"] = np.nan

            same_hour_last_week = origin_idx - 168 + (h % 168)
            if 0 <= same_hour_last_week < origin_idx:
                sample["target_hour_price_last_week"] = price.iloc[same_hour_last_week]
            else:
                sample["target_hour_price_last_week"] = np.nan

            # Demand forecast for target hour (if available)
            if "demand_forecast" in df.columns:
                sample["demand_forecast_target"] = df["demand_forecast"].iloc[target_idx]

            # Target-hour weather features (Phase 1.3)
            # During training: uses actual weather at the target hour
            # During prediction: the caller provides forecast weather in weather_hourly_df
            if weather_hourly_df is not None and not weather_hourly_df.empty:
                if target_dt in weather_hourly_df.index:
                    tw = weather_hourly_df.loc[target_dt]
                    sample["target_weather_temp_c"] = tw.get("temp_c", np.nan)
                    sample["target_weather_wind_kmh"] = tw.get("wind_speed_kmh", np.nan)
                    sample["target_weather_cloud_pct"] = tw.get("cloud_cover_pct", np.nan)
                    _tdr = (tw.get("direct_radiation_wm2", 0) or 0)
                    _tdi = (tw.get("diffuse_radiation_wm2", 0) or 0)
                    sample["target_weather_radiation"] = _tdr + _tdi
                    sample["target_weather_precip_mm"] = tw.get("precipitation_mm", np.nan)
                    # Target weather interaction features
                    _ttc = tw.get("temp_c", np.nan)
                    if _ttc is not None and not np.isnan(_ttc):
                        sample["target_weather_temp_deviation"] = _ttc - 15.0
                        sample["target_weather_heating_dd"] = max(0, 18 - _ttc)
                        sample["target_weather_cooling_dd"] = max(0, _ttc - 24)
                    else:
                        sample["target_weather_temp_deviation"] = np.nan
                        sample["target_weather_heating_dd"] = np.nan
                        sample["target_weather_cooling_dd"] = np.nan
                    _tws = feat.get("wind_share", 0) or 0
                    _tss = feat.get("solar_share", 0) or 0
                    sample["target_weather_wind_x_wind_share"] = (tw.get("wind_speed_kmh", 0) or 0) * _tws
                    sample["target_weather_cloud_x_solar"] = (tw.get("cloud_cover_pct", 0) or 0) * _tss
                    sample["target_weather_ghi_x_solar"] = (_tdr + _tdi) * _tss
                else:
                    sample["target_weather_temp_c"] = np.nan
                    sample["target_weather_wind_kmh"] = np.nan
                    sample["target_weather_cloud_pct"] = np.nan
                    sample["target_weather_radiation"] = np.nan
                    sample["target_weather_precip_mm"] = np.nan
                    sample["target_weather_temp_deviation"] = np.nan
                    sample["target_weather_heating_dd"] = np.nan
                    sample["target_weather_cooling_dd"] = np.nan
                    sample["target_weather_wind_x_wind_share"] = np.nan
                    sample["target_weather_cloud_x_solar"] = np.nan
                    sample["target_weather_ghi_x_solar"] = np.nan

            # D+1 published price features (strategic mode or auto-detect)
            # Available when origin is after OMIE publication (~13:00 UTC)
            if run_mode == "strategic" or (run_mode is None and origin_dt.hour >= 13):
                d1_feat = _extract_d1_price_features(
                    price, origin_dt, target_dt, df.index,
                )
                sample.update(d1_feat)
            else:
                # NaN for dayahead / morning origins — tree models handle this
                sample["d1_mean_price"] = np.nan
                sample["d1_min_price"] = np.nan
                sample["d1_max_price"] = np.nan
                sample["d1_std_price"] = np.nan
                sample["d1_peak_spread"] = np.nan
                sample["d1_same_hour_price"] = np.nan

            # Target price
            sample["target_price"] = price.iloc[target_idx]
            sample["_origin_dt"] = origin_dt
            sample["_target_dt"] = target_dt

            rows.append(sample)

    result = pd.DataFrame(rows)
    logger.info("Built %d direct-prediction samples for horizons %s (run_mode=%s)",
                len(result), f"{min(horizon_hours)}-{max(horizon_hours)}h",
                run_mode or "legacy")
    return result


def build_direct_features_15min(ree_15min_df: pd.DataFrame,
                                 horizon_quarters: list[int],
                                 weather_hourly_df: pd.DataFrame | None = None,
                                 commodity_df: pd.DataFrame | None = None,
                                 origin_step: int = 4,
                                 run_mode: str | None = None) -> pd.DataFrame:
    """
    Build feature matrix for direct 15-min multi-horizon prediction.

    Same approach as build_direct_features but with 15-min resolution lags.
    origin_step=4 means we use every 4th 15-min step as an origin (= hourly origins)
    which keeps training data manageable while still producing 15-min targets.

    Parameters
    ----------
    ree_15min_df : DataFrame indexed by datetime_utc with 15-min REE data.
    horizon_quarters : List of quarter-hours ahead to predict (e.g., range(1,97) for D1).
    weather_hourly_df : Optional hourly weather (broadcast to 15-min via forward-fill).
    commodity_df : Optional daily commodity prices.
    origin_step : Use every Nth 15-min step as origin (4=hourly, 1=every 15 min).
    run_mode : "dayahead", "strategic", or None (legacy).
               Filters origin hours and adds D+1 price features for strategic mode.
    """
    df = ree_15min_df.copy()
    if "day_ahead_price" not in df.columns:
        raise ValueError("ree_15min_df must contain 'day_ahead_price'")

    price = df["day_ahead_price"]
    min_history = 2016  # 3 weeks at 15-min = 2016 steps (for multi-week lags)

    # Determine origin hour filter based on run_mode
    if run_mode == "dayahead":
        allowed_hours = set(DAYAHEAD_ORIGIN_HOURS)
    elif run_mode == "strategic":
        allowed_hours = set(STRATEGIC_ORIGIN_HOURS)
    else:
        allowed_hours = None  # no filter (legacy mode)

    # Pre-compute hourly price series for D+1 feature extraction (strategic)
    hourly_price = None
    if run_mode == "strategic":
        hourly_price = price.resample("h").mean().dropna()

    # Broadcast hourly weather to 15-min index via forward-fill
    weather_15min = None
    if weather_hourly_df is not None and not weather_hourly_df.empty:
        weather_15min = weather_hourly_df.reindex(df.index, method="ffill")

    # Broadcast daily commodities to 15-min index
    commodity_vals = {}
    if commodity_df is not None and not commodity_df.empty:
        comm = commodity_df.copy()
        comm.index = pd.to_datetime(comm.index)
        if comm.index.tz is None:
            comm.index = comm.index.tz_localize("UTC")
        for col in comm.columns:
            # Reindex to daily, ffill, then align to 15-min
            daily = comm[col].reindex(
                pd.date_range(comm.index.min(), df.index.max().normalize(), freq="D"),
                method="ffill",
            )
            # Map each 15-min step to its date
            date_map = df.index.normalize()
            commodity_vals[col] = pd.Series(
                daily.reindex(date_map).values,
                index=df.index,
            ).ffill()

    rows = []
    origins = range(min_history, len(df), origin_step)

    for i in origins:
        origin_dt = df.index[i]

        # Skip origins outside the allowed hour range for this run_mode
        if allowed_hours is not None and origin_dt.hour not in allowed_hours:
            continue

        feat = {}

        # Price lags scaled to 15-min (shift(4)=1h, shift(96)=24h, shift(672)=7d)
        feat["price_lag_15m"] = price.iloc[i - 1]
        feat["price_lag_30m"] = price.iloc[i - 2]
        feat["price_lag_45m"] = price.iloc[i - 3]
        feat["price_lag_1h"] = price.iloc[i - 4]
        feat["price_lag_2h"] = price.iloc[i - 8]
        feat["price_lag_3h"] = price.iloc[i - 12]
        feat["price_lag_24h"] = price.iloc[i - 96]
        feat["price_lag_48h"] = price.iloc[i - 192]
        feat["price_lag_168h"] = price.iloc[i - 672]
        feat["price_lag_336h"] = price.iloc[i - 1344] if i >= 1344 else np.nan
        feat["price_lag_504h"] = price.iloc[i - 2016] if i >= 2016 else np.nan

        # Same-weekday 4-week average and std (15-min: 672 steps = 1 week)
        same_wd_prices = []
        for w in range(1, 5):
            idx = i - (w * 672)
            if idx >= 0:
                same_wd_prices.append(price.iloc[idx])
        feat["price_same_weekday_4w_avg"] = float(np.mean(same_wd_prices)) if same_wd_prices else np.nan
        feat["price_same_weekday_4w_std"] = float(np.std(same_wd_prices)) if len(same_wd_prices) >= 2 else np.nan

        # Rolling stats (in 15-min steps)
        recent_24h = price.iloc[max(0, i - 96):i]  # 96 steps = 24h
        recent_7d = price.iloc[max(0, i - 672):i]   # 672 steps = 7d
        feat["price_rolling_24h"] = recent_24h.mean()
        feat["price_rolling_7d"] = recent_7d.mean()
        feat["price_std_24h"] = recent_24h.std() if len(recent_24h) > 1 else 0.0
        feat["price_std_7d"] = recent_7d.std() if len(recent_7d) > 1 else 0.0
        feat["price_min_24h"] = recent_24h.min()
        feat["price_max_24h"] = recent_24h.max()
        feat["price_range_24h"] = feat["price_max_24h"] - feat["price_min_24h"]
        feat["price_change_1h"] = price.iloc[i - 1] - price.iloc[i - 5] if i >= 5 else 0.0
        feat["price_change_24h"] = price.iloc[i - 1] - price.iloc[i - 97] if i >= 97 else np.nan

        # Volatility & regime indicators (3.3) — 15-min resolution
        q25 = recent_7d.quantile(0.25) if len(recent_7d) >= 4 else np.nan
        q75 = recent_7d.quantile(0.75) if len(recent_7d) >= 4 else np.nan
        feat["price_iqr_7d"] = q75 - q25 if not (np.isnan(q25) or np.isnan(q75)) else np.nan
        feat["price_skewness_7d"] = float(recent_7d.skew()) if len(recent_7d) >= 96 else np.nan
        recent_30d = price.iloc[max(0, i - 2880):i]  # 2880 = 30 days at 15-min
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
            price.iloc[i - 97] - price.iloc[i - 193]
            if i >= 193 else np.nan
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
                h = price.index[max(0, i - 96) + j].hour
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
                        if 6 <= price.index[max(0, i - 96) + j].hour <= 9]
            night = [recent_24h.iloc[j] for j in range(len(recent_24h))
                      if price.index[max(0, i - 96) + j].hour <= 5]
            feat["price_morning_slope"] = (
                (np.mean(morning) - np.mean(night))
                if morning and night else np.nan
            )
            evening = [recent_24h.iloc[j] for j in range(len(recent_24h))
                        if 17 <= price.index[max(0, i - 96) + j].hour <= 21]
            afternoon = [recent_24h.iloc[j] for j in range(len(recent_24h))
                          if 12 <= price.index[max(0, i - 96) + j].hour <= 16]
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
        feat["origin_dow_sin"] = np.sin(2 * np.pi * origin_dt.dayofweek / 7)
        feat["origin_dow_cos"] = np.cos(2 * np.pi * origin_dt.dayofweek / 7)
        feat["origin_month_sin"] = np.sin(2 * np.pi * origin_dt.month / 12)
        feat["origin_month_cos"] = np.cos(2 * np.pi * origin_dt.month / 12)
        feat["origin_is_weekend"] = 1.0 if origin_dt.dayofweek >= 5 else 0.0

        # Demand features (15-min resolution)
        if "real_demand" in df.columns:
            feat["demand_at_origin"] = df["real_demand"].iloc[i]
            feat["demand_lag_24h"] = df["real_demand"].iloc[i - 96] if i >= 96 else np.nan
            # Demand ramp: 15-min and 1h changes
            feat["demand_ramp_15m"] = (
                df["real_demand"].iloc[i] - df["real_demand"].iloc[i - 1]
                if i >= 1 else 0.0
            )
            feat["demand_ramp_1h"] = (
                df["real_demand"].iloc[i] - df["real_demand"].iloc[i - 4]
                if i >= 4 else 0.0
            )
            # Demand ramp 4h (3.2) — 16 steps at 15-min
            feat["demand_ramp_4h"] = (
                df["real_demand"].iloc[i] - df["real_demand"].iloc[i - 16]
                if i >= 16 else np.nan
            )

        if "demand_forecast" in df.columns:
            feat["demand_forecast_at_origin"] = df["demand_forecast"].iloc[i]

        # Generation mix at origin (15-min resolution)
        demand_val = df.get("real_demand", pd.Series(1, index=df.index)).iloc[i]
        demand_safe = demand_val if demand_val != 0 else np.nan
        wind_val = df.get("wind_generation", pd.Series(0, index=df.index)).iloc[i]
        solar_pv_val = df.get("solar_pv_generation", pd.Series(0, index=df.index)).iloc[i]
        solar_th_val = df.get("solar_thermal_gen", pd.Series(0, index=df.index)).iloc[i]
        nuclear_val = df.get("nuclear_generation", pd.Series(0, index=df.index)).iloc[i]

        total_ren = wind_val + solar_pv_val + solar_th_val
        feat["renewable_share"] = total_ren / demand_safe if demand_safe else np.nan
        feat["wind_share"] = wind_val / demand_safe if demand_safe else np.nan
        feat["solar_share"] = (solar_pv_val + solar_th_val) / demand_safe if demand_safe else np.nan
        feat["nuclear_share"] = nuclear_val / demand_safe if demand_safe else np.nan
        feat["residual_demand"] = demand_val - total_ren - nuclear_val

        # Weather features at origin (broadcast hourly -> 15-min)
        if weather_15min is not None and origin_dt in weather_15min.index:
            w_row = weather_15min.loc[origin_dt]
            for col in weather_15min.columns:
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
            hydro_val = df.get("hydro_generation", pd.Series(0, index=df.index)).iloc[i]
            feat["weather_precip_x_hydro"] = _pr * hydro_val / demand_safe if demand_safe else np.nan
            feat["weather_cloud_x_solar"] = _cl * _solar_share
            feat["weather_sunshine_x_solar"] = _sh * _solar_share
            feat["weather_ghi"] = _dr + _di

        # Commodity features
        for col, series in commodity_vals.items():
            feat[f"commodity_{col}"] = series.iloc[i] if i < len(series) else np.nan

        # Commodity derivatives (3.5): momentum, ratios, marginal cost, interactions
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

        # Create one sample per horizon quarter
        for h in horizon_quarters:
            target_idx = i + h
            if target_idx >= len(df):
                continue

            target_dt = df.index[target_idx]
            sample = feat.copy()

            # Target time features — quarter-hour encoding (96-period cycle)
            quarter_of_day = target_dt.hour * 4 + target_dt.minute // 15
            sample["target_quarter_sin"] = np.sin(2 * np.pi * quarter_of_day / 96)
            sample["target_quarter_cos"] = np.cos(2 * np.pi * quarter_of_day / 96)
            sample["target_hour_sin"] = np.sin(2 * np.pi * target_dt.hour / 24)
            sample["target_hour_cos"] = np.cos(2 * np.pi * target_dt.hour / 24)
            sample["target_dow_sin"] = np.sin(2 * np.pi * target_dt.dayofweek / 7)
            sample["target_dow_cos"] = np.cos(2 * np.pi * target_dt.dayofweek / 7)
            sample["target_is_weekend"] = 1.0 if target_dt.dayofweek >= 5 else 0.0
            sample["target_minute"] = target_dt.minute
            sample["quarters_ahead"] = h

            # Price at same target quarter yesterday and last week
            same_quarter_yesterday = i - 96 + (h % 96)
            if 0 <= same_quarter_yesterday < i:
                sample["target_quarter_price_yesterday"] = price.iloc[same_quarter_yesterday]
            else:
                sample["target_quarter_price_yesterday"] = np.nan

            same_quarter_last_week = i - 672 + (h % 672)
            if 0 <= same_quarter_last_week < i:
                sample["target_quarter_price_last_week"] = price.iloc[same_quarter_last_week]
            else:
                sample["target_quarter_price_last_week"] = np.nan

            # Demand forecast for target quarter
            if "demand_forecast" in df.columns and target_idx < len(df):
                sample["demand_forecast_target"] = df["demand_forecast"].iloc[target_idx]

            # Target-hour weather (broadcast from hourly)
            if weather_15min is not None and target_dt in weather_15min.index:
                tw = weather_15min.loc[target_dt]
                sample["target_weather_temp_c"] = tw.get("temp_c", np.nan)
                sample["target_weather_wind_kmh"] = tw.get("wind_speed_kmh", np.nan)
                sample["target_weather_cloud_pct"] = tw.get("cloud_cover_pct", np.nan)
                _tdr = (tw.get("direct_radiation_wm2", 0) or 0)
                _tdi = (tw.get("diffuse_radiation_wm2", 0) or 0)
                sample["target_weather_radiation"] = _tdr + _tdi
                sample["target_weather_precip_mm"] = tw.get("precipitation_mm", np.nan)
                # Target weather interaction features
                _ttc = tw.get("temp_c", np.nan)
                if _ttc is not None and not np.isnan(_ttc):
                    sample["target_weather_temp_deviation"] = _ttc - 15.0
                    sample["target_weather_heating_dd"] = max(0, 18 - _ttc)
                    sample["target_weather_cooling_dd"] = max(0, _ttc - 24)
                else:
                    sample["target_weather_temp_deviation"] = np.nan
                    sample["target_weather_heating_dd"] = np.nan
                    sample["target_weather_cooling_dd"] = np.nan
                _tws = feat.get("wind_share", 0) or 0
                _tss = feat.get("solar_share", 0) or 0
                sample["target_weather_wind_x_wind_share"] = (tw.get("wind_speed_kmh", 0) or 0) * _tws
                sample["target_weather_cloud_x_solar"] = (tw.get("cloud_cover_pct", 0) or 0) * _tss
                sample["target_weather_ghi_x_solar"] = (_tdr + _tdi) * _tss

            # D+1 published price features (strategic mode)
            # At origin (~15:00 UTC), D+1 hourly prices are known from OMIE
            if run_mode == "strategic" and hourly_price is not None:
                d1_feat = _extract_d1_price_features(
                    hourly_price, origin_dt, target_dt, hourly_price.index,
                )
                sample.update(d1_feat)
            elif run_mode == "strategic":
                # NaN fallback — tree models handle missing values
                sample["d1_mean_price"] = np.nan
                sample["d1_min_price"] = np.nan
                sample["d1_max_price"] = np.nan
                sample["d1_std_price"] = np.nan
                sample["d1_peak_spread"] = np.nan
                sample["d1_same_hour_price"] = np.nan

            sample["target_price"] = price.iloc[target_idx]
            sample["_origin_dt"] = origin_dt
            sample["_target_dt"] = target_dt

            rows.append(sample)

    result = pd.DataFrame(rows)
    logger.info("Built %d 15-min direct-prediction samples for horizons %s "
                "(origin_step=%d, run_mode=%s)",
                len(result), f"{min(horizon_quarters)}-{max(horizon_quarters)}q",
                origin_step, run_mode or "legacy")
    return result


class DirectMultiHorizonTrainer:
    """Trains separate models for different forecast horizon groups."""

    def __init__(self, params: dict | None = None, model_type: str = "histgb",
                 gpu: bool = False, quantile: float | None = None,
                 feature_selection: bool = False,
                 feature_selector=None):
        # Only use DEFAULT_PARAMS for histgb; other model types have their own
        # defaults in _create_model. Custom params (e.g. from Optuna) are passed through.
        if params is not None:
            self.params = params
        elif model_type == "histgb":
            self.params = DEFAULT_PARAMS
        else:
            self.params = None  # let _create_model use its own defaults
        self.model_type = model_type
        self.gpu = gpu
        self.quantile = quantile
        self.feature_selection = feature_selection
        self.feature_selector = feature_selector
        self.models: dict[str, object] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.metrics: dict[str, list[dict]] = {}
        self.conformal_calibrator = None

    def train_all(self, ree_df: pd.DataFrame, n_splits: int = 5,
                  weather_df: pd.DataFrame | None = None,
                  commodity_df: pd.DataFrame | None = None,
                  weather_hourly_df: pd.DataFrame | None = None,
                  resolution: str = "hourly",
                  run_mode: str | None = None) -> dict:
        """
        Train one model per horizon group.

        Parameters
        ----------
        ree_df : Raw REE data (hourly or 15-min depending on resolution).
        n_splits : Number of TimeSeriesSplit folds.
        weather_df : Optional daily weather (AEMET, legacy fallback).
        commodity_df : Optional daily commodity prices.
        weather_hourly_df : Optional hourly weather (Open-Meteo historical).
        resolution : "hourly" or "15min". Determines horizon groups and features.
        run_mode : "dayahead", "strategic", or None (legacy).
                   Selects horizon groups and filters training origins.

        Returns dict of {group_name: fold_metrics}.
        """
        if resolution == "15min" and run_mode == "dayahead":
            horizon_groups = HORIZON_GROUPS_15MIN_DAYAHEAD
        elif resolution == "15min" and run_mode == "strategic":
            horizon_groups = HORIZON_GROUPS_15MIN_STRATEGIC
        elif resolution == "15min":
            horizon_groups = HORIZON_GROUPS_15MIN
        elif run_mode == "dayahead":
            horizon_groups = HORIZON_GROUPS_DAYAHEAD
        elif run_mode == "strategic":
            horizon_groups = HORIZON_GROUPS_STRATEGIC
        else:
            horizon_groups = HORIZON_GROUPS

        all_metrics = {}
        all_oof_residuals = []
        all_oof_horizons = []

        for group_name, horizons in horizon_groups.items():
            if resolution == "15min":
                logger.info("Training %s (quarters %d-%d, run_mode=%s)...",
                            group_name, min(horizons), max(horizons),
                            run_mode or "legacy")
                data = build_direct_features_15min(
                    ree_df, horizons,
                    weather_hourly_df=weather_hourly_df,
                    commodity_df=commodity_df,
                    run_mode=run_mode,
                )
            else:
                logger.info("Training %s (hours %d-%d, run_mode=%s)...",
                            group_name, min(horizons), max(horizons),
                            run_mode or "legacy")
                data = build_direct_features(
                    ree_df, horizons, weather_df, commodity_df,
                    weather_hourly_df=weather_hourly_df,
                    run_mode=run_mode,
                )
            if data.empty or len(data) < 200:
                logger.warning("Skipping %s: insufficient data (%d samples)", group_name, len(data))
                continue

            # Training data validation: log target price statistics
            target_stats = data["target_price"].describe()
            logger.info(
                "%s target stats: mean=%.2f, std=%.2f, min=%.2f, max=%.2f, count=%d",
                group_name, target_stats["mean"], target_stats["std"],
                target_stats["min"], target_stats["max"], int(target_stats["count"]),
            )
            if target_stats["max"] > 500:
                logger.warning(
                    "%s: max target price %.2f > 500 — training data may contain inflated prices",
                    group_name, target_stats["max"],
                )

            # Separate features from metadata/target
            meta_cols = ["target_price", "_origin_dt", "_target_dt"]
            feature_cols = [c for c in data.columns if c not in meta_cols]

            X = data[feature_cols].astype(float)
            y = data["target_price"]

            # NaN diagnostics: log per-feature NaN ratio
            nan_ratios = X.isna().mean()
            high_nan = nan_ratios[nan_ratios > 0.05]
            if not high_nan.empty:
                logger.warning(
                    "%s: %d features have >5%% NaN: %s",
                    group_name, len(high_nan),
                    ", ".join(f"{c}={v:.1%}" for c, v in high_nan.items()),
                )
            total_nan_pct = X.isna().any(axis=1).mean()
            logger.info(
                "%s feature matrix: %d samples x %d features, "
                "%.1f%% rows have any NaN",
                group_name, len(X), len(feature_cols), total_nan_pct * 100,
            )

            # Feature selection (optional)
            if self.feature_selection and self.feature_selector is not None:
                from src.models.trainer import _create_model as _fs_create_model

                def _model_factory():
                    return _fs_create_model(
                        self.model_type, self.params,
                        gpu=self.gpu, quantile=self.quantile,
                    )

                selected = self.feature_selector.select_features(
                    group_name=group_name, X=X, y=y,
                    feature_cols=feature_cols,
                    model_factory=_model_factory,
                )
                feature_cols = selected
                X = X[feature_cols]

            # TimeSeriesSplit CV
            tscv = TimeSeriesSplit(n_splits=n_splits)
            from src.models.trainer import _create_model
            model = _create_model(self.model_type, self.params, gpu=self.gpu,
                                  quantile=self.quantile)

            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Drop NaN targets from train and val
                train_valid = y_train.notna()
                val_valid = y_val.notna()
                X_train, y_train = X_train.loc[train_valid], y_train.loc[train_valid]
                X_val, y_val = X_val.loc[val_valid], y_val.loc[val_valid]

                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning("  %s Fold %d: skipped (empty after NaN removal)", group_name, fold)
                    continue

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # Filter NaN predictions
                pred_valid = ~np.isnan(y_pred)
                y_val_clean = y_val.values[pred_valid]
                y_pred_clean = y_pred[pred_valid]

                mae = mean_absolute_error(y_val_clean, y_pred_clean)
                rmse = np.sqrt(mean_squared_error(y_val_clean, y_pred_clean))

                mask = np.abs(y_val_clean) > 1.0
                mape = (
                    float(np.mean(np.abs((y_val_clean[mask] - y_pred_clean[mask]) / y_val_clean[mask])) * 100)
                    if mask.sum() > 0 else None
                )

                fold_metrics.append({"fold": fold, "mae": mae, "rmse": rmse, "mape": mape})
                logger.info("  %s Fold %d: MAE=%.2f, RMSE=%.2f", group_name, fold, mae, rmse)

                # Collect OOF residuals for conformal calibration
                residuals = y_val_clean - y_pred_clean
                all_oof_residuals.append(residuals)
                if "hours_ahead" in X_val.columns:
                    h_vals = X_val.loc[X_val.index[pred_valid], "hours_ahead"].values.astype(int)
                elif "quarters_ahead" in X_val.columns:
                    h_vals = X_val.loc[X_val.index[pred_valid], "quarters_ahead"].values.astype(int)
                else:
                    h_vals = np.full(len(residuals), min(horizons))
                all_oof_horizons.append(h_vals)

            # Final model on all data (drop NaN targets)
            all_valid = y.notna()
            model.fit(X.loc[all_valid], y.loc[all_valid])
            self.models[group_name] = model
            self.feature_names[group_name] = feature_cols
            self.metrics[group_name] = fold_metrics
            all_metrics[group_name] = fold_metrics

            avg_mae = np.mean([m["mae"] for m in fold_metrics])
            logger.info("  %s Average CV MAE: %.2f", group_name, avg_mae)

        # Fit conformal calibrator from all OOF residuals
        if all_oof_residuals:
            from src.models.conformal import ConformalCalibrator
            combined_residuals = np.concatenate(all_oof_residuals)
            combined_horizons = np.concatenate(all_oof_horizons)
            calibrator = ConformalCalibrator()
            calibrator.fit(combined_residuals, combined_horizons, horizon_groups)
            self.conformal_calibrator = calibrator
            logger.info("Conformal calibrator fitted with %d residuals across %d buckets",
                        len(combined_residuals), len(calibrator.residuals_by_bucket))

        return all_metrics

    @staticmethod
    def _model_suffix(resolution: str = "hourly", approach: str = "hourly",
                      run_mode: str | None = None) -> str:
        """Build file suffix from resolution, approach, and run_mode."""
        if run_mode in ("dayahead", "strategic"):
            if approach == "pure15":
                return f"_15min_pure_{run_mode}"
            elif approach == "hybrid15":
                return f"_15min_hybrid_{run_mode}"
            return f"_{run_mode}"
        if approach == "pure15":
            return "_15min_pure"
        elif approach == "hybrid15":
            return "_15min_hybrid"
        elif resolution == "15min":
            return "_15min"
        return ""

    def save_models(self, version: str | None = None,
                    resolution: str = "hourly",
                    approach: str = "hourly",
                    run_mode: str | None = None) -> str:
        """Save all horizon models as a single artifact."""
        if not self.models:
            raise ValueError("No trained models to save")

        if version is None:
            version = datetime.date.today().isoformat()

        suffix = self._model_suffix(resolution, approach, run_mode)
        path = MODELS_DIR / f"direct_model_{self.model_type}{suffix}_{version}.joblib"

        if resolution == "15min":
            horizon_groups = HORIZON_GROUPS_15MIN
        elif run_mode == "dayahead":
            horizon_groups = HORIZON_GROUPS_DAYAHEAD
        elif run_mode == "strategic":
            horizon_groups = HORIZON_GROUPS_STRATEGIC
        else:
            horizon_groups = HORIZON_GROUPS

        artifact = {
            "type": "direct_multi_horizon",
            "model_type": self.model_type,
            "quantile": self.quantile,
            "version": version,
            "resolution": resolution,
            "approach": approach,
            "run_mode": run_mode or "legacy",
            "trained_at": datetime.datetime.utcnow().isoformat(),
            "models": self.models,
            "feature_names": self.feature_names,
            "cv_metrics": self.metrics,
            "horizon_groups": horizon_groups,
            "conformal_calibrator": (
                self.conformal_calibrator.to_dict()
                if self.conformal_calibrator else None
            ),
            "feature_selector": (
                self.feature_selector.to_dict()
                if self.feature_selector else None
            ),
        }
        joblib.dump(artifact, path)
        logger.info("Direct models (%s, run_mode=%s) saved to %s",
                     self.model_type, run_mode or "legacy", path)
        return str(path)

    def load_models(self, version: str = "latest",
                    resolution: str = "hourly",
                    approach: str = "hourly",
                    run_mode: str | None = None) -> dict:
        """Load direct multi-horizon models from disk."""
        mt = self.model_type
        suffix = self._model_suffix(resolution, approach, run_mode)
        if version == "latest":
            # Try model-type-specific + resolution + approach + run_mode artifact
            model_files = sorted(MODELS_DIR.glob(f"direct_model_{mt}{suffix}_*.joblib"))
            if not model_files and suffix:
                # Fall back: try without approach suffix (legacy files)
                legacy_suffix = "_15min" if resolution == "15min" else ""
                model_files = sorted(MODELS_DIR.glob(f"direct_model_{mt}{legacy_suffix}_*.joblib"))
            if not model_files and suffix:
                raise FileNotFoundError(
                    f"No {run_mode or approach} {mt} models found in {MODELS_DIR}"
                )
            if not model_files:
                # Backward compat: fall back to generic direct_model_*.joblib
                model_files = sorted(MODELS_DIR.glob("direct_model_*.joblib"))
            if not model_files:
                raise FileNotFoundError(f"No direct models found in {MODELS_DIR}")
            path = model_files[-1]
        else:
            path = MODELS_DIR / f"direct_model_{mt}{suffix}_{version}.joblib"
            if not path.exists():
                # Backward compat
                path = MODELS_DIR / f"direct_model_{mt}_{version}.joblib"
                if not path.exists():
                    path = MODELS_DIR / f"direct_model_{version}.joblib"

        artifact = joblib.load(path)
        self.models = artifact["models"]
        self.feature_names = artifact["feature_names"]
        self.metrics = artifact.get("cv_metrics", {})

        cal_data = artifact.get("conformal_calibrator")
        if cal_data:
            from src.models.conformal import ConformalCalibrator
            self.conformal_calibrator = ConformalCalibrator.from_dict(cal_data)
            logger.info("Conformal calibrator loaded (%d buckets)",
                        len(self.conformal_calibrator.residuals_by_bucket))
        else:
            self.conformal_calibrator = None

        sel_data = artifact.get("feature_selector")
        if sel_data:
            from src.models.feature_selection import FeatureSelector
            self.feature_selector = FeatureSelector.from_dict(sel_data)
        else:
            self.feature_selector = None

        loaded_type = artifact.get("model_type", "histgb")
        loaded_mode = artifact.get("run_mode", "legacy")
        logger.info("Loaded direct models %s (%s, run_mode=%s, trained %s)",
                     artifact["version"], loaded_type, loaded_mode,
                     artifact["trained_at"])
        return artifact
