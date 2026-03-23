import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import holidays as holidays_lib
    _HOLIDAYS_LIB_AVAILABLE = True
except ImportError:
    _HOLIDAYS_LIB_AVAILABLE = False

# Fallback: Spanish national holidays (fixed dates only)
_SPAIN_HOLIDAYS_FIXED = [
    (1, 1),   # Año Nuevo
    (1, 6),   # Epifanía
    (5, 1),   # Día del Trabajo
    (8, 15),  # Asunción de la Virgen
    (10, 12), # Fiesta Nacional de España
    (11, 1),  # Todos los Santos
    (12, 6),  # Día de la Constitución
    (12, 8),  # Inmaculada Concepción
    (12, 25), # Navidad
]


def _get_spanish_holidays(years: range) -> set:
    """Get all Spanish national holidays (including Easter-dependent) for the given years."""
    if _HOLIDAYS_LIB_AVAILABLE:
        es_holidays = holidays_lib.Spain(years=years)
        return set(es_holidays.keys())
    return set()


def _is_spanish_holiday(dt_index: pd.DatetimeIndex) -> pd.Series:
    """Check if dates are Spanish national holidays."""
    if _HOLIDAYS_LIB_AVAILABLE:
        years = range(dt_index.year.min(), dt_index.year.max() + 1)
        holiday_dates = _get_spanish_holidays(years)
        return pd.Series(
            [1.0 if d.date() in holiday_dates else 0.0 for d in dt_index],
            index=dt_index,
        )
    # Fallback to fixed-date holidays
    month_day = list(zip(dt_index.month, dt_index.day))
    return pd.Series(
        [1.0 if (m, d) in _SPAIN_HOLIDAYS_FIXED else 0.0 for m, d in month_day],
        index=dt_index,
    )


def _calendar_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build extended calendar features: bridge days, vacation periods."""
    result = pd.DataFrame(index=dt_index)

    holiday_series = _is_spanish_holiday(dt_index)

    # Pre-holiday and post-holiday (bridge day detection)
    holiday_shifted_fwd = holiday_series.shift(-1, fill_value=0)
    holiday_shifted_back = holiday_series.shift(1, fill_value=0)
    result["is_pre_holiday"] = holiday_shifted_fwd
    result["is_post_holiday"] = holiday_shifted_back

    # Christmas period (Dec 23 - Jan 6)
    result["is_christmas_period"] = (
        ((dt_index.month == 12) & (dt_index.day >= 23)) |
        ((dt_index.month == 1) & (dt_index.day <= 6))
    ).astype(float)

    # August vacation effect
    result["is_august"] = (dt_index.month == 8).astype(float)

    # Week of year (cyclical)
    week = dt_index.isocalendar().week.values.astype(float)
    result["week_sin"] = np.sin(2 * np.pi * week / 52)
    result["week_cos"] = np.cos(2 * np.pi * week / 52)

    return result


def build_features(ree_df: pd.DataFrame,
                   weather_df: pd.DataFrame | None = None,
                   commodity_df: pd.DataFrame | None = None,
                   weather_hourly_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Transform raw REE hourly data (+ optional weather/commodity) into ML-ready features.

    Parameters
    ----------
    ree_df : DataFrame indexed by datetime_utc (UTC-aware) with indicator columns.
    weather_df : Optional DataFrame indexed by date with daily weather columns
                 (legacy AEMET; used as fallback when weather_hourly_df is not available).
    commodity_df : Optional DataFrame indexed by date with commodity columns
                   (ttf_gas_eur_mwh, ets_carbon_eur).
    weather_hourly_df : Optional DataFrame indexed by datetime_utc (UTC-aware) with
                        hourly weather columns from Open-Meteo historical archive.
                        When provided, takes precedence over daily weather_df.

    Returns
    -------
    DataFrame with feature columns and 'day_ahead_price' as the target.
    Rows with NaN (from lag warmup) are dropped.
    """
    df = ree_df.copy()

    if "day_ahead_price" not in df.columns:
        raise ValueError("ree_df must contain 'day_ahead_price' column")

    # ── Time features (cyclical encoding) ──────────────────────────────
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["is_weekend"] = (dow >= 5).astype(float)
    df["is_holiday"] = _is_spanish_holiday(df.index)

    # Extended calendar features
    cal = _calendar_features(df.index)
    for col in cal.columns:
        df[col] = cal[col].values

    # ── Lag features ───────────────────────────────────────────────────
    price = df["day_ahead_price"]
    df["price_lag_1h"] = price.shift(1)
    df["price_lag_2h"] = price.shift(2)
    df["price_lag_3h"] = price.shift(3)
    df["price_lag_24h"] = price.shift(24)
    df["price_lag_48h"] = price.shift(48)
    df["price_lag_72h"] = price.shift(72)    # 3 days ago same hour
    df["price_lag_168h"] = price.shift(168)  # same hour last week

    # Rolling statistics
    df["price_rolling_6h"] = price.rolling(6, min_periods=1).mean().shift(1)
    df["price_rolling_24h"] = price.rolling(24, min_periods=1).mean().shift(1)
    df["price_rolling_168h"] = price.rolling(168, min_periods=1).mean().shift(1)

    # Volatility features
    df["price_std_24h"] = price.rolling(24, min_periods=1).std().shift(1)
    df["price_std_168h"] = price.rolling(168, min_periods=1).std().shift(1)
    df["price_min_24h"] = price.rolling(24, min_periods=1).min().shift(1)
    df["price_max_24h"] = price.rolling(24, min_periods=1).max().shift(1)
    df["price_range_24h"] = df["price_max_24h"] - df["price_min_24h"]

    # Price momentum
    df["price_change_1h"] = price.diff(1).shift(1)
    df["price_change_24h"] = price.diff(24).shift(1)

    # Demand lags
    if "real_demand" in df.columns:
        df["demand_lag_1h"] = df["real_demand"].shift(1)
        df["demand_lag_24h"] = df["real_demand"].shift(24)
        df["demand_rolling_24h"] = df["real_demand"].rolling(24, min_periods=1).mean().shift(1)

    if "demand_forecast" in df.columns and "real_demand" in df.columns:
        demand_lag = df["real_demand"].shift(24)
        df["demand_forecast_ratio"] = df["demand_forecast"] / demand_lag.replace(0, np.nan)

    # ── Generation mix features ────────────────────────────────────────
    wind = df.get("wind_generation", pd.Series(0, index=df.index))
    solar_pv = df.get("solar_pv_generation", pd.Series(0, index=df.index))
    solar_th = df.get("solar_thermal_gen", pd.Series(0, index=df.index))
    nuclear = df.get("nuclear_generation", pd.Series(0, index=df.index))
    demand = df.get("real_demand", pd.Series(1, index=df.index))  # avoid div by zero

    total_renewable = wind + solar_pv + solar_th
    demand_safe = demand.replace(0, np.nan)

    df["renewable_share"] = total_renewable / demand_safe
    df["wind_share"] = wind / demand_safe
    df["solar_share"] = (solar_pv + solar_th) / demand_safe
    df["nuclear_share"] = nuclear / demand_safe
    df["residual_demand"] = demand - total_renewable - nuclear

    # Thermal generation features (if available from Phase 3)
    combined_cycle = df.get("combined_cycle_gen", pd.Series(0, index=df.index))
    coal = df.get("coal_generation", pd.Series(0, index=df.index))
    cogen = df.get("cogeneration", pd.Series(0, index=df.index))
    if (combined_cycle > 0).any() or (coal > 0).any():
        df["thermal_share"] = (combined_cycle + coal + cogen) / demand_safe
        df["gas_marginal_indicator"] = (combined_cycle > 0).astype(float)

    # Interconnection features (if available from Phase 3)
    france = df.get("france_interconnection", pd.Series(0, index=df.index))
    portugal = df.get("portugal_interconnection", pd.Series(0, index=df.index))
    morocco = df.get("morocco_interconnection", pd.Series(0, index=df.index))
    if (france != 0).any() or (portugal != 0).any():
        df["net_imports"] = france + portugal + morocco
        df["net_import_share"] = df["net_imports"] / demand_safe

    # ── Weather features ──────────────────────────────────────────────
    # Prefer hourly weather (Open-Meteo) over daily (AEMET) when available
    _used_hourly_weather = False
    if weather_hourly_df is not None and not weather_hourly_df.empty:
        # Direct hourly join — no broadcasting
        weather_h = weather_hourly_df.copy()
        # Align on datetime_utc index, interpolate small gaps (up to 3h)
        for col in weather_h.columns:
            aligned = weather_h[col].reindex(df.index)
            # Interpolate gaps of up to 3 consecutive NaN values
            df[col] = aligned.interpolate(method="time", limit=3)

        _used_hourly_weather = True
        logger.info("Joined %d hourly weather columns", len(weather_h.columns))

        # Hourly weather interaction features
        if "temp_c" in df.columns:
            df["cold_x_demand"] = np.maximum(0, 15 - df["temp_c"]) * demand
            df["temp_deviation"] = df["temp_c"] - 15.0
            df["heating_degree_days"] = np.maximum(0, 18 - df["temp_c"])
            df["cooling_degree_days"] = np.maximum(0, df["temp_c"] - 24)
            df["temp_deviation_sq"] = df["temp_deviation"] ** 2
        if "wind_speed_kmh" in df.columns:
            df["wind_speed_x_wind_share"] = df["wind_speed_kmh"] * df["wind_share"]
        if "precipitation_mm" in df.columns and "hydro_generation" in df.columns:
            hydro = df.get("hydro_generation", pd.Series(0, index=df.index))
            df["precip_x_hydro"] = df["precipitation_mm"] * hydro / demand_safe
        if "cloud_cover_pct" in df.columns:
            df["cloud_x_solar"] = df["cloud_cover_pct"] * df["solar_share"]
        if "sunshine_hours" in df.columns:
            df["sunshine_x_solar"] = df["sunshine_hours"] * df["solar_share"]

        # ── Solar irradiance features (Phase 1.2) ────────────────────
        if "direct_radiation_wm2" in df.columns and "diffuse_radiation_wm2" in df.columns:
            direct = df["direct_radiation_wm2"].fillna(0)
            diffuse = df["diffuse_radiation_wm2"].fillna(0)
            df["ghi_wm2"] = direct + diffuse

            # GHI interaction with solar generation share
            df["ghi_x_solar_share"] = df["ghi_wm2"] * df["solar_share"]

            # Solar angle (sun elevation) — inlined, no external dependency
            def _compute_solar_angle(dt_index, latitude=40.53):
                """Approximate solar elevation angle (degrees) for Spain."""
                day_of_year = dt_index.dayofyear
                hour_utc = dt_index.hour + dt_index.minute / 60.0
                B = 2 * np.pi * (day_of_year - 1) / 365.0
                declination = np.degrees(
                    0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
                    - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
                )
                solar_time = hour_utc + 0.23  # longitude correction ~3.5°W
                hour_angle = 15.0 * (solar_time - 12.0)
                lat_rad = np.radians(latitude)
                dec_rad = np.radians(declination)
                ha_rad = np.radians(hour_angle)
                sin_elev = (np.sin(lat_rad) * np.sin(dec_rad)
                            + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
                return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))
            df["solar_elevation_deg"] = _compute_solar_angle(df.index)
            df["is_daylight"] = (df["solar_elevation_deg"] > 0).astype(float)

            # Clear-sky index: ratio of actual GHI to theoretical max
            # Theoretical max GHI ≈ 1361 * sin(elevation) * atmospheric_transmittance
            # Using a simplified model: clear_sky_ghi = max(0, 990 * sin(elevation))
            sin_elev = np.sin(np.radians(df["solar_elevation_deg"].clip(lower=0)))
            theoretical_ghi = 990.0 * sin_elev
            df["clear_sky_index"] = np.where(
                theoretical_ghi > 10,
                (df["ghi_wm2"] / theoretical_ghi).clip(0, 1.5),
                0.0,
            )

    elif weather_df is not None and not weather_df.empty:
        # Fallback: broadcast daily weather to hourly rows (legacy AEMET path)
        df["date"] = df.index.date
        weather = weather_df.copy()
        weather.index = pd.to_datetime(weather.index).date

        df = df.merge(weather, left_on="date", right_index=True, how="left")
        df = df.drop(columns=["date"])

        # Interaction features (daily resolution)
        if "sunshine_hours" in df.columns:
            df["sunshine_x_solar"] = df["sunshine_hours"] * df["solar_share"]
        if "temp_avg" in df.columns:
            df["cold_x_demand"] = np.maximum(0, 15 - df["temp_avg"]) * demand
            df["temp_deviation"] = df["temp_avg"] - 15.0
            df["heating_degree_days"] = np.maximum(0, 18 - df["temp_avg"])
            df["cooling_degree_days"] = np.maximum(0, df["temp_avg"] - 24)
            df["temp_deviation_sq"] = df["temp_deviation"] ** 2
        if "wind_avg" in df.columns:
            df["wind_speed_x_wind_share"] = df["wind_avg"] * df["wind_share"]
        if "precipitation" in df.columns and "hydro_generation" in df.columns:
            hydro = df.get("hydro_generation", pd.Series(0, index=df.index))
            df["precip_x_hydro"] = df["precipitation"] * hydro / demand_safe

    # ── Commodity features ────────────────────────────────────────────
    if commodity_df is not None and not commodity_df.empty:
        df["date"] = df.index.date
        commodity = commodity_df.copy()
        commodity.index = pd.to_datetime(commodity.index).date

        df = df.merge(commodity, left_on="date", right_index=True, how="left")
        df = df.drop(columns=["date"])

        # Forward-fill commodity prices (weekends/gaps)
        if "ttf_gas_eur_mwh" in df.columns:
            df["ttf_gas_eur_mwh"] = df["ttf_gas_eur_mwh"].ffill()
            df["gas_price"] = df["ttf_gas_eur_mwh"]
        if "ets_carbon_eur" in df.columns:
            df["ets_carbon_eur"] = df["ets_carbon_eur"].ffill()
            df["carbon_price"] = df["ets_carbon_eur"]

        # Derived commodity features
        if "gas_price" in df.columns:
            # Marginal cost proxy: gas_price * heat_rate + carbon_cost
            carbon = df.get("carbon_price", pd.Series(0, index=df.index)).fillna(0)
            df["marginal_cost_gas"] = df["gas_price"] * 1.9 + carbon * 0.37
            df["spark_spread"] = price.shift(24) - df["marginal_cost_gas"]
            df["gas_x_residual_demand"] = df["gas_price"] * df["residual_demand"]

    # ── Drop rows with NaN from lag warmup ─────────────────────────────
    initial_rows = len(df)
    df = df.dropna(subset=["price_lag_168h"])  # 7 days of warmup
    dropped = initial_rows - len(df)
    logger.info(
        "Features built: %d rows (%d dropped from lag warmup)",
        len(df), dropped,
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (everything except the target and raw indicators)."""
    target = "day_ahead_price"
    # Raw indicator columns that are replaced by engineered features
    raw_indicators = [
        "intraday_price", "real_demand", "demand_forecast",
        "demand_daily_forecast", "wind_generation", "solar_pv_generation",
        "solar_thermal_gen", "hydro_generation", "nuclear_generation",
        "combined_cycle_gen", "coal_generation", "cogeneration",
        "france_interconnection", "portugal_interconnection", "morocco_interconnection",
        "ttf_gas_eur_mwh", "ets_carbon_eur", "collected_at",
        # Intermediate commodity aliases
        "gas_price", "carbon_price",
        # Raw irradiance kept via derived features (ghi_wm2, clear_sky_index, etc.)
        "direct_radiation_wm2", "diffuse_radiation_wm2",
    ]
    exclude = {target} | set(raw_indicators)
    return [c for c in df.columns if c not in exclude]
