import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray,
              min_price: float = 1.0) -> float | None:
    """MAPE excluding hours where |actual price| < min_price."""
    mask = np.abs(y_true) > min_price
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray,
                   timestamps: pd.DatetimeIndex | None = None) -> dict:
    """
    Comprehensive model evaluation.

    Returns a dict with overall metrics and optional per-hour/per-day breakdowns.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    results = {
        "overall": {
            "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
            "mape": safe_mape(y_true_arr, y_pred_arr),
            "n_samples": int(len(y_true_arr)),
        }
    }

    if timestamps is not None:
        # Per-hour breakdown
        hours = timestamps.hour
        hourly = {}
        for h in range(24):
            mask = hours == h
            if mask.sum() > 0:
                hourly[h] = {
                    "mae": float(mean_absolute_error(y_true_arr[mask], y_pred_arr[mask])),
                    "n_samples": int(mask.sum()),
                }
        results["by_hour"] = hourly

        # Per day-of-week breakdown
        dows = timestamps.dayofweek
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = {}
        for d in range(7):
            mask = dows == d
            if mask.sum() > 0:
                daily[dow_names[d]] = {
                    "mae": float(mean_absolute_error(y_true_arr[mask], y_pred_arr[mask])),
                    "n_samples": int(mask.sum()),
                }
        results["by_day_of_week"] = daily

    return results


def format_metrics_report(metrics: dict) -> str:
    """Format evaluation metrics as a human-readable string."""
    lines = [
        "=== Model Evaluation Report ===",
        f"  MAE:  {metrics['overall']['mae']:.2f} EUR/MWh",
        f"  RMSE: {metrics['overall']['rmse']:.2f} EUR/MWh",
    ]
    if metrics["overall"]["mape"] is not None:
        lines.append(f"  MAPE: {metrics['overall']['mape']:.1f}%")
    lines.append(f"  Samples: {metrics['overall']['n_samples']}")

    if "by_hour" in metrics:
        lines.append("\n--- MAE by Hour ---")
        for h, m in sorted(metrics["by_hour"].items()):
            lines.append(f"  Hour {h:2d}: {m['mae']:.2f}")

    if "by_day_of_week" in metrics:
        lines.append("\n--- MAE by Day of Week ---")
        for day, m in metrics["by_day_of_week"].items():
            lines.append(f"  {day}: {m['mae']:.2f}")

    return "\n".join(lines)


# ── Economic / Trading Quality Metrics ──────────────────────────────


def corr_f_raw(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Pearson correlation between forecast and actual price series."""
    if len(y_true) < 3:
        return None
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return None if np.isnan(r) else round(float(r), 4)


def corr_f_deviation(y_true: np.ndarray, y_pred: np.ndarray,
                     dates: np.ndarray) -> float | None:
    """Correlation after removing daily means (within-day shape).

    Most relevant for BESS/day-ahead: captures whether the forecast
    tracks intraday peaks and valleys, independent of price level.
    """
    if len(y_true) < 3:
        return None
    unique_dates = np.unique(dates)
    dev_true, dev_pred = [], []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < 2:
            continue
        dt = y_true[mask]
        dp = y_pred[mask]
        dev_true.extend(dt - dt.mean())
        dev_pred.extend(dp - dp.mean())
    if len(dev_true) < 3:
        return None
    r = np.corrcoef(dev_true, dev_pred)[0, 1]
    return None if np.isnan(r) else round(float(r), 4)


def corr_f_first_diff(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Correlation of hour-to-hour price changes (first differences).

    Most relevant for intraday trading: measures agreement on the
    direction and magnitude of price movements.
    """
    if len(y_true) < 4:
        return None
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)
    r = np.corrcoef(diff_true, diff_pred)[0, 1]
    return None if np.isnan(r) else round(float(r), 4)


def cov_e(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Correlation between forecast errors and actual price levels.

    Normalized to [-1, +1]. High absolute value means errors are
    systematic w.r.t. price level (e.g., underestimating spikes).
    Low absolute value means errors are random — less harmful.
    """
    if len(y_true) < 3:
        return None
    errors = y_pred - y_true
    std_actual = np.std(y_true)
    std_error = np.std(errors)
    if std_actual == 0 or std_error == 0:
        return None
    r = np.corrcoef(y_true, errors)[0, 1]
    return None if np.isnan(r) else round(float(r), 4)


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Percentage of hours where forecast and actual move in the same direction.

    Computed on hour-to-hour differences. Zero-change hours excluded.
    Returns a percentage (0-100).
    """
    if len(y_true) < 3:
        return None
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)
    mask = diff_true != 0
    if mask.sum() == 0:
        return None
    same_sign = np.sign(diff_true[mask]) == np.sign(diff_pred[mask])
    return round(float(same_sign.mean() * 100), 1)


def spike_recall(y_true: np.ndarray, y_pred: np.ndarray,
                 percentile: float = 90) -> float | None:
    """Percentage of actual top-N% price hours also flagged by forecast.

    Measures ability to identify expensive hours — critical for
    BESS discharge timing. Returns a percentage (0-100).
    """
    if len(y_true) < 10:
        return None
    threshold_actual = np.percentile(y_true, percentile)
    threshold_pred = np.percentile(y_pred, percentile)
    actual_spikes = y_true >= threshold_actual
    pred_spikes = y_pred >= threshold_pred
    n_actual_spikes = actual_spikes.sum()
    if n_actual_spikes == 0:
        return None
    recalled = (actual_spikes & pred_spikes).sum()
    return round(float(recalled / n_actual_spikes * 100), 1)


def spread_capture(y_true: np.ndarray, y_pred: np.ndarray,
                   dates: np.ndarray,
                   n_charge: int = 4, n_discharge: int = 4) -> float | None:
    """Percentage of theoretical maximum daily spread captured.

    For each day, compares the spread from following the forecast's
    charge/discharge ranking vs the theoretical optimum ranking.
    Returns mean capture ratio as percentage (0-100+).

    Args:
        n_charge: number of cheapest slots to charge (4 hours or 16 quarter-hours)
        n_discharge: number of most expensive slots to discharge
    """
    unique_dates = np.unique(dates)
    ratios = []
    min_slots = n_charge + n_discharge
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < min_slots:
            continue
        day_actual = y_true[mask]
        day_pred = y_pred[mask]

        # Theoretical max: sort actual prices
        sorted_actual = np.sort(day_actual)
        theo_max = sorted_actual[-n_discharge:].mean() - sorted_actual[:n_charge].mean()
        if theo_max <= 0:
            continue

        # Forecast-guided: pick slots based on predicted prices
        pred_order = np.argsort(day_pred)
        charge_slots = pred_order[:n_charge]
        discharge_slots = pred_order[-n_discharge:]
        forecast_spread = day_actual[discharge_slots].mean() - day_actual[charge_slots].mean()

        ratios.append(forecast_spread / theo_max)

    if not ratios:
        return None
    return round(float(np.mean(ratios) * 100), 1)
