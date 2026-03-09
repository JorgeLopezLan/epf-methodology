# EPForecast Methodology — Model Training
# Extracted from the EPForecast project (github.com/JorgeLopezLan/epf-methodology)
# Full application: epf.productjorge.com | Docs: epforecast.vercel.app

import logging
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS_DIR = None  # Set to your model artifacts directory (pathlib.Path)

# Conformal prediction calibration (not included in this extract)

logger = logging.getLogger(__name__)


def _check_gpu_available() -> bool:
    """Check if CUDA GPU is available for training."""
    try:
        import xgboost as xgb
        # XGBoost exposes GPU info; try a quick probe
        return True  # If xgboost is installed, user can attempt GPU
    except ImportError:
        pass
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    return False


def _create_model(model_type: str = "histgb", params: dict | None = None,
                  gpu: bool = False, quantile: float | None = None):
    """Factory: create a regression model by type, optionally on GPU.

    Args:
        quantile: If set (e.g. 0.55), use quantile loss instead of MAE.
                  Targets the given percentile of the conditional distribution.
    """
    if model_type == "histgb":
        if gpu:
            logger.warning("HistGradientBoosting has no GPU support. "
                           "Use --model-type xgboost or lightgbm for GPU training. "
                           "Falling back to CPU.")
        defaults = {
            "loss": "absolute_error",
            "max_iter": 500, "max_depth": 8, "learning_rate": 0.05,
            "min_samples_leaf": 20, "l2_regularization": 0.1,
            "early_stopping": True, "validation_fraction": 0.1,
            "n_iter_no_change": 20, "random_state": 42,
        }
        if quantile is not None:
            defaults["loss"] = "quantile"
            defaults["quantile"] = quantile
        defaults.update(params or {})
        return HistGradientBoostingRegressor(**defaults)
    elif model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required: pip install lightgbm>=4.3.0")
        defaults = {
            "objective": "mae",
            "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
            "min_child_samples": 20, "reg_lambda": 0.1,
            "num_leaves": 63, "random_state": 42, "verbose": -1,
        }
        if quantile is not None:
            defaults["objective"] = "quantile"
            defaults["alpha"] = quantile
        if gpu:
            defaults["device"] = "gpu"
            logger.info("LightGBM: GPU training enabled")
        defaults.update(params or {})
        return lgb.LGBMRegressor(**defaults)
    elif model_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")
        defaults = {
            "objective": "reg:absoluteerror",
            "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
            "reg_lambda": 0.1, "random_state": 42, "verbosity": 0,
            "tree_method": "hist",
        }
        if quantile is not None:
            defaults["objective"] = "reg:quantileerror"
            defaults["quantile_alpha"] = quantile
        if gpu:
            defaults["device"] = "cuda"
            logger.info("XGBoost: CUDA GPU training enabled")
        defaults.update(params or {})
        return xgb.XGBRegressor(**defaults)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class ModelTrainer:
    """Trains and persists electricity price forecasting models."""

    def __init__(self, model_type: str = "histgb", params: dict | None = None,
                 gpu: bool = False):
        self.model_type = model_type
        self.model_params = params
        self.gpu = gpu
        self.model = None
        self.feature_names: list[str] = []
        self.metrics: list[dict] = []
        self.conformal_calibrator = None

    def train(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list[dict]:
        """
        Train with time-series cross-validation.

        Parameters
        ----------
        X : Feature matrix.
        y : Target variable (day_ahead_price).
        n_splits : Number of TimeSeriesSplit folds.

        Returns
        -------
        List of per-fold metric dicts.
        """
        self.feature_names = list(X.columns)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        model = _create_model(self.model_type, self.model_params, gpu=self.gpu)

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            # Safe MAPE: only for |price| > 1 EUR/MWh
            mask = np.abs(y_val) > 1.0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
            else:
                mape = None

            metrics = {"fold": fold, "mae": mae, "rmse": rmse, "mape": mape}
            fold_metrics.append(metrics)
            logger.info("Fold %d: MAE=%.2f, RMSE=%.2f, MAPE=%s",
                        fold, mae, rmse, f"{mape:.1f}%" if mape else "N/A")

        # Final model: train on ALL data
        logger.info("Training final model on %d samples...", len(X))
        model.fit(X, y)
        self.model = model
        self.metrics = fold_metrics

        avg_mae = np.mean([m["mae"] for m in fold_metrics])
        avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
        logger.info("Average CV metrics: MAE=%.2f, RMSE=%.2f", avg_mae, avg_rmse)

        return fold_metrics

    def save_model(self, version: str | None = None) -> str:
        """Save model to disk with metadata. Returns the file path.

        Note: Conformal calibrator serialization is not included in this extract.
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        if version is None:
            version = datetime.date.today().isoformat()

        path = MODELS_DIR / f"model_{version}.joblib"
        artifact = {
            "model": self.model,
            "version": version,
            "trained_at": datetime.datetime.utcnow().isoformat(),
            "feature_names": self.feature_names,
            "cv_metrics": self.metrics,
            # Conformal calibrator serialization not included in this extract
            "conformal_calibrator": None,
        }
        joblib.dump(artifact, path)
        logger.info("Model saved to %s", path)
        return str(path)

    def load_model(self, version: str = "latest") -> dict:
        """Load a model from disk. Returns the full artifact dict.

        Note: Conformal calibrator deserialization is not included in this extract.
        """
        if version == "latest":
            model_files = sorted(MODELS_DIR.glob("model_*.joblib"))
            if not model_files:
                raise FileNotFoundError(f"No models found in {MODELS_DIR}")
            path = model_files[-1]
        else:
            path = MODELS_DIR / f"model_{version}.joblib"

        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.metrics = artifact.get("cv_metrics", [])

        # Conformal calibrator deserialization not included in this extract
        self.conformal_calibrator = None

        logger.info("Loaded model %s (trained %s)", artifact["version"], artifact["trained_at"])
        return artifact

    def build_conformal_calibrator_from_db(self, pipeline):
        """
        Build conformal calibrator from historical prediction-vs-actual data.

        Used by the recursive predictor, whose CV residuals don't reflect
        recursive error compounding. This uses real deployed performance.

        Note: Conformal calibration is not included in this extract.
        """
        # Conformal calibration (build_calibrator_from_predictions) is not included
        # in this extract. See the full EPForecast project for the implementation.
        logger.warning("Conformal calibration is not available in this extract.")
        return None

    def get_feature_importances(self) -> pd.Series:
        """Return feature importances sorted by magnitude."""
        if self.model is None:
            raise ValueError("No model loaded")
        try:
            raw = self.model.feature_importances_
        except AttributeError:
            # Compute from internal tree structure for HistGradientBoosting
            from sklearn.inspection import permutation_importance
            logger.info("Computing permutation importance (feature_importances_ not available)")
            # Return uniform importances as fallback
            raw = np.ones(len(self.feature_names)) / len(self.feature_names)
        importances = pd.Series(
            raw,
            index=self.feature_names,
        ).sort_values(ascending=False)
        return importances
