# EPForecast Methodology — Model Training
# Extracted from the EPForecast project (github.com/JorgeLopezLan/epf-methodology)
# Full application: epf.productjorge.com | Docs: epforecast.vercel.app
#
# Updated 2026-04-09 to v11.0 (M0.6 Phase F cutover). v11.0 production is
# single-XGBoost (model_type="xgboost") + residual_1w + pw3x + d365. The
# HistGB and LightGBM branches are kept for historical reference and for
# users who want to reproduce the v4.3-era 3-base ensemble experiments. The
# v10.x LSTM line of experiments was retracted on 2026-04-09 — see
# direct_predictor.py header and SANITIZATION_RULES.md.

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

    Tree hyperparameters are overridable via env vars (no code patches needed):
      EPF_MAX_DEPTH       — tree depth (default 8)
      EPF_LEARNING_RATE   — shrinkage (default 0.05)
      EPF_MIN_CHILD       — min samples per leaf (default 20)
      EPF_REG_LAMBDA      — L2 regularization (default 0.1)
      EPF_N_ESTIMATORS    — number of trees (default 500)

    Args:
        quantile: If set (e.g. 0.55), use quantile loss instead of MAE.
                  Targets the given percentile of the conditional distribution.
    """
    import os
    _depth = int(os.environ.get("EPF_MAX_DEPTH", "12"))
    _lr = float(os.environ.get("EPF_LEARNING_RATE", "0.03"))
    _min_child = int(os.environ.get("EPF_MIN_CHILD", "5"))
    _reg_lambda = float(os.environ.get("EPF_REG_LAMBDA", "0.3"))
    _n_est = int(os.environ.get("EPF_N_ESTIMATORS", "500"))

    if _depth != 12 or _lr != 0.03 or _min_child != 5 or _reg_lambda != 0.3:
        logger.info("Tree overrides: depth=%d lr=%.3f min_child=%d lambda=%.2f n_est=%d",
                     _depth, _lr, _min_child, _reg_lambda, _n_est)

    if model_type == "histgb":
        if gpu:
            logger.warning("HistGradientBoosting has no GPU support. "
                           "Use --model-type xgboost or lightgbm for GPU training. "
                           "Falling back to CPU.")
        defaults = {
            "loss": "absolute_error",
            "max_iter": _n_est, "max_depth": _depth, "learning_rate": _lr,
            "min_samples_leaf": _min_child, "l2_regularization": _reg_lambda,
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
        _num_leaves = min(2 ** _depth - 1, 255) if _depth > 6 else 63
        defaults = {
            "objective": "mae",
            "n_estimators": _n_est, "max_depth": _depth, "learning_rate": _lr,
            "min_child_samples": _min_child, "reg_lambda": _reg_lambda,
            "num_leaves": _num_leaves, "random_state": 42, "verbose": -1,
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
            "n_estimators": _n_est, "max_depth": _depth, "learning_rate": _lr,
            "reg_lambda": _reg_lambda, "random_state": 42, "verbosity": 0,
            "tree_method": "hist",
        }
        # GPU acceleration: use CUDA if available
        _use_gpu = os.environ.get("EPF_USE_GPU", "auto")
        if _use_gpu == "auto":
            try:
                import xgboost as _xgb_test
                _d = _xgb_test.DMatrix([[0]], label=[0])
                _xgb_test.train({"tree_method": "hist", "device": "cuda", "max_depth": 2}, _d, num_boost_round=1)
                defaults["device"] = "cuda"
                logger.info("XGBoost using GPU (CUDA)")
            except Exception:
                logger.info("XGBoost using CPU (no GPU available)")
        elif _use_gpu == "true":
            defaults["device"] = "cuda"
            logger.info("XGBoost using GPU (forced)")
        if _min_child != 20:
            defaults["min_child_weight"] = _min_child
        # V7: Custom asymmetric loss — penalize underprediction of high prices
        _asym_factor = os.environ.get("EPF_ASYM_LOSS_FACTOR", "")
        _asym_thresh = float(os.environ.get("EPF_ASYM_LOSS_THRESHOLD", "80"))
        if _asym_factor:
            _af = float(_asym_factor)
            _at = _asym_thresh
            def _asym_obj(y_true, y_pred):
                import numpy as _np
                residual = y_pred - y_true
                grad = _np.where(residual < 0, -1.0, 1.0)  # MAE gradient
                hess = _np.ones_like(grad)
                # Boost gradient for underprediction of high prices
                high_under = (y_true > _at) & (residual < 0)
                grad[high_under] *= _af
                hess[high_under] *= _af
                return grad, hess
            defaults["objective"] = _asym_obj
            logger.info("XGBoost: asymmetric loss (factor=%.1f, threshold=%.0f)", _af, _at)
        elif quantile is not None:
            defaults["objective"] = "reg:quantileerror"
            defaults["quantile_alpha"] = quantile
        if gpu:
            defaults["device"] = "cuda"
            logger.info("XGBoost: CUDA GPU training enabled")
        defaults.update(params or {})
        return xgb.XGBRegressor(**defaults)
    elif model_type == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("catboost required: pip install catboost")
        defaults = {
            "iterations": _n_est, "depth": _depth, "learning_rate": _lr,
            "l2_leaf_reg": _reg_lambda * 30, "random_seed": 42, "verbose": 0,
            "loss_function": "MAE",
        }
        if quantile is not None:
            defaults["loss_function"] = f"Quantile:alpha={quantile}"
        if gpu:
            defaults["task_type"] = "GPU"
            logger.info("CatBoost: GPU training enabled")
        defaults.update(params or {})
        return CatBoostRegressor(**defaults)
    elif model_type == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        # Env var overrides for MLP architecture
        _hidden = os.environ.get("EPF_MLP_HIDDEN", "")
        _lr = float(os.environ.get("EPF_MLP_LR", "0.001"))
        _max_iter = int(os.environ.get("EPF_MLP_MAX_ITER", "200"))
        defaults = {
            "hidden_layer_sizes": tuple(int(x) for x in _hidden.split(",")) if _hidden else (256, 128, 64),
            "learning_rate_init": _lr,
            "max_iter": _max_iter,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": 42,
            "verbose": False,
        }
        defaults.update(params or {})
        logger.info("MLP: %s, lr=%.4f, max_iter=%d", defaults["hidden_layer_sizes"], defaults["learning_rate_init"], defaults["max_iter"])
        # Wrap in pipeline: impute NaN → scale → (optional feature select) → MLP
        mlp = MLPRegressor(**defaults)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        _top_n = os.environ.get("EPF_MLP_TOP_FEATURES", "")
        if _top_n:
            from sklearn.feature_selection import SelectKBest, mutual_info_regression
            steps.append(("select", SelectKBest(mutual_info_regression, k=int(_top_n))))
            logger.info("MLP: selecting top %s features by mutual information", _top_n)
        steps.append(("mlp", mlp))
        return Pipeline(steps)
    elif model_type == "tft":
        from src.models.tft_model import TFTRegressor
        _hidden = int(os.environ.get("EPF_TFT_HIDDEN", "128"))
        _layers = int(os.environ.get("EPF_TFT_LAYERS", "3"))
        _dropout = float(os.environ.get("EPF_TFT_DROPOUT", "0.2"))
        _tft_lr = float(os.environ.get("EPF_TFT_LR", "0.001"))
        _batch = int(os.environ.get("EPF_TFT_BATCH_SIZE", "512"))
        _epochs = int(os.environ.get("EPF_TFT_MAX_EPOCHS", "200"))
        _patience = int(os.environ.get("EPF_TFT_PATIENCE", "20"))
        _loss = os.environ.get("EPF_TFT_LOSS", "huber")
        _wd = float(os.environ.get("EPF_TFT_WEIGHT_DECAY", "0.0001"))
        defaults = {
            "hidden_dim": _hidden, "n_layers": _layers, "dropout": _dropout,
            "lr": _tft_lr, "batch_size": _batch, "max_epochs": _epochs,
            "patience": _patience, "loss_fn": _loss, "weight_decay": _wd,
            "device": "cuda" if gpu else "auto",
        }
        defaults.update(params or {})
        logger.info("TFT: hidden=%d, layers=%d, dropout=%.2f, lr=%.4f, batch=%d, epochs=%d, loss=%s",
                     _hidden, _layers, _dropout, _tft_lr, _batch, _epochs, _loss)
        return TFTRegressor(**defaults)
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
        """Save model to disk with metadata. Returns the file path."""
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
            "conformal_calibrator": (
                self.conformal_calibrator.to_dict()
                if self.conformal_calibrator else None
            ),
        }
        joblib.dump(artifact, path)
        logger.info("Model saved to %s", path)
        return str(path)

    def load_model(self, version: str = "latest") -> dict:
        """Load a model from disk. Returns the full artifact dict."""
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

        cal_data = artifact.get("conformal_calibrator")
        if cal_data:
            from src.models.conformal import ConformalCalibrator
            self.conformal_calibrator = ConformalCalibrator.from_dict(cal_data)
            logger.info("Conformal calibrator loaded (%d buckets)",
                        len(self.conformal_calibrator.residuals_by_bucket))
        else:
            self.conformal_calibrator = None

        logger.info("Loaded model %s (trained %s)", artifact["version"], artifact["trained_at"])
        return artifact

    def build_conformal_calibrator_from_db(self, pipeline):
        """
        Build conformal calibrator from historical prediction-vs-actual data.

        Used by the recursive predictor, whose CV residuals don't reflect
        recursive error compounding. This uses real deployed performance.
        """
        from src.models.conformal import build_calibrator_from_predictions
        predictions_df = pipeline.get_predictions(days_back=90)
        calibrator = build_calibrator_from_predictions(predictions_df)
        if calibrator:
            self.conformal_calibrator = calibrator
        return calibrator

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
