"""
Feature selection pipeline for EPF direct multi-horizon models.

Two-stage filter applied per horizon group:
  1. Correlation filter: drop one of each pair with |r| > threshold
  2. Permutation importance pruning: drop features below threshold

Operates on the feature matrix X and target y after build_direct_features(),
returning a filtered list of feature column names. The predictor automatically
uses the selected features via the stored feature_names in the model artifact.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def correlation_filter(
    X: pd.DataFrame,
    threshold: float = 0.95,
    importance_ranking: pd.Series | None = None,
) -> list[str]:
    """Drop one of each pair of features with |r| > threshold.

    When two features are highly correlated, keep the one with higher
    importance (if provided) or the one that appears first in the DataFrame.

    Parameters
    ----------
    X : Feature matrix (samples x features), float, may contain NaN.
    threshold : Pairwise Pearson correlation cutoff (default 0.95).
    importance_ranking : Optional Series mapping feature names to importance
        scores. When provided, the more important feature in each
        correlated pair is kept.

    Returns
    -------
    List of feature names to KEEP.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop: set[str] = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for partner in correlated:
            if partner in to_drop or col in to_drop:
                continue
            # Decide which to drop
            if importance_ranking is not None:
                imp_col = importance_ranking.get(col, 0.0)
                imp_partner = importance_ranking.get(partner, 0.0)
                drop = partner if imp_col >= imp_partner else col
            else:
                # Drop the one that appears later in the column list
                drop = partner
            to_drop.add(drop)
            logger.info(
                "Correlation filter: dropped '%s' (r=%.3f with '%s')",
                drop, upper.at[partner, col] if drop == partner else upper.at[col, partner],
                col if drop == partner else partner,
            )

    kept = [c for c in X.columns if c not in to_drop]
    logger.info(
        "Correlation filter: %d -> %d features (dropped %d)",
        len(X.columns), len(kept), len(to_drop),
    )
    return kept


def permutation_importance_filter(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    threshold_pct: float = 0.1,
    n_repeats: int = 5,
    n_splits: int = 3,
) -> tuple[list[str], pd.Series]:
    """Drop features whose permutation importance is below threshold.

    Uses sklearn.inspection.permutation_importance on the last
    TimeSeriesSplit fold's validation set (out-of-sample).

    Parameters
    ----------
    model : Already-fitted model.
    X : Feature matrix (filtered to feature_cols).
    y : Target series, aligned with X.
    feature_cols : Feature names currently in use.
    threshold_pct : Minimum importance as percentage of total.
        Features below this are dropped. Default 0.1%.
    n_repeats : Number of permutation repeats (default 5).
    n_splits : TimeSeriesSplit folds for validation set.

    Returns
    -------
    (kept_features, importance_series)
    """
    # Use last fold's validation set for out-of-sample importance
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_mask = np.zeros(len(X), dtype=bool)
    for _, val_idx in tscv.split(X):
        val_mask[:] = False
        val_mask[val_idx] = True
    # val_mask now has the last fold's validation indices

    X_val = X.loc[val_mask, feature_cols]
    y_val = y.loc[val_mask]

    # Drop rows where target is NaN
    valid = y_val.notna()
    X_val = X_val.loc[valid]
    y_val = y_val.loc[valid]

    result = permutation_importance(
        model, X_val.values, y_val.values,
        n_repeats=n_repeats,
        scoring="neg_mean_absolute_error",
        random_state=42,
    )

    importances = pd.Series(
        result.importances_mean, index=feature_cols, name="importance"
    )
    # Clamp negative importances to zero before normalizing
    importances = importances.clip(lower=0)
    total = importances.sum()
    if total > 0:
        norm_importances = importances / total * 100  # as percentage
    else:
        norm_importances = importances

    # Also compute std for statistical significance check
    importance_std = pd.Series(
        result.importances_std, index=feature_cols, name="std"
    )

    threshold = threshold_pct
    to_drop = []
    for feat in feature_cols:
        imp = norm_importances[feat]
        std = importance_std[feat]
        raw_mean = result.importances_mean[feature_cols.index(feat)]
        # Drop if below threshold OR not statistically different from zero
        if imp < threshold or (raw_mean - 2 * std < 0 and imp < 1.0):
            to_drop.append(feat)
            logger.info(
                "Permutation filter: dropped '%s' (importance=%.3f%%, raw=%.4f±%.4f)",
                feat, imp, raw_mean, std,
            )

    kept = [c for c in feature_cols if c not in to_drop]
    logger.info(
        "Permutation filter: %d -> %d features (dropped %d)",
        len(feature_cols), len(kept), len(to_drop),
    )
    return kept, norm_importances


class FeatureSelector:
    """Two-stage feature selection pipeline.

    Applied per horizon group during training. Results are stored
    for logging and artifact serialization.
    """

    def __init__(
        self,
        corr_threshold: float = 0.95,
        perm_threshold_pct: float = 0.1,
        perm_n_repeats: int = 5,
    ):
        self.corr_threshold = corr_threshold
        self.perm_threshold_pct = perm_threshold_pct
        self.perm_n_repeats = perm_n_repeats

        # Results storage (per group)
        self.selection_reports: dict[str, dict] = {}

    def select_features(
        self,
        group_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        model_factory=None,
    ) -> list[str]:
        """Run the full pipeline for one horizon group.

        Parameters
        ----------
        group_name : e.g. "DA1", "S3". Used as key in selection_reports.
        X : Feature matrix.
        y : Target series.
        feature_cols : Initial feature list (all ~100).
        model_factory : Callable returning an unfitted model instance.

        Returns
        -------
        List of selected feature names.
        """
        initial_count = len(feature_cols)
        report = {"initial_count": initial_count}

        # Stage 1: Correlation filter
        # Quick-fit a model for importance ranking (tiebreaker)
        importance_ranking = None
        if model_factory is not None:
            try:
                quick_model = model_factory()
                valid = y.notna()
                quick_model.fit(X.loc[valid, feature_cols].values, y.loc[valid].values)
                if hasattr(quick_model, "feature_importances_"):
                    importance_ranking = pd.Series(
                        quick_model.feature_importances_,
                        index=feature_cols,
                    )
            except Exception as e:
                logger.warning("Quick-fit for correlation tiebreaker failed: %s", e)

        current_features = correlation_filter(
            X[feature_cols], threshold=self.corr_threshold,
            importance_ranking=importance_ranking,
        )
        report["after_corr"] = len(current_features)
        report["corr_dropped"] = [
            f for f in feature_cols if f not in current_features
        ]

        # Stage 2: Permutation importance
        # Fit model on correlation-filtered features
        if model_factory is not None:
            model = model_factory()
            valid = y.notna()
            model.fit(
                X.loc[valid, current_features].values,
                y.loc[valid].values,
            )

            current_features, importances = permutation_importance_filter(
                model, X, y, current_features,
                threshold_pct=self.perm_threshold_pct,
                n_repeats=self.perm_n_repeats,
            )
            report["importances"] = importances.to_dict()
        else:
            report["importances"] = {}

        report["after_perm"] = len(current_features)
        report["final_count"] = len(current_features)
        report["kept_features"] = current_features
        report["perm_dropped"] = [
            f for f in report.get("kept_features", feature_cols)
            if f not in current_features
        ]

        self.selection_reports[group_name] = report

        logger.info(
            "%s: Feature selection %d -> %d -> %d features "
            "(corr dropped %d, perm dropped %d)",
            group_name, initial_count, report["after_corr"],
            report["final_count"],
            initial_count - report["after_corr"],
            report["after_corr"] - report["final_count"],
        )

        return current_features

    def to_dict(self) -> dict:
        """Serialize selection metadata for model artifact."""
        return {
            "corr_threshold": self.corr_threshold,
            "perm_threshold_pct": self.perm_threshold_pct,
            "perm_n_repeats": self.perm_n_repeats,
            "selection_reports": self.selection_reports,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSelector":
        """Deserialize from model artifact."""
        selector = cls(
            corr_threshold=data.get("corr_threshold", 0.95),
            perm_threshold_pct=data.get("perm_threshold_pct", 0.1),
            perm_n_repeats=data.get("perm_n_repeats", 5),
        )
        selector.selection_reports = data.get("selection_reports", {})
        return selector
