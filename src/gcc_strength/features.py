"""Fold-fitted quality representations from ADR 0005."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from gcc_strength.audit import CATEGORICAL, FEATURES

INPUT_COLUMNS = (CATEGORICAL, *FEATURES)
RAW_COLUMNS = tuple(c for c in FEATURES if c not in {"Physical_Metric_A", "Mineral_Phase_A"})
PHYSICAL_COLUMNS = (
    "Physical_Metric_B",
    "Mineral_Phase_E",
    "Mineral_Phase_G",
    "Mineral_Phase_H",
    "Chemical_Composition_A",
    "Chemical_Composition_B",
)
RATIO_OPERANDS = {
    "log_C_over_B": ("Mineral_Phase_C", "Mineral_Phase_B"),
    "C_over_A": ("Mineral_Phase_C", "Mineral_Phase_A"),
    "log_D_over_A": ("Mineral_Phase_D", "Mineral_Phase_A"),
    "log_F_over_E": ("Mineral_Phase_F", "Mineral_Phase_E"),
    "log_A": ("Mineral_Phase_A",),
    "log_H_over_F": ("Mineral_Phase_H", "Mineral_Phase_F"),
    "F_over_E_plus_F": ("Mineral_Phase_F", "Mineral_Phase_E"),
    "log_H_over_A": ("Mineral_Phase_H", "Mineral_Phase_A"),
}
DEFAULT_RATIOS = ("log_C_over_B", "log_D_over_A", "log_F_over_E")
INTERACTIONS = {
    "blaine_x_c": ("Physical_Metric_B", "Mineral_Phase_C"),
    "f_x_chemical_a": ("Mineral_Phase_F", "Chemical_Composition_A"),
    "blaine_x_d": ("Physical_Metric_B", "Mineral_Phase_D"),
    "chemical_a_squared": ("Chemical_Composition_A", "Chemical_Composition_A"),
}


def validate_inputs(X: pd.DataFrame) -> None:
    """Allow only observed quality columns; reject targets and all metadata."""
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError("Quality inputs must be a DataFrame with unique columns")
    if set(X.columns) - set(INPUT_COLUMNS):
        raise ValueError("Only quality inputs are allowed")


def ratio_features(X: pd.DataFrame, names=tuple(RATIO_OPERANDS)) -> pd.DataFrame:
    """Compute observed positive ratios without epsilon, always returning invalid flags."""
    validate_inputs(X)
    result = pd.DataFrame(index=X.index)
    for name in names:
        operands = X[list(RATIO_OPERANDS[name])].astype(float)
        finite = np.isfinite(operands).all(axis=1)
        if name == "F_over_E_plus_F":
            valid = finite & operands.ge(0).all(axis=1) & operands.sum(axis=1).gt(0)
            value = operands.iloc[:, 0] / operands.sum(axis=1)
        else:
            valid = finite & operands.gt(0).all(axis=1)
            safe = operands.where(valid, np.nan)
            value = safe.iloc[:, 0]
            if safe.shape[1] == 2:
                value = value / safe.iloc[:, 1]
            if name.startswith("log_"):
                value = np.log(value)
        result[name] = value.where(valid, np.nan)
        result[f"flag_invalid_{name}"] = ~valid
    return result


class QualityFeatures(TransformerMixin, BaseEstimator):
    """Select a representation and learn optional interaction centers inside each fold.

    ``physical_metric_a='submodel'`` is routed by PhysicalMetricARegressor, never by
    this transformer. The expert uses ``indicator`` and the fallback uses ``exclude``.
    """

    def __init__(
        self,
        representation="raw",
        physical_metric_a="exclude",
        ratios=DEFAULT_RATIOS,
        ratio_flags=True,
        interactions=False,
    ):
        self.representation = representation
        self.physical_metric_a = physical_metric_a
        self.ratios = ratios
        self.ratio_flags = ratio_flags
        self.interactions = interactions

    def fit(self, X, y=None):
        validate_inputs(X)
        if self.representation not in {"raw", "physical"}:
            raise ValueError("Unknown representation")
        if self.physical_metric_a not in {"exclude", "indicator"}:
            raise ValueError("Use exclude or indicator; submodel requires the routed estimator")
        if set(self.ratios) - set(RATIO_OPERANDS):
            raise ValueError("Unknown ratio")
        if {"log_F_over_E", "F_over_E_plus_F"} <= set(self.ratios):
            raise ValueError("F/(E+F) must replace log(F/E)")
        columns = (
            {c for pair in INTERACTIONS.values() for c in pair} if self.interactions else set()
        )
        self.centers_ = {c: float(X[c].mean()) if X[c].notna().any() else 0.0 for c in columns}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(self._transform(X).columns, dtype=object)
        return self

    def _transform(self, X):
        columns = RAW_COLUMNS if self.representation == "raw" else PHYSICAL_COLUMNS
        result = X[[CATEGORICAL, *columns]].copy()
        if self.representation == "physical":
            ratios = ratio_features(X, self.ratios)
            if not self.ratio_flags:
                ratios = ratios[list(self.ratios)]
            result = pd.concat([result, ratios], axis=1)
        if self.physical_metric_a == "indicator":
            result["Physical_Metric_A"] = X["Physical_Metric_A"]
            result["physical_metric_a_missing"] = X["Physical_Metric_A"].isna()
        if self.interactions:
            for name, (left, right) in INTERACTIONS.items():
                result[name] = (X[left] - self.centers_[left]) * (X[right] - self.centers_[right])
        return result

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        validate_inputs(X)
        return self._transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class OutlierFlags(TransformerMixin, BaseEstimator):
    """Append IQR flags with thresholds fitted exclusively on the training fold."""

    def __init__(self, enabled=False):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.columns_ = (
            [c for c in X.select_dtypes(include="number").columns if c != CATEGORICAL]
            if self.enabled
            else []
        )
        quantiles = X[self.columns_].quantile([0.25, 0.75])
        span = quantiles.loc[0.75] - quantiles.loc[0.25]
        self.low_ = quantiles.loc[0.25] - 1.5 * span
        self.high_ = quantiles.loc[0.75] + 1.5 * span
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        result = X.copy()
        for col in self.columns_:
            result[f"flag_outlier_{col}"] = X[col].lt(self.low_[col]) | X[col].gt(self.high_[col])
        return result
