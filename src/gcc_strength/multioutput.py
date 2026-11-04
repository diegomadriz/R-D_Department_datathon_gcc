"""Train-only multi-output estimators with fold-local quality pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.isotonic import isotonic_regression
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_is_fitted

from gcc_strength.audit import TARGETS
from gcc_strength.cv import _candidates, repeated_group_splits
from gcc_strength.eda import assert_train
from gcc_strength.features import INPUT_COLUMNS, validate_inputs
from gcc_strength.models import PhysicalMetricARegressor, build_model

PREDICTION_COLUMNS = ("predicted_day1", "predicted_day7")


def project_strengths(values):
    """Euclidean isotonic projection along ages, with a nonnegative lower bound."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Finite n x 3 predictions are required")
    return np.maximum(np.array([isotonic_regression(row) for row in values]), 0.0)


class PredictionFeatures(TransformerMixin, BaseEstimator):
    """Append only generated chain predictions after the raw quality transformer."""

    def __init__(self, quality, depth):
        self.quality = quality
        self.depth = depth

    def _validate(self, X):
        expected = {*INPUT_COLUMNS, *PREDICTION_COLUMNS[: self.depth]}
        if set(X.columns) != expected:
            raise ValueError("Chain inputs must contain quality and generated predictions only")
        validate_inputs(X[list(INPUT_COLUMNS)])

    def fit(self, X, y=None):
        self._validate(X)
        self.quality_ = clone(self.quality).fit(X[list(INPUT_COLUMNS)], y)
        return self

    def transform(self, X):
        check_is_fitted(self, "quality_")
        self._validate(X)
        return pd.concat(
            [
                self.quality_.transform(X[list(INPUT_COLUMNS)]),
                X[list(PREDICTION_COLUMNS[: self.depth])],
            ],
            axis=1,
        )

    def get_feature_names_out(self, input_features=None):
        return np.r_[self.quality_.get_feature_names_out(), PREDICTION_COLUMNS[: self.depth]]


class ChainExperts(PhysicalMetricARegressor):
    """Route generated chain inputs without exposing measured targets to either expert."""

    def fit(self, X, y):
        observed = X.Physical_Metric_A.notna().to_numpy()
        self.expert_train_size_ = int(observed.sum())
        self.fallback_ = clone(self.fallback).fit(X, y)
        self.expert_ = None
        if self.expert_train_size_ >= self.min_expert_samples:
            self.expert_ = clone(self.expert).fit(X.loc[observed], np.asarray(y)[observed])
        return self

    def predict(self, X):
        check_is_fitted(self, "fallback_")
        result = self.fallback_.predict(X)
        observed = X.Physical_Metric_A.notna().to_numpy()
        if self.expert_ is not None and observed.any():
            result[observed] = self.expert_.predict(X.loc[observed])
        return result


def _add_predictions(estimator, depth):
    if not depth:
        return estimator
    if isinstance(estimator, PhysicalMetricARegressor):
        return ChainExperts(
            _add_predictions(estimator.fallback, depth),
            _add_predictions(estimator.expert, depth),
            estimator.min_expert_samples,
        )
    estimator.set_params(features=PredictionFeatures(estimator.named_steps["features"], depth))
    return estimator


def parameter_profiles(specs, config):
    """Couple per-output candidate indices; never form an unbounded Cartesian product."""
    pools = []
    for spec in specs:
        setting = config["models"][spec["level"]]
        pools.append(
            _candidates(
                setting.get("search_space", {}),
                setting.get("max_configs", config["hpo"]["max_configs"]),
            )
        )
    budget = min(config["stage3"]["max_configs"], max(map(len, pools)))
    return [tuple(pool[i % len(pool)] for pool in pools) for i in range(budget)]


class FixedArchitecture(RegressorMixin, BaseEstimator):
    """Fit a fixed parameter profile, including recursive group OOF chain inputs.

    ``y`` carries training labels and partition metadata only. Public ``X`` always
    contains quality alone. Grid selection wraps this complete estimator in inner CV;
    no selected parameters from an outer validation fold are reused in its fits.
    """

    def __init__(self, specs, architecture="a", profile=None, chain_splits=3, depth=3):
        self.specs = specs
        self.architecture = architecture
        self.profile = profile
        self.chain_splits = chain_splits
        self.depth = depth

    def fit(self, X, y):
        validate_inputs(X)
        assert_train(y)
        if self.architecture not in {"a", "b", "c"}:
            raise ValueError("Unknown architecture")
        training = y.copy()
        # In cross_val_predict, sklearn slices X and y together, preserving row indices.
        pd.testing.assert_frame_equal(X, training[list(INPUT_COLUMNS)])
        raw_labels = training[TARGETS].to_numpy(float)
        labels = raw_labels.copy()
        if self.architecture == "c":
            with np.errstate(invalid="ignore", divide="ignore"):
                labels[:, :2] = np.log(raw_labels[:, :2] / raw_labels[:, 2:])
            labels[~np.isfinite(labels)] = np.nan
        self.models_ = []
        for j in range(self.depth):
            features = X.copy()
            if self.architecture == "b" and j:
                folds = [
                    (fit, valid)
                    for _, _, fit, valid in repeated_group_splits(
                        training, n_splits=self.chain_splits, n_repeats=1
                    )
                ]
                prefix = FixedArchitecture(self.specs, "b", self.profile, self.chain_splits, j)
                # Recursive prefixes regenerate R7 training inputs within their own folds.
                generated = cross_val_predict(prefix, X, training, cv=folds, n_jobs=1)
                for k in range(j):
                    features[PREDICTION_COLUMNS[k]] = generated[:, k]
            spec = self.specs[j]
            estimator = build_model(
                spec["level"],
                physical_metric_a=spec["physical_metric_a"],
                representation=spec["representation"],
                # A positive R1 effect does not imply a sign for log(R1/R28).
                monotone=bool(spec["monotone"]) and self.architecture != "c",
                target=TARGETS[j],
            )
            estimator = _add_predictions(estimator, j if self.architecture == "b" else 0)
            params = {} if self.profile is None else self.profile[j]
            if spec["physical_metric_a"] == "submodel":
                params = {
                    f"{branch}__{key}": value
                    for branch in ("fallback", "expert")
                    for key, value in params.items()
                }
            observed = np.isfinite(labels[:, j])
            if not observed.any():
                raise ValueError("No observed training labels for an output")
            self.models_.append(
                estimator.set_params(**params).fit(features.loc[observed], labels[observed, j])
            )
        return self

    def predict(self, X):
        check_is_fitted(self, "models_")
        validate_inputs(X)
        predicted = []
        for j, estimator in enumerate(self.models_):
            features = X.copy()
            if self.architecture == "b":
                for k in range(j):
                    features[PREDICTION_COLUMNS[k]] = predicted[k]
            predicted.append(estimator.predict(features))
        result = np.column_stack(predicted)
        if self.depth < 3:
            return result
        if self.architecture == "c":
            # Negative log ratios, ordered in age; no measured R28 is used at inference.
            ratios = np.minimum(result[:, :2], 0)
            ratios = np.array([isotonic_regression(row) for row in ratios])
            anchor = np.maximum(result[:, 2:], np.finfo(float).eps)
            result = np.c_[anchor * np.exp(ratios), anchor]
        return project_strengths(result)


def fit_architecture(train, specs, architecture, config):
    """Select a joint profile exclusively inside training, then refit the architecture."""
    assert_train(train)
    if config["random_state"] != 42:
        raise ValueError("The base random state must be 42")
    profiles = parameter_profiles(specs, config)
    X = train[list(INPUT_COLUMNS)]
    folds = list(repeated_group_splits(train, n_splits=config["hpo"]["inner_splits"], n_repeats=1))
    losses = []
    for profile in profiles:
        scores = []
        for _, _, fit, valid in folds:
            model = FixedArchitecture(
                specs, architecture, profile, config["stage3"]["chain_splits"]
            ).fit(X.iloc[fit], train.iloc[fit])
            prediction = model.predict(X.iloc[valid])
            observed = train.iloc[valid][TARGETS].to_numpy(float)
            # Same target units: equal weight to the three target MAEs.
            scores.append(float(np.nanmean(np.nanmean(abs(observed - prediction), axis=0))))
        losses.append(float(np.mean(scores)))
    best = int(np.argmin(losses))
    fitted = FixedArchitecture(
        specs, architecture, profiles[best], config["stage3"]["chain_splits"]
    ).fit(X, train)
    return fitted, {"profile_index": best, "inner_mae": losses[best], "profile": profiles[best]}
