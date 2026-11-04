"""Model ladder factories; all learned transformations live in pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from gcc_strength.audit import TARGETS
from gcc_strength.features import OutlierFlags, QualityFeatures, validate_inputs


class TypeMeanRegressor(RegressorMixin, BaseEstimator):
    """L1 nominal type means; unseen types fall back to the training global mean."""

    def fit(self, X, y):
        validate_inputs(X)
        values = np.asarray(y, dtype=float)
        self.global_mean_ = float(values.mean())
        self.type_means_ = pd.Series(values).groupby(X.ProductType.to_numpy()).mean().to_dict()
        return self

    def predict(self, X):
        check_is_fitted(self, "type_means_")
        validate_inputs(X)
        return X.ProductType.map(self.type_means_).fillna(self.global_mean_).to_numpy(float)


class PhysicalMetricARegressor(RegressorMixin, BaseEstimator):
    """Fit an observed-A expert plus a no-A fallback on the same training fold."""

    def __init__(self, fallback, expert, min_expert_samples=20):
        self.fallback = fallback
        self.expert = expert
        self.min_expert_samples = min_expert_samples

    def fit(self, X, y):
        validate_inputs(X)
        observed = X.Physical_Metric_A.notna().to_numpy()
        self.expert_train_size_ = int(observed.sum())
        self.fallback_ = clone(self.fallback).fit(X, y)
        self.expert_ = None
        if self.expert_train_size_ >= self.min_expert_samples:
            self.expert_ = clone(self.expert).fit(X.loc[observed], np.asarray(y)[observed])
        return self

    def predict(self, X):
        check_is_fitted(self, "fallback_")
        validate_inputs(X)
        predictions = self.fallback_.predict(X)
        observed = X.Physical_Metric_A.notna().to_numpy()
        if self.expert_ is not None and observed.any():
            predictions[observed] = self.expert_.predict(X.loc[observed])
        return predictions


class NamedLGBMRegressor(RegressorMixin, BaseEstimator):
    """Resolve the sole permitted constraint by the fold's transformed column names."""

    def __init__(self, num_leaves=7, n_estimators=100, learning_rate=0.05, monotone=False):
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.monotone = monotone

    def fit(self, X, y):
        from lightgbm import LGBMRegressor

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.constraints_ = [int(self.monotone and c == "Physical_Metric_B") for c in X]
        if self.monotone and sum(self.constraints_) != 1:
            raise ValueError("The Blaine constraint requires exactly one named column")
        self.model_ = LGBMRegressor(
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_child_samples=15,
            reg_lambda=1.0,
            monotone_constraints=self.constraints_,
            random_state=42,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


class ARDGaussianRegressor(RegressorMixin, BaseEstimator):
    """Exact GP with bounded ARD optimization inside each training fold.

    The axis vector is built after fold-specific encoding. Quality axes have the
    candidate initial length scale, nominal dummy axes start at unit scale.
    Each axis then varies independently under a fixed optimization budget.
    """

    def __init__(self, length_scale=2.0, noise_level=0.1):
        self.length_scale = length_scale
        self.noise_level = noise_level

    def fit(self, X, y):
        scales = np.array([1.0 if c.startswith("ProductType_") else self.length_scale for c in X])
        self.model_ = GaussianProcessRegressor(
            kernel=Matern(scales, length_scale_bounds=(0.1, 100.0), nu=1.5)
            + WhiteKernel(self.noise_level, noise_level_bounds=(0.01, 2.0)),
            optimizer=_bounded_gp_optimizer,
            normalize_y=True,
            random_state=42,
            alpha=1e-8,
        ).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)


def _bounded_gp_optimizer(objective, initial_theta, bounds):
    """Limit marginal-likelihood optimization without consulting validation data."""
    result = minimize(
        objective,
        initial_theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 8, "maxfun": 16, "ftol": 1e-6},
    )
    return result.x, result.fun


def design_size(fitted):
    """Count trained design columns, summing both experts for parsimonious ties."""
    if isinstance(fitted, PhysicalMetricARegressor):
        return design_size(fitted.fallback_) + (
            design_size(fitted.expert_) if fitted.expert_ is not None else 0
        )
    if "preprocess" in fitted.named_steps:
        return len(fitted.named_steps["preprocess"].get_feature_names_out())
    return int(isinstance(fitted.named_steps["model"], TypeMeanRegressor))


def build_model(
    level,
    *,
    physical_metric_a="exclude",
    interactions=False,
    ratio_flags=True,
    outlier_flags=False,
    ratios=None,
    min_expert_samples=20,
    representation=None,
    monotone=False,
    target=None,
):
    """Build L0-L5 or robust Huber; A variants share evaluation rows."""
    if level not in {"L0", "L1", "L2", "L3", "L4", "L5", "Huber"}:
        raise ValueError("Unknown model level")
    if monotone and (level != "L4" or target != TARGETS[0] or interactions):
        raise ValueError("Only additive Blaine -> Strength1 in L4 may be constrained")
    if physical_metric_a not in {"exclude", "indicator", "submodel"}:
        raise ValueError("Unknown Physical_Metric_A variant")
    if physical_metric_a == "submodel":
        if level in {"L0", "L1"}:
            raise ValueError("A submodel requires a quality estimator")
        options = dict(
            interactions=interactions,
            ratio_flags=ratio_flags,
            outlier_flags=outlier_flags,
            ratios=ratios,
            representation=representation,
            monotone=monotone,
            target=target,
        )
        return PhysicalMetricARegressor(
            build_model(level, physical_metric_a="exclude", **options),
            build_model(level, physical_metric_a="indicator", **options),
            min_expert_samples,
        )
    features = QualityFeatures(
        representation=representation or ("physical" if level == "L3" else "raw"),
        physical_metric_a=physical_metric_a,
        interactions=interactions,
        ratio_flags=ratio_flags,
        **({"ratios": tuple(ratios)} if ratios is not None else {}),
    )
    if level in {"L0", "L1"}:
        estimator = DummyRegressor(strategy="mean") if level == "L0" else TypeMeanRegressor()
        return Pipeline([("features", features), ("model", estimator)])
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                numeric,
                make_column_selector(
                    pattern=r"^(?!ProductType$)",
                    dtype_exclude=["object", "category", "string"],
                ),
            ),
            (
                "type",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                ["ProductType"],
            ),
        ],
        verbose_feature_names_out=False,
    )
    if level in {"L4", "L5"}:
        preprocess.set_output(transform="pandas")
    estimators = {
        "L2": ElasticNet(random_state=42, max_iter=20000, tol=1e-6),
        "L3": ElasticNet(random_state=42, max_iter=20000, tol=1e-6),
        "L4": NamedLGBMRegressor(monotone=monotone),
        "L5": ARDGaussianRegressor(),
        "Huber": TransformedTargetRegressor(
            regressor=HuberRegressor(max_iter=1000), transformer=StandardScaler()
        ),
    }
    return Pipeline(
        [
            ("features", features),
            ("outliers", OutlierFlags(enabled=outlier_flags)),
            ("preprocess", preprocess),
            ("model", estimators[level]),
        ]
    )
