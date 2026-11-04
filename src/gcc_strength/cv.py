"""Train-only repeated stratified group CV, nested HPO and temporal sensitivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold

from gcc_strength.audit import TARGETS
from gcc_strength.config import CONF_DIR, load_medallion_conf, load_yaml
from gcc_strength.eda import assert_train
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.metrics import regression_metrics
from gcc_strength.models import PhysicalMetricARegressor, build_model, design_size
from gcc_strength.split import GROUP_COL, assign_strata


@dataclass
class CVResult:
    """Long-form OOF predictions, per-fold scores and per-fold selected parameters."""

    oof: pd.DataFrame
    metrics: pd.DataFrame
    selections: pd.DataFrame


def load_model_config():
    """Load the declared search budget, never infer it from validation results."""
    return load_yaml(CONF_DIR / "models.yaml")


def repeated_group_splits(train, *, n_splits=5, n_repeats=5):
    """Yield repeat, fold and positional indices using deterministic seeds 42..46.

    Strata are recomputed on the received training population via split.assign_strata.
    Rare strata may be too small in inner CV: group isolation still holds and sklearn
    emits its standard warning. No resampling or target-guided regrouping is performed.
    """
    assert_train(train)
    if n_splits < 2 or n_repeats < 1 or train[GROUP_COL].nunique() < n_splits:
        raise ValueError("Not enough groups or invalid CV dimensions")
    if train[GROUP_COL].isna().any():
        raise ValueError("Groups must be observed")
    split_conf = load_medallion_conf()["split"]
    strata = assign_strata(
        train, n_bins=split_conf["n_bins"], rare_types=split_conf["rare_types"], min_groups=n_splits
    )
    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        for fold, (fit, valid) in enumerate(
            cv.split(np.zeros(len(train)), strata, train[GROUP_COL])
        ):
            yield repeat, fold, fit, valid


def temporal_group_splits(train, *, n_splits=5):
    """Forward-chain contiguous group blocks in row order, with no group crossing a cut.

    Split the ordered groups into n_splits+1 blocks; the initial block is warm-up.
    Validation rows are returned once; the warm-up rows have no temporal OOF prediction.
    """
    assert_train(train)
    if n_splits < 1 or train[GROUP_COL].nunique() < n_splits + 1:
        raise ValueError("Not enough groups for temporal CV")
    if train[[GROUP_COL, "_row_id"]].isna().any().any() or not train._row_id.is_unique:
        raise ValueError("Observed groups and unique row ids are required")
    ordered = train.sort_values("_row_id", kind="stable")
    sequence = ordered[GROUP_COL].to_numpy()
    runs = sequence[np.r_[True, sequence[1:] != sequence[:-1]]]
    if len(runs) != len(pd.unique(runs)):
        raise ValueError("Temporal CV requires contiguous groups in row order")
    blocks = np.array_split(runs, n_splits + 1)
    for fold in range(n_splits):
        previous = np.concatenate(blocks[: fold + 1])
        fit = np.flatnonzero(train[GROUP_COL].isin(previous))
        valid = np.flatnonzero(train[GROUP_COL].isin(blocks[fold + 1]))
        yield 0, fold, fit, valid


def _candidates(space, budget):
    if not space:
        return [{}]
    if not 1 <= budget <= 20:
        raise ValueError("HPO budget must be between 1 and 20 configurations")
    size = int(np.prod([len(values) for values in space.values()]))
    return list(ParameterSampler(space, n_iter=min(budget, size), random_state=42))


def _tune(estimator, train, target, candidates, inner_splits):
    """Choose by mean inner-fold MAE; outer validation is never received here."""
    if candidates == [{}]:
        return {}, np.nan
    folds = list(repeated_group_splits(train, n_splits=inner_splits, n_repeats=1))
    X, y = train[list(INPUT_COLUMNS)], train[target].to_numpy(float)
    scores = []
    for params in candidates:
        losses = []
        for _, _, fit, valid in folds:
            fitted = clone(estimator).set_params(**params).fit(X.iloc[fit], y[fit])
            predicted = fitted.predict(X.iloc[valid])
            losses.append(float(np.abs(y[valid] - predicted).mean()))
        scores.append(float(np.mean(losses)))
    best = int(np.argmin(scores))
    return candidates[best], scores[best]


def cross_validate_model(
    train,
    target,
    model="L2",
    *,
    config=None,
    temporal=False,
    physical_metric_a="exclude",
    model_options=None,
) -> CVResult:
    """Evaluate one baseline on observed train labels, returning every OOF prediction.

    The caller loads real data exclusively with eda.load_train. Missing labels are
    excluded per target before partitioning. All A variants use the same target rows
    and folds. HPO is repeated inside every outer training fold, including temporal CV;
    its random inner folds contain only the past of the temporal outer validation.
    """
    assert_train(train)
    if target not in TARGETS:
        raise ValueError("Unknown target")
    config = load_model_config() if config is None else config
    if config["random_state"] != 42:
        raise ValueError("The base random state must be 42")
    if not train._row_id.is_unique:
        raise ValueError("Row ids must be unique")
    known = (
        train.loc[train[target].notna()]
        .sort_values("_row_id", kind="stable")
        .reset_index(drop=True)
    )
    if known.empty or not np.isfinite(known[target]).all():
        raise ValueError("A nonempty finite observed target population is required")
    options = {} if model_options is None else dict(model_options)
    if model == "L4":
        options["target"] = target
    estimator = build_model(model, physical_metric_a=physical_metric_a, **options)
    spec = config["models"][model]
    space = spec.get("search_space", {})
    if physical_metric_a == "submodel":
        # Couple the expert/fallback hyperparameters to keep the fixed budget unchanged.
        base = _candidates(space, spec.get("max_configs", config["hpo"]["max_configs"]))
        candidates = [
            {
                f"{branch}__{key}": value
                for branch in ("fallback", "expert")
                for key, value in params.items()
            }
            for params in base
        ]
    else:
        candidates = _candidates(space, spec.get("max_configs", config["hpo"]["max_configs"]))
    cv_conf = config["cv"]
    folds = (
        temporal_group_splits(known, n_splits=cv_conf["temporal_splits"])
        if temporal
        else (
            repeated_group_splits(
                known, n_splits=cv_conf["n_splits"], n_repeats=cv_conf["n_repeats"]
            )
        )
    )
    X, y = known[list(INPUT_COLUMNS)], known[target].to_numpy(float)
    predictions, scores, selections = [], [], []
    for repeat, fold, fit, valid in folds:
        training = known.iloc[fit].copy()
        params, inner_mae = _tune(
            estimator, training, target, candidates, config["hpo"]["inner_splits"]
        )
        fitted = clone(estimator).set_params(**params).fit(X.iloc[fit], y[fit])
        pred = fitted.predict(X.iloc[valid])
        identifiers = {
            "model": model,
            "target": target,
            "repeat": repeat,
            "fold": fold,
            "scheme": "temporal" if temporal else "repeated_group",
            "physical_metric_a": physical_metric_a,
        }
        part = known.iloc[valid][["_row_id", GROUP_COL, "ProductType"]].copy()
        for key, value in identifiers.items():
            part[key] = value
        part["observed"], part["predicted"] = y[valid], pred
        predictions.append(part)
        metric = regression_metrics(y[valid], pred, known.iloc[valid].ProductType)
        training_metrics = regression_metrics(
            y[fit], fitted.predict(X.iloc[fit]), training.ProductType
        )
        scores.append(
            {
                **identifiers,
                **metric,
                "train_n": len(fit),
                "train_mae": training_metrics["mae"],
                "train_r2": training_metrics["r2"],
                "overfit_gap": training_metrics["r2"] - metric["r2"],
            }
        )
        selection = {
            **identifiers,
            "params": params,
            "inner_mae": inner_mae,
            "n_candidates": len(candidates),
            "n_features": design_size(fitted),
        }
        if isinstance(fitted, PhysicalMetricARegressor):
            selection.update(
                expert_train_size=fitted.expert_train_size_,
                expert_fitted=fitted.expert_ is not None,
                observed_a_coverage=float(X.iloc[valid].Physical_Metric_A.notna().mean()),
                expert_prediction_coverage=float(X.iloc[valid].Physical_Metric_A.notna().mean())
                if fitted.expert_ is not None
                else 0.0,
            )
        selections.append(selection)
    return CVResult(
        pd.concat(predictions, ignore_index=True), pd.DataFrame(scores), pd.DataFrame(selections)
    )
