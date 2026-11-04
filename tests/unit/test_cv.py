"""Group/temporal isolation, nested fit boundaries and deterministic baseline CV."""

import ast
import copy
import inspect

import numpy as np
import pandas as pd
import pytest
from _modeling import synthetic_train

from gcc_strength import cv, features, metrics, models
from gcc_strength.eda import PROVENANCE
from gcc_strength.features import QualityFeatures


@pytest.fixture
def train(tmp_path):
    return synthetic_train(tmp_path)


@pytest.fixture
def quick_config():
    config = copy.deepcopy(cv.load_model_config())
    config["cv"].update(n_splits=3, n_repeats=2, temporal_splits=2)
    config["hpo"].update(inner_splits=2, max_configs=2)
    return config


def test_default_cv_has_25_folds_intact_groups_and_one_oof_per_repeat(train):
    folds = list(cv.repeated_group_splits(train))
    assert len(folds) == 25
    repeat_indices = {repeat: [] for repeat in range(5)}
    first_partitions = []
    for repeat, fold, fit, valid in folds:
        assert set(train.iloc[fit].strength_group_id).isdisjoint(
            train.iloc[valid].strength_group_id
        )
        assert set(fit).isdisjoint(valid)
        assert len(fit) + len(valid) == len(train)
        repeat_indices[repeat].extend(valid)
        if fold == 0:
            first_partitions.append(tuple(valid))
    for valid in repeat_indices.values():
        assert sorted(valid) == list(range(len(train)))
    assert len(set(first_partitions)) == 5
    for first, second in zip(folds, cv.repeated_group_splits(train), strict=True):
        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_array_equal(first[3], second[3])


def test_temporal_splits_preserve_order_groups_and_work_with_shuffled_input(train):
    shuffled = train.sample(frac=1, random_state=42)
    for frame in (train, shuffled):
        folds = list(cv.temporal_group_splits(frame))
        seen = set()
        for _, _, fit, valid in folds:
            fitting, validation = frame.iloc[fit], frame.iloc[valid]
            assert fitting._row_id.max() < validation._row_id.min()
            assert set(fitting.strength_group_id).isdisjoint(validation.strength_group_id)
            assert seen.isdisjoint(validation._row_id)
            seen.update(validation._row_id)
        assert len(seen) < len(frame)  # warm-up has no OOF predictions
    train.loc[10, "strength_group_id"] = train.loc[0, "strength_group_id"]
    with pytest.raises(ValueError, match="contiguous"):
        list(cv.temporal_group_splits(train))


@pytest.mark.parametrize(
    "function", [cv.repeated_group_splits, cv.temporal_group_splits, cv.cross_validate_model]
)
def test_cv_entrypoints_require_train_provenance(train, function):
    bad = train.copy()
    bad.attrs.pop(PROVENANCE)
    with pytest.raises(ValueError, match="Use load_train"):
        if function is cv.cross_validate_model:
            function(bad, "Strength1")
        else:
            list(function(bad))


def test_modeling_modules_have_no_ingestion_or_lockbox_executor():
    for module in (cv, features, metrics, models):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"read_parquet", "read_csv", "read_table", "run_sql"}
            if isinstance(node, ast.FunctionDef):
                assert "lockbox" not in node.name


@pytest.mark.parametrize("target", ["Strength1", "Strength7", "Strength28"])
@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_all_baselines_are_reproducible_on_all_targets(train, quick_config, target, level):
    first = cv.cross_validate_model(train, target, level, config=quick_config)
    second = cv.cross_validate_model(train, target, level, config=quick_config)
    pd.testing.assert_frame_equal(first.oof, second.oof, check_exact=True)
    pd.testing.assert_frame_equal(first.metrics, second.metrics, check_exact=True)
    pd.testing.assert_frame_equal(first.selections, second.selections, check_exact=True)
    assert len(first.metrics) == 6
    assert len(first.oof) == train[target].notna().sum() * 2
    assert first.oof.observed.notna().all()
    assert first.oof.groupby(["repeat", "_row_id"]).size().eq(1).all()
    assert np.isfinite(first.oof.predicted).all()
    assert set(first.oof._row_id) == set(train.loc[train[target].notna(), "_row_id"])


def test_nested_fits_never_receive_outer_validation(train, quick_config, monkeypatch):
    fits = []
    original = QualityFeatures.fit

    def record(self, X, y=None):
        fits.append(set(X.index))
        return original(self, X, y)

    monkeypatch.setattr(QualityFeatures, "fit", record)
    result = cv.cross_validate_model(train, "Strength7", "L2", config=quick_config)
    fits_per_outer = 2 * 2 + 1  # candidates × inner folds + outer refit
    assert len(fits) == len(result.metrics) * fits_per_outer
    folds = list(cv.repeated_group_splits(train, n_splits=3, n_repeats=2))
    for number, (_, _, fit, valid) in enumerate(folds):
        calls = fits[number * fits_per_outer : (number + 1) * fits_per_outer]
        assert calls[-1] == set(fit)
        for received in calls:
            assert received <= set(fit)
            assert received.isdisjoint(valid)
        assert all(len(received) < len(fit) for received in calls[:-1])


def test_a_variants_evaluate_identical_rows_and_report_expert_coverage(train, quick_config):
    results = [
        cv.cross_validate_model(
            train, "Strength7", "L2", config=quick_config, physical_metric_a=variant
        )
        for variant in ("exclude", "indicator", "submodel")
    ]
    keys = ["repeat", "fold", "_row_id", "observed"]
    for result in results[1:]:
        pd.testing.assert_frame_equal(results[0].oof[keys], result.oof[keys])
    expert = results[-1].selections
    assert expert.expert_train_size.gt(0).all()
    assert expert.expert_prediction_coverage.between(0, 1).all()
    assert all(len(params) == 4 for params in expert.params)


def test_temporal_hpo_runs_and_initial_missing_a_uses_fallback(train, quick_config):
    result = cv.cross_validate_model(
        train, "Strength7", "L3", config=quick_config, temporal=True, physical_metric_a="submodel"
    )
    assert len(result.metrics) == 2
    assert not result.oof._row_id.isin(train.iloc[:40]._row_id).any()
    assert not result.selections.expert_fitted.any()
    assert result.selections.expert_prediction_coverage.eq(0).all()
