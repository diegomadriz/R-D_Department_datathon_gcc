"""Quality representation boundaries, invalid ratios and fold-fitted transformations."""

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from _modeling import synthetic_train
from sklearn.base import clone

from gcc_strength import features, models
from gcc_strength.features import INPUT_COLUMNS, OutlierFlags, QualityFeatures, ratio_features
from gcc_strength.gold import NON_FEATURE_COLUMNS
from gcc_strength.models import build_model


@pytest.fixture
def X(tmp_path):
    return synthetic_train(tmp_path)[list(INPUT_COLUMNS)]


def test_feature_contract_is_static_quality_only():
    forbidden = set(NON_FEATURE_COLUMNS) | {"block"}
    for columns in (
        INPUT_COLUMNS,
        features.RAW_COLUMNS,
        features.PHYSICAL_COLUMNS,
        *features.RATIO_OPERANDS.values(),
        *features.INTERACTIONS.values(),
    ):
        assert forbidden.isdisjoint(columns)
        assert set(columns) <= set(INPUT_COLUMNS)
    assert "Mineral_Phase_A" not in features.RAW_COLUMNS
    for module in (features, models):
        tree = ast.parse(inspect.getsource(module))
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert forbidden.isdisjoint(strings)


@pytest.mark.parametrize("column", NON_FEATURE_COLUMNS)
def test_transformer_rejects_targets_and_metadata(X, column):
    X[column] = 1
    with pytest.raises(ValueError, match="quality inputs"):
        QualityFeatures().fit(X)


def test_all_ratio_formulas_and_invalid_operands(X):
    X.loc[
        0,
        [
            "Mineral_Phase_A",
            "Mineral_Phase_B",
            "Mineral_Phase_C",
            "Mineral_Phase_D",
            "Mineral_Phase_E",
            "Mineral_Phase_F",
            "Mineral_Phase_H",
        ],
    ] = [8, 2, 6, 4, 3, 1, 2]
    result = ratio_features(X)
    expected = [
        np.log(3),
        0.75,
        np.log(0.5),
        np.log(1 / 3),
        np.log(8),
        np.log(2),
        0.25,
        np.log(0.25),
    ]
    np.testing.assert_allclose(result.loc[0, list(features.RATIO_OPERANDS)].astype(float), expected)
    X.loc[1:3, "Mineral_Phase_B"] = [0, -1, np.nan]
    X.loc[4, "Mineral_Phase_A"] = 0
    X.loc[5, "Mineral_Phase_H"] = np.inf
    invalid = ratio_features(X)
    assert invalid.loc[1:3, "log_C_over_B"].isna().all()
    assert invalid.loc[1:3, "flag_invalid_log_C_over_B"].all()
    assert np.isnan(invalid.loc[4, "log_A"])
    assert np.isnan(invalid.loc[5, "log_H_over_F"])


@pytest.mark.parametrize(
    "e,f,expected", [(0, 1, 1), (1, 0, 0), (0, 0, None), (-1, 2, None), (1, np.nan, None)]
)
def test_fraction_zero_exception(X, e, f, expected):
    X.loc[0, ["Mineral_Phase_E", "Mineral_Phase_F"]] = [e, f]
    row = ratio_features(X).iloc[0]
    if expected is None:
        assert np.isnan(row.F_over_E_plus_F)
        assert row.flag_invalid_F_over_E_plus_F
    else:
        assert row.F_over_E_plus_F == expected
        assert not row.flag_invalid_F_over_E_plus_F


def test_physical_substitution_and_redundant_ratio_guard(X):
    result = QualityFeatures(representation="physical").fit_transform(X)
    assert "Mineral_Phase_C" not in result
    assert "Mineral_Phase_A" not in result
    assert set(features.DEFAULT_RATIOS) <= set(result)
    with pytest.raises(ValueError, match="replace"):
        QualityFeatures(ratios=("log_F_over_E", "F_over_E_plus_F")).fit(X)
    fraction = QualityFeatures(representation="physical", ratios=("F_over_E_plus_F",))
    assert "log_F_over_E" not in fraction.fit_transform(X)


def test_interactions_center_only_on_fit_and_keep_inputs_unchanged(X):
    before = X.copy(deep=True)
    transform = QualityFeatures(interactions=True).fit(X.iloc[:50])
    centers = dict(transform.centers_)
    heldout = X.iloc[50:].copy()
    heldout["Physical_Metric_B"] = 1000
    output = transform.transform(heldout)
    expected = (heldout.Physical_Metric_B - X.iloc[:50].Physical_Metric_B.mean()) * (
        heldout.Mineral_Phase_C - X.iloc[:50].Mineral_Phase_C.mean()
    )
    np.testing.assert_allclose(output.blaine_x_c, expected)
    assert transform.centers_ == centers
    pd.testing.assert_frame_equal(X, before)


def test_outlier_flags_do_not_learn_from_validation(X):
    transform = OutlierFlags(enabled=True).fit(X.iloc[:50])
    high = transform.high_.copy()
    future = X.iloc[50:].copy()
    future["Physical_Metric_B"] = 1000
    output = transform.transform(future)
    assert output.flag_outlier_Physical_Metric_B.all()
    pd.testing.assert_series_equal(high, transform.high_)


def test_imputation_and_scaling_are_fit_inside_pipeline_with_all_missing_a(X):
    model = build_model("L2", physical_metric_a="indicator")
    train = X.iloc[:50]
    model.fit(train, np.arange(len(train)))
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    names = model.named_steps["features"].get_feature_names_out()
    numeric = [name for name in names if name != "ProductType"]
    index = numeric.index("Physical_Metric_B")
    assert imputer.statistics_[index] == pytest.approx(train.Physical_Metric_B.median())
    assert imputer.statistics_[numeric.index("Physical_Metric_A")] == 0
    stats = imputer.statistics_.copy()
    assert np.isfinite(model.predict(X.iloc[80:])).all()
    np.testing.assert_array_equal(stats, imputer.statistics_)
    encoder = model.named_steps["preprocess"].named_transformers_["type"]
    assert list(encoder.categories_[0]) == ["Type 1", "Type 2"]


def test_expert_fallback_routing_and_minimum_size(X):
    y = np.arange(len(X), dtype=float)
    fallback = build_model("L2", physical_metric_a="submodel")
    fallback.fit(X.iloc[:80], y[:80])
    assert fallback.expert_ is None
    np.testing.assert_array_equal(fallback.predict(X), fallback.fallback_.predict(X))
    expert = clone(fallback).fit(X, y)
    assert expert.expert_train_size_ == 40
    observed = X.Physical_Metric_A.notna()
    predicted = expert.predict(X)
    np.testing.assert_array_equal(predicted[~observed], expert.fallback_.predict(X.loc[~observed]))
    np.testing.assert_array_equal(predicted[observed], expert.expert_.predict(X.loc[observed]))


def test_l0_l1_are_fold_means_with_unseen_nominal_type_fallback(X):
    fitting = X.iloc[:4].copy()
    fitting["ProductType"] = ["Type 1", "Type 1", "Type 2", "Type 2"]
    y = [1, 3, 7, 9]
    query = fitting.iloc[:3].copy()
    query["ProductType"] = ["Type 1", "Type 2", "Type 5"]
    np.testing.assert_allclose(build_model("L0").fit(fitting, y).predict(query), [5, 5, 5])
    np.testing.assert_allclose(build_model("L1").fit(fitting, y).predict(query), [2, 8, 5])


def test_numeric_category_labels_never_enter_scaler_or_outlier_flags(X):
    X["ProductType"] = X.ProductType.map({"Type 1": 1, "Type 2": 2})
    model = build_model("L2", outlier_flags=True).fit(X, np.arange(len(X)))
    assert "ProductType" not in model.named_steps["outliers"].columns_
    numeric_columns = model.named_steps["preprocess"].transformers_[0][2]
    assert "ProductType" not in numeric_columns
    assert all("ProductType" not in name for name in numeric_columns)
    assert list(model.named_steps["preprocess"].named_transformers_["type"].categories_[0]) == [
        1,
        2,
    ]
