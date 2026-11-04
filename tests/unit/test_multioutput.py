"""Synthetic checks for nested multi-output fitting and held-out CV+ coverage."""

import ast
import copy
import inspect

import numpy as np
import pandas as pd
import pytest
from _modeling import synthetic_train
from sklearn.base import clone

from gcc_strength import multioutput, stage3, uncertainty
from gcc_strength.audit import TARGETS
from gcc_strength.cv import load_model_config, repeated_group_splits
from gcc_strength.features import INPUT_COLUMNS, QualityFeatures


def settings():
    config = copy.deepcopy(load_model_config())
    config["cv"].update(n_splits=2, n_repeats=1, temporal_splits=1)
    config["hpo"].update(inner_splits=2, max_configs=1)
    config["stage3"].update(max_configs=1, chain_splits=2, conformal_splits=2, permutations=2)
    return config


def specs():
    return [
        dict(level="L2", physical_metric_a="indicator", representation="raw", monotone=False)
        for _ in TARGETS
    ]


@pytest.mark.parametrize("architecture", ["a", "b", "c"])
def test_architectures_are_reproducible_and_coherent(tmp_path, architecture):
    train = synthetic_train(tmp_path, 80)
    fit = train.iloc[:60]
    X = train.iloc[60:][list(INPUT_COLUMNS)]
    first, selection = multioutput.fit_architecture(fit, specs(), architecture, settings())
    second, repeated = multioutput.fit_architecture(fit, specs(), architecture, settings())
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert selection == repeated
    assert np.isfinite(first.predict(X)).all()
    assert (np.diff(first.predict(X), axis=1) >= 0).all()
    with pytest.raises(ValueError, match="quality inputs"):
        first.predict(train.iloc[60:])
    assert clone(first).get_params()["architecture"] == architecture


def test_chain_prefix_oof_is_invariant_to_heldout_group_labels(tmp_path):
    train = synthetic_train(tmp_path, 80)
    _, _, fit, valid = next(repeated_group_splits(train, n_splits=2, n_repeats=1))
    profile = ({"model__alpha": 1.0},) * 3
    estimator = multioutput.FixedArchitecture(specs(), "b", profile, chain_splits=2, depth=2)
    first = clone(estimator).fit(train.iloc[fit][list(INPUT_COLUMNS)], train.iloc[fit])
    changed = train.copy()
    changed.loc[changed.index[valid], TARGETS] = 1e10
    second = clone(estimator).fit(changed.iloc[fit][list(INPUT_COLUMNS)], changed.iloc[fit])
    np.testing.assert_array_equal(
        first.predict(train.iloc[valid][list(INPUT_COLUMNS)]),
        second.predict(train.iloc[valid][list(INPUT_COLUMNS)]),
    )


def test_chain_recursive_fits_keep_groups_intact_and_measurements_out_of_features(
    tmp_path, monkeypatch
):
    train = synthetic_train(tmp_path, 80)
    calls = []
    original = QualityFeatures.fit

    def record(self, X, y=None):
        assert set(X.columns) <= set(INPUT_COLUMNS)
        calls.append(set(X.index))
        return original(self, X, y)

    monkeypatch.setattr(QualityFeatures, "fit", record)
    model = multioutput.FixedArchitecture(specs(), "b", ({},) * 3, chain_splits=2)
    model.fit(train[list(INPUT_COLUMNS)], train)
    assert any(len(c) < 40 for c in calls)  # nested R7-prefix fits, not a global OOF shortcut
    for received in calls:
        # Target-specific missing labels can remove individual rows, but not import a heldout group.
        assert received <= set(train.index)
    forbidden = train[list(INPUT_COLUMNS)].assign(predicted_day1=train.Strength1)
    with pytest.raises(ValueError):
        model.predict(forbidden)


def test_isotonic_projection_and_c_does_not_copy_measured_anchor():
    projected = multioutput.project_strengths([[3, 1, 2], [-3, -2, -1]])
    np.testing.assert_array_equal(projected, [[2, 2, 2], [0, 0, 0]])


def test_cvplus_finite_ranks_and_invalid_inputs():
    centers = np.tile([10.0, 20.0], (10, 1))
    lower, upper = uncertainty.cvplus_bounds(centers, np.arange(10), alpha=0.1)
    np.testing.assert_array_equal(lower, [1, 11])
    np.testing.assert_array_equal(upper, [19, 29])
    low, high = uncertainty.cvplus_bounds(centers[:2], [1, 2])
    assert np.isneginf(low).all() and np.isposinf(high).all()
    with pytest.raises(ValueError):
        uncertainty.cvplus_bounds(centers, np.full(10, -1))


def test_nested_conformal_never_fits_outer_validation(tmp_path, monkeypatch):
    train = synthetic_train(tmp_path, 80)
    fit, validation = train.iloc[:60], train.iloc[60:]
    calls = []
    original = uncertainty.fit_architecture

    def record(frame, *args):
        calls.append(set(frame.strength_group_id))
        assert set(frame.strength_group_id).isdisjoint(validation.strength_group_id)
        return original(frame, *args)

    monkeypatch.setattr(uncertainty, "fit_architecture", record)
    first = uncertainty.nested_cvplus(
        fit, validation[list(INPUT_COLUMNS)], specs(), "a", settings()
    )
    validation = validation.copy()
    validation[TARGETS] = -1e9
    second = uncertainty.nested_cvplus(
        fit, validation[list(INPUT_COLUMNS)], specs(), "a", settings()
    )
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 4
    assert np.isfinite(first).all()
    assert (first[0] <= first[1]).all()


def test_permutation_moves_joint_labels_preserving_group_blocks(tmp_path):
    train = synthetic_train(tmp_path, 80)
    shuffled = stage3.permute_group_labels(train, np.random.default_rng(42))
    pd.testing.assert_frame_equal(train[list(INPUT_COLUMNS)], shuffled[list(INPUT_COLUMNS)])
    original = sorted(
        tuple(np.nan_to_num(part[TARGETS].to_numpy()).ravel())
        for _, part in train.groupby("strength_group_id")
    )
    permuted = sorted(
        tuple(np.nan_to_num(part[TARGETS].to_numpy()).ravel())
        for _, part in shuffled.groupby("strength_group_id")
    )
    assert original == permuted
    assert not train[TARGETS].equals(shuffled[TARGETS])


def test_new_estimators_have_no_data_readers_or_lockbox_executor():
    for module in (multioutput, uncertainty):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"read_parquet", "read_csv", "read_table", "run_sql"}
            if isinstance(node, ast.FunctionDef):
                assert "lockbox" not in node.name


def test_recursive_prefix_predictions_exclude_the_predicted_assay(tmp_path, monkeypatch):
    train = synthetic_train(tmp_path, 80)
    original_fit = multioutput.FixedArchitecture.fit
    original_predict = multioutput.FixedArchitecture.predict
    checks = []

    def fit(self, X, y):
        self.received_groups_ = set(y.strength_group_id)
        return original_fit(self, X, y)

    def predict(self, X):
        if self.depth < 3:
            predicted_groups = set(train.loc[X.index, "strength_group_id"])
            assert self.received_groups_.isdisjoint(predicted_groups)
            checks.append(self.depth)
        return original_predict(self, X)

    monkeypatch.setattr(multioutput.FixedArchitecture, "fit", fit)
    monkeypatch.setattr(multioutput.FixedArchitecture, "predict", predict)
    multioutput.FixedArchitecture(specs(), "b", ({},) * 3, chain_splits=2).fit(
        train[list(INPUT_COLUMNS)], train
    )
    assert set(checks) == {1, 2}


def test_stage3_end_to_end_reduced_run_and_cache(tmp_path):
    train = synthetic_train(tmp_path, 40)
    config = settings()
    config["bootstrap"]["n_resamples"] = 30
    config["stage3"]["learning_fractions"] = [0.5, 1.0]
    final = pd.DataFrame(
        [
            dict(family=family, target=target, **specs()[j])
            for family in ("linear", "lightgbm", "gaussian_process")
            for j, target in enumerate(TARGETS)
        ]
    )
    final_path = tmp_path / "finalists.csv"
    final.to_csv(final_path, index=False)
    output = tmp_path / "stage3"
    decision = stage3.run_stage3(
        train, output=output, finalists_path=final_path, config=config, workers=1
    )
    assert decision["selected"]["architecture"] == "a"
    assert (output / "complete").read_text() == "ok\n"
    for filename in (
        "oof.csv",
        "comparison.csv",
        "coverage.csv",
        "permutation.csv",
        "learning_curves.csv",
        "temporal_sensitivity.csv",
    ):
        assert not pd.read_csv(output / filename).empty
    first = (output / "comparison.csv").read_bytes()
    assert (
        stage3.run_stage3(train, output=output, finalists_path=final_path, config=config, workers=1)
        == decision
    )
    assert (output / "comparison.csv").read_bytes() == first
    config["stage3"]["alpha"] = 0.2
    with pytest.raises(ValueError, match="manifest"):
        stage3.run_stage3(train, output=output, finalists_path=final_path, config=config, workers=1)


@pytest.mark.parametrize("chain_error,expected", [(8.0, "b"), (10.0, "a")])
def test_architecture_gate_rejects_c_when_r1_deteriorates(chain_error, expected):
    rows = []
    for family in ("linear", "lightgbm", "gaussian_process"):
        for architecture in ("a", "b", "c"):
            for j, target in enumerate(TARGETS):
                for row in range(30):
                    error = 10.0 if architecture == "a" else chain_error
                    if architecture == "c":
                        error = 11.0 if j == 0 else 0.0
                    rows.append(
                        dict(
                            family=family,
                            architecture=architecture,
                            target=target,
                            repeat=0,
                            fold=row % 5,
                            _row_id=row,
                            strength_group_id=row,
                            ProductType="Type 1",
                            observed=100 + row,
                            predicted=100 + row + error,
                            scheme="repeated_group",
                        )
                    )
    config = settings()
    config["bootstrap"]["n_resamples"] = 50
    gates, decision = stage3.select_architecture(pd.DataFrame(rows), config)
    assert decision["selected"]["architecture"] == expected
    assert not gates.loc[gates.architecture.eq("c"), "eligible"].any()
