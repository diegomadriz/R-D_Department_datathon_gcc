"""Stage 2 estimators, named monotonicity, reproducibility and ensemble selection."""

import copy

import numpy as np
import pandas as pd
import pytest
from _modeling import synthetic_train
from sklearn.base import clone

from gcc_strength.cv import cross_validate_model, load_model_config
from gcc_strength.features import INPUT_COLUMNS, QualityFeatures
from gcc_strength.ladder import (
    candidate_specs,
    ensemble_gate,
    fixed_ensemble,
    run_ladder,
    select_finalists,
)
from gcc_strength.models import build_model


@pytest.fixture
def train(tmp_path):
    return synthetic_train(tmp_path, n=90)


@pytest.fixture
def config():
    config = copy.deepcopy(load_model_config())
    config["cv"].update(n_splits=3, n_repeats=1, temporal_splits=2)
    config["hpo"].update(inner_splits=2, max_configs=1)
    for spec in config["models"].values():
        spec["max_configs"] = 1
    return config


@pytest.mark.parametrize("level", ["L4", "L5", "Huber"])
@pytest.mark.parametrize("target", ["Strength1", "Strength7", "Strength28"])
def test_stage2_all_targets_reproduce_exactly(train, config, level, target):
    first = cross_validate_model(train, target, level, config=config)
    second = cross_validate_model(train, target, level, config=config)
    pd.testing.assert_frame_equal(first.oof, second.oof, check_exact=True)
    pd.testing.assert_frame_equal(first.metrics, second.metrics, check_exact=True)
    pd.testing.assert_frame_equal(first.selections, second.selections, check_exact=True)
    assert first.selections.n_features.gt(0).all()


@pytest.mark.parametrize("a", ["exclude", "indicator", "submodel"])
@pytest.mark.parametrize("representation", ["raw", "physical"])
def test_constraint_uses_fold_column_names_and_monotone_predictions(train, a, representation):
    known = train.loc[train.Strength1.notna()]
    model = build_model(
        "L4",
        target="Strength1",
        monotone=True,
        physical_metric_a=a,
        representation=representation,
        min_expert_samples=3,
    )
    fitted = clone(model).fit(known[list(INPUT_COLUMNS)], known.Strength1)
    branches = [fitted] if a != "submodel" else [fitted.fallback_, fitted.expert_]
    for branch in branches:
        if branch is None:
            continue
        estimator = branch.named_steps["model"]
        assert sum(estimator.constraints_) == 1
        index = list(estimator.feature_names_in_).index("Physical_Metric_B")
        assert estimator.constraints_[index] == 1
    X = pd.concat([known[list(INPUT_COLUMNS)].iloc[[0]]] * 50, ignore_index=True)
    X.Physical_Metric_B = np.linspace(-5, 5, len(X))
    assert (np.diff(fitted.predict(X)) >= -1e-10).all()


@pytest.mark.parametrize("target", ["Strength7", "Strength28", None])
def test_monotonic_constraint_cannot_be_used_on_other_targets(target):
    with pytest.raises(ValueError, match="Only additive Blaine"):
        build_model("L4", target=target, monotone=True)
    with pytest.raises(ValueError, match="Only additive Blaine"):
        build_model("L4", target="Strength1", monotone=True, interactions=True)


def test_gp_kernel_has_scaled_ard_axes_and_white_noise(train):
    model = build_model("L5").fit(train[list(INPUT_COLUMNS)], train.Strength7)
    gp = model.named_steps["model"].model_
    assert gp.kernel_.k1.nu == 1.5
    assert (
        len(gp.kernel_.k1.length_scale)
        == model.named_steps["preprocess"]
        .transform(
            model.named_steps["outliers"].transform(
                model.named_steps["features"].transform(train[list(INPUT_COLUMNS)])
            )
        )
        .shape[1]
    )
    assert gp.kernel_.k2.noise_level > 0
    assert callable(gp.optimizer)
    assert np.ptp(gp.kernel_.k1.length_scale) > 0
    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert "scale" in numeric.named_steps


@pytest.mark.parametrize("level", ["L4", "L5", "Huber"])
def test_new_models_fit_features_only_inside_outer_training(train, config, monkeypatch, level):
    calls = []
    original = QualityFeatures.fit

    def record(self, X, y=None):
        calls.append(set(X.index))
        return original(self, X, y)

    monkeypatch.setattr(QualityFeatures, "fit", record)
    cross_validate_model(train, "Strength7", level, config=config)
    from gcc_strength.cv import repeated_group_splits

    folds = list(repeated_group_splits(train, n_splits=3, n_repeats=1))
    for number, (_, _, fit, valid) in enumerate(folds):
        # Even one nonempty parameter configuration is checked on all inner folds.
        batch = calls[number * 3 : (number + 1) * 3]
        assert len(batch) == 3
        assert batch[-1] == set(fit)
        assert all(received <= set(fit) and received.isdisjoint(valid) for received in batch)


def test_full_candidate_factorial_and_prior_ensemble_members():
    config = load_model_config()
    for target in ("Strength1", "Strength7", "Strength28"):
        specs = candidate_specs(config, target)
        names = [s["candidate"] for s in specs]
        assert len(names) == len(set(names)) == (32 if target == "Strength1" else 26)
        assert set(config["stage2"]["ensemble_members"]) <= set(names)
        assert sum(s["monotone"] for s in specs) == (6 if target == "Strength1" else 0)


def test_fixed_ensemble_gate_rejects_an_identical_member(train, config):
    result = cross_validate_model(train, "Strength7", "L1", config=config)
    results = {name: copy.deepcopy(result) for name in ("first", "second", "third")}
    for item in results.values():
        item.selections["n_features"] = 1
    ensemble = fixed_ensemble(results, list(results))
    np.testing.assert_allclose(
        ensemble.oof.predicted,
        result.oof.sort_values(["repeat", "_row_id"]).predicted,
    )
    accepted, comparison = ensemble_gate(ensemble, {"L1": result}, n_bootstrap=200)
    assert not accepted
    assert comparison["L1"]["ci_high"] == pytest.approx(0, abs=1e-12)
    ensemble.oof["predicted"] = ensemble.oof.observed
    accepted, comparison = ensemble_gate(ensemble, {"L1": result}, n_bootstrap=200)
    assert accepted and comparison["L1"]["ci_high"] < 0
    results["second"].oof.loc[0, "fold"] = 99
    with pytest.raises(AssertionError):
        fixed_ensemble(results, list(results))


def test_finalist_selection_uses_mean_mae_then_parsimony_and_ignores_temporal():
    config = load_model_config()
    rows = []
    for level, mae, size, a in [
        ("L2", 10, 14, "exclude"),
        ("L3", 10 + 5e-10, 12, "exclude"),
        ("Huber", 11, 8, "exclude"),
        ("L4", 9, 14, "exclude"),
        ("L5", 12, 12, "exclude"),
    ]:
        rows.append(
            dict(
                target="Strength1",
                scheme="repeated_group",
                level=level,
                candidate=level,
                mae_mean=mae,
                n_features_mean=size,
                physical_metric_a=a,
                monotone=False,
            )
        )
    rows.append({**rows[0], "scheme": "temporal", "mae_mean": 0})
    selected = select_finalists(pd.DataFrame(rows), config).set_index("family")
    assert selected.loc["linear", "candidate"] == "L3"
    assert set(selected.level) == {"L3", "L4", "L5"}


def test_ladder_persists_complete_artifacts_and_refuses_a_different_manifest(tmp_path, config):
    train = synthetic_train(tmp_path, n=40)
    config["cv"].update(n_splits=2, n_repeats=1, temporal_splits=1)
    config["stage2"]["physical_metric_a"] = ["exclude"]
    config["bootstrap"]["n_resamples"] = 50
    directory = tmp_path / "ladder"
    summary, finalists = run_ladder(train, output=directory, config=config, workers=1)
    assert len(finalists) == 9
    assert set(finalists.family) == {"linear", "lightgbm", "gaussian_process"}
    assert summary.n_folds.isin([1, 2]).all()
    assert summary.n_bootstrap.eq(50).all()
    for name in [
        "manifest.json",
        "ladder.csv",
        "ladder.md",
        "finalists.csv",
        "temporal_sensitivity.csv",
        "ensemble_gate.csv",
    ]:
        assert (directory / name).is_file()
    first = pd.read_csv(directory / "ladder.csv")
    run_ladder(train, output=directory, config=config, workers=1)
    pd.testing.assert_frame_equal(
        first, pd.read_csv(directory / "ladder.csv"), rtol=1e-12, atol=1e-12
    )
    config["bootstrap"]["n_resamples"] = 51
    with pytest.raises(ValueError, match="manifest differs"):
        run_ladder(train, output=directory, config=config, workers=1)
