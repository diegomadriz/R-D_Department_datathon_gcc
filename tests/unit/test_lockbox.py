"""Only synthetic, temporary Gold can be authorized in these tests."""

import copy
import hashlib
import json
import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

from gcc_strength import lockbox
from gcc_strength.audit import FEATURES, TARGETS
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.split import GROUP_COL, lockbox_digest


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(42)
    n = 100
    frame = pd.DataFrame({col: rng.uniform(0.1, 1, n) for col in FEATURES})
    frame[GROUP_COL] = np.arange(n) // 2
    frame["_row_id"] = np.arange(n)
    frame["ProductType"] = [f"Type {1 + g % 3}" for g in frame[GROUP_COL]]
    frame["split"] = np.where(frame[GROUP_COL] < 40, "train", "test")
    for j, target in enumerate(TARGETS):
        frame[target] = (j + 1) * 100 + 15 * frame.Physical_Metric_B + rng.normal(0, 2, n)
    frame.loc[::13, "Physical_Metric_A"] = np.nan
    frame.loc[::17, "Strength1"] = np.nan
    gold = tmp_path / "synthetic_gold.parquet"
    frame.to_parquet(gold)
    spec = dict(
        level="L2",
        physical_metric_a="indicator",
        representation="raw",
        monotone=False,
        params={"model__alpha": 0.1, "model__l1_ratio": 0.1},
    )
    config = dict(
        random_state=42,
        cv=dict(n_splits=2),
        stage3=dict(conformal_splits=2),
        final=dict(
            architecture="a",
            families={
                family: {t: copy.deepcopy(spec) for t in TARGETS}
                for family in ("linear", "lightgbm", "gaussian_process")
            },
        ),
    )
    models = tmp_path / "models.yaml"
    models.write_text(yaml.safe_dump(config))
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text(
        yaml.safe_dump(
            dict(
                lockbox=dict(
                    adversarial_auc_max=0.6,
                    near_duplicate_distance=0.1,
                    bootstrap_resamples=30,
                    coverage_target=0.9,
                )
            )
        )
    )
    digest = tmp_path / "lockbox.sha256"
    digest.write_text(lockbox_digest(frame.loc[frame.split.eq("test"), "_row_id"]))
    cv = tmp_path / "cv.json"
    cv.write_text(
        json.dumps(
            dict(
                protocol=dict(quick=False),
                cv=[
                    dict(family=f, target=t, **{f"cv_{m}_mean": 1.0 for m in lockbox.METRICS})
                    for f in config["final"]["families"]
                    for t in TARGETS
                ],
            )
        )
    )
    return dict(
        gold=gold,
        output=tmp_path / "result",
        models_path=models,
        thresholds_path=thresholds,
        lockbox_file=digest,
        cv_metrics_path=cv,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        git_check=lambda: "synthetic-commit",
    )


def test_no_authorization_has_no_io(synthetic, monkeypatch):
    monkeypatch.delenv("GCC_LOCKBOX_AUTHORIZED", raising=False)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Data opened"))
    with pytest.raises(PermissionError):
        lockbox.evaluate_lockbox(**synthetic)
    assert not synthetic["output"].exists()
    assert not (synthetic["gold"].parent / "mlruns").exists()


def test_complete_and_second_call_refused(synthetic, monkeypatch):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")
    result = lockbox.evaluate_lockbox(**synthetic)
    output = synthetic["output"]
    assert result["status"] == "ok"
    assert set(result) == {
        "status",
        "metrics",
        "models",
        "coverage",
        "monotonicity",
        "adversarial",
        "run_id",
        "coverage_target",
        "architecture",
    }
    assert len(result["metrics"]) == 9
    for row in result["metrics"]:
        assert set(row["ci95"]) == set(lockbox.METRICS)
        assert set(row["test_minus_cv"]) == set(lockbox.METRICS)
        assert set(row["by_type"]) == {"Type 1", "Type 2", "Type 3"}
        for bounds in row["ci95"].values():
            assert bounds["lower"] <= bounds["upper"]
        delta = row["delta_mae_vs_l1"]
        assert delta["ci_low"] <= delta["ci_high"]
    assert all(r["after_pct"] == 100 for r in result["monotonicity"])
    assert {r["scope"] for r in result["coverage"]} == {"Todos", "Type 1", "Type 2", "Type 3"}
    assert {r["family"] for r in result["coverage"]} == {"linear", "lightgbm", "gaussian_process"}
    saved = (output / "result.json").read_bytes()
    assert json.loads(saved)["status"] == "ok"
    assert len(pd.read_csv(output / "test_3x3.csv")) == 9
    assert (output / "test_3x3.png").stat().st_size > 1000
    assert "_row_id" not in json.dumps(json.loads(saved))
    with pytest.raises(FileExistsError):
        lockbox.evaluate_lockbox(**synthetic)
    assert (output / "result.json").read_bytes() == saved
    client = MlflowClient(tracking_uri=synthetic["tracking_uri"])
    run = client.get_run(result["run_id"])
    assert run.data.tags["mlflow.runName"] == "lockbox_evaluation"
    assert run.info.status == "FINISHED"
    assert set(run.data.params) >= {
        "git_commit",
        "config_hash",
        "lockbox_sha256",
        "test_size",
        "train_size",
    }
    assert run.data.params["train_size"] == "80" and run.data.params["test_size"] == "20"
    assert (
        run.data.params["config_hash"]
        == hashlib.sha256(synthetic["models_path"].read_bytes()).hexdigest()
    )
    assert any(k.startswith("test_metrics_linear_Strength1_ci95_mae") for k in run.data.metrics)
    assert {a.path for a in client.list_artifacts(run.info.run_id)} >= {
        "result.json",
        "test_3x3.csv",
        "test_3x3.png",
        "attempt.json",
    }
    assert (
        len(
            mlflow.search_logged_models(
                experiment_ids=[run.info.experiment_id], output_format="list"
            )
        )
        == 3
    )


def test_previous_attempt_is_not_retried(synthetic, monkeypatch):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")
    synthetic["output"].mkdir()
    attempt = synthetic["output"] / "attempt.json"
    attempt.write_text("reserved")
    with pytest.raises(FileExistsError):
        lockbox.evaluate_lockbox(**synthetic)
    assert attempt.read_text() == "reserved"
    assert not (synthetic["output"] / "result.json").exists()


def test_previous_mlflow_run_prevents_attempt(synthetic, monkeypatch):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")
    client = MlflowClient(tracking_uri=synthetic["tracking_uri"])
    experiment = client.create_experiment("other-experiment")
    client.create_run(experiment, tags={"mlflow.runName": "lockbox_evaluation"})
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Data opened"))
    with pytest.raises(FileExistsError, match="lockbox evaluation"):
        lockbox.evaluate_lockbox(**synthetic)
    assert not (synthetic["output"] / "attempt.json").exists()


def test_cli_requires_parent_without_evaluation(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lockbox"])
    monkeypatch.setattr(lockbox, "evaluate_lockbox", lambda **kwargs: pytest.fail("Evaluated"))
    with pytest.raises(SystemExit) as error:
        lockbox.main()
    assert error.value.code == 2
    assert "--parent-run-id" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["shift", "duplicate"])
def test_adversarial_abort_never_loads_test_targets(synthetic, monkeypatch, failure):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")
    frame = pd.read_parquet(synthetic["gold"])
    if failure == "shift":
        frame.loc[frame.split.eq("test"), "Physical_Metric_B"] += 100
    else:
        frame.loc[80, list(INPUT_COLUMNS)] = frame.loc[1, list(INPUT_COLUMNS)].to_numpy()
    frame.to_parquet(synthetic["gold"])
    reader = pd.read_parquet

    def read(*args, **kwargs):
        if kwargs.get("filters") == [("split", "==", "test")]:
            assert not set(kwargs["columns"]) & set(TARGETS)
        return reader(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(lockbox, "_fit", lambda *a, **k: pytest.fail("Model fitted after abort"))
    result = lockbox.evaluate_lockbox(**synthetic)
    assert set(result) == {"status", "adversarial"}
    assert result["status"] == "aborted"
    if failure == "shift":
        assert result["adversarial"]["auc"] > 0.6
    else:
        assert result["adversarial"]["exact_duplicate_pairs"] == 1
    assert not (synthetic["output"] / "test_3x3.csv").exists()


def test_dirty_git_prevents_any_data_access(synthetic, monkeypatch):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")

    def dirty():
        raise RuntimeError("dirty tree")

    synthetic["git_check"] = dirty
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Data opened"))
    with pytest.raises(RuntimeError, match="dirty"):
        lockbox.evaluate_lockbox(**synthetic)
    assert not synthetic["output"].exists()


def test_bad_split_consumes_attempt_before_any_feature_read(synthetic, monkeypatch):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")
    synthetic["lockbox_file"].write_text("0" * 64)
    reader = pd.read_parquet

    def read(*args, **kwargs):
        assert kwargs == {"columns": ["_row_id", "split", GROUP_COL]}
        assert (synthetic["output"] / "attempt.json").exists()
        return reader(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    with pytest.raises(RuntimeError, match="differs"):
        lockbox.evaluate_lockbox(**synthetic)
    assert not (synthetic["output"] / "result.json").exists()
    with pytest.raises(FileExistsError):
        lockbox.evaluate_lockbox(**synthetic)


@pytest.mark.parametrize("bad_param", ["config_hash", "lockbox_sha256", "quick"])
def test_parent_mismatch_prevents_data_access(synthetic, monkeypatch, bad_param):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")
    mlflow.set_tracking_uri(synthetic["tracking_uri"])
    mlflow.set_experiment("synthetic-parent")
    params = dict(
        config_hash=hashlib.sha256(synthetic["models_path"].read_bytes()).hexdigest(),
        lockbox_sha256=synthetic["lockbox_file"].read_text(),
        quick=False,
    )
    params[bad_param] = "mismatch"
    with mlflow.start_run() as parent:
        mlflow.log_params(params)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Data opened"))
    with pytest.raises(ValueError, match="Parent"):
        lockbox.evaluate_lockbox(**synthetic, parent_run_id=parent.info.run_id)
    assert not synthetic["output"].exists()


def test_matching_parent_tag_and_experiment(synthetic, monkeypatch):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")
    mlflow.set_tracking_uri(synthetic["tracking_uri"])
    mlflow.set_experiment("synthetic-parent")
    with mlflow.start_run() as parent:
        mlflow.log_params(
            dict(
                config_hash=hashlib.sha256(synthetic["models_path"].read_bytes()).hexdigest(),
                lockbox_sha256=synthetic["lockbox_file"].read_text(),
                quick=False,
            )
        )
    # Exercise parent linkage through an adversarial abort without a second full fit.
    frame = pd.read_parquet(synthetic["gold"])
    frame.loc[frame.split.eq("test"), "Physical_Metric_B"] += 100
    frame.to_parquet(synthetic["gold"])
    lockbox.evaluate_lockbox(**synthetic, parent_run_id=parent.info.run_id)
    client = MlflowClient(tracking_uri=synthetic["tracking_uri"])
    runs = client.search_runs(
        [parent.info.experiment_id], "tags.mlflow.runName = 'lockbox_evaluation'"
    )
    assert len(runs) == 1
    assert runs[0].data.tags["parent_run_id"] == parent.info.run_id


def test_bootstrap_resamples_complete_unequal_groups():
    result = lockbox.bootstrap_metrics(
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 5.0],
        ["Type 1"] * 4,
        [0, 0, 0, 1],
        n_resamples=1000,
    )
    assert result["mae"] == 1
    assert result["ci95"]["mae"] == {"lower": 0, "upper": 4}
    assert np.isnan(result["r2"])


def test_real_git_checker_rejects_dirty(monkeypatch):
    monkeypatch.setattr(lockbox.subprocess, "check_output", lambda *a, **k: " M synthetic.py\n")
    with pytest.raises(RuntimeError, match="clean git"):
        lockbox.check_git_clean()


def test_frozen_real_families_fit_synthetic_training(synthetic):
    from gcc_strength.config import ROOT, load_yaml
    from gcc_strength.eda import train_only
    from gcc_strength.multioutput import project_strengths

    frame = pd.read_parquet(synthetic["gold"])
    training = train_only(frame.loc[frame.split.eq("train")])
    X = training[list(INPUT_COLUMNS)].head(4)
    final = load_yaml(ROOT / "conf/models.yaml")["final"]
    for settings in final["families"].values():
        specs = [settings[t] for t in TARGETS]
        fitted = lockbox._fit(training, specs)
        assert fitted.profile == tuple(s["params"] for s in specs)
        raw = np.column_stack([m.predict(X) for m in fitted.models_])
        np.testing.assert_array_equal(fitted.predict(X), project_strengths(raw))


def test_bootstrap_matches_explicit_whole_group_draws():
    y = np.array([2.0, 3.0, 6.0, 8.0, 13.0, 15.0])
    pred = y + [1.0, -1.0, 2.0, 3.0, -2.0, 1.0]
    groups = np.array([0, 0, 0, 1, 2, 2])
    types = np.array(["Type 1"] * len(y))
    result = lockbox.bootstrap_metrics(y, pred, types, groups, n_resamples=30)
    rng = np.random.default_rng(42)
    draws = {m: [] for m in lockbox.METRICS}
    for _ in range(30):
        indices = np.concatenate([np.flatnonzero(groups == g) for g in rng.integers(0, 3, 3)])
        scores = lockbox.regression_metrics(y[indices], pred[indices], types[indices])
        for metric in draws:
            draws[metric].append(scores[metric])
    for metric, values in draws.items():
        np.testing.assert_allclose(
            list(result["ci95"][metric].values()), np.quantile(values, [0.025, 0.975])
        )


def test_post_reservation_failure_is_not_retried(synthetic, monkeypatch):
    monkeypatch.setenv("GCC_LOCKBOX_AUTHORIZED", "1")

    def unavailable(*args, **kwargs):
        assert (synthetic["output"] / "attempt.json").exists()
        raise OSError("synthetic read failure")

    monkeypatch.setattr(pd, "read_parquet", unavailable)
    with pytest.raises(OSError, match="synthetic"):
        lockbox.evaluate_lockbox(**synthetic)
    with pytest.raises(FileExistsError):
        lockbox.evaluate_lockbox(**synthetic)
