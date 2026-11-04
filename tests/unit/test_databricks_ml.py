"""Local contracts for ML orchestration, replay comparison and train-only publication."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from gcc_strength import catalog, lockbox
from gcc_strength.config import ROOT
from gcc_strength.databricks_ml import build_consumption_tables, verified_replay
from gcc_strength.eda import load_train, train_only
from gcc_strength.job_parameters import ML_TASKS, parse_notebook_parameters
from gcc_strength.provenance import train_fingerprint


def parameters():
    return dict(
        experiment_path="/Workspace/Shared/gcc-test",
        train_run_id="a1",
        evaluate_run_id="b2",
        register_run_id="c3",
    )


@pytest.mark.parametrize("task", ML_TASKS)
def test_notebook_parameters(task):
    parsed = parse_notebook_parameters(task, parameters())
    assert parsed.train_mode == "full" and parsed.lockbox_mode == "replay"
    assert parsed.catalog == "datathon" and parsed.schema == "gcc"
    assert (
        parse_notebook_parameters(task, {**parameters(), "train_mode": "quick"}).train_mode
        == "quick"
    )


@pytest.mark.parametrize("task", ML_TASKS)
@pytest.mark.parametrize(
    "override",
    [
        {"lockbox_mode": "first"},
        {"lockbox_mode": ""},
        {"train_mode": "fast"},
        {"catalog": "x;drop"},
        {"schema": "x.y"},
        {"experiment_path": "relative"},
        {"experiment_path": "/Workspace/${unresolved}"},
    ],
)
def test_invalid_notebook_parameters_fail(task, override):
    with pytest.raises(ValueError):
        parse_notebook_parameters(task, {**parameters(), **override})


@pytest.mark.parametrize("task", ["evaluate", "register", "publish"])
def test_upstream_ids_required(task):
    with pytest.raises(ValueError, match="upstream"):
        parse_notebook_parameters(task, {"experiment_path": "/Workspace/Shared/gcc"})


def synthetic_aggregates():
    return dict(
        metrics=[
            dict(
                family="linear",
                target="Strength1",
                mae=100.0,
                ci95=dict(mae=dict(lower=90.0, upper=110.0)),
                other=None,
            )
        ],
        coverage=[dict(coverage=0.9, n_rows=20, reliable=True)],
        monotonicity=[dict(after_pct=100.0)],
        adversarial=dict(auc=0.5, exact=0),
    )


def test_synthetic_result_replay_equal(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(synthetic_aggregates()))
    expected = json.loads(path.read_text())
    comparison = lockbox.compare_replay(copy.deepcopy(expected), expected, rel_tol=0.02)
    assert comparison["passed"]
    assert all(row.get("absolute_difference", 0) == 0 for row in comparison["differences"])


@pytest.mark.parametrize(
    "mutation", ["metric", "ci", "missing", "extra", "null", "zero", "nan", "flag"]
)
def test_replay_fails_closed(mutation):
    expected = synthetic_aggregates()
    actual = copy.deepcopy(expected)
    match mutation:
        case "metric":
            actual["metrics"][0]["mae"] = 103.0
        case "ci":
            actual["metrics"][0]["ci95"]["mae"]["upper"] = 115.0
        case "missing":
            actual.pop("coverage")
        case "extra":
            actual["metrics"].append({})
        case "null":
            actual["metrics"][0]["other"] = 0
        case "zero":
            actual["adversarial"]["exact"] = 1
        case "nan":
            actual["adversarial"]["auc"] = float("nan")
        case "flag":
            actual["coverage"][0]["reliable"] = False
    assert not lockbox.compare_replay(actual, expected, rel_tol=0.02)["passed"]


def test_tolerance_with_zero_reference():
    assert lockbox.compare_replay({"a": 101.0}, {"a": 100.0}, rel_tol=0.02)["passed"]
    assert not lockbox.compare_replay({"a": 0.001}, {"a": 0.0}, rel_tol=0.02)["passed"]


class FakeTable:
    def __init__(self, frame, calls, partition=None):
        self.frame, self.calls, self.partition = frame, calls, partition

    def where(self, predicate):
        self.calls.append(("where", predicate))
        return FakeTable(self.frame, self.calls, predicate)

    def groupBy(self, column):
        return SimpleNamespace(
            count=lambda: SimpleNamespace(
                collect=lambda: [
                    {"split": value, "count": count}
                    for value, count in self.frame[column].value_counts().items()
                ]
            )
        )

    def toPandas(self):
        assert self.partition is not None, "Unfiltered payload collected"
        self.calls.append(("toPandas", self.partition))
        label = self.partition.split("'")[1]
        return self.frame.loc[self.frame["split"].eq(label)].copy()

    def select(self, *columns):
        return FakeTable(self.frame[list(columns)], self.calls, self.partition)


def test_uc_filters_train_before_pandas():
    calls = []
    spark = SimpleNamespace(
        table=lambda name: FakeTable(
            pd.DataFrame({"split": ["train", "test"], "payload": [1, 999]}), calls
        )
    )
    train = load_train((spark, "datathon", "gcc"))
    assert train.payload.tolist() == [1] and train.attrs["lockbox_size"] == 1
    assert calls == [("where", "split = 'train'"), ("toPandas", "split = 'train'")]


def test_replay_invalid_mode_and_official_output_never_open_data(monkeypatch):
    monkeypatch.setattr(lockbox, "_read_replay_gold", lambda *a: pytest.fail("Opened Gold"))
    with pytest.raises(ValueError, match="explicit mode"):
        lockbox.replay_lockbox(mode="first", parent_run_id="a", output="unused")
    for path in (
        lockbox.DEFAULT_OUTPUT,
        lockbox.DEFAULT_OUTPUT / "replay",
        lockbox.DEFAULT_OUTPUT.parent,
    ):
        with pytest.raises(ValueError, match="separate"):
            lockbox.replay_lockbox(mode="replay", parent_run_id="a", output=path)


def test_frozen_result_is_aggregate_only_and_protocol_unchanged(medallion):
    frozen = json.loads((ROOT / "conf/lockbox_result_frozen.json").read_text())
    assert set(frozen) == set(lockbox.AGGREGATE_KEYS)
    assert len(frozen["metrics"]) == 9
    assert not {"run_id", "models", "_row_id", "predictions"} & set(frozen)
    protocol = json.loads((ROOT / "conf/lockbox_protocol_frozen.json").read_text())
    assert (
        train_fingerprint(train_only(medallion.gold.loc[medallion.gold["split"].eq("train")]))
        == protocol["train_sha256"]
    )


def test_bundle_replay_and_pinned_ml_environment():
    bundle = yaml.safe_load((ROOT / "databricks.yml").read_text())
    job = bundle["resources"]["jobs"]["gcc_end_to_end"]
    params = {p["name"]: p["default"] for p in job["parameters"]}
    assert params["lockbox_mode"] == "replay" and params["train_mode"] == "full"
    assert params["experiment_path"] == "${var.experiment_path}"
    assert job["max_concurrent_runs"] == 1
    ml = next(e["spec"] for e in job["environments"] if e["environment_key"] == "ml")
    assert {"shap==0.51.0", "lightgbm==4.6.0", "statsmodels==0.14.5"} <= set(ml["dependencies"])
    assert all("==" in d for d in ml["dependencies"])
    assert not any(k in bundle for k in ("host", "profile"))
    for task in job["tasks"]:
        for key, value in task["notebook_task"].get("base_parameters", {}).items():
            upstream = key.removesuffix("_run_id")
            assert value == "{{tasks." + upstream + ".values." + key + "}}"


def test_verified_replay_rejects_failed_or_unrelated_runs(monkeypatch, tmp_path):
    import gcc_strength.databricks_ml as module

    for status, tags in [
        ("FAILED", {}),
        ("FINISHED", {"replay_passed": "false"}),
        ("FINISHED", {"replay_passed": "true", "train_run_id": "other"}),
    ]:
        monkeypatch.setattr(
            module,
            "MlflowClient",
            lambda status=status, tags=tags: SimpleNamespace(
                get_run=lambda _: SimpleNamespace(
                    info=SimpleNamespace(status=status), data=SimpleNamespace(tags=tags)
                )
            ),
        )
        with pytest.raises(ValueError, match="verified replay"):
            verified_replay("a1", "b2", tmp_path)


def test_consumption_tables_synthetic_train_only(tmp_path):
    from gcc_strength.audit import FEATURES
    from gcc_strength.split import GROUP_COL

    n = 12
    frame = pd.DataFrame({col: np.linspace(0.1, 1.0, n) for col in FEATURES})
    frame["ProductType"], frame["split"] = "Type 1", "train"
    frame["_row_id"], frame[GROUP_COL] = np.arange(n), np.arange(n) // 2
    train = train_only(frame)
    rows, cv_rows, replay_rows = [], [], []
    for target in ("Strength1", "Strength7", "Strength28"):
        rows.extend(
            dict(
                family="gaussian_process",
                target=target,
                repeat=0,
                fold=i % 2,
                _row_id=i,
                strength_group_id=i // 2,
                ProductType="Type 1",
                observed=10.0 + i,
                predicted=11.0 + i,
            )
            for i in range(n)
        )
        cv_rows.append(
            dict(
                family="gaussian_process",
                target=target,
                **{f"cv_{m}_{s}": 1.0 for m in lockbox.METRICS for s in ("mean", "sd")},
            )
        )
        replay_rows.append(
            dict(
                family="gaussian_process",
                target=target,
                **{m: 1.0 for m in lockbox.METRICS},
                ci95={m: {"lower": 0.8, "upper": 1.2} for m in lockbox.METRICS},
            )
        )
        directory = tmp_path / "artifacts/gaussian_process" / target
        directory.mkdir(parents=True)
        pd.DataFrame(
            [dict(target=target, branch="main", feature=FEATURES[0], importance=1.0)]
        ).to_csv(directory / "importance.csv", index=False)
    oof = pd.DataFrame(rows)
    oof.to_csv(tmp_path / "oof.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(tmp_path / "cv_3x3.csv", index=False)
    oof.assign(lower=oof.predicted - 2.0, upper=oof.predicted + 2.0, covered=True).to_csv(
        tmp_path / "gaussian_process_intervals.csv", index=False
    )
    tables = build_consumption_tables(
        train, tmp_path, {"metrics": replay_rows}, train_mode="quick", n_resamples=20
    )
    assert len(tables["model_metrics"]) == 24
    assert set(tables["predictions"]._row_id) == set(train._row_id)
    assert set(tables["train_ranges"].feature) == set(FEATURES)
    for key, table in tables.items():
        assert table.train_mode.eq("quick").all()
        for column in table:
            assert catalog.column_comment(key, column)


def test_registration_keeps_train_model_and_links_both_runs(monkeypatch, tmp_path):
    import gcc_strength.databricks_ml as module

    calls = []
    client = SimpleNamespace(
        set_model_version_tag=lambda *args: calls.append(("tag", args)),
        set_registered_model_alias=lambda *args: calls.append(("alias", args)),
    )
    monkeypatch.setattr(module, "configure_mlflow", lambda path: None)
    monkeypatch.setattr(module, "verified_replay", lambda *args: {"status": "verified"})
    monkeypatch.setattr(module, "audit_run", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module,
        "download_json",
        lambda *args: {
            "champion": "models:/frozen-train-model",
            "families": {"gaussian_process": "models:/frozen-train-model"},
        },
    )
    monkeypatch.setattr(module, "MlflowClient", lambda: client)
    from contextlib import nullcontext

    monkeypatch.setattr(
        module.mlflow,
        "start_run",
        lambda **kwargs: nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="registration1"))),
    )
    monkeypatch.setattr(module.mlflow, "set_tags", lambda tags: calls.append(("run_tags", tags)))
    monkeypatch.setattr(module.mlflow, "log_dict", lambda *args: calls.append(("artifact", args)))

    def register(source, name, **kwargs):
        calls.append(("register", source, name))
        return SimpleNamespace(version="7")

    monkeypatch.setattr(module.mlflow, "register_model", register)
    result = module.register_champion(parse_notebook_parameters("register", parameters()), tmp_path)
    assert ("register", "models:/frozen-train-model", "datathon.gcc.gcc_strength_champion") in calls
    assert ("alias", ("datathon.gcc.gcc_strength_champion", "champion", "7")) in calls
    assert result["train_run_id"] == "a1" and result["evaluate_run_id"] == "b2"


def test_replay_uc_reader_filters_before_select_and_collect():
    calls = []

    class Frame:
        def where(self, predicate):
            calls.append(("where", predicate))
            return self

        def select(self, *columns):
            assert calls == [("where", "split = 'test'")]
            calls.append(("select", columns))
            return self

        def toPandas(self):
            assert calls[-1] == ("select", ("_row_id", "payload"))
            calls.append(("collect",))
            return pd.DataFrame({"_row_id": [1], "payload": [2.0]})

    spark = SimpleNamespace(table=lambda name: Frame())
    result = lockbox._read_replay_gold((spark, "datathon", "gcc"), ["_row_id", "payload"], "test")
    assert len(result) == 1 and calls[-1] == ("collect",)
