"""The audit must reject a parent with no logged model children."""

from types import SimpleNamespace

import mlflow
import pytest
from mlflow.store.entities import PagedList

from gcc_strength.mlflow_audit import ARTIFACTS, FAMILIES, METRICS, PARAMS, TARGETS, audit_run


def test_missing_children_fail_audit(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.create_experiment("audit-test")
    with mlflow.start_run(experiment_id=experiment) as parent:
        parent_id = parent.info.run_id
    errors = audit_run(parent_id, tracking_uri=uri)
    assert len([error for error in errors if "Expected exactly one child" in error]) == 9


@pytest.mark.parametrize("model_count", [0, 1, 2])
def test_searches_all_logged_model_pages_and_checks_counts(monkeypatch, model_count):
    children = []
    for index, (family, target) in enumerate(
        [(family, target) for family in sorted(FAMILIES) for target in TARGETS]
        + [("multioutput_a", "all")]
    ):
        params = dict.fromkeys(PARAMS, "value")
        params.update(
            model_type=family, target=target, seed="42", test_size="152", train_size="621"
        )
        children.append(
            SimpleNamespace(
                info=SimpleNamespace(run_id=f"child-{index}", status="FINISHED"),
                data=SimpleNamespace(params=params, metrics=dict.fromkeys(METRICS, 1)),
                inputs=SimpleNamespace(dataset_inputs=[object()]),
            )
        )
    models = [
        SimpleNamespace(source_run_id=f"other-{i}", model_uri=f"other-{i}") for i in range(105)
    ]
    models += [
        SimpleNamespace(source_run_id=child.info.run_id, model_uri=child.info.run_id)
        for child in children[1:]
    ]
    models += [
        SimpleNamespace(source_run_id="child-0", model_uri=f"child-0-{i}")
        for i in range(model_count)
    ]
    calls = []

    class Client:
        def get_run(self, run_id):
            return SimpleNamespace(info=SimpleNamespace(experiment_id="exp", status="FINISHED"))

        def search_runs(self, *args, **kwargs):
            return children

        def list_artifacts(self, run_id):
            paths = (
                {"cv_3x3.csv", "cv_3x3.png", "metrics.json", "manifest.json"}
                if run_id == "parent"
                else ARTIFACTS
            )
            return [SimpleNamespace(path=path) for path in paths]

        def search_logged_models(self, *, experiment_ids, max_results, page_token):
            assert experiment_ids == ["exp"]
            assert max_results == 50
            calls.append(page_token)
            start = int(page_token or 0)
            end = start + max_results
            return PagedList(models[start:end], str(end) if end < len(models) else None)

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr("gcc_strength.mlflow_audit.MlflowClient", Client)
    monkeypatch.setattr(
        mlflow.models.Model,
        "load",
        lambda uri: SimpleNamespace(signature=object(), saved_input_example_info=object()),
    )
    errors = audit_run("parent", tracking_uri="local")
    assert calls == [None, "50", "100"]
    if model_count == 1:
        assert errors == []
    else:
        assert errors == [f"child-0: expected one logged model, got {model_count}"]
