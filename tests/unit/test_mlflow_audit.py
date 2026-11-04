"""The audit must reject a parent with no logged model children."""

import mlflow

from gcc_strength.mlflow_audit import audit_run


def test_missing_children_fail_audit(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.create_experiment("audit-test")
    with mlflow.start_run(experiment_id=experiment) as parent:
        parent_id = parent.info.run_id
    errors = audit_run(parent_id, tracking_uri=uri)
    assert len([error for error in errors if "Expected exactly one child" in error]) == 9
