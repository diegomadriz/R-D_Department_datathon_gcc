"""Synthetic fitting/logging contracts; no historical or lockbox data is evaluated."""

import hashlib
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from gcc_strength import guided_training as guided
from gcc_strength import train as full_training
from gcc_strength.audit import TARGETS
from gcc_strength.eda import train_only
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.mlflow_audit import ARTIFACTS, FAMILIES, METRICS, audit_run
from gcc_strength.provenance import train_fingerprint
from gcc_strength.split import GROUP_COL


class SyntheticArchitecture(RegressorMixin, BaseEstimator):
    """A cheap three-output estimator for transport checks, never a cement model."""

    def fit(self, X, y):
        self.means_ = y[list(TARGETS)].mean().to_numpy()
        self.models_ = [self, self, self]
        return self

    def predict(self, X):
        return np.tile(self.means_, (len(X), 1))


@pytest.fixture
def synthetic_train():
    frame = pd.DataFrame({c: np.linspace(1.0, 2.0, 621) for c in INPUT_COLUMNS})
    frame["ProductType"] = "Type 1"
    frame["split"] = "train"
    frame["_row_id"] = np.arange(621)
    frame[GROUP_COL] = np.arange(621) // 2
    for j, target in enumerate(TARGETS):
        frame[target] = np.arange(621) + (j + 1) * 1000
    frame = train_only(frame)
    frame.attrs["lockbox_size"] = 152
    return frame


@pytest.mark.parametrize("reason", ["missing", "tampered", "fingerprint"])
def test_verification_precedes_fit_tracking_and_output(
    monkeypatch, synthetic_train, tmp_path, reason
):
    monkeypatch.setattr(guided, "load_train", lambda source: synthetic_train)

    def reject(*args, **kwargs):
        raise ValueError(reason)

    monkeypatch.setattr(guided, "verify_evidence", reject)
    monkeypatch.setattr(guided, "_fit", lambda *args: pytest.fail("Fit before verification"))
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *args: pytest.fail("Tracking opened"))
    with pytest.raises(ValueError, match=reason):
        guided.run_guided_training(evidence=tmp_path / "evidence", output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def make_synthetic_evidence(directory, train):
    """Explicit synthetic full-evidence fixture; verifier is replaced only for this test."""
    diagnostics = directory / "diagnostics"
    diagnostics.mkdir(parents=True)
    (diagnostics / "manifest.json").write_text(
        json.dumps({"fixture": "synthetic", "quick": False, "git_commit": "producer-commit"})
    )
    rows = [
        dict(
            family=family, target=target, rule_within_type_pass=True, **dict.fromkeys(METRICS, 1.0)
        )
        for family in sorted(FAMILIES)
        for target in TARGETS
    ]
    pd.DataFrame(rows).to_csv(diagnostics / "cv_3x3.csv", index=False)
    (diagnostics / "cv_3x3.png").write_bytes(b"synthetic-image-fixture")
    (diagnostics / "metrics.json").write_text(
        json.dumps(dict(protocol=dict(quick=False), fixture="synthetic"))
    )
    for family in [*sorted(FAMILIES), "multioutput"]:
        for target in TARGETS if family != "multioutput" else [None]:
            folder = diagnostics / "artifacts" / family
            if target:
                folder /= target
            folder.mkdir(parents=True)
            for name in ARTIFACTS:
                (folder / name).write_bytes(
                    (diagnostics / name).read_bytes()
                    if (diagnostics / name).exists()
                    else b"synthetic-diagnostic-fixture"
                )
    manifest = dict(
        source_run_id="synthetic-producer",
        train_fingerprint=train_fingerprint(train),
        config_hash=hashlib.sha256((guided.ROOT / "conf/models.yaml").read_bytes()).hexdigest(),
        lockbox_sha256=(guided.ROOT / "conf/lockbox.sha256").read_text().strip(),
        files={
            p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in diagnostics.rglob("*")
            if p.is_file()
        },
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_guided_fits_full_train_and_transports_models_historical_diagnostics(
    monkeypatch, synthetic_train, tmp_path
):
    # Loading a model with code_paths prepends its artifact folder to sys.path, which spawned
    # workers in later tests would inherit; restore it when this test ends.
    monkeypatch.setattr(sys, "path", list(sys.path))
    evidence = tmp_path / "evidence"
    manifest = make_synthetic_evidence(evidence, synthetic_train)
    monkeypatch.setattr(guided, "load_train", lambda source: synthetic_train)
    monkeypatch.setattr(guided, "verify_evidence", lambda *args, **kwargs: manifest)
    fit_rows = []

    def fit(frame, specs):
        fit_rows.append(frame._row_id.tolist())
        return SyntheticArchitecture().fit(frame[list(INPUT_COLUMNS)], frame)

    monkeypatch.setattr(guided, "_fit", fit)
    monkeypatch.setattr(guided, "design_size", lambda model: len(INPUT_COLUMNS))
    for name in (
        "_evaluate",
        "_cvplus",
        "_permutation_task",
        "_artifacts",
        "explain_model",
        "repeated_group_splits",
        "temporal_group_splits",
    ):
        monkeypatch.setattr(
            full_training, name, lambda *args, **kwargs: pytest.fail("Recomputed diagnostics")
        )
    uri = (tmp_path / "mlruns").as_uri()
    parent_id = guided.run_guided_training(
        evidence=evidence, tracking_uri=uri, output=tmp_path / "output"
    )
    assert fit_rows == [list(range(621))] * 3
    client = mlflow.tracking.MlflowClient()
    parent = client.get_run(parent_id)
    children = client.search_runs(
        [parent.info.experiment_id], f"tags.mlflow.parentRunId = '{parent_id}'"
    )
    assert len(children) == 10
    for run in [parent, *children]:
        for key in guided.PROVENANCE_KEYS:
            assert run.data.params[key] == run.data.tags[key]
        assert run.data.params["model_fit_size"] == "621"
        assert run.data.params["model_train_sha256"] == train_fingerprint(synthetic_train)
        assert run.data.params["diagnostics_source_run_id"] == "synthetic-producer"
    for run in children:
        if run.data.params["target"] != "all":
            assert run.data.metrics["rule_within_type_pass"] == 1.0
    refs = json.loads(Path(client.download_artifacts(parent_id, "models.json")).read_text())
    assert refs["champion"] == refs["families"]["gaussian_process"]
    for model_uri in refs["families"].values():
        prediction = mlflow.sklearn.load_model(model_uri).predict(
            synthetic_train[list(INPUT_COLUMNS)].head(3)
        )
        assert prediction.shape == (3, 3)
        assert np.all(np.diff(prediction, axis=1) >= 0)
    producer = json.loads(
        Path(client.download_artifacts(parent_id, "training/diagnostic_manifest.json")).read_text()
    )
    assert producer["git_commit"] == "producer-commit"
    # The audit still validates copied bytes, metrics and lineage after its package gate.
    import gcc_strength.config as config_module
    import gcc_strength.training_evidence as evidence_module

    monkeypatch.setattr(evidence_module, "verify_evidence_files", lambda *args, **kwargs: manifest)
    config_root = tmp_path / "config-root"
    (config_root / "conf").mkdir(parents=True)
    (config_root / "conf/lockbox_protocol_frozen.json").write_text(
        json.dumps({"train_sha256": train_fingerprint(synthetic_train)})
    )
    monkeypatch.setattr(config_module, "ROOT", config_root)
    assert audit_run(parent_id, tracking_uri=uri) == []
    child_id = children[0].info.run_id
    client.delete_tag(child_id, "diagnostics_origin")
    assert any(
        "guided tag diagnostics_origin" in error for error in audit_run(parent_id, tracking_uri=uri)
    )
    client.set_tag(child_id, "diagnostics_origin", "local")
    client.log_metric(child_id, "coverage", 0.123)
    assert any(
        "historical metric mismatch: coverage" in error
        for error in audit_run(parent_id, tracking_uri=uri)
    )
    tamper = tmp_path / "residuals.png"
    tamper.write_bytes(b"modified")
    client.log_artifact(child_id, str(tamper))
    assert any(
        "historical diagnostic artifact mismatch: residuals.png" in error
        for error in audit_run(parent_id, tracking_uri=uri)
    )
