"""Train-only CLI integration, deterministic metrics and logged serving contracts."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from gcc_strength import train
from gcc_strength.config import ROOT
from gcc_strength.eda import load_train
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.mlflow_audit import audit_run


def test_quick_cli_reproducible_and_auditable(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    outputs = [tmp_path / f"execution-{i}" for i in range(2)]
    for output in outputs:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gcc_strength.train",
                "--quick",
                "--tracking-uri",
                uri,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        run = json.loads((output / "run.json").read_text())
        assert audit_run(run["parent_id"], tracking_uri=uri) == []
    assert (outputs[0] / "metrics.json").read_bytes() == (outputs[1] / "metrics.json").read_bytes()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    parent = client.get_run(run["parent_id"])
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    assert len(children) == 10
    for child in children:
        if child.data.params.get("target") != "all":
            assert "rule_within_type_pass" in child.data.metrics
    models = mlflow.search_logged_models(
        experiment_ids=[parent.info.experiment_id], output_format="list"
    )
    multi = next(r for r in children if r.data.params["model_type"] == "multioutput_a")
    logged = next(m for m in models if m.source_run_id == multi.info.run_id)
    restored = mlflow.sklearn.load_model(logged.model_uri)
    prediction = restored.predict(load_train()[list(INPUT_COLUMNS)].head(5))
    assert prediction.shape == (5, 3)
    assert np.isfinite(prediction).all()
    assert (np.diff(prediction, axis=1) >= 0).all()
    metrics = json.loads((outputs[0] / "metrics.json").read_text())
    assert len(metrics["cv"]) == 9
    assert {row["family"] for row in metrics["coverage"]} == {
        "linear",
        "lightgbm",
        "gaussian_process",
    }
    for row in metrics["cv"]:
        assert all(key in row for key in ("delta_mae", "mae_ci_low", "coverage", "overfit_gap"))
    assert pd.read_csv(outputs[0] / "folds.csv").groupby(["family", "target"]).size().eq(4).all()


def test_train_has_no_partition_reader_or_test_filter():
    tree = ast.parse(Path(train.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Name) and call.func.id == "load_train" for call in calls)
    for call in calls:
        name = (
            call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
        )
        assert name not in {
            "read_csv",
            "read_parquet",
            "read_table",
            "read_sql",
            "evaluate_lockbox",
            "train_only",
            "query",
        }
        if name in {"read_text", "read_bytes"}:
            text = ast.unparse(call.func.value)
            assert "data/" not in text and "gold" not in text
    assert not any(
        isinstance(node, ast.Constant) and node.value in {"test", "split"}
        for node in ast.walk(tree)
    )
