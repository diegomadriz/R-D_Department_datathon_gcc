"""Check the completeness of a local Phase 3 MLflow execution.

This is an artifact-completeness check, not an independent leakage or model audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from gcc_strength.audit import TARGETS

PARAMS = frozenset(
    {
        "model_type",
        "target",
        "n_features",
        "train_size",
        "test_size",
        "seed",
        "config_hash",
        "git_commit",
        "lockbox_sha256",
    }
)
METRICS = frozenset(
    {
        *(
            f"cv_{name}_{stat}"
            for name in ("r2", "mae", "mape_pct", "rmse")
            for stat in ("mean", "sd")
        ),
        "cv_r2_type1_mean",
        "overfit_gap",
        "coverage",
    }
)
ARTIFACTS = frozenset(
    {
        "oof_actual_vs_predicted.png",
        "residuals.png",
        "residuals_by_type.png",
        "importance.png",
        "shap_global.png",
        "shap_local.png",
        "learning_curve.png",
        "cv_3x3.csv",
        "cv_3x3.png",
    }
)
FAMILIES = frozenset({"linear", "lightgbm", "gaussian_process"})


def audit_run(parent_id, *, tracking_uri="file:./mlruns"):
    """Return errors; never interpret missing fields as a passing audit."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    parent = client.get_run(parent_id)
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        max_results=5000,
    )
    errors = []
    if parent.info.status != "FINISHED":
        errors.append("Parent run is not FINISHED")
    if len(children) != 10:
        errors.append(f"Expected exactly 10 children, got {len(children)}")
    multi = [r for r in children if r.data.params.get("model_type") == "multioutput_a"]
    if len(multi) != 1 or multi[0].data.params.get("target") != "all":
        errors.append("Expected one multioutput_a/all child")
    parent_artifacts = {item.path for item in client.list_artifacts(parent_id)}
    for artifact in sorted(
        {"cv_3x3.csv", "cv_3x3.png", "metrics.json", "manifest.json"} - parent_artifacts
    ):
        errors.append(f"Parent: missing artifact {artifact}")
    counts = Counter(
        (r.data.params.get("model_type"), r.data.params.get("target")) for r in children
    )
    for family in FAMILIES:
        for target in TARGETS:
            if counts[(family, target)] != 1:
                errors.append(f"Expected exactly one child for {family}/{target}")
    models = []
    page_token = None
    while True:
        page = client.search_logged_models(
            experiment_ids=[parent.info.experiment_id], max_results=50, page_token=page_token
        )
        models.extend(page)
        page_token = page.token
        if not page_token:
            break
    for run in children:
        name = run.info.run_id
        if run.info.status != "FINISHED":
            errors.append(f"{name}: run is not FINISHED")
        for param in sorted(PARAMS - run.data.params.keys()):
            errors.append(f"{name}: missing parameter {param}")
        for metric in sorted(METRICS - run.data.metrics.keys()):
            errors.append(f"{name}: missing metric {metric}")
        if run.data.params.get("seed") != "42":
            errors.append(f"{name}: seed is not 42")
        if run.data.params.get("test_size") != "152":
            errors.append(f"{name}: declared lockbox size differs from protocol")
        if run.data.params.get("train_size") != "621":
            errors.append(f"{name}: declared train size differs from protocol")
        if not run.inputs or not run.inputs.dataset_inputs:
            errors.append(f"{name}: missing dataset input")
        present = {item.path for item in client.list_artifacts(name)}
        for artifact in sorted(ARTIFACTS - present):
            errors.append(f"{name}: missing artifact {artifact}")
        attached = [model for model in models if model.source_run_id == name]
        if len(attached) != 1:
            errors.append(f"{name}: expected one logged model, got {len(attached)}")
            continue
        try:
            metadata = mlflow.models.Model.load(attached[0].model_uri)
            if metadata.signature is None or metadata.saved_input_example_info is None:
                errors.append(f"{name}: model lacks signature or input example")
        except Exception as exc:
            errors.append(f"{name}: cannot inspect model metadata: {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_id", nargs="?", help="MLflow parent run ID (default: latest)")
    parser.add_argument("--tracking-uri", default=Path("mlruns").absolute().as_uri())
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    parent_id = args.parent_id
    if parent_id is None:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("gcc-phase3-final")
        if experiment is None:
            raise SystemExit("No final training experiment")
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'final-training'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise SystemExit("No parent run")
        parent_id = runs[0].info.run_id
    errors = audit_run(parent_id, tracking_uri=args.tracking_uri)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"MLflow audit passed: parent={parent_id}; 11 runs; 0 missing fields")


if __name__ == "__main__":
    main()
