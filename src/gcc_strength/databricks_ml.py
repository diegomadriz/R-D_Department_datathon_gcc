"""Workspace orchestration and pure consumption-table builders for ML job tasks."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from gcc_strength.audit import FEATURES, TARGETS
from gcc_strength.catalog import quote_ident
from gcc_strength.eda import assert_train
from gcc_strength.lockbox import METRICS, bootstrap_metrics
from gcc_strength.mlflow_audit import audit_run
from gcc_strength.split import GROUP_COL


def configure_mlflow(experiment):
    """Use workspace tracking and UC registry; credentials come from the runtime."""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)


def download_json(run_id, artifact, destination):
    path = MlflowClient().download_artifacts(run_id, artifact, str(destination))
    return json.loads(Path(path).read_text())


def verified_replay(train_run_id, evaluate_run_id, destination):
    """Require a successful replay of this exact training execution before promotion."""
    client = MlflowClient()
    run = client.get_run(evaluate_run_id)
    if (
        run.info.status != "FINISHED"
        or run.data.tags.get("replay_passed") != "true"
        or run.data.tags.get("train_run_id") != train_run_id
    ):
        raise ValueError("A verified replay linked to this training run is required")
    result = download_json(evaluate_run_id, "result.json", destination)
    if result.get("status") != "verified" or result.get("train_run_id") != train_run_id:
        raise ValueError("Replay artifact lineage mismatch")
    return result


def register_champion(parameters, destination):
    """Register the existing multioutput pyfunc flavor; never refit or select a model."""
    p = parameters
    configure_mlflow(p.experiment_path)
    verified_replay(p.train_run_id, p.evaluate_run_id, destination)
    errors = audit_run(p.train_run_id, tracking_uri="databricks")
    if errors:
        raise ValueError(f"Training artifact audit failed: {errors}")
    refs = download_json(p.train_run_id, "models.json", destination)
    # selected_family and architecture are decisions frozen before the official lockbox.
    if refs["families"]["gaussian_process"] != refs["champion"]:
        raise ValueError("Champion must be the frozen GPR multioutput model from train")
    for identifier in (p.catalog, p.schema):
        quote_ident(identifier)
    name = f"{p.catalog}.{p.schema}.gcc_strength_champion"
    with mlflow.start_run(run_name="register") as run:
        lineage = dict(
            train_run_id=p.train_run_id,
            evaluate_run_id=p.evaluate_run_id,
            train_mode=p.train_mode,
            architecture="a",
            selected_family="gaussian_process",
        )
        mlflow.set_tags(lineage)
        version = mlflow.register_model(refs["champion"], name, await_registration_for=300)
        client = MlflowClient()
        for key, value in lineage.items():
            client.set_model_version_tag(name, version.version, key, value)
        client.set_registered_model_alias(name, "champion", version.version)
        record = dict(
            name=name,
            version=str(version.version),
            alias="champion",
            model_uri=refs["champion"],
            register_run_id=run.info.run_id,
            **lineage,
        )
        mlflow.log_dict(record, "registration.json")
        return record


def build_consumption_tables(train, training_dir, replay, *, train_mode, n_resamples=10000):
    """Build aggregate scores and train-only OOF/ranges without opening Gold here.

    CV CIs use whole-group bootstrap over repeated OOF predictions, retaining all
    repeats in each selected group. They are conditional descriptive intervals.
    """
    assert_train(train)
    directory = Path(training_dir)
    cv = pd.read_csv(directory / "cv_3x3.csv")
    oof = pd.read_csv(directory / "oof.csv")
    scores = []
    for (family, target), part in oof.groupby(["family", "target"], sort=True):
        estimate = bootstrap_metrics(
            part.observed,
            part.predicted,
            part.ProductType,
            part[GROUP_COL],
            n_resamples=n_resamples,
        )
        row = cv.loc[cv.family.eq(family) & cv.target.eq(target)].iloc[0]
        for metric in METRICS:
            scores.append(
                dict(
                    family=family,
                    target=target,
                    metric=metric,
                    scope="cv",
                    value=estimate[metric],
                    ci_low=estimate["ci95"][metric]["lower"],
                    ci_high=estimate["ci95"][metric]["upper"],
                    cv_mean=row[f"cv_{metric}_mean"],
                    cv_sd=row[f"cv_{metric}_sd"],
                )
            )
    for row in replay["metrics"]:
        for metric in METRICS:
            scores.append(
                dict(
                    family=row["family"],
                    target=row["target"],
                    metric=metric,
                    scope="lockbox_replay",
                    value=row[metric],
                    ci_low=row["ci95"][metric]["lower"],
                    ci_high=row["ci95"][metric]["upper"],
                    cv_mean=float("nan"),
                    cv_sd=float("nan"),
                )
            )
    importance = []
    for family in sorted(cv.family.unique()):
        for target in TARGETS:
            frame = pd.read_csv(directory / "artifacts" / family / target / "importance.csv")
            frame["family"] = family
            importance.append(frame)
    selected = "gaussian_process"
    intervals = pd.read_csv(directory / f"{selected}_intervals.csv")
    keys = ["family", "repeat", "fold", "target", "_row_id"]
    predictions = oof.loc[oof.family.eq(selected)].merge(
        intervals[[*keys, "lower", "upper", "covered"]], on=keys, validate="one_to_one"
    )
    if not set(predictions._row_id).issubset(set(train._row_id)):
        raise ValueError("OOF predictions must belong exclusively to training rows")
    ranges = []
    for product, part in train.groupby("ProductType", observed=True, sort=True):
        for feature in FEATURES:
            values = part[feature].dropna()
            ranges.append(
                dict(
                    ProductType=product,
                    feature=feature,
                    n_rows=len(part),
                    n_observed=len(values),
                    minimum=values.min(),
                    maximum=values.max(),
                    p05=values.quantile(0.05),
                    p95=values.quantile(0.95),
                )
            )
    tables = {
        "model_metrics": pd.DataFrame(scores),
        "feature_importance": pd.concat(importance, ignore_index=True),
        "predictions": predictions,
        "train_ranges": pd.DataFrame(ranges),
    }
    for frame in tables.values():
        frame["train_mode"] = train_mode
    return tables


def publish_tables(spark, parameters, destination):
    """Consume only artifacts from this job and attach run lineage to every Gold row."""
    from gcc_strength.eda import load_train
    from gcc_strength.spark_io import write_table

    p = parameters
    configure_mlflow(p.experiment_path)
    replay = verified_replay(p.train_run_id, p.evaluate_run_id, destination)
    registration = download_json(p.register_run_id, "registration.json", destination)
    if (
        registration["train_run_id"] != p.train_run_id
        or registration["evaluate_run_id"] != p.evaluate_run_id
    ):
        raise ValueError("Registration lineage mismatch")
    folder = MlflowClient().download_artifacts(p.train_run_id, "training", str(destination))
    tables = build_consumption_tables(
        load_train((spark, p.catalog, p.schema)), folder, replay, train_mode=p.train_mode
    )
    with mlflow.start_run(run_name="publish") as run:
        for key, frame in tables.items():
            frame["train_run_id"] = p.train_run_id
            frame["evaluate_run_id"] = p.evaluate_run_id
            frame["model_version"] = registration["version"]
            write_table(spark, frame, p.catalog, p.schema, key)
            mlflow.log_metric(f"{key}_rows", len(frame))
        mlflow.log_dict(registration, "registration.json")
        return run.info.run_id
