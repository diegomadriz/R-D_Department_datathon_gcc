# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · train
# MAGIC Envoltorio del job: parámetros validados y artefactos compartidos mediante MLflow.
# MAGIC La lógica y los contratos viven en `src/gcc_strength/` (ADR 0009).

# COMMAND ----------

import os
import sys

dbutils.widgets.text("bundle_root", "")
bundle_root = dbutils.widgets.get("bundle_root") or os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(bundle_root, "src"))

# COMMAND ----------

import tempfile
from pathlib import Path

import mlflow

from gcc_strength.job_parameters import parse_notebook_parameters

defaults = {
    "catalog": "datathon",
    "schema": "gcc",
    "experiment_path": "",
    "train_mode": "guided",
    "lockbox_mode": "replay",
    "train_run_id": "",
    "evaluate_run_id": "",
    "register_run_id": "",
}
for name, default in defaults.items():
    dbutils.widgets.text(name, default)
parameters = parse_notebook_parameters(
    "train", {name: dbutils.widgets.get(name) for name in defaults}
)

# COMMAND ----------

from gcc_strength.guided_training import run_guided_training
from gcc_strength.mlflow_audit import audit_run
from gcc_strength.train import run_training

with tempfile.TemporaryDirectory(prefix="gcc-train-") as work:
    trainer = run_guided_training if parameters.train_mode == "guided" else run_training
    options = (
        {"evidence": Path(bundle_root) / "data/training_evidence"}
        if parameters.train_mode == "guided"
        else {"quick": parameters.train_mode == "quick", "permutation_workers": 1}
    )
    parent_id = trainer(
        source=(spark, parameters.catalog, parameters.schema),
        tracking_uri="databricks",
        experiment=parameters.experiment_path,
        output=Path(work) / "training",
        **options,
    )
    errors = audit_run(parent_id, tracking_uri="databricks")
    mlflow.tracking.MlflowClient().set_tag(
        parent_id, "artifact_audit_passed", str(not errors).lower()
    )
    if errors:
        raise RuntimeError(f"MLflow artifact audit failed: {errors}")
    dbutils.jobs.taskValues.set(key="train_run_id", value=parent_id)
