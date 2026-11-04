# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · eda
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

from gcc_strength.databricks_ml import configure_mlflow
from gcc_strength.job_parameters import parse_notebook_parameters

defaults = {
    "catalog": "datathon",
    "schema": "gcc",
    "experiment_path": "",
    "train_mode": "full",
    "lockbox_mode": "replay",
    "train_run_id": "",
    "evaluate_run_id": "",
    "register_run_id": "",
}
for name, default in defaults.items():
    dbutils.widgets.text(name, default)
parameters = parse_notebook_parameters(
    "eda", {name: dbutils.widgets.get(name) for name in defaults}
)

# COMMAND ----------

from gcc_strength.eda import analyze, load_train, make_figures
from threadpoolctl import threadpool_limits

configure_mlflow(parameters.experiment_path)
with tempfile.TemporaryDirectory(prefix="gcc-eda-") as work:
    directory = Path(work)
    train = load_train((spark, parameters.catalog, parameters.schema))
    with mlflow.start_run(run_name="eda") as run, threadpool_limits(limits=1):
        tables = analyze(train)
        for name, table in tables.items():
            table.to_csv(directory / f"{name}.csv", index=False)
        make_figures(train, tables, directory / "figures")
        mlflow.log_param("train_size", len(train))
        mlflow.log_artifacts(str(directory))
        dbutils.jobs.taskValues.set(key="eda_run_id", value=run.info.run_id)
