# Databricks notebook source
# MAGIC %md
# MAGIC # 09 · publish
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
    "publish", {name: dbutils.widgets.get(name) for name in defaults}
)

# COMMAND ----------

from gcc_strength.databricks_ml import publish_tables

with tempfile.TemporaryDirectory(prefix="gcc-publish-") as work:
    publish_run_id = publish_tables(spark, parameters, Path(work))
    dbutils.jobs.taskValues.set(key="publish_run_id", value=publish_run_id)
