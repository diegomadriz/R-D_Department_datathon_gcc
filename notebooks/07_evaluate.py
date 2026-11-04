# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · evaluate
# MAGIC Envoltorio del job: parámetros validados y artefactos compartidos mediante MLflow.
# MAGIC La lógica y los contratos viven en `src/gcc_strength/` (ADR 0009).

# COMMAND ----------

import os
import sys

dbutils.widgets.text("bundle_root", "")
bundle_root = dbutils.widgets.get("bundle_root") or os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(bundle_root, "src"))

# COMMAND ----------

import json
import tempfile
from pathlib import Path


from gcc_strength.databricks_ml import configure_mlflow
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
    "evaluate", {name: dbutils.widgets.get(name) for name in defaults}
)

# COMMAND ----------

from gcc_strength.lockbox import replay_lockbox

configure_mlflow(parameters.experiment_path)
with tempfile.TemporaryDirectory(prefix="gcc-replay-") as work:
    result = replay_lockbox(
        mode=parameters.lockbox_mode,
        parent_run_id=parameters.train_run_id,
        gold=(spark, parameters.catalog, parameters.schema),
        tracking_uri="databricks",
        experiment=parameters.experiment_path,
        output=Path(work) / "replay",
    )
    dbutils.jobs.taskValues.set(key="evaluate_run_id", value=result["run_id"])
    print(f"Replay {result['status']}; lockbox SHA-256 {result['lockbox_sha256']}")

dbutils.notebook.exit(
    json.dumps({key: result[key] for key in ("status", "run_id", "lockbox_sha256")})
)
