# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver
# MAGIC Aplica las reglas deterministas 1–7 del ADR 0002 y escribe `datathon_data_silver` y
# MAGIC `silver_quarantine` (con `reason`).

# COMMAND ----------

import os
import sys

dbutils.widgets.text("catalog", "datathon")
dbutils.widgets.text("schema", "gcc")
dbutils.widgets.text("bundle_root", "")
dbutils.widgets.text("run_id", "")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
bundle_root = dbutils.widgets.get("bundle_root") or os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(bundle_root, "src"))

# COMMAND ----------

from datetime import UTC, datetime

from gcc_strength import dq
from gcc_strength.silver import build_silver
from gcc_strength.spark_io import read_table, write_table

run_id = dbutils.widgets.get("run_id") or dq.new_run_id()
now = datetime.now(UTC)

bronze = read_table(spark, catalog, schema, "bronze")
silver, quarantine, rows = build_silver(bronze)
dq.assert_no_errors(rows)

write_table(spark, silver, catalog, schema, "silver")
write_table(spark, quarantine, catalog, schema, "quarantine")
write_table(spark, dq.to_frame(rows, run_id, now), catalog, schema, "dq", mode="append")
print(f"silver rows: {len(silver)}; quarantine: {quarantine['reason'].value_counts().to_dict()}")
