# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Reconciliación y gate de calidad
# MAGIC Comprueba `bronze = silver + quarantine + duplicados_eliminados`, los conteos esperados
# MAGIC del data audit y que ninguna regla `error` de esta ejecución haya fallado.

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
from gcc_strength.catalog import table_fqn
from gcc_strength.config import load_medallion_conf
from gcc_strength.spark_io import write_table

run_id = dbutils.widgets.get("run_id")
if not run_id.isalnum():
    raise ValueError(f"run_id must be a non-empty alphanumeric id, got {run_id!r}")
now = datetime.now(UTC)

bronze = spark.table(table_fqn(catalog, schema, "bronze"))
n_bronze = bronze.count()
file_hash = bronze.select("_file_hash").first()[0]
n_silver = spark.table(table_fqn(catalog, schema, "silver")).count()
by_reason = {
    r["reason"]: int(r["n"])
    for r in spark.table(table_fqn(catalog, schema, "quarantine"))
    .groupBy("reason")
    .count()
    .withColumnRenamed("count", "n")
    .collect()
}
run_rows = [
    r.asDict()
    for r in spark.table(table_fqn(catalog, schema, "dq")).where(f"run_id = '{run_id}'").collect()
]
dups = dq.rows_failed(run_rows, "silver", "exact_duplicate")

try:
    rows = dq.run_reconciliation(
        n_bronze, n_silver, by_reason, dups, file_hash, load_medallion_conf().get("expected_counts")
    )
    dq.assert_no_errors(run_rows + rows)
except dq.DataQualityError:
    failed = [dq.dq_row("dq", "reconciliation_gate", "error", n_bronze, 1, "fail")]
    write_table(spark, dq.to_frame(failed, run_id, now), catalog, schema, "dq", mode="append")
    raise

write_table(spark, dq.to_frame(rows, run_id, now), catalog, schema, "dq", mode="append")
print(f"reconciled: bronze={n_bronze} silver={n_silver} quarantine={by_reason} dups={dups}")
