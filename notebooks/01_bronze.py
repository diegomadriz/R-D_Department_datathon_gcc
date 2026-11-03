# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze
# MAGIC Lee el CSV del volumen con todas las columnas como string, agrega el linaje
# MAGIC (`_row_id`, `_ingest_ts`, `_source_file`, `_file_hash`) y escribe `datathon_data_bronze`.

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
from pathlib import Path

from gcc_strength import catalog as uc
from gcc_strength import dq
from gcc_strength.bronze import expected_sha256, read_bronze
from gcc_strength.config import CHECKSUMS_FILE, load_medallion_conf
from gcc_strength.spark_io import write_table

run_id = dbutils.widgets.get("run_id") or dq.new_run_id()
now = datetime.now(UTC)
raw_file = load_medallion_conf()["raw_file"]
path = Path(uc.volume_path(catalog, schema, uc.RAW_VOLUME)) / raw_file

bronze = read_bronze(path, source_file=str(path), ingest_ts=now)
file_hash = bronze["_file_hash"].iloc[0]
ok = file_hash == expected_sha256(CHECKSUMS_FILE, raw_file)
rows = [
    dq.dq_row(
        "bronze",
        "source_sha256_matches_checksums",
        "error",
        len(bronze),
        0 if ok else len(bronze),
        "fail",
    ),
    dq.dq_row("bronze", "rows_ingested", "info", len(bronze), 0, "none"),
]
dq.assert_no_errors(rows)

write_table(spark, bronze, catalog, schema, "bronze")
write_table(spark, dq.to_frame(rows, run_id, now), catalog, schema, "dq", mode="append")
print(f"bronze rows: {len(bronze)}")
