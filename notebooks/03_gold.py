# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold
# MAGIC Calcula `strength_group_id`, asigna el split persistido (ADR 0002, reglas 8–11) y
# MAGIC escribe `datathon_data_gold`. El split debe reproducir exactamente `conf/lockbox.sha256`;
# MAGIC si no, el job falla. Este notebook solo reporta conteos, nunca valores del test.

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
from gcc_strength.config import LOCKBOX_FILE, load_medallion_conf
from gcc_strength.gold import build_gold, gold_lockbox_digest
from gcc_strength.spark_io import read_table, write_table
from gcc_strength.split import SPLIT_COL, lockbox_check, read_lockbox

run_id = dbutils.widgets.get("run_id") or dq.new_run_id()
now = datetime.now(UTC)

silver = read_table(spark, catalog, schema, "silver")
gold, rows = build_gold(silver, load_medallion_conf()["split"])
n_test = int((gold[SPLIT_COL] == "test").sum())
rows += lockbox_check(gold_lockbox_digest(gold), read_lockbox(LOCKBOX_FILE), n_test)
dq.assert_no_errors(rows)

write_table(spark, gold, catalog, schema, "gold")
write_table(spark, dq.to_frame(rows, run_id, now), catalog, schema, "dq", mode="append")
print(f"gold rows by split: {gold[SPLIT_COL].value_counts().to_dict()}")
