# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · Reset del esquema (protegido)
# MAGIC Borra las tablas del esquema (`reset_mode=tables`) o el esquema completo con su volumen
# MAGIC (`reset_mode=schema`) para probar el job desde un esquema vacío. Solo actúa si
# MAGIC `confirm_reset` es exactamente `<catalog>.<schema>`; con cualquier otro valor falla sin
# MAGIC borrar nada.

# COMMAND ----------

import os
import sys

dbutils.widgets.text("catalog", "datathon")
dbutils.widgets.text("schema", "gcc")
dbutils.widgets.text("bundle_root", "")
dbutils.widgets.text("confirm_reset", "")
dbutils.widgets.dropdown("reset_mode", "tables", ["tables", "schema"])
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
bundle_root = dbutils.widgets.get("bundle_root") or os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(bundle_root, "src"))

# COMMAND ----------

from gcc_strength import catalog as uc
from gcc_strength.spark_io import run_sql

confirm = dbutils.widgets.get("confirm_reset")
mode = dbutils.widgets.get("reset_mode")
# Validate the confirmation before touching the catalog.
uc.reset_statements(catalog, schema, [], confirm, mode)

if spark.catalog.databaseExists(f"{catalog}.{schema}"):
    tables = [
        r["tableName"]
        for r in spark.sql(
            f"SHOW TABLES IN {uc.quote_ident(catalog)}.{uc.quote_ident(schema)}"
        ).collect()
        if not r["isTemporary"]
    ]
    statements = uc.reset_statements(catalog, schema, tables, confirm, mode)
    run_sql(spark, statements)
    print("\n".join(statements))
else:
    print(f"{catalog}.{schema} does not exist: nothing to reset")
