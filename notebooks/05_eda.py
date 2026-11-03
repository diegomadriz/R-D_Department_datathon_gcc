# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · EDA y física (solo train)
# MAGIC Lee únicamente `split = 'train'` antes de convertir a pandas. Genera los mismos
# MAGIC resultados locales de Fase 2. No entrena modelos ni evalúa el lockbox.
# MAGIC Este notebook todavía no forma parte del job del bundle.

# COMMAND ----------

import os
import sys
from pathlib import Path

dbutils.widgets.text("catalog", "datathon")
dbutils.widgets.text("schema", "gcc")
dbutils.widgets.text("bundle_root", "")
dbutils.widgets.text("output_dir", "/tmp/gcc_fase2")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
bundle_root = dbutils.widgets.get("bundle_root") or os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(bundle_root, "src"))

# COMMAND ----------

from gcc_strength.eda import train_only, write_report

# Filter in Spark: no test payload is collected or summarized on the driver.
train = train_only(
    spark.table(f"`{catalog}`.`{schema}`.`datathon_data_gold`").where("split = 'train'").toPandas()
)
tables = write_report(train, Path(dbutils.widgets.get("output_dir")))
display(tables["19_fisica"])
display(tables["18_cinetica"])
