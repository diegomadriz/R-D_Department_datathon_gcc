# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Bootstrap del catálogo
# MAGIC Crea el catálogo, el esquema y el volumen `raw` con SQL (`IF NOT EXISTS`), copia el CSV
# MAGIC desde los archivos del bundle al volumen si aún no existe y verifica su SHA-256.
# MAGIC La lógica vive en `src/gcc_strength/` (ADR 0003).

# COMMAND ----------

import os
import sys

dbutils.widgets.text("catalog", "datathon")
dbutils.widgets.text("schema", "gcc")
dbutils.widgets.text("bundle_root", "")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
bundle_root = dbutils.widgets.get("bundle_root") or os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(bundle_root, "src"))

# COMMAND ----------

import shutil
from pathlib import Path

from gcc_strength import catalog as uc
from gcc_strength.audit import sha256
from gcc_strength.bronze import expected_sha256
from gcc_strength.config import CHECKSUMS_FILE, DATA_DIR, load_medallion_conf
from gcc_strength.spark_io import run_sql

run_sql(spark, uc.bootstrap_statements(catalog, schema, uc.RAW_VOLUME))

raw_file = load_medallion_conf()["raw_file"]
src = DATA_DIR / raw_file
dst = Path(uc.volume_path(catalog, schema, uc.RAW_VOLUME)) / raw_file
expected = expected_sha256(CHECKSUMS_FILE, raw_file)
if not dst.exists():
    shutil.copyfile(src, dst)
    print(f"copied {raw_file} to {dst}")
actual = sha256(dst)
if actual != expected:
    raise RuntimeError(f"{dst} SHA-256 {actual} does not match CHECKSUMS.txt {expected}")
print(f"{dst} verified (sha256 {actual})")
