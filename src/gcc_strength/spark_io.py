"""Thin Spark I/O used by the Databricks notebooks (pandas <-> Delta).

The medallion logic is pandas-only (the dataset has ~800 rows); Spark only reads and writes
Unity Catalog tables. ``pyspark`` is imported lazily so this module also imports locally,
where only the pure helper :func:`spark_ddl` is unit-tested.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pandas.api import types as ptypes

from gcc_strength.catalog import governance_statements, table_fqn


def spark_ddl(pdf: pd.DataFrame) -> str:
    """Explicit Spark schema (DDL string) for a pandas frame, so empty frames also write."""
    parts = []
    for col, dtype in pdf.dtypes.items():
        if ptypes.is_bool_dtype(dtype):
            sql_type = "BOOLEAN"
        elif ptypes.is_integer_dtype(dtype):
            sql_type = "BIGINT"
        elif ptypes.is_float_dtype(dtype):
            sql_type = "DOUBLE"
        elif ptypes.is_datetime64_any_dtype(dtype):
            sql_type = "TIMESTAMP"
        elif ptypes.is_object_dtype(dtype) or ptypes.is_string_dtype(dtype):
            sql_type = "STRING"
        else:
            raise TypeError(f"unsupported dtype {dtype} for column {col!r}")
        parts.append(f"`{col}` {sql_type}")
    return ", ".join(parts)


def to_spark(spark, pdf: pd.DataFrame):
    """pandas -> Spark with an explicit schema; float NaN becomes SQL NULL."""
    from pyspark.sql import functions as F

    clean = pdf.copy()
    for col in clean.columns:
        if ptypes.is_object_dtype(clean[col]):
            clean[col] = clean[col].astype("object").where(clean[col].notna(), None)
    sdf = spark.createDataFrame(clean, schema=spark_ddl(pdf))
    for col, dtype in pdf.dtypes.items():
        if ptypes.is_float_dtype(dtype):
            sdf = sdf.withColumn(col, F.when(F.isnan(F.col(col)), None).otherwise(F.col(col)))
    return sdf


def write_table(
    spark, pdf: pd.DataFrame, catalog: str, schema: str, table_key: str, mode: str = "overwrite"
) -> str:
    """Write a pandas frame as a managed Delta table and apply comments and layer tag."""
    fqn = table_fqn(catalog, schema, table_key)
    writer = to_spark(spark, pdf).write.format("delta").mode(mode)
    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(fqn)
    run_sql(spark, governance_statements(catalog, schema, table_key, pdf.columns))
    return fqn


def read_table(spark, catalog: str, schema: str, table_key: str) -> pd.DataFrame:
    return spark.table(table_fqn(catalog, schema, table_key)).toPandas()


def run_sql(spark, statements: Iterable[str]) -> None:
    for stmt in statements:
        spark.sql(stmt)
