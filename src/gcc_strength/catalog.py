"""Unity Catalog naming, data dictionary (table/column comments, layer tags) and SQL builders.

Everything here is pure: functions return SQL strings that the notebooks execute with
``spark.sql``. Identifiers are validated, and comments are escaped for Spark SQL literals.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from gcc_strength.audit import CATEGORICAL, FEATURES, PHASES, TARGETS
from gcc_strength.silver import ZERO_FLAG_COLUMNS

RAW_VOLUME = "raw"

TABLES = {
    "bronze": "datathon_data_bronze",
    "silver": "datathon_data_silver",
    "quarantine": "silver_quarantine",
    "gold": "datathon_data_gold",
    "dq": "dq_results",
}
LAYER_TAG = {
    "bronze": "bronze",
    "silver": "silver",
    "quarantine": "silver",
    "gold": "gold",
    "dq": "dq",
}

TABLE_COMMENTS = {
    "bronze": (
        "Bronze: CSV original de GCC tal cual (todas las columnas como string) más linaje "
        "(_row_id, _ingest_ts, _source_file, _file_hash). Sin transformaciones."
    ),
    "silver": (
        "Silver: tipos validados, ProductType normalizado, sin filas en cuarentena ni "
        "duplicados exactos, y reglas deterministas del ADR 0002 (banderas y nulos). "
        "Sin imputación ni banderas estadísticas."
    ),
    "quarantine": (
        "Filas rechazadas por Silver, con su valor crudo (string) y el motivo en reason "
        "(empty_row, cast_error, unknown_product_type, no_features, no_target)."
    ),
    "gold": (
        "Gold: features crudas, targets en columnas separadas, banderas de Silver, "
        "strength_group_id y split train/test persistido (lockbox en conf/lockbox.sha256)."
    ),
    "dq": (
        "Resultados de calidad de datos por ejecución: regla, severidad, filas evaluadas, "
        "filas fallidas y acción. Incluye la reconciliación bronze = silver + quarantine + "
        "duplicados."
    ),
}

_FEATURE_COMMENTS = {
    "Physical_Metric_A": (
        "Métrica física A, pre-escalada min-max [0,1] por el proveedor (unidades originales "
        "desconocidas). Solo existe desde _row_id 586; no se imputa fuera del Pipeline."
    ),
    "Physical_Metric_B": (
        "Métrica física B, pre-estandarizada (z-score) por el proveedor; unidades originales "
        "desconocidas."
    ),
    "Mineral_Phase_A": (
        "Fase mineral A, min-max [0,1]. Variable derivada: A = B + C a la resolución de cada "
        "columna (ADR 0002); se excluye de modelos lineales y del VIF."
    ),
    "Chemical_Composition_A": "Composición química A, pre-escalada min-max [0,1].",
    "Chemical_Composition_B": "Composición química B, pre-escalada min-max [0,1].",
}
for _p in PHASES[1:]:
    _FEATURE_COMMENTS[_p] = f"Fase mineral {_p[-1]}, pre-escalada min-max [0,1]."
_FEATURE_COMMENTS["Mineral_Phase_F"] += " Sus ceros exactos se marcan en flag_zero_Mineral_Phase_F."
_FEATURE_COMMENTS["Mineral_Phase_H"] += " Sus ceros exactos se marcan en flag_zero_Mineral_Phase_H."

_TARGET_COMMENTS = {
    t: (
        f"Target: resistencia a compresión a {t.removeprefix('Strength')} día(s), "
        "probablemente en psi (múltiplos de 10; no verificado). Nunca es feature."
    )
    for t in TARGETS
}

_COMMON = {
    "_row_id": (
        "Número de fila del registro en el CSV original (0 = primera fila de datos). Orden "
        "reproducible; no es feature."
    ),
    "_ingest_ts": "Momento (UTC) de la ingesta a Bronze.",
    "_source_file": "Ruta del archivo fuente leído en la ingesta.",
    "_file_hash": "SHA-256 del archivo fuente (verificado contra data/CHECKSUMS.txt).",
    CATEGORICAL: "Tipo de producto (categórico, Type 1..5). Nunca se trata como ordinal.",
    "flag_phases_anomalous": (
        "True si >= 7 de las 8 fases valen exactamente 0 (mínimo escalado); en ese caso las 8 "
        "fases se pusieron en nulo (ADR 0002, regla 3)."
    ),
    **{
        f"flag_zero_{c}": (
            f"True si {c} vale exactamente 0 (posible no medido); el valor se conserva."
        )
        for c in ZERO_FLAG_COLUMNS
    },
    "flag_r1_suspect": (
        "True si Strength1 == Strength7 en el CSV; ese Strength1 se puso en nulo. Derivada de "
        "targets: nunca es feature."
    ),
    "strength_group_id": (
        "Grupo de filas consecutivas con el mismo ensayo de resistencia (ADR 0002, regla 10). "
        "Metadato de partición: nunca es feature."
    ),
    "split": (
        "train o test, asignado una vez con StratifiedGroupKFold (random_state=42) y "
        "persistido. El test es el lockbox: no se usa para decidir nada."
    ),
    "reason": (
        "Motivo de cuarentena: empty_row, cast_error, unknown_product_type, no_features o "
        "no_target (uno por fila, por prioridad)."
    ),
    "run_id": "Identificador de la ejecución (run del job o uuid local).",
    "ts": "Momento (UTC) en que se evaluó la regla.",
    "layer": "Capa evaluada: bronze, silver, gold o dq.",
    "rule": "Nombre de la regla de calidad.",
    "severity": "info (descriptiva), warn (regla aplicada) o error (gate que falla el job).",
    "rows_evaluated": "Filas evaluadas por la regla.",
    "rows_failed": "Filas que no cumplen la regla (o a las que se aplicó la acción).",
    "action": "Acción tomada: quarantine, drop, flag, set_null_and_flag, normalize, fail, etc.",
}


CONSUMPTION_COMMENTS = {
    "model_metrics": "Métricas 3×3 de CV y replay; IC95% por grupos. Resultado oficial conservado.",
    "feature_importance": "Importancia SHAP descriptiva por familia, target, rama y feature.",
    "predictions": "Predicciones OOF de train del GPR con intervalos CV+ externos.",
    "train_ranges": "Rangos de calidad por tipo solo en train; soporte del simulador.",
}
for _key, _comment in CONSUMPTION_COMMENTS.items():
    TABLES[_key] = f"gold_{_key}"
    TABLE_COMMENTS[_key] = _comment
    LAYER_TAG[_key] = "gold"

_COMMON.update(
    {
        "family": "Familia congelada del modelo.",
        "target": "Horizonte objetivo; nunca se usa como feature.",
        "metric": "Métrica: MAE, MAPE porcentual, RMSE o R².",
        "scope": "CV descriptiva de train o replay verificado del lockbox oficial.",
        "value": "Estimación puntual: OOF agrupado para CV; replay para lockbox.",
        "ci_low": "Extremo inferior del IC95% bootstrap por grupos.",
        "ci_high": "Extremo superior del IC95% bootstrap por grupos.",
        "cv_mean": "Media de métricas de folds externos; puede diferir del OOF agrupado.",
        "cv_sd": "Desviación de métricas entre folds externos.",
        "feature": "Nombre de la variable del modelo o de calidad.",
        "branch": "Rama experta/fallback del predictor explicada antes de la proyección.",
        "importance": "Media del valor absoluto SHAP; unidades del target, descriptiva.",
        "repeat": "Repetición de CV externa.",
        "fold": "Fold externo: el predictor no vio este grupo al ajustar.",
        "observed": "Etiqueta observada de train; nunca es feature.",
        "predicted": "Predicción OOF de train después de proyección monótona.",
        "lower": "Límite inferior del intervalo CV+ externo nominal 90%.",
        "upper": "Límite superior del intervalo CV+ externo nominal 90%.",
        "covered": "Indica si el intervalo OOF contiene la etiqueta observada.",
        "n_rows": "Número de filas de train del tipo.",
        "n_observed": "Valores no nulos de calidad en train.",
        "minimum": "Mínimo observado en train, sin imputación.",
        "maximum": "Máximo observado en train, sin imputación.",
        "p05": "Percentil 5 de calidad observado en train.",
        "p95": "Percentil 95 de calidad observado en train.",
        "train_mode": (
            "guided: diagnósticos locales verificados y ajuste final; "
            "full: recalcula diagnósticos completos; quick: diagnósticos reducidos."
        ),
        "diagnostics_origin": "Entorno que produjo los diagnósticos de train: local o databricks.",
        "diagnostics_source_run_id": "Run MLflow original que calculó los diagnósticos de train.",
        "evidence_manifest_sha256": (
            "SHA-256 del manifiesto del paquete de diagnósticos importado; "
            "vacío cuando se recalculan en esta ejecución."
        ),
        "train_run_id": "Run padre MLflow del entrenamiento de esta ejecución.",
        "evaluate_run_id": "Run MLflow del replay verificado de esta ejecución.",
        "model_version": "Versión UC del campeón registrado en esta ejecución.",
    }
)


def column_comment(table_key: str, column: str) -> str:
    """Comment for ``column`` in ``table_key``; raises ``KeyError`` if it is not documented."""
    raw_layer = table_key in {"bronze", "quarantine"}
    if column in FEATURES or column in TARGETS or column == CATEGORICAL:
        base = {**_FEATURE_COMMENTS, **_TARGET_COMMENTS}.get(column, _COMMON.get(column))
        if raw_layer:
            return f"Valor crudo (string, sin transformar) del CSV. {base}"
        return base
    if column in _COMMON:
        return _COMMON[column]
    raise KeyError(f"column {column!r} of {table_key!r} has no comment in the data dictionary")


# --------------------------------------------------------------------------------------
# SQL builders
# --------------------------------------------------------------------------------------
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"invalid identifier {name!r}")
    return f"`{name}`"


def sql_string(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def table_fqn(catalog: str, schema: str, table_key: str) -> str:
    return ".".join(quote_ident(x) for x in (catalog, schema, TABLES[table_key]))


def volume_path(catalog: str, schema: str, volume: str = RAW_VOLUME) -> str:
    for x in (catalog, schema, volume):
        quote_ident(x)
    return f"/Volumes/{catalog}/{schema}/{volume}"


def bootstrap_statements(catalog: str, schema: str, volume: str = RAW_VOLUME) -> list[str]:
    c, s, v = quote_ident(catalog), quote_ident(schema), quote_ident(volume)
    return [
        f"CREATE CATALOG IF NOT EXISTS {c} COMMENT "
        + sql_string("Datathon MICAI 2026: catálogo del proyecto"),
        f"CREATE SCHEMA IF NOT EXISTS {c}.{s} COMMENT "
        + sql_string("Reto GCC: medallón de resistencias del cemento"),
        f"CREATE VOLUME IF NOT EXISTS {c}.{s}.{v} COMMENT "
        + sql_string("CSV original inmutable (SHA-256 en data/CHECKSUMS.txt)"),
    ]


def governance_statements(
    catalog: str, schema: str, table_key: str, columns: Iterable[str]
) -> list[str]:
    """COMMENT on the table and every column, plus the ``layer`` tag."""
    fqn = table_fqn(catalog, schema, table_key)
    stmts = [f"COMMENT ON TABLE {fqn} IS {sql_string(TABLE_COMMENTS[table_key])}"]
    for col in columns:
        stmts.append(
            f"ALTER TABLE {fqn} ALTER COLUMN {quote_ident(col)} COMMENT "
            + sql_string(column_comment(table_key, col))
        )
    stmts.append(f"ALTER TABLE {fqn} SET TAGS ('layer' = '{LAYER_TAG[table_key]}')")
    return stmts


def reset_statements(
    catalog: str, schema: str, tables: Iterable[str], confirm: str, mode: str = "tables"
) -> list[str]:
    """DROP statements for a reset. ``confirm`` must equal ``'<catalog>.<schema>'``.

    ``mode='tables'`` drops every listed table; ``mode='schema'`` drops the whole schema
    (tables and volume) with CASCADE, so the next run starts from a missing schema.
    """
    if confirm != f"{catalog}.{schema}":
        raise PermissionError(f"reset refused: confirm must be exactly '{catalog}.{schema}'")
    c, s = quote_ident(catalog), quote_ident(schema)
    if mode == "schema":
        return [f"DROP SCHEMA IF EXISTS {c}.{s} CASCADE"]
    if mode != "tables":
        raise ValueError(f"unknown reset mode {mode!r}")
    return [f"DROP TABLE IF EXISTS {c}.{s}.{quote_ident(t)}" for t in sorted(tables)]
