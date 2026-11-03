# `gcc_strength`

Paquete con la lógica pura y testeable del proyecto: limpieza, features físicas, modelos, métricas y utilidades de MLflow. Los notebooks y pipelines solo llaman a este código. Aquí no va lógica de presentación ni rutas de Databricks fijas.

| Módulo | Contenido |
|---|---|
| `audit.py` | Data audit reproducible de la Fase 0 y `strength_groups` |
| `bronze.py` | Lectura del CSV como string + linaje (`_row_id`, `_ingest_ts`, `_source_file`, `_file_hash`) |
| `silver.py` | Reglas deterministas 1–7 del ADR 0002 y cuarentena con `reason` |
| `split.py` | Estratos, `StratifiedGroupKFold`, comprobaciones de viabilidad y lockbox |
| `gold.py` | Tabla Gold y separación entre features candidatas y metadatos |
| `dq.py` | Esquema de `dq_results`, gates y reconciliación |
| `catalog.py` | Nombres de Unity Catalog, diccionario de datos y SQL (bootstrap, comentarios, tags, reset) |
| `spark_io.py` | E/S pandas ↔ Delta para los notebooks (pyspark se importa solo ahí) |
| `pipeline_local.py` | Medallón completo en local → Parquet en `outputs/medallion/` |
| `config.py` | Rutas del proyecto y carga de YAML |
