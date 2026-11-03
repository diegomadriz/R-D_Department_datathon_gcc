# `notebooks/`

Envoltorios delgados que orquestan el código de `src/gcc_strength/`. La lógica no vive aquí: cada notebook lee sus parámetros (`catalog`, `schema`, `bundle_root`, `run_id`), agrega `<bundle_root>/src` a `sys.path` (ADR 0004), llama a las funciones del paquete y escribe Delta. Antes de cada commit, limpia las salidas pesadas.

| Notebook | Tarea del job | Qué hace |
|---|---|---|
| `00_bootstrap.py` | `bootstrap` | `CREATE CATALOG/SCHEMA/VOLUME IF NOT EXISTS`; copia y verifica el CSV en el volumen `raw` |
| `01_bronze.py` | `bronze` | `datathon_data_bronze` (CSV como string + linaje) |
| `02_silver.py` | `silver` | `datathon_data_silver` y `silver_quarantine` |
| `03_gold.py` | `gold` | `datathon_data_gold` con el split persistido; verifica `conf/lockbox.sha256` |
| `04_dq_reconcile.py` | `reconcile` | Reconciliación `bronze = silver + quarantine + duplicados` y gate de `dq_results` |
| `99_reset_schema.py` | job `gcc_reset_schema` | Borra tablas o el esquema; exige `confirm_reset=<catalog>.<schema>` |

Los notebooks de las fases siguientes (EDA, entrenamiento, evaluación, ROI) se agregarán con numeración posterior. El equivalente local de las tareas 01–04 es `uv run python -m gcc_strength.pipeline_local`.
