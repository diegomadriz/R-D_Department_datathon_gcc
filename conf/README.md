# `conf/`

Configuración en YAML: semillas (`random_state=42`), umbrales, hiperparámetros, espacio de búsqueda y supuestos de ROI. Aquí no van secretos ni tokens.

- `medallion.yaml`: parámetros del split (folds, fold de test, estratos, comprobaciones de viabilidad) y conteos esperados del data audit (ADR 0002 y 0003).
- `lockbox.sha256`: SHA-256 de los `_row_id` del test ordenados, uno por línea. Lo escribe `uv run python -m gcc_strength.pipeline_local` la primera vez; después, toda ejecución (local o Databricks) debe reproducirlo o falla.
- `thresholds.yaml`: umbrales de desempeño (provisionales).
