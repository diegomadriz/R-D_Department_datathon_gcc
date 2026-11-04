# `notebooks/`

Envoltorios delgados de `src/gcc_strength/`, sincronizados por el bundle.
`bundle_root` agrega `src/` a `sys.path`; `catalog` y `schema` seleccionan UC.
Los parámetros de ML se validan con funciones puras antes de leer datos.

| Notebook | Tarea | Resultado |
|---|---|---|
| `00_bootstrap.py` | `bootstrap` | Catálogo/esquema/volumen por SQL; CSV copiado y SHA-256 verificado |
| `01_bronze.py` | `bronze` | CSV crudo y linaje |
| `02_silver.py` | `silver` | Limpieza determinista y cuarentena |
| `03_gold.py` | `gold` | Split persistido; verifica `conf/lockbox.sha256` |
| `04_dq_reconcile.py` | `reconcile` | Conteos y gate de calidad |
| `05_eda.py` | `eda` | Tablas y figuras solo de train en un run MLflow `eda` |
| `06_train.py` | `train` | Tres familias × tres targets, GPR multisalida y auditoría de artefactos |
| `07_evaluate.py` | `evaluate` | Replay de agregados congelados, con diferencias registradas |
| `08_register.py` | `register` | Mismo GPR del entrenamiento en UC, alias `champion` y linaje |
| `09_publish.py` | `publish` | Cuatro tablas Gold de consumo con comentarios |
| `99_reset_schema.py` | `gcc_reset_schema` | Reset protegido por `confirm_reset=<catalog>.<schema>` |

La cadena es secuencial. Tracking `databricks`, registry `databricks-uc` y
experimento `experiment_path`. MLflow transfiere artefactos entre tareas; los
IDs concretos viajan con task values, sin depender de archivos temporales compartidos.
`train_mode` admite `full` (por defecto) o `quick` (diagnósticos reducidos, modelos
finales siempre con 621 filas). `lockbox_mode` admite exclusivamente `replay`.

`07_lockbox.py` conserva el envoltorio local histórico de un único intento, fuera
del bundle: **no ejecutarlo**, el intento oficial ya se consumió. El replay nunca
lo invoca ni modifica sus salidas. Ver [ADR 0008](../docs/adr/0008-protocolo-lockbox.md)
y [ADR 0009](../docs/adr/0009-entrenamiento-y-registro-en-databricks.md).

El medallón local equivalente es `uv run python -m gcc_strength.pipeline_local`.
Para reproducir el job en cualquier cuenta: [runbook](../docs/databricks_runbook.md).
