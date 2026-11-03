# ADR 0004: Dependencias de la Fase 1 y empaquetado de `src/` para serverless

- **Estado:** Propuesto (Fase 1)
- **Fecha:** 2026-11-03
- **Relacionados:** ADR 0001 y `docs/fase0/databricks_capabilities.md` §3 (versiones del environment 6)

## Contexto

- El ADR 0001 dejó pendientes mlflow, lightgbm, shap y pyspark hasta validar las versiones del runtime.
- La Fase 0 midió el environment 6 de serverless:
  - `mlflow-skinny` 3.12.0, `databricks-sdk` 0.122.0, `pyyaml` 6.0.3;
  - **sin** mlflow completo, lightgbm, shap, xgboost, statsmodels, seaborn ni plotly;
  - pyspark es el cliente de Databricks Connect 19.1, no PySpark OSS.
- Además, hay que decidir cómo llega el paquete `gcc_strength` a las tareas serverless.

## Decisión

### 1. Dependencias (agregadas con `uv add`, versiones exactas)

| Paquete | Versión | Origen de la versión | Nota |
|---|---|---|---|
| `mlflow-skinny` | 3.12.0 | env 6 | Basta para tracking, el flavor sklearn y el registro en UC |
| `databricks-sdk` | 0.122.0 | env 6 | |
| `pyyaml` | 6.0.3 | env 6 | Lee `conf/*.yaml` |
| `lightgbm` | 4.6.0 | no viene en env 6 | Se declarará con la misma versión en el environment `ml` del job |
| `shap` | 0.51.0 | no viene en env 6 | Trae numba 0.62.1 y llvmlite 0.45.1, compatibles con numpy 2.3.4 |

- **mlflow-skinny y no mlflow completo:** así local y serverless tienen el mismo paquete.
  - Con skinny, el backend local por defecto (sqlite) no funciona, porque no incluye SQLAlchemy ni el registro local. En local hay que usar `MLFLOW_TRACKING_URI=file:./mlruns` (ya ignorado en git) o el tracking de Databricks.
  - Si la Fase 3 necesita la UI o el registro en local, se cambia a `mlflow==3.12.0` solo en local, con un ADR.
- **lightgbm en macOS** necesita la librería del sistema OpenMP (`brew install libomp`). Sin ella, `import lightgbm` falla en local. En serverless (Linux) la rueda la trae. Es un requisito del entorno de desarrollo, no del proyecto.
- **pyspark NO se agrega.** Las razones:
  - la lógica es pandas y Spark solo hace E/S (ADR 0003);
  - el pyspark del runtime es Databricks Connect, así que un PySpark OSS local no reproduciría el runtime;
  - PySpark local exige Java 17 o 21 (la Java global del equipo es la 25).

  Se reconsidera si hace falta probar `spark_io` en local.

### 2. Empaquetado: `sys.path` a los archivos del bundle, sin wheel

- `databricks.yml` sincroniza solo `src/`, `notebooks/`, `conf/` y `data/` (`sync.paths`).
- Cada notebook recibe `bundle_root = ${workspace.file_path}` como parámetro del job y agrega `<bundle_root>/src` a `sys.path`.
- `conf/` y `data/` quedan en la misma ruta relativa que en el repo, así que `gcc_strength.config.ROOT` funciona igual en local y en serverless.
- Las dependencias que el runtime no trae se declaran en el environment del job. Un test (`tests/unit/test_bundle.py`) exige que cada una coincida con su pin de `pyproject.toml`.

## Alternativas consideradas

- **Wheel construido con `uv build` como artifact del bundle:** se descarta por ahora, por tres razones:
  1. `pyproject.toml` incluye dependencias de notebooks y análisis (jupyter, ipykernel, seaborn, plotly, statsmodels, openpyxl). El wheel las arrastraría a cada tarea serverless: instalación lenta y riesgo de chocar con paquetes del runtime.
  2. Con la versión fija 0.1.0, el environment serverless podría reutilizar un wheel en caché y correr código viejo (**NO VERIFICADO**; evitarlo exigiría versiones dinámicas).
  3. Con `sys.path`, el código que corre es exactamente el que sincronizó el último deploy.

  Se reconsidera si las dependencias de desarrollo pasan a un `dependency-group`.
- **mlflow completo en local:** se descarta para no separarse del runtime mientras skinny alcance.
- **pyspark OSS local para pruebas de E/S:** se pospone (ver la decisión 1).

## Consecuencias

- `uv sync` resuelve 162 paquetes con Python 3.12.3.
- Las dependencias del job se mantienen a mano en `databricks.yml`; el test de paridad evita que se desalineen.
- Las tareas de entrenamiento (Fase 3) agregarán un environment `ml` con `lightgbm==4.6.0` y `shap==0.51.0`. La instalación de esas ruedas en serverless queda **NO VERIFICADA** hasta esa fase.
