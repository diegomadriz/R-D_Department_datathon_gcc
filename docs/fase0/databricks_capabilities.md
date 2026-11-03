# Fase 0: capacidades reales de Databricks Free Edition

Pruebas hechas el 2026-11-03 con Databricks CLI v1.23.0 contra el workspace de prueba (Free Edition), con autenticación OAuth de usuario. El usuario de prueba es administrador del workspace (grupos `users` y `admins`). En este documento se omiten a propósito el host, los correos, los identificadores de cuenta y los tokens.

## 1. Resumen

- **Casi todo lo que necesita la solución está disponible:** catálogo propio, esquema, volumen, jobs serverless, pipelines declarativos con expectations, Asset Bundles, MLflow con registro en UC y alias, dashboards AI/BI publicables con parámetros, Databricks Apps y Genie.
- **El catálogo `datathon` no se puede crear con `databricks catalogs create`** (falla porque el metastore no tiene storage root), pero sí con SQL: `CREATE CATALOG IF NOT EXISTS datathon` desde el SQL warehouse. Queda como `MANAGED_CATALOG` sobre Default Storage. Lo mismo aplica a cualquier otra cuenta Free Edition del equipo.
- **El runtime serverless es Python 3.12.3** (environment version 6, `client.6.5`), y la versión de entorno se puede fijar de forma explícita en el spec del job, incluso en tareas de notebook. **lightgbm, shap, xgboost, statsmodels, seaborn y plotly NO vienen preinstalados**, y de MLflow solo está `mlflow-skinny` 3.12.0. Hay que declararlos como dependencias del environment.
- **Sorpresas operativas:** `databricks pipelines list-pipeline-events` (CLI 1.23.0) no devuelve el campo `details`, así que las métricas de expectations hay que leerlas con `databricks api get /api/2.0/pipelines/<id>/events`. Además, borrar un pipeline **no** borra sus tablas (ST/MV): hay que eliminarlas aparte.
- **Límites de Free Edition** (doc oficial): solo hay cómputo serverless; hay un tope diario de uso sin cifra publicada (si se agota, el cómputo se apaga el resto del día); 1 SQL warehouse 2X-Small; 5 tareas de job concurrentes; 1 pipeline activo por tipo; 3 apps, con auto-stop a las 24 h; Model Serving limitado y sin GPU.

## 2. Matriz de capacidades

| Capacidad | Estado | Evidencia (comando → salida corta) | Límite observado | Fallback elegido | Impacto en el plan |
|---|---|---|---|---|---|
| Catálogo propio | **disponible, con límite** | `catalogs create datathon` → `Error: Metastore storage root URL does not exist. Default Storage is enabled in your account. You can use the UI to create a new catalog using Default Storage, or please provide a storage location…`. Luego, `POST /api/2.0/sql/statements` con `CREATE CATALOG IF NOT EXISTS datathon COMMENT '…'` → `SUCCEEDED`; `catalogs get datathon` → `MANAGED_CATALOG`, `storage_root` asignado | La API REST de catálogos no sirve; hay que crearlo con SQL (warehouse o notebook) o desde la UI | Crear el catálogo con SQL `CREATE CATALOG IF NOT EXISTS` en el primer paso del job (o en un script de bootstrap) | Se mantiene `datathon.gcc`. El bundle no puede declarar el catálogo con la API de catálogos: lo crea una tarea SQL o un paso previo |
| Esquema + volumen | disponible | `schemas create gcc datathon` → `datathon.gcc`; `volumes create datathon gcc raw MANAGED` → `datathon.gcc.raw MANAGED` | Ninguno | — | Ninguno: el esquema y el volumen pueden ir como recursos del bundle o crearse por CLI |
| Upload a volumen | disponible | `fs cp data/datathon_data_gcc.csv dbfs:/Volumes/datathon/gcc/raw/…` → `FILE 181648`. Al descargarlo de vuelta, SHA-256 = `e410cbb1…cc83b83` (coincide); 792 líneas (791 filas + header) | Ninguno | — | El volumen `raw` sirve como fuente inmutable; `_file_hash` en Bronze es verificable |
| Job serverless (notebook) | disponible | `jobs submit` sin cluster, con `environment_key` → `TERMINATED SUCCESS`; setup 4 s, ejecución 72 s, run total ≈ 77 s | Máximo 5 tareas concurrentes (doc) | — | El job de punta a punta es viable; conviene no paralelizar más de 5 tareas |
| Versión de Python serverless y de las librerías | disponible | `sys.version` → `3.12.3 (main, Aug 31 2026…) [GCC 13.3.0]`; `DATABRICKS_RUNTIME_VERSION=client.6.5`; Spark 4.2.0 (ver tabla de la sección 3) | lightgbm, shap y xgboost no vienen instalados; mlflow solo en su versión `skinny` | Declarar `dependencies` en el environment del job | La Fase 1 fija las versiones locales con `uv` según la sección 3 y agrega lightgbm y shap como dependencias del environment |
| Environment version seleccionable | disponible | Spec `"environments":[{"environment_key":"env6","spec":{"environment_version":"6"}}]` en una tarea `notebook_task` → la API lo acepta; dentro del run, `DATABRICKS_ENV_VERSION=6`. En el bundle, `environment_version: "6"` en una tarea de notebook pasa `validate` y `deploy` | NO VERIFICADO cuál es la versión por defecto si no se especifica (habría requerido otra ejecución) | Fijar siempre `environment_version: "6"` | Paridad local/serverless garantizada a nivel de spec |
| Tags y comments en UC | disponible | `COMMENT ON TABLE`, `ALTER COLUMN … COMMENT`, `SET TAGS ('layer'='spike')` y tag de columna. Leídos de `information_schema.tables/columns/table_tags/column_tags` → `table_comment: "spike test table"`, `table_tags: [{layer: spike}]`, `column_tags: [{ProductType, role, categorical}]` | Ninguno | — | El diccionario de datos y los tags `layer=` se pueden hacer como está planeado |
| Pipelines declarativos + expectations | **disponible, con límite** | Pipeline `serverless: true`, `catalog: datathon`, `schema: gcc`, con un archivo SQL (ST con `read_files` + MV). El update → `COMPLETED`. El event log (`api get …/events`) trae `data_quality` por flow: ST `product_type_not_null` passed 791 / failed 0 (DROP ROW), `metric_a_positive` (WARN) passed 191 / failed 600, `warned_records: 600`; MV `n_positive` passed 5 / failed 0. ST `num_output_rows: 791` | 1 pipeline activo por tipo (doc). El CLI `list-pipeline-events` no devuelve `details`. `pipelines delete` deja vivas las tablas ST/MV. El primer update falló en el análisis por un error en el SQL de prueba (una expectation sobre una columna inexistente en la MV); el segundo pasó | No hace falta fallback. Las métricas se leen con `api get`, y la limpieza borra las tablas aparte | Bronze/Silver pueden ir como pipeline declarativo con expectations, y `dq_results` se puede poblar desde el event log. Solo un pipeline activo a la vez: el medallón debe ser **un solo pipeline** |
| Asset Bundles validate/deploy/destroy | disponible | `bundle validate` → `Validation OK!`; `bundle deploy` → `Created jobs.spike_bundle_job … Resources: 1 created`; `bundle destroy --auto-approve` → `Destroy: 1 deleted` | `mode: development` antepone `[dev <usuario>]` al nombre del job. El host del perfil, con ruta extra, **no** dio problemas: la URL del recurso salió bien formada | Si el host con ruta falla en otro entorno, pasar `DATABRICKS_HOST` con solo el origen en el comando | El bundle multi-target es viable. Queda la carpeta vacía `.bundle/` en el home del usuario |
| MLflow tracking | disponible | `tracking_uri: databricks`; `set_experiment('/Users/<me>/…')` → `experiment_id` creado; run con `log_params` y `log_metric` → `run_id` | mlflow completo no viene instalado (solo `mlflow-skinny` 3.12.0, suficiente para tracking y sklearn) | Agregar `mlflow` completo a las dependencias solo si se necesita algo que skinny no trae | Funciona tal cual; la Fase 1 debe fijar `mlflow-skinny==3.12.0` o una versión compatible en local |
| Registro UC + alias | disponible | `registry_uri` por defecto: `databricks-uc`. `log_model(…, signature, input_example, registered_model_name='datathon.gcc._spike_model')` → `registered_version: 1`; `set_registered_model_alias(…, 'champion', 1)` → `alias_version: 1` | Ninguno | — | El modelo `cement_strength_predictor@champion` es viable |
| Carga por alias | disponible | `mlflow.pyfunc.load_model('models:/datathon.gcc._spike_model@champion')` → predice; la signature conserva `Physical_Metric_A: double, Physical_Metric_B: double` | Ninguno | — | El simulador y el job pueden cargar el modelo por alias |
| SQL warehouse | **disponible, con límite** | `warehouses list` → 1 warehouse, `Serverless Starter Warehouse`, `PRO`, `enable_serverless_compute: true`, `2X-Small`, `auto_stop_mins: 10`, `max_num_clusters: 1` | 1 warehouse, máximo 2X-Small (doc). Arrancó solo para el `CREATE CATALOG` y se detuvo al final (`state: STOPPED`) | — | El dashboard y Genie usan este único warehouse; las queries deben ser ligeras |
| AI/BI dashboards (+ parámetros/filtros, publicación) | disponible | `lakeview create` con un dataset parametrizado (`:ptype`) y widgets `filter-single-select` + `table` → `ACTIVE`, y el `serialized_dashboard` conserva `parameters`. `lakeview publish --embed-credentials=false` → revisión publicada; `get-published` OK. Después, `unpublish` + `trash` → `TRASHED` | Los queries no se ejecutaron (no se abrió el dashboard). La doc oficial confirma filtros de campo, parámetros (single/multi, fecha, texto) y parámetros basados en query, y que los dashboards publicados guardan la selección en la URL | — | Se puede hacer el dashboard con filtro `ProductType` y versionarlo como `.lvdash.json` |
| Databricks Apps | **disponible, con límite** | `apps list` → `[]`; `apps create spike-hello` → compute `ACTIVE` en ≈ 2 min, `compute_size: MEDIUM`; `apps deploy` (Streamlit) → `App started successfully`, `app_status: RUNNING`; `apps stop` → `STOPPED`; `apps delete` → eliminada (`apps list` → 0) | 3 apps por cuenta; cada una se detiene sola 24 h después de arrancar o redesplegarse (doc). Streamlit funcionó sin `requirements.txt` | — | El simulador como App es viable. Hay que reiniciarlo antes de la demo, y la malla `gold_sim_grid` queda como respaldo opcional |
| Model Serving | **con límite** (no se creó nada) | `serving-endpoints list` → 12 endpoints de foundation models del sistema, `READY` (p. ej. `databricks-meta-llama-3-3-70b-instruct`, `databricks-gte-large-en`) | Doc: límite de endpoints activos (sin cifra), sin GPU, sin provisioned throughput y sin batch inference. NO VERIFICADO si se puede crear un endpoint CPU para un modelo propio: no se intentó, por costo | Cargar el modelo por alias dentro de la App (sin endpoint) | La solución no depende de Serving; la App carga el pyfunc directamente |
| Genie | disponible | `genie list-spaces` → existe un space de ejemplo (`Bakehouse Sales Starter Space`); el CLI expone `create-space`, `ask`, `create-message`… | La doc de límites de Free Edition no menciona Genie; requiere el warehouse PRO/serverless (disponible) | — | El Genie Space sobre las tablas Gold es viable como extra |
| Cuota diaria | **con límite** (sin cifra) | Doc oficial: hay un tope de uso diario de cómputo sin cifra publicada. Si se agota, el cómputo se apaga el resto del día (o del mes, en casos extremos), sin perder datos | No se agotó en estas pruebas | Bundle multi-target para pasar a la cuenta de otro integrante | Se confirma el enfoque local-first: Databricks solo para integración y la ejecución final |

## 3. Versiones en serverless (environment version 6)

Las lee el notebook del run 1 con `importlib.metadata.version()` (`DATABRICKS_ENV_VERSION=6`, `DATABRICKS_RUNTIME_VERSION=client.6.5`, `IS_SERVERLESS=TRUE`).

| Paquete | Versión |
|---|---|
| Python | 3.12.3 |
| Spark (`spark.version`) | 4.2.0 |
| pyspark | 4.3.0.dev0+databricks.connect.19.1 |
| databricks-connect | 19.1 |
| databricks-sdk | 0.122.0 |
| pandas | 2.3.3 |
| numpy | 2.3.4 |
| scikit-learn | 1.7.2 |
| scipy | 1.16.3 |
| pyarrow | 21.0.0 |
| matplotlib | 3.10.6 |
| cloudpickle | 3.1.1 |
| pyyaml | 6.0.3 |
| mlflow-skinny | 3.12.0 |
| mlflow (completo) | no instalado |
| lightgbm | no instalado |
| shap | no instalado |
| xgboost | no instalado |
| statsmodels | no instalado |
| seaborn | no instalado |
| plotly | no instalado |
| streamlit / dash | no instalados en el job (la App de Streamlit sí corrió en el runtime de Apps) |

Nota: el `pyspark` del runtime es el cliente de Databricks Connect (19.1), no un PySpark OSS; en local conviene usar `pyspark` OSS de la serie 4.x solo para pruebas.

## 4. Recursos en el workspace

**Quedaron (los usa la Fase 1):**
- Catálogo `datathon` (`MANAGED_CATALOG`, Default Storage).
- Esquema `datathon.gcc` (sin tablas).
- Volumen gestionado `datathon.gcc.raw` con `datathon_data_gcc.csv` (181 648 bytes; SHA-256 verificado).
- Carpeta vacía `.bundle/` en el home del usuario (la deja `bundle destroy`; no tiene contenido).
- El dashboard de prueba quedó en estado `TRASHED` (papelera del workspace, no publicado) y el experimento de MLflow está borrado de forma lógica (`lifecycle_stage: deleted`), que es como los borra la API.
- El historial de los runs (el run de `jobs submit` y los dos updates del pipeline) queda en el workspace, como cualquier historial de ejecución.

**Se borraron:**
- Tabla `datathon.gcc._spike_test`.
- Modelo `datathon.gcc._spike_model` (versión 1, alias `champion` y el modelo).
- Experimento de prueba de MLflow.
- Pipeline `_spike_pipeline` y sus tablas `_spike_dlt_st` (streaming table) y `_spike_dlt_mv` (materialized view), estas eliminadas aparte con `tables delete`.
- Job del bundle (`bundle destroy`) y los archivos del bundle. El run 1 se hizo con `jobs submit`, así que no quedó un job persistente (`jobs list` → 0).
- Dashboard `_spike_dashboard` (despublicado y en papelera).
- App `spike-hello` (detenida y eliminada; `apps list` → 0).
- Carpeta temporal del usuario con el notebook de prueba, el SQL del pipeline y el código de la App.
- SQL warehouse: detenido (`STOPPED`), no borrado (es el único que permite la cuenta).

**Cómputo usado:** 1 run de job serverless (≈ 77 s), 2 updates del pipeline (el primero falló en la fase de análisis por un error en el SQL de prueba y el segundo completó), 1 arranque corto del warehouse para el `CREATE CATALOG` y una App MEDIUM encendida unos 4 minutos. Ninguno de estos recursos quedó encendido.

## 5. Recomendaciones para la Fase 1

1. **Bootstrap del catálogo:** como `catalogs create` no funciona sin storage root, el primer paso reproducible debe ser SQL: `CREATE CATALOG IF NOT EXISTS ${catalog}`, más `CREATE SCHEMA IF NOT EXISTS` y `CREATE VOLUME IF NOT EXISTS` (en una tarea SQL o de notebook del job, o en un script de bootstrap). No declares el catálogo como recurso del bundle. Esto debe quedar en un ADR.
2. **Environment del bundle:** declarar un environment con `environment_version: "6"` y `dependencies` explícitas para lo que no viene preinstalado (`lightgbm`, `shap` y, si hace falta, `mlflow` completo), y usarlo en todas las tareas, también en las de notebook. En local, fijar con `uv` las versiones de la sección 3 (pandas 2.3.3, numpy 2.3.4, scikit-learn 1.7.2, scipy 1.16.3, mlflow-skinny 3.12.0, pyarrow 21.0.0, databricks-sdk 0.122.0). Las de lightgbm y shap se eligen y se fijan igual en los dos lados.
3. **Host:** el perfil con ruta extra en el host funcionó para `validate`, `deploy`, `destroy` y para todos los comandos del CLI. Aun así, en `databricks.yml` no se escribe host ni perfil. Cada integrante usa su propio target, y si algún comando falla por el host, se pasa `DATABRICKS_HOST` con solo el origen en ese comando, sin guardarlo en archivos.
4. **Pipeline declarativo:** usar **un solo pipeline** para Bronze → Silver (límite de 1 pipeline activo por tipo), con expectations `ON VIOLATION DROP ROW` o `WARN` según la regla. Para `dq_results`, leer el event log con la API REST (`/api/2.0/pipelines/<id>/events`), porque el comando `list-pipeline-events` del CLI 1.23.0 pierde `details`; también se puede habilitar la publicación del event log como tabla. En la limpieza o el destroy, borrar explícitamente las ST/MV, porque eliminar el pipeline no las borra. La cuarentena con `reason` se puede implementar como una tabla aparte con la condición invertida.
5. **Expectations de prueba:** la expectation `CAST(Physical_Metric_A AS DOUBLE) > 0` falló en 600 de 791 filas. Lo explica el data audit: son los 599 nulos de esa columna más 1 valor igual a 0, que es el mínimo de su escalado min-max. Es decir, no es un error de datos. Solo muestra que `WARN` registra las fallas sin descartar filas. Las reglas reales deben salir de la auditoría de datos.
6. **Apps:** desplegar el simulador como App de Streamlit. Se apaga sola a las 24 h, así que hay que reiniciarla (`apps start`) antes de la evaluación. La App carga el modelo por alias, sin Model Serving.
7. **Cuota:** como la cifra no está publicada, mantener el modo local-first y hacer en Databricks solo ejecuciones de integración. Si una cuenta agota su cuota, se despliega en otra con `bundle deploy -t <integrante>`.

## Fuentes

- Limitaciones de Free Edition: https://docs.databricks.com/aws/en/getting-started/free-edition-limitations
- Filtros y parámetros en dashboards AI/BI: https://docs.databricks.com/aws/en/dashboards/filters
