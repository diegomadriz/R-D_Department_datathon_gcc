# ADR 0009: entrenamiento y registro en Databricks

- **Estado:** Aceptado
- **Fecha:** 2026-11-03
- **Relacionados:** ADR 0001, 0003, 0004, 0006, 0007 y 0008 (enmienda 2).

## Contexto

El medallón ya tiene un job reproducible; EDA, entrenamiento y evaluación todavía
usaban rutas y tracking locales. Las tareas serverless no comparten archivos en
`/tmp`, ni reciben un checkout con `.git`. Free Edition limita el cómputo diario.
La evaluación oficial del lockbox ya ocurrió y no se puede sustituir ni repetir
para seleccionar modelos.

## Decisión

1. El bundle ejecuta `bootstrap → bronze → silver → gold → reconcile → eda → train
   → evaluate → register → publish`, con `max_concurrent_runs: 1`. Los notebooks
   05–09 son envoltorios; la validación de sus parámetros es pura y vive en
   `job_parameters.py`. Solo se admite `lockbox_mode=replay`.
2. `load_train(source)` sigue siendo el único lector del entrenamiento. Acepta
   Parquet local o `(spark, catalog, schema)`; filtra `split = 'train'` en Spark
   antes de `toPandas`. Solo se leen conteos de partición para declarar el tamaño
   del lockbox. `lockbox.py` sigue siendo el único lector de su contenido.
3. Tracking `databricks`, registry `databricks-uc`; experimento configurable con
   `experiment_path`, por defecto
   `/Workspace/Users/${workspace.current_user.userName}/gcc_strength`.
   No se sincronizan outputs locales, credenciales ni perfiles.
4. EDA registra un run `eda`, con tablas CSV y figuras solo de train. El entorno
   `ml` fija `lightgbm==4.6.0`, `shap==0.51.0`, `statsmodels==0.14.5` y
   `pyyaml==6.0.3`, igual que pyproject. Environment 6 ya trae
   `mlflow-skinny==3.12.0`; no se instala MLflow completo ni PySpark OSS.
5. `train_mode=full` reproduce los diagnósticos congelados, sin HPO nueva.
   `quick` reduce a 15% de grupos, CV 2×2, dos permutaciones y SHAP reducido.
   **Ambos ajustan los predictores finales sobre las 621 filas completas** con
   los mismos perfiles; quick no constituye evidencia válida de desempeño.
   Las tablas y versiones llevan `train_mode` para conservar esa distinción.
   En serverless las permutaciones corren secuencialmente; en local se conserva
   el ejecutor de dos procesos. BLAS se limita a un hilo.
6. Se conservan parámetros, diez runs hijos, firmas, ejemplos y artefactos
   obligatorios. `mlflow_audit` valida completitud al terminar train y antes del
   replay/registro; cualquier error falla el job. Este control no sustituye una
   revisión independiente de fuga u overfitting.
7. Cada tarea usa un directorio temporal nuevo. MLflow almacena `models.json`,
   los modelos congelados y `training/` (OOF, intervalos, diagnósticos y SHAP).
   Los IDs concretos se transfieren mediante task values y referencias del job;
   nunca se busca el último run global. Sin Git en workspace, `git_commit`
   conserva un identificador `source-sha256:` calculado sobre el paquete; el
   manifiesto mantiene hashes individuales de código/configuración.
8. `evaluate` verifica agregados oficiales conforme a la enmienda 2. Carga
   exactamente los predictores ajustados por train; no los reajusta con test.
   CV+ se recalibra exclusivamente con train según el protocolo congelado.
   Una diferencia fuera de tolerancia bloquea registro y publicación.
9. `register` registra el **mismo** modelo GPR (a) del hijo multisalida de train,
   usando su flavor pyfunc de sklearn, firma e input example existentes. No se
   reentrena, ni se elige el mejor modelo con test. El nombre es
   `${catalog}.${schema}.gcc_strength_champion`; se asigna el alias `champion`.
   Los tags de versión y el run `register` enlazan train y evaluate. La selección
   de GPR viene de train y no cambia aunque otra familia gane una celda del test.
10. `publish` escribe cuatro tablas con comentarios de tabla/columna y tag Gold:
    `gold_model_metrics`, `gold_feature_importance`, `gold_predictions` y
    `gold_train_ranges`. Todas llevan run IDs y versión del modelo. Las métricas
    incluyen nueve parejas familia/target por ámbito, con cuatro métricas e IC95%.
    En CV, `value` e IC describen el OOF agrupado; `cv_mean/cv_sd` mantienen el
    diagnóstico por fold. El bootstrap conserva juntas todas las repeticiones de
    cada grupo; los IC son descriptivos y condicionales a la selección previa.
    Predicciones e intervalos son exclusivamente OOF de train del GPR; en quick
    cubren solo el subconjunto diagnóstico. Los rangos siempre usan train completo.
    SHAP es descriptivo, antes de la proyección, y no es causal ni OOF.

## Alternativas y consecuencias

- Compartir `/tmp` entre tareas se descarta: no es persistente entre entornos.
- Subir `mlruns/` local se descarta: cada cuenta genera su propio tracking y
  reconstruye los perfiles congelados, verificando su paridad con los agregados.
- Reabrir el evaluador local se descarta: el candado de un intento sigue intacto.
- El registro es automático tras los gates técnicos de esta fase, autorizado por
  esta decisión; no certifica utilidad industrial, unidades físicas, tipos raros,
  cobertura futura ni ausencia de deriva.
- Cada repetición sobrescribe tablas de consumo y agrega runs/versiones UC; el
  alias avanza solo al finalizar el registro. La publicación de cuatro tablas
  no es una transacción conjunta: si falla, el job queda fallido y debe repetirse;
  los IDs de linaje permiten detectar versiones mezcladas.
- Despliegue, APIs MLflow del workspace y gobierno UC quedan **sin verificación
  de integración** hasta una ejecución autorizada en Databricks. No se presume
  una cuota numérica ni un tiempo garantizado. Guía: [runbook](../databricks_runbook.md).
