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
   La búsqueda de modelos registrados recorre todas las páginas mediante tokens,
   con un máximo de 50 por página, y asocia cada modelo por su `source_run_id`.
   Así conserva la detección de modelos faltantes o duplicados aunque el
   experimento acumule modelos de otras corridas.
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

## Enmienda 1: diagnósticos locales verificados y ajuste final guiado

**Aceptada el 2026-11-03.** Esta enmienda sustituye el modo predeterminado del
punto 5 y acota la excepción al transporte de resultados del punto 3. Se conserva
el resto del contrato del job, incluido el replay y sus tolerancias. El propósito
es aprovechar la evidencia completa ya calculada sobre train, evitando repetir
diagnósticos costosos en cada prueba de integración de Free Edition.

### Modos y separación del cómputo

- `train_mode=guided` es el modo predeterminado del job. Verifica un paquete de
  diagnósticos completos locales, carga únicamente train de Gold y ajusta los
  nueve componentes finales de las tres familias, con arquitectura (a), las 621
  filas y los perfiles congelados. No vuelve a calcular CV, CV+ externa,
  permutaciones, curvas de aprendizaje, sensibilidad temporal ni SHAP de F3.
- `train_mode=full` conserva el cálculo completo anterior. Sirve para reproducir
  los diagnósticos desde cero sin depender del paquete local; no realiza HPO
  nueva ni cambia el campeón.
- `train_mode=quick` conserva los diagnósticos reducidos para comprobaciones
  técnicas. No produce evidencia completa admisible para el paquete de guided.
- EDA sigue calculándose sobre train. La tarea evaluate conserva su replay
  verificado y la recalibración CV+ sobre train prevista en la enmienda 2 del
  ADR 0008. El ahorro afecta a los diagnósticos F3; no elimina todo el cómputo del
  job. Los tiempos y el ahorro real permanecen **NO VERIFICADOS** hasta medirlos.

### Paquete y procedencia

1. Se exportan exclusivamente los diagnósticos de un run local completo y
   finalizado, conservando su ID, manifiesto histórico, versiones y hashes.
   El manifiesto distingue los archivos contrastados byte a byte con artefactos
   del run, los reconstruidos desde sus hijos y los CSV locales no archivados.
   No se cambia la atribución histórica por el commit o entorno del nuevo ajuste.
2. El paquete vive en `data/training_evidence/`, ignorado por Git y sincronizado
   explícitamente por el bundle. No se sincronizan `mlruns/` ni otros outputs.
   Contiene resultados y figuras de train; no contiene el lockbox, modelos
   serializados ni credenciales. Su transferencia entre integrantes es privada.
3. `conf/training_evidence_frozen.json` fija el hash del manifiesto del paquete.
   Antes de abrir un run o ajustar modelos se comprueban ese pin, el inventario
   exacto y los hashes de los archivos, la configuración, los componentes
   analíticos compatibles y la identidad estable de train. Los identificadores
   y etiquetas OOF deben corresponder exclusivamente a train.
4. El hash histórico de train del productor y el fingerprint estable usado en
   integración se conservan como identidades distintas. Los cambios conocidos
   de integración no justifican alterar los hashes históricos ni admitir
   modificaciones de la lógica analítica sin una revisión explícita.
5. Paquete ausente, incompleto, modificado, quick o incompatible implica un
   fallo antes del ajuste. No existe fallback automático a full ni actualización
   silenciosa del pin. Exportar una nueva evidencia es una acción explícita;
   no modifica los archivos congelados del lockbox.

El productor histórico no archivó una carpeta `training/` completa: conservó
métricas y figuras del padre/hijos, pero `folds.csv` y los CSV de intervalos por
familia solo quedaron en disco local. Se admiten como **snapshot local fijado al
exportar**, después de comprobar pertenencia y etiquetas de train y concordancia
con los agregados archivados: métricas CV y brecha de ajuste para folds,
cobertura y anchos por tipo para intervalos. No se afirma que
MLflow hubiese registrado sus hashes. El manifiesto declara el nivel de
verificación por archivo; el pin garantiza integridad desde esta exportación,
sin constituir una prueba retrospectiva de cuándo se generó cada archivo.

### Registro, consumo y límites

El nuevo run describe un **ajuste final guiado por evidencia previa**, con un
padre y diez hijos, firmas y ejemplos actuales. Los artefactos y métricas de CV
importados se atribuyen al run local productor. Cada run declara
`diagnostics_origin`, `diagnostics_source_run_id`, `evidence_manifest_sha256` y
el entorno del ajuste. El auditor exige procedencia coherente además de la
completitud de artefactos. Las tablas Gold, el registro UC y la publicación
conservan el origen del diagnóstico y el vínculo con el ajuste nuevo.

Los modelos finales se ajustan en el entorno del job y se transfieren a evaluate
mediante MLflow. El replay conserva todos sus gates; importar diagnósticos no
autoriza omitirlo, repetir la evaluación oficial ni elegir con test. Registrar y
publicar requieren el replay aprobado del ajuste de esta misma ejecución.

Un clon nuevo puede recibir el paquete privado y verificarlo contra el pin
versionado, o usar `full` para reproducir el cómputo sin ese paquete. Guided
demuestra la integración del ajuste y la procedencia de evidencia previa;
no debe presentarse como una repetición completa de los diagnósticos dentro de
Databricks. Su compatibilidad remota queda pendiente de integración autorizada.

## Enmienda 2: huella de train estable entre plataformas

**Aceptada el 2026-11-03.** La primera corrida guiada en Databricks se detuvo antes
de ajustar porque la huella de train no coincidía con la congelada. Una comparación
de solo lectura, filtrada a `split = 'train'`, midió la causa: 621 filas, `_row_id`,
grupos, tipos y targets idénticos, y **2 de 9 315 celdas de calidad distintas**, ambas en
`Physical_Metric_B`, por un máximo de 4.4e-16 (2 ULP). El parseo de texto a float de
Silver no es idéntico entre el entorno local y serverless; el origen exacto (CPU o
versión de pandas) no se aisló. Serializar con `%.17g` convertía ese ruido en un fallo.

`train_fingerprint` serializa ahora con 12 dígitos significativos, con los que ambas
copias producen la misma huella. `train_sha256` de `conf/lockbox_protocol_frozen.json`
pasa de `aafa1894c81a…` (exacta, `%.17g`) a `b72024737d90…`, recalculada solo con train
sobre el mismo Gold local, sin cambiar datos, modelos ni resultados oficiales. El paquete
de evidencia se reexportó para fijar la nueva huella. Se conservan el hash del lockbox,
las tolerancias del replay y los hashes de los componentes congelados.

La huella ya no detecta cambios por debajo de 12 dígitos. El ajuste en Databricks usa
esos valores remotos, por lo que sus predicciones pueden diferir mínimamente de las
locales; el replay sigue siendo el control de que los agregados coinciden. Corregir el
parseo de Silver queda como mejora posterior, porque cambiaría los valores de Gold.
