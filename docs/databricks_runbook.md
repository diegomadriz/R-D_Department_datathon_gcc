# Reproducir el pipeline completo en Databricks Free Edition

Esta guía sirve para macOS y Windows, en la cuenta propia de cualquier integrante.
Los textos entre `<...>` se sustituyen; no se copian literalmente. No requiere
subir el CSV a mano: el bootstrap lo verifica y copia desde los archivos del bundle.
La ejecución realiza medallón, EDA, entrenamiento, replay del resultado oficial,
registro UC y publicación de tablas. La integración en Databricks está pendiente
hasta que un integrante ejecute estos pasos; los tests locales no la certifican.
El modo predeterminado `guided` reutiliza diagnósticos completos locales
verificados y ajusta los modelos finales en el job. `full` recalcula también
todos los diagnósticos, sin requerir ese paquete.

## 1. Requisitos

- Acceso al repositorio privado de GitHub y Git instalado. Para clonar por HTTPS,
  autenticar Git con el gestor de credenciales del sistema; no poner tokens en URLs.
- `uv`, que descarga Python 3.12.3 según `.python-version` y resuelve `uv.lock`.
- Databricks CLI **≥1.23.0**, la versión medida en
  [capacidades de Fase 0](fase0/databricks_capabilities.md). No instalar el paquete
  legado `databricks-cli` de pip. Comparar versiones numéricas, no como texto.
- Cuenta Free Edition activa con acceso a workspace y permisos para crear
  catálogo, esquema, volumen, tablas, experimentos y modelos UC.

macOS con Homebrew:

```bash
brew install git uv libomp
brew install databricks/tap/databricks
```

Alternativa para uv y CLI mediante sus instaladores oficiales (Git debe existir):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
```

Windows, desde PowerShell con winget:

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id Databricks.DatabricksCLI -e
```

Alternativa de uv en PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cerrar y volver a abrir la terminal después de instalar. Verificar:

```text
git --version
uv --version
databricks version
```

`libomp` permite importar LightGBM local en macOS. Databricks Linux usa la rueda
correspondiente; no necesita Homebrew. El PC solo despliega y recibe resultados:
los modelos del job se ajustan en Databricks.

## 2. Clonar y preparar

Usar la URL privada entregada al equipo, con acceso concedido en GitHub:

```text
git clone <url-del-repo-privado> datathon
cd datathon
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
```

Usar una revisión que incluya el ADR 0009 y los notebooks 05–09. Si el mantenedor
indica una rama publicada, seleccionarla con `git switch <rama-publicada>`.
`pytest` usa datos sintéticos para el evaluador oficial y no abre un nuevo intento.
En una cuenta nueva no hace falta generar outputs locales ni ejecutar el lockbox:
el job reconstruye Gold desde el CSV versionado y usa los agregados congelados.

## 3. Autenticación

Copiar la URL del workspace propio desde el navegador; no es la URL de la consola
de cuentas. Autenticación OAuth de usuario, en ambos sistemas:

```text
databricks auth login --host <url-del-workspace> --profile <nombre>
databricks current-user me -p <nombre>
```

Las credenciales quedan en la configuración local del CLI, nunca en el repo.
El archivo de perfiles está en `~/.databrickscfg` (macOS) o
`%USERPROFILE%\.databrickscfg` (Windows); el CLI administra también su caché OAuth
local. No copiar estos archivos al repositorio ni compartirlos entre integrantes.

## 4. Validar y desplegar

### Preparar la evidencia de train para guided

En la máquina que conserva la corrida local completa, exportar una vez:

```text
uv run python -m gcc_strength.training_evidence export --source outputs/fase3/final
uv run python -m gcc_strength.training_evidence verify
```

La exportación contrasta los diagnósticos con el run MLflow local de origen y
genera `data/training_evidence/` y el pin
`conf/training_evidence_frozen.json`. No entrena ni abre el lockbox. El directorio
del paquete queda fuera de Git; el pin sí forma parte del proyecto. No exportar
desde una corrida quick ni reemplazar un pin para aceptar resultados distintos.

Los demás integrantes reciben **solo esa carpeta**, por un canal privado del
equipo, y la colocan en `data/training_evidence/`. Para verificarla necesitan Gold
local con el mismo train; si ya está disponible, ejecutar el comando `verify`
anterior. La verificación del job usa directamente train de Gold en Databricks.
El manifiesto conserva la procedencia por archivo, incluidas las limitaciones de
los CSV locales que no se archivaron en el run histórico. No transferir `mlruns/`,
credenciales ni el resultado oficial del lockbox.

Un clon sin paquete puede usar `train_mode=full`. Su despliegue no depende de que
exista `data/training_evidence/`; intentar guided sin ella falla antes del ajuste.
No hay fallback automático que empiece a consumir cómputo de full.

### Despliegue del bundle

Desde la raíz del repo:

```text
databricks bundle validate -t <target> -p <perfil>
databricks bundle deploy -t <target> -p <perfil>
```

La validación necesita autenticación para resolver el usuario actual; no es una
validación offline. Deploy sincroniza `src`, `notebooks`, `conf` y `data`, y crea
los jobs. Una inclusión explícita sincroniza el paquete privado de evidencia
cuando existe; no se suben otros outputs ni el tracking local. Catálogo y esquema
se crean por SQL al ejecutar bootstrap.

Targets disponibles: `diego` (por defecto), `ricardo` y `christopher`. Son iguales;
no contienen hosts ni perfiles. **El perfil `-p` selecciona la cuenta**, el target
separa el despliegue. Ejemplo: Ricardo usa `-t ricardo -p <su-perfil>`.
Para agregar un integrante, añadir a `targets` de `databricks.yml`:

```yaml
nuevo_integrante:
  mode: development
```

Actualizar el contrato de targets de `tests/unit/test_bundle.py`. No agregar host
ni profile. Para usar otro catálogo/esquema o experimento, definir las variables
con `--var` tanto en validate como en deploy, por ejemplo:

```text
databricks bundle deploy -t <target> -p <perfil> --var="catalog=datathon,schema=gcc,experiment_path=/Workspace/Shared/gcc_strength"
```

La ruta por defecto es
`/Workspace/Users/<usuario-actual>/gcc_strength`; no hace falta crear a mano
el experimento. Se recomienda esa ruta personal; la compartida exige permisos.

## 5. Ejecutar

Con el paquete verificado, el comando predeterminado usa guided:

```text
databricks bundle run gcc_end_to_end -t <target> -p <perfil>
```

Para reproducir todos los diagnósticos desde cero, sin paquete:

```text
databricks bundle run gcc_end_to_end -t <target> -p <perfil> --params train_mode=full
```

También se conserva la prueba con diagnósticos reducidos:

```text
databricks bundle run gcc_end_to_end -t <target> -p <perfil> --params train_mode=quick
```

`lockbox_mode` vale `replay` y no admite `first`. Los tres modos conservan las 621
filas de train y los perfiles congelados para el ajuste final. Guided no recalcula
CV, CV+ externa, permutaciones, aprendizaje, sensibilidad temporal ni SHAP de F3:
importa sus resultados con `diagnostics_origin=local`, el run productor y el hash
del manifiesto. EDA y el replay (incluida su calibración de train) siguen corriendo.
Quick produce métricas CV y OOF reducidas y no sustituye evidencia completa.
Cualquiera de los modos registra una versión UC y mueve el alias si pasa los gates.

| Tarea | Tiempo orientativo por ejecución |
|---|---|
| bootstrap | 1–3 min, incluida disponibilidad del entorno |
| bronze, silver, gold, reconcile | 0.5–2 min cada una |
| eda | 1–5 min, con 2 000 remuestreos descriptivos |
| train full | 10–30 min; referencia local histórica ≈8 min, serverless usa permutaciones secuenciales |
| train quick | 1–5 min; prueba local ≈19 s sin arranque del entorno |
| train guided | Tiempo remoto NO VERIFICADO; verifica evidencia y hace solo el ajuste final y logging |
| evaluate | 1–5 min; conserva bootstrap 10 000 y calibración CV+ completa |
| register | 0.5–3 min |
| publish | 1–5 min |

**Estimaciones, no mediciones de integración:** presupuestar aproximadamente
20–60 minutos de cómputo serverless para full y 8–30 para quick, con arranques e
instalación incluidos. El consumo depende del hardware y de la caché del entorno;
no existe una conversión verificada a DBU ni un importe monetario para este job.
La cuota diaria gratuita no tiene una cifra publicada. Un solo full puede agotarla;
quick tampoco es gratuito en términos de cuota. No lanzar varios jobs a la vez.
No hay una cifra de ahorro medida para guided. El primer intento quick falló en
la consulta de modelos de MLflow; la paginación se corrigió localmente, pero eso
no certifica todavía el resto de la integración.

## 6. Verificar

El comando run devuelve el enlace y el ID del run del job. Confirmar diez tareas
`SUCCESS` y `max_concurrent_runs=1`. Para revisar el estado desde terminal:

```text
databricks jobs get-run <job-run-id> -p <perfil>
databricks tables list datathon gcc -p <perfil>
```

Debe haber Bronze, Silver, cuarentena, Gold, `dq_results` y estas cuatro tablas:
`gold_model_metrics`, `gold_feature_importance`, `gold_predictions`,
`gold_train_ranges`. Consultar comentarios con, por ejemplo:

```text
databricks tables get datathon.gcc.gold_model_metrics -p <perfil>
```

En el SQL Editor del workspace (warehouse iniciado), ejecutar estas consultas
acotadas; sustituyendo catálogo/esquema si se configuraron otros:

```sql
SELECT 'bronze' AS layer, count(*) AS n FROM datathon.gcc.datathon_data_bronze
UNION ALL SELECT 'silver', count(*) FROM datathon.gcc.datathon_data_silver
UNION ALL SELECT 'quarantine', count(*) FROM datathon.gcc.silver_quarantine;
SELECT split, count(*) AS n FROM datathon.gcc.datathon_data_gold GROUP BY split;
SELECT scope, count(*) AS n FROM datathon.gcc.gold_model_metrics GROUP BY scope;
SELECT train_mode, train_run_id, evaluate_run_id, model_version, count(*) AS n
FROM datathon.gcc.gold_predictions GROUP BY ALL;
SELECT diagnostics_origin, diagnostics_source_run_id, evidence_manifest_sha256,
       count(*) AS n
FROM datathon.gcc.gold_predictions GROUP BY ALL;
```

Esperado: **791 / 773 / 18**, Gold **773**, train **621**, test **152**.
Las métricas contienen **36 filas por ámbito**, nueve parejas familia/target ×
cuatro métricas; `scope=cv` y `scope=lockbox_replay`. Las predicciones contienen
solo train con etiquetas disponibles, repetición/fold e intervalos. No consultar
calidad, etiquetas ni predicciones individuales del lockbox fuera del replay.

Para el hash, obtener el ID específico de la tarea `evaluate` en `jobs get-run`
(no el del job padre) y ejecutar:

```text
databricks jobs get-run-output <evaluate-task-run-id> -p <perfil>
uv run python -c "from pathlib import Path; print(Path('conf/lockbox.sha256').read_text().strip())"
```

`notebook_output.result` debe mostrar `status=verified`, el run MLflow de replay y
`lockbox_sha256` idéntico al archivo local. Gold también falla antes si cambia ese
hash. No editar el archivo para lograr coincidencia.

En **Experiments** de la UI, abrir
`/Workspace/Users/<usuario-actual>/gcc_strength` (o `experiment_path` configurado).
Verificar `eda`, `final-training`, sus diez hijos, `lockbox_replay`, `register` y
`publish`. Train conserva firma/input example, params y artefactos. El tag
`artifact_audit_passed=true` confirma completitud técnica. En replay:
`replay_passed=true`, `comparison.json.passed=true`; revisar diferencias, tolerancias,
`frozen_result_sha256`, `lockbox_sha256` y el enlace `train_run_id`.

Modelo y alias:

```text
databricks registered-models get datathon.gcc.gcc_strength_champion --include-aliases -p <perfil>
```

Debe existir el alias `champion` y su versión. En Catalog Explorer comprobar los
tags `train_run_id`, `evaluate_run_id`, `selected_family=gaussian_process`,
`architecture=a`, `train_mode` y la procedencia del diagnóstico. En guided, el
origen debe ser local y el run productor debe coincidir con el paquete fijado;
el nuevo `train_run_id` identifica el ajuste del job, no el cálculo histórico de CV.
Los valores deben coincidir con el run del job y las tablas.
El alias se consulta como `models:/datathon.gcc.gcc_strength_champion@champion`.

## 7. Reset y repetición desde cero

El reset borra objetos del proyecto; confirmar que se usa la cuenta y el esquema
correctos. Para borrar tablas y conservar volumen, historial y versiones UC:

```text
databricks bundle run gcc_reset_schema -t <target> -p <perfil> --params confirm_reset=datathon.gcc,reset_mode=tables
databricks bundle run gcc_end_to_end -t <target> -p <perfil>
```

Para reconstruir también esquema, volumen y objetos dependientes:

```text
databricks bundle run gcc_reset_schema -t <target> -p <perfil> --params confirm_reset=datathon.gcc,reset_mode=schema
databricks bundle run gcc_end_to_end -t <target> -p <perfil>
```

`confirm_reset` debe ser exactamente `<catalog>.<schema>`; vacío o distinto falla
antes de borrar. Bootstrap recrea el esquema y verifica el CSV. El reset no toca
el resultado oficial local ni `conf/lockbox.sha256`. Las repeticiones agregan runs
MLflow; las tablas se sobrescriben. Nunca borrar intentos locales para reintentarlos.

## 8. Solución de problemas

- **Cuota agotada:** detener las pruebas; esperar la renovación de cuota o
  autenticar otra cuenta propia del equipo y desplegar allí con su target/perfil.
  No modificar datos ni umbrales para reducir errores. No hay cifra diaria garantizada.
- **`catalogs create` falla por storage root:** no usarlo para bootstrap. El
  notebook 00 crea el catálogo con `CREATE CATALOG IF NOT EXISTS` por SQL sobre
  Default Storage, conforme a Fase 0.
- **Permisos:** comprobar el usuario con `current-user me`. Revisar creación de
  catálogo, USE CATALOG/SCHEMA, CREATE TABLE/VOLUME/MODEL y acceso al experimento.
  Una ruta de experimento compartida puede requerir permisos adicionales.
- **CLI antiguo o comando desconocido:** ejecutar `databricks version`, actualizar
  con brew/winget o reinstalar desde el instalador oficial y reabrir la terminal.
  Comprobar que no se invoca el CLI legado instalado con pip.
- **Dependencia ausente:** revisar el environment `ml`, versión 6 y sus pins.
  `statsmodels` es necesario para EDA; MLflow skinny ya viene en el runtime.
  Revisar el log de instalación antes de volver a gastar cuota.
- **Evidencia ausente o incompatible en guided:** comprobar la carpeta privada,
  el pin versionado y sus hashes. No actualizar el pin ni cambiar la configuración
  para que pase. Usar full solo como una ejecución deliberada con su presupuesto;
  guided nunca cambia de modo automáticamente.
- **Replay falla:** abrir `comparison.json` y revisar hashes, versiones y linaje.
  El job debe quedar fallido; no ampliar tolerancia ni cambiar agregados para
  promover el modelo. Documentar la discrepancia antes de repetir una integración.
- **App o warehouse apagados:** el job no depende de la App ni de Model Serving.
  Para la demo, iniciar la App con `databricks apps start <app> -p <perfil>` y el
  warehouse con `databricks warehouses start <warehouse-id> -p <perfil>`.
  Free Edition detiene Apps automáticamente; un dashboard no inicia una App.
- **Publish falla tras registrar:** las cuatro escrituras no son atómicas. Revisar
  linaje de tablas y repetir el job completo tras corregir la causa; no presentar
  tablas de ejecuciones distintas como un conjunto coherente.

## 9. Qué no hacer

- No editar `data/` ni reemplazar el CSV del volumen por una versión distinta.
- No subir credenciales, hosts o archivos de perfiles al repo.
- No ejecutar `gcc_strength.lockbox` en modo oficial ni `notebooks/07_lockbox.py`:
  el único intento local ya se usó.
- No cambiar `conf/lockbox.sha256`, los agregados congelados o el protocolo para
  hacer pasar un replay. No usar test para elegir modelo, features ni parámetros.

Referencias oficiales para instalación y sintaxis:
[CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install),
[CLI Windows](https://learn.microsoft.com/azure/databricks/dev-tools/cli/install),
[uv](https://docs.astral.sh/uv/getting-started/installation/),
[bundle run y parámetros](https://docs.databricks.com/aws/en/dev-tools/cli/bundle-commands),
[registro UC](https://learn.microsoft.com/azure/databricks/dev-tools/cli/reference/registered-models-commands).
