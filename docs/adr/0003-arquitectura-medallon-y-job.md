# ADR 0003: Arquitectura del medallón y del job de punta a punta

- **Estado:** Propuesto (Fase 1)
- **Fecha:** 2026-11-03
- **Relacionados:** ADR 0001 (local-first), ADR 0002 (reglas de limpieza y split), `docs/fase0/databricks_capabilities.md`

## Contexto

- El dataset tiene 791 filas y 16 columnas. Las reglas de limpieza y de split ya están fijadas en el ADR 0002.
- En Free Edition hay pipelines declarativos con expectations, pero con límites: un solo pipeline activo por tipo, las métricas de expectations solo se leen por la API REST del event log, y borrar un pipeline no borra sus tablas.
- `databricks catalogs create` falla en Free Edition (el metastore no tiene storage root). En cambio, `CREATE CATALOG IF NOT EXISTS` por SQL sí funciona.
- El job tiene que poder recrearse desde un esquema vacío en la cuenta de cualquier integrante, y la cuota diaria obliga a desarrollar en local (ADR 0001).

## Decisión

### 1. Lógica en `src/`, notebooks delgados y Spark solo para E/S

- Toda la lógica del medallón es pandas puro, en `src/gcc_strength/`:
  - `bronze.py`: lectura del CSV y linaje;
  - `silver.py`: reglas 1–7 del ADR 0002;
  - `split.py` y `gold.py`: reglas 8–11;
  - `dq.py`: resultados de calidad y reconciliación;
  - `catalog.py`: diccionario de datos y SQL.
- Las mismas funciones corren en local (`uv run python -m gcc_strength.pipeline_local`, que escribe Parquet en `outputs/medallion/`) y en Databricks.
- Los notebooks `00_bootstrap` … `04_dq_reconcile` solo leen parámetros, llaman a esas funciones y escriben Delta en `${catalog}.${schema}` con `gcc_strength.spark_io`.
- `spark_io` convierte pandas ↔ Spark con un esquema explícito (así una tabla vacía también se escribe) y pasa los `NaN` a `NULL`.

### 2. Sin pipeline declarativo por ahora

Es una elección de diseño, no la falta de una capacidad. Las razones:

- **Una sola fuente de verdad:** con un pipeline declarativo, las reglas vivirían en SQL o en decoradores del pipeline, duplicadas respecto de las pruebas locales. Además, el split por grupo, el lockbox y la reconciliación no se pueden expresar como expectations.
- **Testeabilidad:** `pytest` cubre las reglas sin Spark ni cuota.
- **Límite de 1 pipeline activo por tipo** y una limpieza más compleja: el event log solo se lee por REST y, al borrar el pipeline, las tablas quedan vivas.

Se puede revisar si en las fases siguientes un pipeline aporta algo que el job no cubra.

### 3. `dq_results` cumple el rol de las expectations

| Expectation de un pipeline declarativo | Equivalente aquí |
|---|---|
| `WARN` | Regla `severity = 'warn'`: la regla se aplica (bandera, nulo) y se registran las filas afectadas |
| `DROP ROW` | Cuarentena: la fila va a `silver_quarantine` con su `reason` y sus valores crudos. No se pierde |
| `FAIL UPDATE` | Regla `severity = 'error'`: si `rows_failed > 0`, falla la tarea (`assert_no_errors`). Las comprobaciones del split y del lockbox lanzan una excepción |

- Esquema: `run_id`, `ts`, `layer`, `rule`, `severity`, `rows_evaluated`, `rows_failed` y `action`.
- Cada tarea agrega sus filas (`append`) con `run_id = {{job.run_id}}`.
- La última tarea (`reconcile`) hace tres comprobaciones:
  - `bronze = silver + quarantine + duplicados_eliminados`, leyendo los conteos de las tablas;
  - que los conteos coincidan con los del data audit (791 / 773 / 18 / 0). Solo aplica si el hash del CSV es el auditado; si no, la comprobación se registra como omitida;
  - que ninguna regla `error` de la ejecución haya fallado.

### 4. Bootstrap del catálogo por SQL

- `00_bootstrap` ejecuta `CREATE CATALOG IF NOT EXISTS`, `CREATE SCHEMA IF NOT EXISTS` y `CREATE VOLUME IF NOT EXISTS raw`.
- El catálogo no se declara como recurso del bundle.
- El CSV se copia de los archivos del bundle (`data/`) al volumen **solo si no existe** y su SHA-256 se verifica contra `data/CHECKSUMS.txt`. Si el archivo del volumen no coincide, el job falla: nunca se sobrescribe.

### 5. Tablas y gobierno

| Tabla | Escritura | Tag `layer` |
|---|---|---|
| `datathon_data_bronze` | overwrite | `bronze` |
| `datathon_data_silver` | overwrite | `silver` |
| `silver_quarantine` | overwrite | `silver` |
| `datathon_data_gold` | overwrite | `gold` |
| `dq_results` | append | `dq` |

- Cada tabla y cada columna recibe un `COMMENT` tomado del diccionario de `catalog.py`. Un test exige que toda columna de salida tenga comentario.

### 6. Lockbox: se asigna en local y en Databricks solo se verifica

- `pipeline_local` escribe `conf/lockbox.sha256` en la primera ejecución: el SHA-256 de los `_row_id` de test ordenados, uno por línea. Ese archivo se versiona.
- Las ejecuciones siguientes, en local o en Databricks, deben reproducir exactamente ese hash; si no, fallan.
- La tarea `gold` de Databricks nunca escribe el archivo. Si el split de serverless difiere del local, el job falla, y con eso se prueba la paridad.
- Reasignar el split exige `--rewrite-lockbox` y un ADR.

### 7. Reset para la prueba "desde un esquema vacío"

- El job `gcc_reset_schema` (notebook `99_reset_schema`) solo actúa si `confirm_reset` es exactamente `<catalog>.<schema>`.
- Tiene dos modos:
  - `reset_mode=tables` borra todas las tablas del esquema;
  - `reset_mode=schema` hace `DROP SCHEMA … CASCADE`, que incluye el volumen, y el bootstrap lo recrea.
- Por defecto `confirm_reset` está vacío, así que una ejecución accidental falla sin borrar nada.

### 8. Job `gcc_end_to_end`

- Cadena de tareas: `bootstrap → bronze → silver → gold → reconcile`.
- Todas las tareas usan el environment `medallion`: `environment_version: "6"` y `pyyaml==6.0.3`, que ya está en el runtime.
- Las tareas `eda → train → evaluate → register → publish_tables → rubric_audit` quedan como TODO comentado, no como tareas rotas.
- Hay un target por integrante (`diego` por defecto, y `ricardo` y `christopher`), sin host ni perfil. La cuenta se elige con `-p <perfil>`.

### 9. Precisiones de implementación del ADR 0002

El ADR 0002 deja abiertos estos puntos; aquí se fijan:

- **Prioridad de `reason`** (un motivo por fila): `empty_row` > `cast_error` > `unknown_product_type` > `no_features` > `no_target`.
  - `empty_row` se evalúa sobre el texto crudo, así que una fila con un token inválido no cuenta como vacía.
- **`cast_error`:**
  - el único token nulo es la celda vacía (o solo espacios);
  - cualquier otro valor que no sea un número finito (`abc`, `NaN`, `inf`) manda la fila a cuarentena.
- **Normalización de `ProductType`:**
  - se aplica `strip`, se colapsan los espacios y se ignoran las mayúsculas;
  - se aceptan `type`/`tipo`, los separadores `_`/`-` y los ceros a la izquierda (`'  tipo_03'` → `Type 3`);
  - un valor fuera de `Type 1..5` va a cuarentena.
- **Duplicados exactos:** mismo `ProductType` normalizado y mismos 15 valores numéricos ya casteados. Se evalúan después de la cuarentena y antes de las reglas de nulos, y se conserva el `_row_id` menor.
- **Orden de las reglas deterministas:** primero fases anómalas, después ceros de F/H y al final R1 = R7.
  - Como los ceros se marcan después de anular las fases, la fila 619 no lleva `flag_zero_*`: sus fases ya son nulas.
  - Resultado: F tiene 2 ceros marcados (312 y 491) y H tiene 5 (241, 242, 243, 246 y 491).
- **Estratos:**
  - los cuantiles se calculan sobre las filas de cada tipo, y un valor igual a un punto de corte va al bin inferior;
  - las filas sin R28 usan los bins de R7 de su tipo y comparten la etiqueta del bin (`Type 1|q2`);
  - las filas sin R28 ni R7 forman el estrato `<tipo>|na`. Si tuviera menos de 5 grupos, se uniría al bin con menos grupos de su tipo; hoy tiene 6 grupos y queda solo;
  - la fusión de bins pequeños une el bin con menos grupos con su vecino más pequeño (en empate, el inferior);
  - los estratos no se persisten: `split.assign_strata` se recalcula sobre train para la CV (regla 12).
- **Fold de test:** `test_fold = 0` en `conf/medallion.yaml`. Se fijó antes de ver cualquier resultado y no se cambia.
- **Comprobaciones de reporte (sin acción):** se registran en `dq_results` targets ≤ 0, violaciones de monotonía y filas que se quedan sin features o sin targets después de las reglas. Hoy todas dan 0.

## Conteos (ejecución local reproducible)

| Concepto | Filas |
|---|---|
| Bronze | 791 |
| Cuarentena `empty_row` | 8 |
| Cuarentena `no_features` | 5 |
| Cuarentena `no_target` | 5 |
| Duplicados eliminados | 0 |
| **Silver = Gold** | **773** (791 = 773 + 18 + 0) |
| `flag_phases_anomalous` | 1 (fila 619) |
| `flag_r1_suspect` | 1 (fila 265) |
| Grupos (`strength_group_id`) | 750, de ellos 21 multi-fila (44 filas) |
| Train | 621 filas / 600 grupos |
| Test | 152 filas (19.7%) / 150 grupos |

- Estratos (grupos por estrato, sobre las 773 filas): Type 1 `q0`–`q3` (170 / 157 / 156 / 157), `Type 1|na` (6), Type 2 `q0`/`q1` (20 / 15), Type 3 `q0`/`q1` (33 / 31) y `rare` (5). No hizo falta fusionar.
- **Consecuencia estructural:** el estrato `rare` tiene 5 grupos repartidos en 5 folds, y el fold 0 no recibió ninguno. Types 4 y 5 quedan completos en train, y el lockbox no evalúa esos tipos. Esto es compatible con el ADR 0002, que solo exige Types 1–3 en ambos lados y ya declara que el modelo no es confiable para Types 4 y 5.

## Alternativas consideradas

- **Pipeline declarativo con expectations para Bronze → Silver:** se descarta por ahora, por la duplicación de lógica y el límite de pipelines (ver la decisión 2).
- **PySpark puro en los notebooks:** se descarta porque duplicaría la lógica pandas que se prueba en local, sin ninguna ganancia con 791 filas.
- **`DROP ROW` sin tabla de cuarentena:** se descarta porque pierde la trazabilidad de la reconciliación.
- **Escribir `conf/lockbox.sha256` desde Databricks:** se descarta. El hash se fija una vez en el repo, y el job solo lo verifica.

## Consecuencias

- La E/S con Spark (`spark_io` y los notebooks) no se prueba en local. Solo `spark_ddl` tiene tests. Queda **NO VERIFICADA** hasta la primera ejecución de integración en Databricks.
- El diseño en pandas no escala a millones de filas. Aquí no importa, y si hiciera falta, el cambio queda acotado a `spark_io`.
- Cualquier cambio en las reglas que mueva el split rompe el lockbox a propósito: obliga a un ADR y a `--rewrite-lockbox`.
