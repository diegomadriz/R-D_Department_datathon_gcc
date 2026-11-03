# ADR 0002: Reglas de limpieza y de split derivadas del data audit de la Fase 0

- **Estado:** Propuesto (pendiente del gate de la Fase 0; se implementa en la Fase 1)
- **Fecha:** 2026-11-03
- **Evidencia:** `docs/fase0/data_audit.md` y `docs/fase0/data_audit_results.md` (`uv run python -m gcc_strength.audit`)

## Contexto

El audit del CSV crudo encontró lo siguiente:

1. Las features vienen pre-escaladas: min-max [0, 1], y `Physical_Metric_B` en z-score.
2. `Mineral_Phase_A` = B + C, demostrado en conteos enteros a la resolución de cada columna. Su VIF implícito es ≈ 7·10⁵.
3. Las 8 fases no cierran a una constante.
4. Hay 8 filas vacías, 5 filas sin features pero con algún target y 5 filas con features pero sin ningún target.
5. La fila 619 tiene 7 fases en el mínimo escalado y G en el máximo (13 veces su p99), y comparte targets con la fila 620. Sin unidades originales es una anomalía, no una imposibilidad demostrada.
6. Hay 21 grupos de filas consecutivas con el mismo trío (R1, R7, R28) y fases distintas.
7. `Physical_Metric_A` solo existe desde la fila 586.
8. Las variables derivan de forma fuerte a lo largo del archivo, con un orden probablemente cronológico.
9. Type 4 tiene n = 6 y Type 5 tiene n = 1.

## Decisión

### Silver: solo transformaciones deterministas

1. `ProductType` se normaliza con `strip`, se colapsan los espacios y se mapea a `Type N`. Un valor fuera del catálogo `Type 1..5` va a cuarentena con `reason = 'unknown_product_type'`.
2. Cuarentena con motivo:
   - sin features ni targets: `reason = 'empty_row'`;
   - sin features pero con algún target: `reason = 'no_features'`;
   - con features pero sin ningún target: `reason = 'no_target'`.
3. **Composición anómala** (≥ 7 de las 8 fases exactamente en el mínimo escalado): se marca con `flag_phases_anomalous = true`. Hay dos opciones y la decisión es del gate F0:
   - (a) las 8 fases pasan a nulo, que es la opción propuesta;
   - (b) solo se marcan.

   Hoy aplica a la fila 619. Es una regla determinista, no depende de estadísticas.
4. Se **marcan, sin borrar**, los ceros exactos de `Mineral_Phase_F` y `H` (`flag_zero_<col>`), por ser posibles "no medido". Es una regla determinista.
   - Si en una fila R1 = R7 (resistencia nula entre 1 y 7 días, físicamente implausible), se marca `flag_r1_suspect`. Hoy aplica a la fila 265 (Type 4). Hay que decidir si ese R1 se excluye del entrenamiento de R1.
5. **Las banderas de outlier estadístico (IQR, z robusto) NO se persisten en Silver**: dependen de cuantiles y MAD del dataset completo, que incluye el test. Se calculan dentro del `Pipeline` de sklearn, con umbrales ajustados solo en el fold de entrenamiento.
6. Los duplicados exactos se eliminan después de la cuarentena de filas vacías. Hoy, entre filas con datos, son 0. El conteo queda en `dq_results`.
7. No se imputa por balance de masa, porque no hay cierre. Tampoco se imputa nada con estadísticas globales: cualquier imputación va dentro del `Pipeline` de sklearn.

### Gold: features y split

0. **Orden reproducible.** `_row_id` es el número de línea del CSV (0 = primera fila de datos) y se asigna en Bronze. Toda operación que dependa del orden (`strength_groups`, el split y la CV temporal) ordena primero por `_row_id`, porque Spark no garantiza el orden de lectura. `conf/lockbox.sha256` se calcula sobre los `_row_id` del test ordenados.

8. `Mineral_Phase_A` se conserva en Gold, pero se documenta como derivada (A = B + C). Los modelos lineales y el análisis de VIF la excluyen. La decisión final de conservarla en modelos de árbol se toma por CV en la Fase 3.
9. Se crea `strength_group_id` con `gcc_strength.audit.strength_groups`. La fila i se une al grupo de la fila i−1 si ambas comparten al menos 2 targets no nulos y todos los targets observados en ambas son iguales. Las demás filas forman un grupo propio. Hoy da 21 grupos multi-fila (44 filas).
   - Es **metadato de partición**: no se usa como feature, y un test verifica que no esté en `X`.
   - Es una hipótesis (un mismo ensayo con varios registros de calidad). Si GCC entrega un identificador de lote, lo reemplaza.
10. **El split 80/20 se hace por grupo**, nunca por fila, con `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. Se toma uno de los 5 folds como test.
    - **Estrato** = `ProductType` × cuartil de `Strength28` dentro del tipo, con estas reglas:
      - Types 4 y 5 (7 filas en total) forman un único estrato `rare`, solo para estratificar; el `ProductType` real no cambia.
      - Type 2 y Type 3 usan solo 2 bins (mediana), por su n.
      - Las filas sin `Strength28` usan el cuartil de `Strength7`; si tampoco lo tienen, solo el tipo.
      - Los cuantiles se calculan con los valores de R28 por tipo, que ya se expusieron en el perfil de la Fase 0, y no se usan para modelar.
    - **Comprobaciones de viabilidad** (fallan el job si no se cumplen):
      - todo estrato tiene ≥ 5 grupos (si no, se fusionan bins adyacentes del mismo tipo hasta cumplirlo);
      - el test tiene entre 18% y 22% de las **filas**;
      - Type 1, 2 y 3 aparecen en train y en test;
      - ningún `strength_group_id` aparece en los dos lados.
11. Toda la CV del entrenamiento usa `StratifiedGroupKFold` (o `GroupKFold`) con `strength_group_id`, con las mismas reglas de estrato.

### Validación adicional

12. Además de la CV repetida 5×5, se reporta una **CV temporal de sensibilidad** solo sobre train: *forward-chaining* por bloques contiguos de filas en el orden original (`_row_id`).
    - **Los cortes se ponen solo en fronteras de grupo.** Como los grupos son de filas consecutivas, esto preserva a la vez el orden y la separación de grupos. Un test verifica que ningún grupo quede a ambos lados de un corte.
    - La brecha entre la CV aleatoria y la temporal se documenta.
    - El número de fila **no** se usa como feature sin una decisión explícita del equipo.

## Alternativas consideradas

- **Banderas de outlier en Silver con umbrales globales:** se descarta, porque fugan información del test.
- **Imputar `Physical_Metric_A` con la mediana global en Silver:** se descarta porque fuga estadísticas del test e inventa una medición que no existía antes de la fila 586.
- **Eliminar la fila 619 completa:** se descarta porque sus targets y sus `Chemical_Composition` son válidos. Solo se invalidan las fases.
- **Split aleatorio por fila:** se descarta porque los grupos de trío idéntico filtrarían el target entre train y test.
- **Split temporal como lockbox oficial:** se descarta como test oficial, porque el reto pide un 80/20 y no hay fechas confirmadas. Se mantiene como sensibilidad dentro de train.

## Consecuencias

- La reconciliación de conteos queda así: `bronze = silver + quarantine + duplicados_eliminados`, con la cuarentena separada por `reason`.
- El split persistido depende de `strength_group_id`, así que `conf/lockbox.sha256` se calcula sobre los `_row_id` del test una vez agrupados.
- Si GCC confirma un identificador de lote o fechas, este ADR se revisa: el grupo pasaría a ser el lote y la CV temporal usaría fechas reales.
