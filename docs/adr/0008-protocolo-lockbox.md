# ADR 0008: protocolo de evaluación del lockbox

Estado: Aceptado

## Decisión

La Fase 4 ejecutará una sola vez la evaluación en el job, después del gate de
auditoría. Ninguna ejecución de desarrollo abre ni evalúa el lockbox.

Antes de calcular cualquier métrica de test, el job ejecutará validación adversarial
train frente a test usando exclusivamente features y CV agrupada. Emitirá y registrará
en MLflow solo el AUC y el conteo de casi-duplicados cruzados; abortará antes de las
métricas si AUC > 0.6 o existen duplicados exactos cruzados. La detección de duplicados
exactos se utiliza como condición de parada sin publicar filas ni características.

Si pasa el control, se calcularán la tabla 3×3 de MAE, MAPE y R², IC al 95% por
bootstrap de grupos, ΔMAE pareado frente a L1 según ADR 0002 punto 14, cobertura
conformal, R² por tipo y monotonía. Se registrarán los modelos en Unity Catalog y se
asignará el alias `@champion` al seleccionado.

No hay reintentos para mejorar resultados: si decepcionan, se documentan con sus
limitaciones. El split aleatorio por grupo mide interpolación dentro del periodo y no
constituye una validación del desempeño futuro bajo deriva.

## Enmienda 1

Por decisión del equipo, la evaluación única se ejecuta **en local**, después del
gate de auditoría independiente. La dispara un integrante humano y se registra en
MLflow local, en el experimento `gcc-phase3-final` (o el del run padre indicado).
Esta enmienda sustituye la ejecución en el job y el registro inmediato en Unity Catalog
previstos arriba. El notebook `07_lockbox.py` es un envoltorio opcional, fuera del bundle.

Se conservan las siguientes salvaguardas:

- `GCC_LOCKBOX_AUTHORIZED` debe valer exactamente `1`; se verifica antes de cualquier
  lectura de datos o creación de salidas.
- Árbol Git limpio, incluidos archivos sin seguimiento no ignorados, y perfiles de
  `conf/models.yaml:final` congelados. No se busca ni selecciona con test.
- `attempt.json` se crea de forma exclusiva antes de abrir Gold; conserva timestamp,
  commit y hash SHA-256 de `conf/models.yaml`. Se registra también el hash de umbrales.
  Tanto un intento interrumpido como un `result.json` previo impiden otra ejecución.
  Antes de reservarlo se buscan runs `lockbox_evaluation` en todos los experimentos
  del tracking, incluidos eliminados; cualquier coincidencia impide el intento.
  No existe bandera para sobreescribir ni reintentar; no se deben borrar o cambiar
  las salidas para repetir la evaluación.
- Antes de leer features de test, el hash de sus identificadores ordenados debe coincidir
  con `conf/lockbox.sha256`, usando la función del split. Solo se escanean metadatos para
  esa comprobación y para validar grupos disjuntos. Con `--parent-run-id` obligatorio se exige
  un run de entrenamiento completo, finalizado, con `config_hash` y `lockbox_sha256`
  coincidentes. Esa comprobación técnica no reemplaza el gate independiente.
- Validación adversarial previa: bosque clasificador con semilla 42, cinco folds
  estratificados por partición y agrupados por `strength_group_id`, solo calidad.
  La imputación, estandarización y codificación nominal se ajustan dentro de cada fold.
  Si AUC > 0.6 o hay duplicados exactos cruzados, se aborta sin abrir etiquetas del test
  ni ajustar/evaluar regresores. Solo se publica estado y agregados adversariales.
- Casi-duplicados: pares cruzados con distancia euclídea ≤0.10 después de mediana y
  z-score ajustados únicamente en train y one-hot nominal; se excluyen los pares exactos.
  Se incluyen las doce features crudas y el tipo; nulos coincidentes cuentan como iguales
  para duplicados exactos. El umbral está fijado en `conf/thresholds.yaml`, no se modifica
  después de la apertura y el conteo no añade una condición de parada.

Si pasa el gate adversarial, se reajustan los mismos nueve perfiles de la arquitectura
(a) sobre todo train, más L1 por tipo. CV+ conserva cinco folds agrupados de calibración
solo sobre train, sin etiquetas de test. MAE, MAPE porcentual, RMSE y R² llevan IC95%
percentiles de 10 000 remuestreos de grupos íntegros, semilla 42; se conservan pesos por
fila y la máscara observada por target. ΔMAE contra L1 usa el mismo bootstrap pareado.
R² constante y MAPE sin targets no nulos son indefinidos y se serializan como `null`.
Los intervalos percentiles excluyen remuestreos indefinidos; si todos lo son, quedan `null`.
Las métricas por tipo, cobertura, monotonía antes/después y brechas test−CV son
agregados; los CSV/PNG publicados no contienen filas ni predicciones individuales.

El registro en Unity Catalog y el alias `@champion` se difieren a la fase de producto,
**sin reentrenar ni volver a abrir el lockbox**: se registrará el mismo modelo ya evaluado,
conservado como artefacto de modelo de MLflow local por familia. El resultado no habilita
por sí mismo promoción automática. El comando y la plantilla de resultados se conservan
en [Protocolo de Fase 4](../fase4/lockbox.md).
El CLI usa rutas fijas para Gold y salida. Tras el gate, el comando humano previsto es:

```bash
GCC_LOCKBOX_AUTHORIZED=1 uv run python -m gcc_strength.lockbox --tracking-uri file:./mlruns --parent-run-id d90aa4d34750448b9b608383237660d5
```
