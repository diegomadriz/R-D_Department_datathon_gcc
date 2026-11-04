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
