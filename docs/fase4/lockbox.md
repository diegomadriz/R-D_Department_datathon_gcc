# Fase 4: evaluación única del lockbox

Estado: **NO EJECUTADO**. Este documento no contiene resultados del test.
Protocolo: [ADR 0008, Enmienda 1](../adr/0008-protocolo-lockbox.md#enmienda-1).

Un integrante humano ejecuta la evaluación en local tras el gate independiente de
fuga y sobreajuste y con todos los cambios revisados e integrados en un árbol Git limpio.
El desarrollo y las pruebas se limitan a Gold sintético. `07_lockbox.py` no está en el bundle.

Desde la raíz del repositorio, el integrante humano ejecuta exactamente:

```bash
GCC_LOCKBOX_AUTHORIZED=1 uv run python -m gcc_strength.lockbox --tracking-uri file:./mlruns --parent-run-id d90aa4d34750448b9b608383237660d5
```

El `config_hash` es el SHA-256 de los bytes de `conf/models.yaml`, igual que en train.
Se verifican el hash del split y ambos parámetros del padre antes de abrir calidad del
test. No se cambian perfiles ni umbrales tras autorizar. Si el run auditado no coincide,
se resuelve su procedencia antes de la apertura, sin sustituir el padre por un modo rápido.
El gate humano independiente sigue siendo obligatorio aunque las comprobaciones técnicas pasen.

`attempt.json` se reserva antes de leer Gold. La presencia de ese archivo **o** de
`result.json` impide una segunda llamada, incluso tras fallo, interrupción o aborto.
Antes de reservarlo se buscan runs `lockbox_evaluation` en todos los experimentos del
tracking, incluidos eliminados: si existe alguno, se rechaza el intento. El CLI exige
el run padre y usa rutas fijas para Gold y salida; las rutas inyectables son para pruebas.
No hay override: no borrar archivos, cambiar el directorio ni copiar Gold para repetir.
Los fallos previos a la reserva (autorización, árbol sucio, configuración/padre) no abren Gold.
Un fallo posterior consume el intento y requiere documentar el incidente sin reabrir test.

Primero se mide AUC train-vs-test con CV agrupada, sin targets ni metadatos como inputs.
AUC >0.6 o un duplicado exacto cruzado produce `status: aborted`, sin métricas de resistencia.
Los casi-duplicados son pares cruzados no exactos a distancia euclídea ≤0.10 en calidad
estandarizada con train y tipo one-hot; no son una condición adicional de parada.

Si pasa, se ajustan los finalistas congelados (a) con proyección isotónica y L1 en train.
Se calculan métricas con IC95% de bootstrap de grupos (10 000, semilla 42), ΔMAE pareado,
cobertura CV+ nominal 90%, R²/MAE por tipo, monotonía y brecha test−CV frente a
`outputs/fase3/final/metrics.json`. CV+ calibra exclusivamente con train. Las advertencias
sobre deriva y cobertura nominal del ADR 0007 siguen vigentes. Un IC indefinido queda `null`.

Las salidas agregadas son `result.json`, `test_3x3.csv` y `test_3x3.png`; todas bajo
`outputs/fase4/lockbox/`, ignorado por Git. MLflow registra `lockbox_evaluation`, params,
métricas `test_*`, el padre como tag y los modelos ajustados. No se escriben predicciones
individuales. Conservar `mlruns/` y sus artefactos para registrar posteriormente el mismo
modelo en Unity Catalog y aplicar `@champion`, sin reajuste ni nueva apertura del lockbox.
El CLI imprime únicamente estado y ruta de `result.json`; MLflow puede emitir sus logs.

## Plantilla de resultados

| Campo | Resultado |
|---|---|
| Responsable y gate de auditoría | NO EJECUTADO |
| Fecha, commit y hashes | NO EJECUTADO |
| Run padre / run lockbox | NO EJECUTADO |
| Estado y AUC adversarial | NO EJECUTADO |
| Pares exactos / casi-duplicados | NO EJECUTADO |
| Tabla 3 familias × 3 targets: MAE, MAPE, RMSE, R² + IC95% | NO EJECUTADO |
| ΔMAE pareado frente a L1 + IC95% | NO EJECUTADO |
| Cobertura CV+ por familia, target y tipo | NO EJECUTADO |
| R² y MAE de Types 1/2/3 | NO EJECUTADO |
| Monotonía antes / después | NO EJECUTADO |
| Brecha test − CV | NO EJECUTADO |
| Limitaciones e incidentes | NO EJECUTADO |
| Modelo conservado para registro posterior | NO EJECUTADO |
