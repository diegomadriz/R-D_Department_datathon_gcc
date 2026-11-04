# Fase 3: modelado y ejecución final sobre train

La etapa 4 se ejecutó dos veces con `uv run python -m gcc_strength.train --tracking-uri file:./mlruns`.
Se recalcularon los ajustes y diagnósticos completos en ambas ejecuciones, sin caché de OOF.
Una tercera ejecución incorporó el R² por tipo y su IC sin cambiar los campos previos.
Solo se cargó train mediante `load_train`: 621 filas, 600 grupos. El lockbox sigue cerrado.
La escalera previa y la comparación de arquitecturas se conservan en [Etapa 2](escalera.md)
y [Etapa 3](multisalida.md). Las estimaciones principales son las de esas etapas con
HPO anidada; las cifras de `train.py` son diagnóstico condicional a la selección previa.
La selección de variantes y arquitectura entre etapas sobre el mismo train introduce sesgo
optimista incluso en las estimaciones anidadas; el lockbox es la estimación honesta posterior.

En la Etapa 3, GPR (a) obtiene MAE/MAPE/R² de 106.69/4.709/0.8316,
259.65/4.449/0.5798 y 307.51/4.349/0.5614 para R1/R7/R28. La tabla completa
con HPO anidada de las tres familias está en [Etapa 3](multisalida.md).

## Configuración final

`conf/models.yaml:final` persiste las elecciones de la CV anidada de Etapa 2:
Huber crudo (submodelo de A en R1/R28, indicador en R7), LightGBM con indicador de A
(crudo y restricción positiva de Blaine en R1, crudo en R7, físico en R28), y GPR
crudo con indicador de A en las tres salidas. Huber usa epsilon=1.75; LightGBM usa
7/15/15 hojas; GPR escalas iniciales 1/3/3 y ruido inicial 0.1. El resto de los
hiperparámetros constantes se declara también en YAML. Se copia el perfil modal
entre las selecciones internas anidadas, con desempate lexicográfico; no se ajusta
otra búsqueda con las cifras finales. Huber crudo ganó a sus alternativas físicas.
La selección modal de hiperparámetros de GPR en R1 es inestable: aparece en 8 de 25 folds.

Las tres familias usan (a): modelos independientes y proyección isotónica no negativa
R1 ≤ R7 ≤ R28. Se mantiene GPR (a) como selección provisional de Etapa 3.
Los nueve modelos registrados exponen cada salida del vector proyectado; SHAP e
importancia explican sus componentes antes de la proyección. No se usan targets medidos
como entradas. Imputación, escalado, one-hot y features se ajustan dentro del Pipeline.
Con `OneHotEncoder(drop="first", handle_unknown="ignore")`, un tipo ausente en el
fold se codifica igual que la categoría base. Esta limitación conocida afecta a los
tipos raros 4/5, cuyo soporte ya se considera no confiable.

## Diagnóstico condicional: tabla 3×3

Cada celda contiene MAE / MAPE %. CV agrupada estratificada 5×5 sobre folds comunes,
semilla base 42, perfiles congelados. Las unidades de resistencia siguen sin confirmación.

| Familia | Strength1 | Strength7 | Strength28 |
|---|---:|---:|---:|
| Huber | 114.85 / 5.128 | 271.88 / 4.540 | 329.41 / 4.467 |
| LightGBM | 118.81 / 5.248 | 267.94 / 4.636 | 321.94 / 4.623 |
| GPR | 106.37 / 4.694 | 259.44 / 4.445 | 306.77 / 4.340 |

| Familia | Target | R² ± SD | MAE ± SD [IC95%] | MAPE % ± SD | RMSE ± SD | R² Type 1 | Brecha ajuste–CV |
|---|---|---:|---:|---:|---:|---:|---:|
| Huber | Strength1 | 0.6616 ± 0.3846 | 114.85 ± 12.04 [104.35, 128.85] | 5.128 ± 0.689 | 175.26 ± 78.45 | 0.5563 | 0.1989 |
| Huber | Strength7 | 0.5355 ± 0.1684 | 271.88 ± 18.38 [252.53, 292.20] | 4.540 ± 0.379 | 363.13 ± 49.65 | 0.1615 | 0.0940 |
| Huber | Strength28 | 0.5593 ± 0.1086 | 329.41 ± 22.05 [307.68, 351.43] | 4.467 ± 0.321 | 425.59 ± 33.88 | 0.3661 | 0.0580 |
| LightGBM | Strength1 | 0.7957 ± 0.0816 | 118.81 ± 7.84 [111.42, 126.46] | 5.248 ± 0.370 | 150.38 ± 9.93 | 0.4734 | 0.1019 |
| LightGBM | Strength7 | 0.5506 ± 0.0811 | 267.94 ± 20.02 [249.46, 287.81] | 4.636 ± 0.532 | 363.82 ± 40.94 | 0.2481 | 0.2687 |
| LightGBM | Strength28 | 0.5058 ± 0.0697 | 321.94 ± 25.49 [296.52, 349.04] | 4.623 ± 0.637 | 459.08 ± 71.75 | 0.4485 | 0.2987 |
| GPR | Strength1 | 0.8325 ± 0.0696 | 106.37 ± 7.87 [99.63, 113.31] | 4.694 ± 0.363 | 135.73 ± 10.42 | 0.5687 | 0.0663 |
| GPR | Strength7 | 0.5807 ± 0.1029 | 259.44 ± 23.26 [241.11, 278.67] | 4.445 ± 0.615 | 350.89 ± 54.80 | 0.2780 | 0.1781 |
| GPR | Strength28 | 0.5627 ± 0.1064 | 306.77 ± 30.27 [282.99, 331.56] | 4.340 ± 0.719 | 430.31 ± 84.17 | 0.4726 | 0.2069 |

La SD describe dispersión entre folds correlacionados, no error estándar. El IC percentil
usa 10 000 remuestreos de grupos íntegros, semilla 42 y todas sus repeticiones juntas.
El IC estima MAE ponderado por fila de las OOF; la media mostrada promedia folds sin
ponderar. La selección previa de variantes y perfiles sobre el mismo train introduce
optimismo: esta CV final y sus IC son diagnósticos condicionales, no evaluación
independiente de todo el procedimiento de selección.

### R² OOF dentro de cada tipo

Estimación agrupada por tipo sobre todas las predicciones OOF y sus repeticiones;
IC percentil al 95% con 10 000 remuestreos de grupos, semilla 42. Estos resultados
son diagnósticos condicionales a la selección previa, no HPO anidada. La estimación
agrupada difiere de la media de R² por fold de la tabla anterior.

| Familia | Target | Type 1 [IC95%] | Type 2 [IC95%] | Type 3 [IC95%] |
|---|---|---:|---:|---:|
| Huber | Strength1 | 0.562 [0.499, 0.617] | 0.261 [-0.413, 0.558] | 0.352 [-0.141, 0.593] |
| Huber | Strength7 | 0.163 [0.101, 0.217] | -0.138 [-0.533, 0.052] | 0.454 [0.149, 0.641] |
| Huber | Strength28 | 0.367 [0.288, 0.432] | 0.071 [-0.429, 0.254] | -0.018 [-0.653, 0.357] |
| LightGBM | Strength1 | 0.479 [0.407, 0.539] | 0.153 [-0.210, 0.294] | 0.373 [0.062, 0.521] |
| LightGBM | Strength7 | 0.249 [0.171, 0.319] | -0.290 [-1.129, 0.001] | 0.388 [0.094, 0.570] |
| LightGBM | Strength28 | 0.451 [0.363, 0.526] | -0.268 [-1.517, 0.193] | -0.140 [-0.600, 0.111] |
| GPR | Strength1 | 0.574 [0.512, 0.629] | 0.264 [-0.295, 0.517] | 0.531 [0.264, 0.662] |
| GPR | Strength7 | 0.279 [0.214, 0.337] | -0.146 [-0.830, 0.190] | 0.376 [0.121, 0.536] |
| GPR | Strength28 | 0.474 [0.394, 0.544] | -0.010 [-0.735, 0.347] | -0.072 [-0.464, 0.182] |

La regla `rule_within_type` pasa para las nueve combinaciones: cada R² de Type 1 es
positivo y su IC excluye cero. Los intervalos de Types 2/3 reflejan su menor soporte.

## Comparación pareada contra L1

L1 ajusta medias nominales por tipo exclusivamente en el entrenamiento de cada fold,
sobre sus etiquetas disponibles, y predice las mismas filas/folds que cada finalista.
ΔMAE = modelo − L1; IC95% pareado por grupo, 10 000 remuestreos. Orden R1 / R7 / R28.

| Familia | Strength1 | Strength7 | Strength28 |
|---|---:|---:|---:|
| Huber | -44.73 [-57.31, -30.16] | -32.14 [-45.28, -19.24] | -81.16 [-101.32, -60.41] |
| LightGBM | -40.81 [-49.37, -31.93] | -36.06 [-52.93, -18.46] | -88.45 [-117.38, -58.62] |
| GPR | -53.25 [-62.86, -43.40] | -44.50 [-59.36, -29.40] | -103.66 [-129.07, -77.52] |

## Cobertura conformal CV+ nominal 90%

Se recalcula para las tres familias. Cada entrenamiento externo contiene cinco folds
agrupados de calibración; ningún ajuste de calibración recibe sus etiquetas retenidas
ni las de validación externa. Se conservan los perfiles previamente seleccionados.
Un grupo aporta el máximo residual por target; los límites usan órdenes finitos de CV+.
No se proyectan los extremos. Coberturas por fila, porcentajes en orden R1 / R7 / R28:

| Familia | Global | Type 1 | Type 2 | Type 3 | Type 4 | Type 5 |
|---|---:|---:|---:|---:|---:|---:|
| Huber | 90.7/90.4/90.7 | 92.3/90.5/91.0 | 83.2/83.7/89.6 | 80.0/96.5/91.1 | s/d/75.0/75.0 | 0.0/0.0/0.0 |
| LightGBM | 92.8/92.0/91.7 | 93.9/93.0/93.2 | 80.0/82.2/80.7 | 89.4/95.4/92.0 | s/d/0.0/0.0 | 0.0/0.0/0.0 |
| GPR | 90.8/91.0/90.5 | 91.8/91.8/91.9 | 80.0/83.0/82.2 | 88.6/93.8/88.9 | s/d/30.0/15.0 | 0.0/0.0/0.0 |

`s/d`: sin etiquetas observadas. Types 4/5 carecen de soporte fiable: R7/R28 de Type 4
tienen cuatro filas y Type 5 una fila; R1 de Type 5 también una. No se debe interpretar
su tasa como estimación estable. Type 2 presenta subcobertura, especialmente en GPR
y LightGBM; no se aprueban bandas por tipo por alcanzar la tasa global. El CSV guarda
filas/grupos únicos, predicciones, cobertura de grupo, ancho y alarmas por segmento.
El 90% es nominal, sin garantía exacta bajo deriva, estratificación dependiente de
etiquetas y selección previa; CV+ tiene además una cota teórica general más débil.

## Permutación, aprendizaje y sensibilidad temporal

Permutación conjunta de etiquetas y nulos entre grupos de igual tamaño, B=200,
cinco folds de la primera repetición, GPR (a) seleccionado. Perfiles congelados;
no repite la selección ni HPO de etapas anteriores. Semillas derivadas de 42.
p=(1+#R²_perm≥R²_real)/201, resolución mínima 0.004975.

| Target | R² observado | R² barajado medio ± SD | p |
|---|---:|---:|---:|
| Strength1 | 0.8319 | -0.0374 ± 0.0268 | 0.004975 |
| Strength7 | 0.5616 | -0.0180 ± 0.0125 | 0.004975 |
| Strength28 | 0.5332 | -0.0198 ± 0.0136 | 0.004975 |

La nulidad puede arrojar R² negativo por sobreajuste; no se fuerza a cero. Las curvas
registradas para las nueve salidas usan 25/50/75/100% de grupos anidados del entrenamiento
externo con validación fija y modelos reajustados; se guardan MAE de ajuste y validación.
La brecha de R² ajuste–CV está en la tabla. Ningún MAPE medio final cae bajo su piso de ruido.
Los pisos de ruido se estimaron con el dataset completo, incluidos targets del lockbox;
solo sirven como alarma y no intervienen en la selección ni el ajuste.
La proyección garantiza monotonía en todas las OOF y al servir el predictor.

Sensibilidad temporal: cinco cortes forward-chaining solo en fronteras de grupo y orden
`_row_id`; el primer bloque es calentamiento. Diferencia MAE temporal − CV aleatoria:

| Familia | Strength1 | Strength7 | Strength28 |
|---|---:|---:|---:|
| Huber | +26.31 | +128.25 | +140.45 |
| LightGBM | +35.75 | +117.56 | +176.37 |
| GPR | +47.75 | +152.36 | +168.43 |

Las poblaciones difieren por el calentamiento temporal y no hay fechas confirmadas.
La brecha es sensibilidad al orden y posible deriva, sin identificación causal.
El lockbox se asignó aleatoriamente por grupo: medirá interpolación dentro del periodo,
no desempeño futuro. La CV temporal, con brechas GPR de +48/+152/+168 MAE,
es la mejor guía disponible para ese caso, con las salvedades de población y fechas.

## Explicación y consistencia física

SHAP global y local usa TreeExplainer en LightGBM y permutación acotada en Huber/GPR,
32 filas de train y fondo de hasta 16, semilla 42. Se explican experto y fallback por
separado donde corresponde. Son asociaciones descriptivas sobre modelos ajustados a
train, antes de proyectar, no explicaciones OOF ni efectos causales.
Por tanto, los SHAP son in-sample y descriptivos.

Signo comprobable provisional Blaine→R1 (correlación valor–SHAP):

| Familia | Componente | rho | Estado |
|---|---|---:|---|
| Huber | expert | 0.9992 | coherente |
| Huber | fallback | 0.9994 | coherente |
| LightGBM | model | 0.9215 | coherente |
| GPR | model | 0.9868 | coherente |

La identidad de Blaine es provisional; cualquier discrepancia puede reflejar entradas
correlacionadas, interacciones, deriva o la muestra limitada. Las fases F/D y demás
features no tienen un signo físico identificado de forma suficiente: su estado queda
indeterminado. Los CSV `physics.csv` registran esas limitaciones por variable.

## MLflow y reproducibilidad

- Primera ejecución completa: `54c595149c9f486dadba8f541b9e068a`, 483.6 s.
- Segunda ejecución completa: `77c0ee126bc04bba89f5808858bdf3f0`, 482.0 s.
- Tercera ejecución completa: `d90aa4d34750448b9b608383237660d5`, 489.3 s.
- La tercera preservó todos los campos anteriores de `metrics.json` y agregó los R² e IC
  por tipo y la regla. Los indicadores de la regla se registraron después en los nueve hijos:
  una corrección posterior del código incorpora `numpy.bool_` al registro numérico. El hash
  del manifiesto corresponde al código ejecutado, anterior a esa corrección de logging.
- Presupuesto íntegro en ambas: CV 5×5, CV+ con cinco folds de calibración y B=200.
- `metrics.json` redondea a ocho decimales: comparación byte a byte idéntica, diff vacío.
  No se incluyen IDs ni tiempos en ese archivo; quedan separados en `run.json`.
- Los manifiestos guardan hashes de código, configuración y train, commit y hash del lockbox.
  `git_commit` identifica la base del checkout; los hashes de código identifican estos cambios
  locales todavía sin commit. No se ha hecho commit ni push.
- Cada ejecución tiene 11 runs: padre, nueve hijos familia×target y un hijo multisalida.
  Todos los modelos tienen firma, ejemplo de cinco filas de calidad y dataset de train.
  Params obligatorios incluyen tamaños declarados 621/152, seed 42 y procedencia.
- Artefactos: OOF real–predicho, residuos globales/por tipo, importancia, SHAP global/local,
  curva de aprendizaje y tabla CSV/PNG. El padre guarda métricas, manifiesto y diagnósticos.
- `mlflow_audit --tracking-uri file:./mlruns` sobre el último padre: **11 runs, 0 faltantes**.
  La auditoría también pasa para el primero. Verifica completitud, no fuga u overfitting.
- `uv run pytest -q`: **237 passed**; `uv run ruff check` y `uv run ruff format --check`: verdes.
  El test de integración ejecuta `--quick` dos veces con tracking temporal, compara métricas,
  audita los runs y recarga el predictor para comprobar sus tres salidas monótonas.

Artefactos locales: `outputs/fase3/final/` y `mlruns/` (fuera de git). El notebook
`06_train.py` es un envoltorio delgado y no se incorporó al bundle.

## Lo que queda sin conocer

Las métricas y la cobertura del lockbox siguen sin conocerse. No se ha abierto ni evaluado.
Sigue pendiente la revisión independiente de fuga/sobreajuste y la aprobación del predictor.
La subcobertura por tipo, la escasez de Types 4/5, el optimismo de selección y la sensibilidad
al orden impiden presentar estas cifras como garantía de operación futura.
