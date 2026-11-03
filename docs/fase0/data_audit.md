# Fase 0: data audit del CSV crudo

- **Fecha:** 2026-11-03
- **Archivo:** `data/datathon_data_gcc.csv` (SHA-256 `e410cbb1…3b83`, verificado contra `data/CHECKSUMS.txt`)
- **Script reproducible:** `uv run python -m gcc_strength.audit`, que regenera [`data_audit_results.md`](data_audit_results.md) con todas las tablas. Este documento interpreta esas tablas; cada cifra se puede rastrear ahí.
- **Alcance:** calidad del dato sobre el dataset completo. Aquí **no** se analiza la relación features → targets; eso es la Fase 2 y se hace solo sobre train. Las únicas miradas a los targets son su consistencia interna, su perfil por `ProductType`, la diferencia de targets entre casi-duplicados (piso de ruido) y su deriva a lo largo del archivo.
- **Exposición declarada:** el perfil de targets por tipo y los baselines L0/L1 in-sample (§6) usan **todas** las filas, incluidas las que serán test. Esa exposición se limita a la distribución marginal de los targets por tipo, no a la relación features → targets. Ninguna cifra de este audit se usa para seleccionar features, outliers, hiperparámetros ni modelos. Todo lo que siga a la Fase 0 se calcula solo sobre train.

---

## 1. Esquema y tipos

- Son 791 filas × 16 columnas, con los nombres y el orden esperados. Todas las columnas numéricas se parsean como `float64` y no hay tokens no numéricos: los nulos son celdas vacías.
- `ProductType` trae exactamente 5 valores, todos ya canónicos (`Type 1` … `Type 5`), sin espacios ni variantes de escritura:

  | Tipo | Filas |
  |---|---|
  | Type 1 | 680 |
  | Type 3 | 66 |
  | Type 2 | 38 |
  | Type 4 | 6 |
  | Type 5 | 1 |

  Aun así, Silver debe normalizar (con `strip` y mayúsculas) para que el pipeline no dependa de este archivo en particular.
- **Hallazgo crítico: las 12 features vienen pre-escaladas por el proveedor del dato.**
  - Once columnas están en min-max [0, 1]: exactamente un 0 y un 1 por columna, salvo los empates en 0 de F (3) y H (6).
  - `Physical_Metric_B` viene estandarizada: su media es exactamente 0 (−3·10⁻¹⁶) y su desviación estándar, con ddof=0, es exactamente 1, sobre sus 769 valores no nulos.
  - El escalado se calculó sobre **todo** el dataset. Como es una transformación afín y monótona por columna, no fuga información de los targets; sí implica que el mínimo y el máximo de cada columna los fija una sola fila.
  - **No se pueden recuperar las unidades físicas**, así que no hay rangos absolutos para comparar contra normas. Esto cambia el método de identificación de variables (ver `variable_fingerprint.md`).
- Los targets **no** están escalados. Todos son múltiplos de 10 y sus rangos son R1 1600–4010, R7 2490–8210 y R28 2800–9130. Esto es consistente con **psi** redondeado a 10 psi, como reporta ASTM C109 (R28 mediano de Type 1 = 7470 psi ≈ 51.5 MPa). Las unidades quedan como hipótesis: **NO VERIFICADO** con GCC.

## 2. Nulos

| Columna | % nulos | Patrón |
|---|---|---|
| `Physical_Metric_A` | **75.7%** | **Estructural/temporal:** no existe ningún valor antes de la fila 586; a partir de ahí está presente en 192 de 205 filas, mientras que en las filas 0–585 vale nulo siempre. Además, su ausencia depende del tipo (χ², p = 4·10⁻⁷). **No es MCAR.** |
| `Mineral_Phase_A..H` | 9.0% | Las 8 fases faltan **en bloque** (Jaccard = 1.0; 71 filas sin ninguna fase y 1 fila solo sin E). La prueba χ² contra el tipo da p = 0.087. Es orientativa: Types 4 y 5 dejan frecuencias esperadas < 5. No hay evidencia de dependencia del tipo, pero eso **no demuestra** que sea MCAR. |
| `Chemical_Composition_A/B` | 3.5–3.7% | Faltan juntas (Jaccard = 0.97); su ausencia sí depende del tipo (p < 10⁻¹⁴, sobre todo en Type 4). |
| `Physical_Metric_B` | 2.8% | Depende del tipo (Type 4 y el único Type 5). |
| `Strength1` / `7` / `28` | 6.5% / 2.8% / 7.5% | Los targets **no** faltan juntos. Hay 693 filas con los 3 targets, 36 sin solo R1, 38 sin solo R28, 13 sin ninguno y 11 con otras combinaciones. R28 falta más en Type 3 (19.7%). |

- **8 filas vacías** (solo traen `ProductType`): 128, 262, 264, 300, 368, 783, 789 y 790. Explican los 5 "duplicados exactos" del archivo (ver §3).
- **Por fila:** 157 filas están completas, 488 tienen 1 nulo (casi siempre `Physical_Metric_A`) y 53 tienen 9 (las 8 fases + `Physical_Metric_A`).
- **Implicaciones:**
  - `Physical_Metric_A` falta por un cambio de protocolo o instrumento, no al azar. Imputarla con la mediana a lo largo de 586 filas en que nunca se midió sería inventar señal. Hay que comparar en CV tres opciones: excluirla, conservarla con un indicador de "medido/no medido" o usarla solo en el submodelo del periodo en que existe.
  - Las fases faltan en bloque, así que **no hay una fase única que recuperar por balance de masa** (ver §5). La imputación de fases va dentro del `Pipeline` (iterativa/KNN) o se usa un modelo que tolere nulos.
  - Para el entrenamiento, cada target usa sus filas con target no nulo. Las 13 filas sin ningún target (incluidas las 8 vacías) no aportan nada y van a cuarentena.

## 3. Duplicados y piso de ruido

- **No hay duplicados reales.**
  - Los 5 duplicados exactos son filas vacías (§2).
  - Los 4 "duplicados en features" entre filas no vacías son filas **sin ninguna feature, pero con algún target**: 169, 329, 549 y 601 (Type 1). Con la 263 (Type 2) suman 5 filas así, que no sirven para modelar y van a cuarentena (`no_features`).
  - Entre las filas con alguna feature no hay ningún duplicado.
- **Filas con features pero sin ningún target:** 220, 314, 339, 519 y 591. Van a cuarentena (`no_target`).
- **Hallazgo importante: hay 21 grupos (44 filas) que comparten exactamente el mismo trío (R1, R7, R28) con features distintas.** La regla (`strength_groups`): filas **consecutivas** que comparten al menos 2 targets no nulos y en las que todos los targets observados son iguales. Con la regla tolerante a nulos parciales se obtienen los mismos 21 grupos.
  - Todos son de filas **consecutivas** (salto máximo de 1).
  - En 17 grupos coinciden 4 o 5 features (`Physical_Metric_A/B` y `Chemical_Composition_A/B`) y difieren las fases. En 4 grupos coinciden 10 de las 12.
  - Esto parece **un mismo ensayo de resistencia asociado a varios registros de calidad**, por ejemplo varias muestras de XRD del mismo lote.
  - **Implicación de fuga:** si una fila del grupo cae en train y otra en test (o en otro fold), el modelo "ve" el target. **El split y la CV deben agrupar estos registros.**
  - El grupo es una **hipótesis**: podría haber coincidencias accidentales, aunque es poco probable que tres valores coincidan en filas contiguas. Los targets se usan aquí solo como **metadato de partición**. `strength_group_id` nunca entra en `X`.
- **Casi-duplicados:** con distancia de Chebyshev < 0.01 en las features escaladas, dentro del mismo tipo, hay 0 pares; con < 0.02 hay 2; con < 0.05 hay 160.
  - En esos 160 pares, la diferencia relativa mediana de targets es 5.3% (R1), 3.4% (R7) y 3.9% (R28). Esa diferencia mezcla el ruido del ensayo con el efecto de las features que aún difieren, así que solo es **una cota superior** del ruido.
  - Si ese ruido fuera solo del ensayo, el MAPE de un predictor perfecto rondaría **~4.4% (R1), ~2.9% (R7) y ~3.2% (R28)**.
    - La conversión supone errores normales independientes: mediana|Δ| = √2·0.6745·σ y MAE = √(2/π)·σ, así que MAE ≈ 0.8365 · mediana|Δ|.
    - En la distancia se excluye `Physical_Metric_A` y `Physical_Metric_B` se divide entre 10 para hacerla comparable con las columnas min-max.
  - **No hay réplicas verdaderas**, así que el piso de ruido empírico queda **acotado, no medido**. Referencia de norma: ASTM C109/C109M-21, tabla de precisión. El CV de un solo laboratorio es ≈ 3.1% (1 d), 3.9% (7 d) y 3.8% (28 d); el de varios laboratorios, ≈ 7.3%, 6.6% y 6.5%. Eso da un MAPE de un predictor perfecto de ≈ 2.5% / 3.1% / 3.0% (MAE = 0.798σ), del mismo orden que la cota empírica. Un MAPE muy por debajo de ~3% en R7 o R28 apuntaría a fuga. Las cifras de EN 196-1 solo vienen de una fuente secundaria: **NO VERIFICADO**.

## 4. Rangos, outliers y valores imposibles

- **Fila 619 (Type 1): registro anómalo, muy probablemente un error de captura (pendiente de confirmar con GCC).**
  - Las fases A–F y H valen 0, es decir, el mínimo de cada columna, y G vale 1, el máximo. El siguiente valor más alto de G es 0.093 y su p99 es 0.075.
  - Comparte el trío de targets con la fila 620, que sí tiene fases normales.
  - En min-max, 0 significa "el mínimo observado", no necesariamente una fase ausente. Por eso **no se puede afirmar que sea físicamente imposible**. Lo que sí se puede afirmar:
    - es la única fila con 7 fases en el mínimo a la vez;
    - G vale 13 veces su p99;
    - repite exactamente los targets de la fila vecina.
  - **Esta fila fija el min-max de 8 columnas.**
  - Propuesta para Silver: las 8 fases de la fila 619 pasan a nulo, con la bandera `flag_phases_anomalous`. La alternativa es conservarlas y solo marcarlas. Decisión pendiente (ver ADR 0002).
- **Fila 491 (Type 1):** B = 1 y E = 1 (máximos), y F = 0 y H = 0 (mínimos). Es sospechosa, pero no imposible sin unidades. Se marca y no se borra.
- **Ceros de `Mineral_Phase_H`** en las filas 241, 242, 243 y 246, que son casi consecutivas. Pueden ser "no medido" codificado como 0. Se marcan y se le pregunta a GCC.
- **Type 4 (6 filas):** `Physical_Metric_B` tiene z = 6.5–8.6, `Mineral_Phase_D` ≈ 0.95 y `Chemical_Composition_A` ≈ 0.01. Son outliers globales, pero **coherentes como producto distinto** (ver huella de variables). No se tratan como error.
- **Outliers estadísticos** (IQR 1.5×):
  - Hay muchos en D (120), E (76) y B (52).
  - Las curtosis más extremas están en G (556, por la fila 619) y en `Chemical_Composition_B` (200, por la fila 121, el único Type 5).
  - Su tratamiento (conservar, recortar o usar una pérdida robusta) se decide dentro de la CV, nunca a ojo sobre el dataset completo.
- **No se pueden aplicar límites físicos absolutos** (p. ej. "C₃S entre 40 y 75%") porque las features están escaladas. Las reglas deterministas de Silver se limitan a tres: composición degenerada (≥ 7 fases en el mínimo escalado), targets ≤ 0 y violaciones de monotonía de los targets (hoy no hay ninguna).
- **Las banderas de outlier (IQR, z robusto) no se persisten en Silver**, porque dependen de estadísticas del dataset completo y traerían información del test. Se calculan dentro del `Pipeline`, ajustadas solo con train o con el fold de entrenamiento. Las tablas de outliers de este audit son descriptivas.

## 5. Estructura composicional de las fases

- **La suma de las 8 fases escaladas no es constante** (media 3.48, desviación estándar 0.33). En unidades escaladas esto es lo esperado aunque en origen cerraran a 100%, así que la prueba correcta es buscar dependencias lineales.
- **Hallazgo: `Mineral_Phase_A` es una combinación lineal casi exacta de B y C.**
  - `A = −0.000016 + 0.538688·B + 0.817807·C`, con R² = 0.9999986 y residuo máximo 1.5·10⁻⁴ (redondeo) en 720 filas.
  - **Prueba en conteos enteros:** cada columna es un entero entre un divisor fijo (A = k/7328, B = k/3947, C = k/5993), y `k_A − k_B − k_C ∈ {−1, 0, +1}` en las 720 filas. Por lo tanto, A es la suma redondeada de B y C a la misma resolución. Esa prueba es independiente de la fila 619. El intercepto ≈ 0 y la fila 619, con las tres fases en el mínimo, son consistentes con esto. **A es una variable derivada (B + C)**, no una medición independiente. Que sean silicatos totales (C₃S + C₂S) sigue siendo una hipótesis (ver `variable_fingerprint.md`).
  - Consecuencias:
    - El VIF implícito de A es ≈ 7·10⁵ (enorme, aunque finito). Hay que quitar A (o B y C) antes de los modelos lineales y de los análisis de VIF.
    - A no aporta información nueva.
    - Reconstruir A desde B y C con los coeficientes estimados sería una imputación **aprendida**, que tendría que ir dentro del `Pipeline`. Hoy no hace falta, porque las fases faltan siempre juntas.
- **No existe otra dependencia exacta.** Sin A, el valor singular mínimo de B..H es 0.56, contra 0.0013 de la dependencia A–B–C. Por lo tanto, **las 8 fases no cierran a una constante** (no suman 100%).
  - O faltan fases (amorfo, otras menores) o algunas no son fases cristalinas.
  - **El análisis composicional de Aitchison (log-ratios) no aplica de forma directa**, porque no hay cierre verificable ni unidades.
  - **La imputación por balance de masa se descarta.** Se registrará en un ADR.

## 6. Consistencia de los targets

- **0% de violaciones de monotonía:** ninguna fila tiene R1 > R7, R7 > R28 ni R1 > R28, en total ni por tipo. La única igualdad es la fila 265 (Type 4), con R1 = R7 = 3190.
- **Fila 265 (Type 4):** R1 = R7 = 3190 psi. Probablemente el R7 se copió en el R1. Ese R1 se marca como sospechoso.
- Correlación entre targets (Pearson): R1–R7 0.70, R7–R28 0.77, **R1–R28 0.23**. La resistencia temprana explica poco de la de 28 días, y eso refuerza la necesidad de modelar los horizontes por separado (o con una cinética flexible), no con una sola escala.
- **Perfil por tipo** (medianas; es el único análisis de targets por grupo permitido aquí):

  | Tipo | n (R28) | R1 | R7 | R28 | R1/R28 | Lectura |
  |---|---|---|---|---|---|---|
  | Type 1 | 639 | 2240 | 5980 | 7470 | 0.30 | Cemento de uso general |
  | Type 2 | 35 | **3670** | 7570 | 8540 | **0.43** | Alta resistencia temprana |
  | Type 3 | 53 | 2230 | 5990 | **7900** | 0.28 | Temprana similar a Type 1, R28 mayor |
  | Type 4 | 4 | 3190* | 2955 | **3485** | — | Resistencia muy baja: no es un cemento estructural |
  | Type 5 | 1 | 1710 | 4240 | 5580 | 0.31 | Una sola fila: no modelable por sí sola |

  \* Una sola fila con R1.

- **Baseline orientativo** (L1, media por tipo, **in-sample sobre todo el dataset**; solo sirve para dimensionar, no para seleccionar):

  | Target | R² | MAE | MAPE |
  |---|---|---|---|
  | R1 | 0.66 | 161 | 7.1% |
  | R7 | 0.50 | 302 | 5.0% |
  | R28 | 0.36 | 407 | 5.6% |

  - Buena parte del R² global "gratis" viene del tipo, sobre todo en R1. El **R² dentro de cada tipo** es la métrica honesta.
  - El baseline oficial se recalculará con CV solo sobre train.

## 7. ¿El orden de las filas es temporal?

**Sí, con alta probabilidad.** Dentro de Type 1 (para no confundir con la mezcla de tipos), casi todas las variables derivan de forma fuerte y monótona a lo largo del archivo (Spearman contra número de fila). En `data_audit_results.md` §7, las medianas por bloque también son solo de Type 1:

- `Physical_Metric_B` ρ = −0.76.
- `Mineral_Phase_D` ρ = +0.62.
- `Mineral_Phase_A` ρ = −0.59.
- `Chemical_Composition_A` ρ = −0.53.
- R1 ρ = −0.53.
- R28 ρ = +0.35.
- Kruskal-Wallis entre 8 bloques: p < 10⁻²⁵ en casi todas.

Además, `Physical_Metric_A` empieza a medirse en la fila 586. Todo apunta a datos ordenados por fecha, con cambios de proceso o materia prima en el tiempo.

**Implicaciones y propuesta:**

1. El split oficial 80/20 aleatorio estratificado (como pide el reto) es válido, pero mide **interpolación** dentro del mismo periodo. Sobreestima el desempeño que tendría el modelo en producción, sobre lotes futuros.
2. **Validación temporal de sensibilidad** (solo dentro del 80% de train, sin tocar el lockbox): CV por bloques contiguos de filas (*forward-chaining*: entrenar con bloques 1..k y validar con el k+1) y reportar la brecha entre la CV aleatoria y la temporal. **Los cortes entre bloques se hacen solo entre grupos completos**, para que un mismo ensayo nunca quede a ambos lados; un test lo verificará. Si la brecha es grande, se documenta y se considera una feature de periodo, por ejemplo el número de bloque. Esto último **solo** con aprobación del equipo, porque el número de fila no es una variable de calidad.
3. Los grupos de §3 caen en filas consecutivas, así que cortar por fronteras de grupo preserva tanto la separación de grupos como el orden temporal.

## 8. Decisiones que este audit pone sobre la mesa (cada una irá en un ADR)

1. **Fila 619:** sus fases pasan a nulo con bandera, o solo se marcan. Decisión del equipo.
2. **Cuarentena con motivo:**
   - filas vacías (8);
   - filas sin features pero con algún target (5);
   - filas con features pero sin ningún target (5).
3. **Grupos de trío idéntico:** se agrupan en el split y en la CV (y, por tanto, también en el split persistido de Gold).
4. **`Physical_Metric_A`:** no se imputa con estadística global. Su tratamiento (excluir, indicador o submodelo) se decide en la CV.
5. **`Mineral_Phase_A`:** es redundante (A = B + C). Se excluye de los modelos lineales y del VIF, y queda documentada.
6. **Se descarta la imputación por balance de masa**: no hay cierre.
7. **Type 4 y Type 5** tienen muy pocas filas. Para estratificar, se fusionan en un estrato "raro" sin cambiar el `ProductType` real (regla exacta en el ADR 0002). Hay que reportar que el modelo **no es confiable** para esos tipos.
8. **CV temporal de sensibilidad**, además de la CV repetida 5×5.
9. **Banderas de outlier:** se calculan dentro del `Pipeline`, no en Silver.
10. **Orden de filas reproducible:** `_row_id` = número de línea del CSV (sin contar el encabezado), asignado en Bronze. Antes de agrupar, partir o cortar bloques se ordena por `_row_id`, porque Spark no garantiza el orden.
