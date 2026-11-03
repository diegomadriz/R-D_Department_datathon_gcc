# Fase 2: EDA y pruebas de física

Fecha: 2026-11-03. Alcance: **621 filas y 600 grupos de ensayo de train**, cargados desde Gold persistido. El lockbox no participa en ninguna tabla, figura ni decisión de este documento. Las propuestas son candidatas para CV; no hay resultados predictivos ni selección de modelo en esta fase.

Reproducción local:

```bash
uv run python -m gcc_strength.eda
```

El comando regenera [`eda_results.md`](eda_results.md) y 22 figuras PNG a 90 dpi. Se reutilizan los Parquet existentes; no se ejecuta el pipeline ni se cambia el split. La entrada del EDA lee las etiquetas de partición y solicita únicamente filas `split == 'train'` a Parquet. Verifica sus etiquetas y elimina `split`. Las funciones analíticas rechazan entradas sin esta validación o que aún contengan `split`. El notebook aplica el filtro en Spark antes de llevar los datos a pandas; su ejecución en UC queda pendiente.

## 1. Hallazgos principales

1. **La ausencia de `Physical_Metric_A` sigue siendo estructural:** faltan 467/621 valores (**75.20%**). En Type 1 no hay observaciones en los primeros cinco bloques y solo hay dos en el sexto. Su Spearman con R7 de 0.2611 usa apenas 120 parejas y representa el periodo reciente, no todo train ([tablas 02 y 20](eda_results.md#02_nulos)).
2. **La deriva produce un efecto Simpson en R1–R28 dentro de Type 1.** El Pearson sin ajuste es −0.0646 (global +0.2723), pero el Spearman parcial por bloques del orden de `_row_id` es **+0.4087 / +0.4323 / +0.4240 con 8/24/48 bloques**, sobre 493 parejas. El Spearman sin ajuste es −0.0496: el cambio no depende solo de cambiar Pearson por Spearman. La relación ajustada positiva debilita el argumento contra la arquitectura (b), que debe compararse en CV usando exclusivamente predicciones fuera de fold ([tabla 05](eda_results.md#05_correlacion_targets)).
3. **La deriva altera sustancialmente las asociaciones.** En Type 1, Blaine–R1 pasa de Spearman 0.5021 a parcial 0.1461, y A–R1 de 0.5121 a 0.1477. No interpretar las correlaciones sin controlar por orden como efectos del proceso ([tabla 04](eda_results.md#04_correlaciones)).
4. **Blaine→R1 (+) queda como candidata de CV condicionada a estabilidad.** El efecto original es pequeño: rho parcial 0.1461, IC95% marginal [0.0517, 0.2428], sin corrección por multiplicidad. Su signo positivo persiste en la sensibilidad de bloques/controles, pero esto no valida causalidad ni mejora predictiva. D y F son no concluyentes ([tablas 19–19b](eda_results.md#19b_sensibilidad_blaine)).
5. **La redundancia de A es extrema incluso excluyendo Types 4/5:** VIF(A)=426 760.65 y VIF(C)=708 044.08; sin A, el máximo numérico es 4.0268, con 558 casos completos. Con dummies de Types 1–3, el máximo es 6.1598 (D); el anterior 8.7836 incorporaba la dummy de Type 4 ([tablas 06–17d](eda_results.md#06_vif_con_A)).
6. **La forma cinética de raíz cuadrada deja sesgo en R1.** Ajustando simultáneamente las tres edades, sobreestima R1 en 98.57% de Type 1 y en 100% de Type 2, con sesgos medios de +5.92% y +4.19%. Con exponente libre por tipo, el MAPE de ajuste R1 cae de 5.96% a 2.38% en Type 1 y de 4.19% a 0.62% en Type 2. Estos MAPE quedan por debajo del piso de ruido de `conf/thresholds.yaml` (R1: 4.4%): son **sobreajuste por construcción**, con dos parámetros por muestra para tres puntos más un exponente compartido; no justifican priorizar (c) ([tabla 18](eda_results.md#18_cinetica)).
7. **La advertencia química sobre R1 depende del ajuste empleado.** Si f28 se fija al R28 observado y s se obtiene de R7, el sesgo medio de R1 es **+35.97% / +39.69%** en Types 1/2, del orden del 28–39% advertido. Al optimizar las tres edades, el ajuste sacrifica R7/R28 para reducir ese error; por eso los porcentajes no son intercambiables ([tabla 18b](eda_results.md#18b_cinetica_anclada_r7)).
8. **Los tipos raros son aún menos numerosos que la aproximación inicial:** Type 4 tiene **4 filas**, ninguna con los tres targets; Type 5 tiene **1 fila**. No se puede estimar una cinética de Type 4; el ajuste libre exacto de Type 5 tiene tres parámetros efectivos para tres puntos y no aporta evidencia de generalización ([tablas 18c y 21](eda_results.md#18c_cobertura_cinetica)).

## 2. Descriptivos, distribuciones y horizontes

La [tabla 01](eda_results.md#01_descriptivos) contiene n observado, media, desviación, mínimo, cuartiles, máximo, asimetría y curtosis de exceso, globalmente y por tipo, para las 12 features y los tres targets. `ProductType` se describe con frecuencias en la [tabla 21](eda_results.md#21_tipos); no tiene media ni correlación ordinal. Types 4/5 y las estadísticas con menos de 20 observaciones aparecen como no confiables. Medias y medianas se calculan sobre los valores disponibles de cada variable, sin rellenar nulos.

Las medianas R1/R7/R28 son 2240/5980/7460 en Type 1, 3670/7610/8540 en Type 2 y 2220/5975/7920 en Type 3. Las unidades originales siguen **NO VERIFICADAS**; no se convierten a MPa. Los 15 gráficos `distribution_*.png` reúnen histograma global de train y boxplots por tipo. Los outliers se describen sin eliminación: D tiene 95 casos IQR y 74 por z robusto; E tiene 62 y 44. Esos conteos incluyen diferencias legítimas entre productos, además de extremos dentro de tipo ([tabla 03](eda_results.md#03_outliers)).

Las tres primeras features por |Spearman| dentro de Type 1 se usan para los scatter, con su n disponible ([tabla 10](eda_results.md#10_top_features)). El orden siguiente es descriptivo, no una lista definitiva de inputs:

| Horizonte | Top 3 sin ajuste temporal (rho) | Lectura después de controlar por bloque |
|---|---|---|
| R1 | A +0.5121; Blaine +0.5021; H +0.4846 | H +0.2958 y `Chemical_Composition_B` +0.2926 destacan; Blaine conserva +0.1461. Las identidades de H y del indicador químico siguen abiertas. |
| R7 | `Physical_Metric_A` +0.2611; G −0.1861; B +0.1478 | G −0.2871 es la asociación parcial más fuerte; la métrica física A +0.2107 usa solo 120 parejas. |
| R28 | A −0.4188; Blaine −0.4142; C −0.3879 | G −0.2405 y Blaine −0.1877 destacan; A se atenúa a −0.0738 y C a −0.0561. |

Estos efectos parciales eliminan la media de los rangos por bloque, equivalentes a residualizar contra intercepto y siete indicadores. Se recalculan sobre los casos completos de cada pareja. No controlan simultáneamente por el resto de la composición ni demuestran causalidad. Blaine en R7 es aproximadamente cero, sin una anomalía física que explicar. Su asociación con R28 es sensible al ajuste y C es débil en R1; estas asociaciones no permiten confirmar identidades.

## 3. Asociaciones exploratorias de física

La [tabla 19](eda_results.md#19_fisica) usa 2000 remuestreos, semilla 42, de grupos de ensayo dentro de cada bloque. Cada remuestreo conserva juntas las filas del mismo grupo, recalcula rangos y residualización, y produce IC percentil 95%. Los bloques de igual anchura se construyen exclusivamente con el mínimo/máximo de `_row_id` de train; un grupo que cruce una frontera queda asignado al bloque de su primera fila. No son fechas comprobadas. Los IC son marginales exploratorios, sin corrección por multiplicidad; no abarcan toda dependencia serial residual.

| Hipótesis | Efecto parcial e IC95% | Veredicto |
|---|---|---|
| Blaine → R1 positivo | +0.1461 [0.0517, 0.2428], n=512, 493 grupos | Apoyo exploratorio pequeño, IC marginal sin corrección por multiplicidad |
| Blaine más asociado con R1 que con R28 | Diferencia +0.3244 [0.2251, 0.4256], n=488, 469 grupos | **apoya** |
| D → R28 negativo, bajo la hipótesis de caliza | +0.0428 [−0.0521, 0.1362], n=470 | **no concluyente** |
| F → R1 positivo, bajo la hipótesis de C₃A | −0.0391 [−0.1259, 0.0479], n=475 | **no concluyente** |
| E (C₄AF) aproximadamente neutro | R1: −0.0864 [−0.1761, 0.0051]; R7: −0.0176 [−0.1045, 0.0707]; R28: −0.0710 [−0.1601, 0.0158] | **no concluyente** en las tres edades |

Para E, «apoya» requiere que todo el IC esté dentro de la región práctica de equivalencia [−0.10, +0.10], fijada como criterio exploratorio. Que un intervalo cruce cero no basta para afirmar neutralidad. Cambiar el margen después de ver el resultado sería una nueva decisión. Como diagnóstico adicional, Blaine–R28 es negativo: −0.1877 [−0.2872, −0.0864], por lo que **no apoya** un efecto positivo a esa edad. No se propone imponer ese signo negativo: su magnitud depende del número de bloques, los controles y los casos completos. [H] En un IL intermolido, la caliza puede concentrarse en la fracción fina; la capa de hidratos, la demanda de agua y la distribución granulométrica también podrían influir. Son mecanismos plausibles, no identificados en estos datos.

D→R28 y F→R1 no son pruebas de identidad: tienen poca potencia (D tiene sd ≈0.035 dentro de Type 1, con dilución menor que el ruido), y filler/balance de sulfato impiden esperar signos inequívocos. No refutan ni confirman las identidades; su sustento procede de la prueba de dilución de F0.

La [tabla 19b](eda_results.md#19b_sensibilidad_blaine) reproduce el método del dictamen: cuantiles del rango de `_row_id` **en los casos completos de cada prueba**, dentro de Type 1; se residualizan los rangos de Blaine y del target contra dummies de bloque y los controles también en rangos. Los bloques se recalculan por prueba y pueden separar grupos de ensayo. Se distingue del diagnóstico original (tabla 19), que usa ocho intervalos de igual anchura en todo train y conserva grupos íntegros. «Ninguno» significa sin controles de composición; todas las filas controlan por bloque. Son estimaciones puntuales exploratorias, sin IC nuevos. Los controles cambian la población observada, por lo que las diferencias no aíslan solo el efecto de ajustar.

| Bloques | Controles | n R1 / R28 | Blaine→R1 | Blaine→R28 |
|---|---|---|---:|---:|
| 8 | Ninguno | 512 / 506 | +0.1786 | −0.1422 |
| 8 | D | 471 / 466 | +0.1552 | −0.2035 |
| 8 | D + Chem_A + C + G + H | 465 / 461 | +0.1322 | −0.1895 |
| 24 | Ninguno | 512 / 506 | +0.2347 | −0.0592 |
| 24 | D | 471 / 466 | +0.1644 | −0.1993 |
| 24 | D + Chem_A + C + G + H | 465 / 461 | +0.1028 | −0.2196 |
| 48 | Ninguno | 512 / 506 | +0.2491 | −0.1134 |
| 48 | D | 471 / 466 | +0.2121 | −0.1469 |
| 48 | D + Chem_A + C + G + H | 465 / 461 | +0.1435 | −0.1701 |

R1 conserva el signo positivo, con magnitud modesta y sensible a controles. R28 conserva signo negativo en esta grilla, pero su magnitud varía de −0.0592 a −0.2196: no valida una restricción causal ni un signo universal. La única restricción aprobada como **candidata de CV** es Blaine→R1 (+), condicionada a mantener estabilidad en los folds y por tipo. No imponer D→R28, F→R1, Blaine→R28, SO₃ ni H. Probar Blaine×D como ablation.

## 4. Multicolinealidad y representación candidata

Todos los VIF se recalculan **solo con Types 1–3**, intercepto, casos completos por diseño y sin imputación. Types 4/5 se tratan como grupo aparte sin soporte suficiente para validar un diseño; no participan en esta selección. El 8.7836 previo de D estaba influido por la dummy de Type 4 (cuatro filas con D≈0.93; dos completas para este diseño). No usar targets ni metadatos como inputs.

| Diseño | Inputs numéricos más `ProductType` nominal | n completo | Máximo VIF incluyendo dummies |
|---|---|---:|---:|
| Crudo | Blaine, fases B/C/D/E/F/G/H, composiciones químicas A/B | 558 | **6.1598** (D) |
| Ratios por sustitución | Blaine, fases E/G/H, químicas A/B, log(C/B), log(D/A), log(F/E) | 557 | **7.7949** (log(D/A)) |

Sin A, el máximo VIF numérico es 4.0268; con A/C es 426 760.65/708 044.08. Las [tablas 06–17d](eda_results.md#06_vif_con_A) documentan todos los diseños. El descarte iterativo es diagnóstico, no una selección final para CV. Los casos completos no son un filtro de entrenamiento.

Se implementan ocho candidatos y sus banderas: `log(C/B)`, `C/A`, `log(D/A)`, `log(F/E)`, **`log(A)`, `log(H/F)`, `F/(E+F)` y `log(H/A)`**. Las tablas 11–12 incluyen inválidos y correlaciones globales/Type 1; las 13–17d, VIF solo Types 1–3. `log(A)` representa el nivel de clínker en el diseño por sustitución; D es una alternativa de nivel. `log(H/F)` es un sustituto hipotético de SO₃/C₃A y `log(H/A)` otro balance candidato; las identidades y escalas siguen sin confirmar.

La tabla 17d añade **una candidata a la vez** al diseño de sustitución: máximos VIF 9.3714 con log(A), 8.4373 con log(H/F), 23.5473 con F/(E+F) y 8.2161 con log(H/A). El cociente F/(E+F) se propone como **reemplazo** de log(F/E), no para coexistir: la tabla muestra su redundancia al apilarlos. La unión indiscriminada llega a VIF 1514.42 (C) y 890.80 (log(A)). No se recomienda apilar todos los derivados.

Logs y cocientes positivos requieren operandos observados >0, sin epsilon. F/(E+F) admite F=0 o E=0 si ambos son no negativos y su suma >0; resuelve el cero de F que invalida log(F/E). Las transformaciones y banderas son deterministas. Las correlaciones no prueban una mejora predictiva; se comparan por CV. Los log-ratios no son invariantes a desplazamientos aditivos, y su interpretación requiere confirmar los mínimos originales.

Como ablation de CV, probar interacciones **centradas** Blaine×C, F×Chemical_Composition_A y Blaine×D, y el cuadrático centrado de Chemical_Composition_A. El centrado se aprende exclusivamente en el fold de entrenamiento dentro del Pipeline; no se introduce una media global del EDA en las features productivas. Son candidatos empíricos con las escalas disponibles, no mediciones físicas confirmadas.

`Physical_Metric_A` se excluye del diseño de referencia. Comparar en CV: exclusión; conservación con indicador de ausencia e imputación dentro del Pipeline; y un experto para filas con medición, con fallback al modelo de referencia. Los folds de comparación deben evaluar las mismas filas. Si el experto no tiene suficientes muestras en un fold temporal, se usa el fallback y se informa la cobertura. Las banderas de ratios pueden evaluarse como ablation adicional; su VIF no está incluido en los diseños numéricos de las tablas. No incorporar banderas redundantes sin revisar su dependencia.

## 5. Cinética y arquitectura multi-salida

La [tabla 18](eda_results.md#18_cinetica) informa distribución de s (media, desviación, mínimo, cuartiles, máximo), exponente, MAPE y sesgo por edad. Se ajustan las 559 muestras completas de train: 491/25/42/0/1 por Type 1/2/3/4/5. Los mínimos cuadrados son en unidades originales del target, con igual peso por observación; se optimiza también f28, que puede diferir del R28 observado. En la variante libre se comparte a por tipo y cada muestra conserva su par f28/s; hay tres inicios deterministas. Los límites de a son [0.05, 1.5]; ninguno de los ajustes reportados toca un límite.

| Tipo | a libre | Mediana s con raíz / libre | MAPE raíz R1/R7/R28 (%) | MAPE libre R1/R7/R28 (%) |
|---|---:|---|---|---|
| Type 1 | **0.6947** | 0.2750 / 0.1336 | 5.9643 / 4.0018 / 1.9688 | **2.3838 / 1.8717 / 1.0248** |
| Type 2 | **0.8501** | 0.1935 / 0.0529 | 4.1899 / 4.6256 / 2.6237 | **0.6221 / 1.0340 / 0.7074** |
| Type 3 | **0.5739** | 0.2900 / 0.2192 | 3.5546 / 2.2097 / 1.0319 | **3.4014 / 2.2222 / 1.0557** |

La variante libre mejora el error cuadrático total, pero no tiene por qué mejorar cada MAPE: Type 3 empeora ligeramente en R7/R28. s cambia de escala al cambiar a, por lo que no deben compararse ambos parámetros como si fueran la misma constante química. Type 5 no permite una estimación confiable por tipo y Type 4 carece de R1 tras las reglas de Silver.

Los MAPE libres de Types 1–3 quedan por debajo de los pisos de ruido de `conf/thresholds.yaml`: **4.4% / 2.9% / 3.2%** para R1/R7/R28 (cotas empíricas aproximadas, no mediciones de repetibilidad). Son **sobreajuste por construcción**: dos parámetros por muestra para tres puntos más un exponente por tipo; reutilizan los targets reconstruidos. No son evidencia a favor de (c). Los exponentes 0.6947/0.8501/0.5739 son razonables y reproducen aproximadamente F0 (0.67/0.84/0.55); Type 2 es el menos estable, con 25 muestras.

**Recomendación:** construir **(a) modelos independientes + proyección isotónica** como referencia; usar **(c) como curva del simulador etiquetada «interpolación»**, sin prioridad predictiva derivada de su error de ajuste. La forma fija de raíz cuadrada no se recomienda como única arquitectura. Los tres tiempos por muestra permiten detectar incompatibilidad de forma, pero no validan una curva entre edades ni fuera de [1,28] días. La arquitectura **(b)** puede compararse mediante predicciones fuera de fold en la cadena, nunca targets medidos; la relación parcial R1–R28 ≈+0.41–0.43 elimina el argumento anterior contra (b), pero no demuestra que la cadena mejore CV. Considerar también modelar **log(R1/R28) y log(R7/R28)** como salidas auxiliares junto con R28, jamás como features medidas; imponer coherencia al reconstruir y evaluar los tres targets.

Toda elección definitiva exige CV repetida por grupo y sensibilidad temporal sobre train. El exponente por tipo, los parámetros cinéticos usados como etiquetas auxiliares, la imputación y cualquier selección deben volver a estimarse exclusivamente en la porción de entrenamiento de cada fold. Los parámetros ajustados aquí son diagnósticos, nunca inputs del predictor de calidad. No reutilizar los exponentes de todo train como constantes «validadas» en CV. El uso de (c) para interpolación no implica que gane como predictor; el manejo de tipos raros requiere pooling/fallback definido y validado dentro de los folds.

## 6. Riesgos y pendientes

La [tabla 20](eda_results.md#20_deriva) y [`drift.png`](figures/drift.png) muestran, dentro de Type 1, mediana R1 de **2400 → 2230** entre bloques 1 y 8 (−7.08%), R28 **6730 → 7400** (+9.96%) y Blaine **0.7136 → −0.5924** en escala del proveedor. La evolución no es monótona: R28 alcanza 7940 en el bloque 5. No sustituir estas diferencias por una tendencia temporal causal. El bloque 8 solo tiene 38 R28 observados, frente a 57 R1; la ausencia de targets también limita la lectura del cambio.

Los casos completos de VIF y cinética seleccionan poblaciones distintas; su exclusión es descriptiva, no un filtro de entrenamiento. Types 4/5 quedan fuera de todos los VIF y requieren tratamiento aparte y fallback evaluado en CV. Faltan confirmar con GCC las identidades, unidades, mínimos originales, fechas y grupos reales. La auditoría independiente de fuga/overfitting, la CV de las propuestas y la integración Databricks no se han ejecutado. No se ha entrenado ni evaluado un predictor final, ni abierto el lockbox.
