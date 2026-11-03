# ADR 0005: Features físicas y preprocesamiento para la Fase 3

- **Estado:** Propuesto; revisión química aprobada con ajustes incorporados
- **Fecha:** 2026-11-03
- **Relacionados:** ADR 0002 y ADR 0003
- **Evidencia:** [interpretación Fase 2](../fase2/eda.md) y [tablas reproducibles de train](../fase2/eda_results.md)

## Contexto

Gold contiene 621 filas de train y un split persistido que no se modifica. Las variables originales están pre-escaladas y sus identidades físicas no están confirmadas por GCC. A es redundante con B/C; las fases no tienen cierre comprobado. Hay deriva de orden, grupos de ensayo y tipos con muy pocos datos. Los análisis de esta fase generan propuestas; no eligen un predictor por desempeño.

## Propuesta

### 1. Dos representaciones candidatas, sin targets como inputs

En ambos diseños `ProductType` es nominal: one-hot dentro del Pipeline para modelos lineales, o tratamiento categórico nativo compatible con el estimador. Ningún diseño recibe `Strength1/7/28`, parámetros cinéticos derivados de ellos, `_row_id`, bloque, `split`, `strength_group_id` ni `flag_r1_suspect` como features.

**Referencia cruda:** `Physical_Metric_B`; `Mineral_Phase_B/C/D/E/F/G/H`; `Chemical_Composition_A/B`; y `ProductType`. VIF máximo numérico 4.0268, o **6.1598 incluyendo dummies**, n=558, solo Types 1–3, casos completos ([tablas 07 y 17b](../fase2/eda_results.md#17b_vif_crudo_con_tipo)).

**Representación física por sustitución:** `Physical_Metric_B`; `Mineral_Phase_E/G/H`; `Chemical_Composition_A/B`; `log_C_over_B`, `log_D_over_A`, `log_F_over_E`; y `ProductType`. Sustituye B/C/D/F por relaciones que resumen sus balances, conservando E como nivel de ferrita. VIF máximo **7.7949 incluyendo dummies**, n=557, solo Types 1–3, ([tabla 17c](../fase2/eda_results.md#17c_vif_ratios_sustitucion)).

La biblioteca de derivados incluye `C_over_A` como alternativa a `log_C_over_B`, y agrega las candidatas **`log_A` (log(A)), `log_H_over_F` (log(H/F)), `F_over_E_plus_F` (F/(E+F)) y `log_H_over_A` (log(H/A))**. `log(A)` aporta nivel de clínker al diseño por sustitución; D es una alternativa de nivel. `log(H/F)` es un sustituto hipotético de SO₃/C₃A; las identidades/escalas no están confirmadas. F/(E+F) reemplaza log(F/E) para admitir ceros aislados, sin apilar ambos.

Se implementan sus banderas, correlaciones y VIF en el EDA ([tablas 11–17d](../fase2/eda_results.md#17d_vif_nuevas_candidatas)). Añadir individualmente log(A), log(H/F), F/(E+F) o log(H/A) al diseño de sustitución da máximos VIF **9.3714 / 8.4373 / 23.5473 / 8.2161**. El tercero revela redundancia al coexistir con log(F/E), por lo que debe evaluarse como reemplazo. No adoptar todos los derivados por su correlación exploratoria.

Como **ablation**, evaluar interacciones centradas **Blaine×C**, **F×Chemical_Composition_A** y **Blaine×D**, y un término cuadrático centrado de **Chemical_Composition_A**. Aprender las medias de centrado exclusivamente en cada fold de entrenamiento dentro del Pipeline; este EDA no fija medias globales para producción. Con las escalas disponibles son candidatos estadísticos, sin asignarles unidades o identidades físicas confirmadas.

### 2. Exclusiones y regla de VIF

- Excluir `Mineral_Phase_A` como input de los modelos lineales: VIF **426 760.65**; C llega a **708 044.08** al coexistir con A/B. A puede usarse como operando observado de ratios deterministas y en la candidata log(A) de sustitución, sin agregar A cruda como input independiente.
- No introducir todas las crudas y todos los ratios a la vez. La unión ampliada de crudas y derivados alcanza VIF 1514.42 (C) y 890.80 (log(A)); el descarte iterativo es diagnóstico y no se adopta como diseño final.
- La regla propuesta es **VIF < 10 para todas las columnas del diseño lineal completo**, incluyendo dummies con categoría de referencia e intercepto. Los dos diseños de referencia cumplen en sus casos completos de train de Types 1–3; las ablations requieren su propia comprobación. Recalcular dentro del fold de entrenamiento; no fijar selección aprendida a partir de validación.
- VIF sobre casos completos no justifica entrenar solo con ellos. La imputación/indicadores cambian el diseño y requieren revisar singularidades dentro del fold; una constante o dummy ausente se maneja como tal. Todos los VIF excluyen Types 4/5: se tratan como grupo aparte, sin estimación estable de multicolinealidad ni validación por estas tablas. El anterior 8.7836 de D estaba influido por la dummy de Type 4 (cuatro filas con D≈0.93, dos completas para ese diseño); en Types 1–3 baja a 6.1598.
- Para árboles, conservar B/C/D/F y explorar ratios como ablation por CV; no eliminar señal solo por un VIF que diagnostica el diseño lineal. A en árboles sigue siendo una alternativa redundante prevista en ADR 0002, nunca una incorporación automática.

### 3. Nulos, ceros y `Physical_Metric_A`

La referencia **excluye `Physical_Metric_A`** por 75.20% de nulos y cobertura concentrada al final del archivo. Comparar en CV tres variantes con exactamente las mismas filas de evaluación:

1. Exclusión.
2. Valor observado + indicador `physical_metric_a_missing`, con imputación ajustada dentro del Pipeline/fold.
3. Experto para filas con medición y fallback al modelo sin A. Informar cobertura, n de entrenamiento del experto y desempeño en la población común; si no puede ajustarse en un fold, usar el fallback.

No imponer una mediana aprendida global ni usar el indicador como proxy temporal sin medir su estabilidad en la sensibilidad por bloques.

Los logs y cocientes positivos se calculan solo con operandos positivos observados. Cero, negativo o faltante → NaN y `flag_invalid_<ratio>`; no hay epsilon. Excepción explícita: F/(E+F) admite E=0 o F=0 si ambos son no negativos observados y E+F>0; suma cero, negativos o faltantes se marcan inválidos. Son transformaciones deterministas que deberán formar parte del Pipeline predictivo. La ausencia total de fases no se imputa por balance de masa. Los indicadores son candidatos de ablation: evitar introducir banderas idénticas/redundantes sin revisar el diseño. Escaladores, medianas, KNN o imputación iterativa se ajustan dentro del Pipeline y solo con el fold de entrenamiento. Comparar contra tolerancia nativa a nulos cuando el modelo la permita.

Los log-ratios solo conservan asociaciones ante cambios multiplicativos de escala; no corrigen el desplazamiento de un min-max con mínimo original distinto de cero. Su sentido físico permanece condicionado a esa confirmación. log(H/F) y las interacciones propuestas se evalúan como ablations con las escalas disponibles, sin afirmar que midan SO₃/C₃A o superficie reactiva física.

### 4. Restricciones monótonas candidatas

Queda aprobada **solo para R1 y como candidata de CV** la restricción **`Physical_Metric_B` → `Strength1` positiva**, condicionada a estabilidad en los folds y por tipo. El diagnóstico original tiene rho +0.1461, IC95% marginal [0.0517, 0.2428]: efecto pequeño sin corrección por multiplicidad. La [tabla 19b de sensibilidad](../fase2/eda_results.md#19b_sensibilidad_blaine) usa cuantiles del rango de `_row_id` por casos completos (8/24/48 bloques), distintos de los intervalos originales con grupos íntegros: sin controles adicionales da **+0.1786 / +0.2347 / +0.2491**; con D, **+0.1552 / +0.1644 / +0.2121**; con D + Chem_A + C + G + H, **+0.1322 / +0.1028 / +0.1435**. Conserva signo, con magnitud modesta y sensible a controles/población. Comparar con la versión sin restricción en CV; no extrapolar a Types 4/5 ni a las tres salidas cinéticas.

No aprobar restricciones D→R28, F→R1, Blaine→R28, SO₃ ni H. D→R28 (IC [−0.0521, 0.1362]) y F→R1 (IC [−0.1259, 0.0479]) son no concluyentes, con poca potencia: D tiene sd≈0.035 en Type 1 y su dilución es menor que el ruido. Filler y balance de sulfato tampoco permiten signos inequívocos. Estas pruebas no confirman ni refutan identidades, sustentadas en la prueba de dilución de F0. E no alcanza equivalencia práctica; G/H y química B no identifican especies.

Blaine→R28 varía entre −0.0592 y −0.2196 en la tabla 19b: conserva signo en esta grilla, pero no demuestra un signo causal robusto. [H] La caliza puede concentrarse en la fracción fina de un IL intermolido; también podrían intervenir capa de hidratos, demanda de agua o distribución granulométrica. Son hipótesis sin verificación en estos datos. Blaine en R7 es aproximadamente cero.

### 5. Forma cinética y arquitectura

Para (c), como curva del simulador etiquetada **«interpolación»**, considerar `f(t)=f28·exp(s·(1−(28/t)^a))`, con **a compartido por tipo y estimado dentro del fold**, no la raíz cuadrada fija. Diagnóstico de train: a=0.6947/0.8501/0.5739 en Types 1/2/3; f28>0, s≥0 y a>0 garantizan crecimiento para t>0. Mantener los límites exploratorios 0.05≤a≤1.5 y 0≤s≤5 como propuesta a validar, sin asignarles significado químico confirmado.

La raíz cuadrada ajustada a las tres edades sesga R1 +5.92%/+4.19% en Types 1/2. Si se ancla en R7/R28, los sesgos son +35.97%/+39.69%; son diagnósticos distintos, no métricas predictivas. La variante libre reduce MAPE in-sample de R1 a 2.38%/0.62% y R28 de Type 2 a 0.71%. Los errores libres de Types 1–3 quedan por debajo del piso de ruido de `conf/thresholds.yaml` (R1/R7/R28: **4.4%/2.9%/3.2%**, cotas empíricas aproximadas). Son **sobreajuste por construcción**: dos parámetros por muestra para tres puntos más un exponente compartido, reutilizando los targets que se reconstruyen. No aportan evidencia para priorizar (c). Los exponentes son razonables y cercanos a F0 (0.67/0.84/0.55); Type 2 es el menos estable, con 25 muestras.

Construir **(a) modelos independientes + proyección isotónica como referencia**; reservar (c) para interpolación del simulador, sin prioridad predictiva por su ajuste in-sample. Comparar (b) con predicciones fuera de fold y nunca targets medidos en la cadena. El Pearson R1–R28 de −0.0646 dentro de Type 1 refleja un efecto Simpson de la deriva: el Spearman parcial por bloques es **+0.4087 / +0.4323 / +0.4240 con 8/24/48 bloques y 493 parejas** ([tabla 05](../fase2/eda_results.md#05_correlacion_targets)); el Spearman sin ajustar es −0.0496. Esto debilita el argumento contra (b), sin demostrar ventaja predictiva. Considerar modelar **log(R1/R28) y log(R7/R28)** como salidas auxiliares junto con R28, reconstruyendo predicciones coherentes; nunca como features medidas ni usando targets de una edad para predecir otra. Los parámetros ajustados desde targets son etiquetas de la candidata cinética, jamás features del predictor principal. El exponente y las etiquetas cinéticas deben estimarse sin usar los targets de validación para ajustar la forma compartida. Reportar errores en los tres targets, no solo en parámetros.

No fijar exponentes independientes de Types 4/5: Type 4 tiene 0 muestras completas y Type 5 solo 1. Definir pooling/fallback dentro del entrenamiento de cada fold y declarar la falta de soporte. No extrapolar curvas fuera de las edades observadas como si estuvieran verificadas.

### 6. Criterio de aceptación en la Fase 3

Comparar diseños y arquitecturas con CV repetida 5×5 agrupada por `strength_group_id`, y sensibilidad temporal con fronteras de grupo dentro de train. Estimar cualquier preprocesamiento/selección por fold; usar CV anidada para búsquedas. Semilla 42 en toda aleatoriedad. El ganador predictivo lo determina CV; el uso de (c) como interpolación del simulador no determina la arquitectura predictiva ni valida extrapolación. No cambiar el split ni usar resultados del lockbox para estas decisiones.

## Consecuencias y pendientes

Se obtienen dos diseños lineales interpretables sin multicolinealidad extrema, con la imputación aún por comparar y sin prometer mejoras de precisión. Se conserva la trazabilidad de la física que apoya los datos y de las hipótesis no concluyentes. Quedan pendientes confirmación de identidades/unidades/mínimos/fechas, auditoría independiente de fuga y overfitting, CV de todas las propuestas e integración en UC. Esta propuesta no agrega el notebook al bundle ni aprueba un predictor final.
