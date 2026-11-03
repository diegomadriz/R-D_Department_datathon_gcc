# Preguntas para GCC y para los organizadores (borrador)

- **Fecha:** 2026-11-03 · **Estado:** borrador, falta enviarlo
- **Contexto:** surgen del data audit de la Fase 0 (`data_audit.md`). Están ordenadas por impacto en la solución.

## A. Evaluación (organizadores)

1. **Benchmark mínimo.** La rúbrica (criterio 4) habla de "alcanzar o superar el benchmark de precisión (R², MAE, MAPE) para R1, R7 y R28", pero no da cifras. ¿Cuáles son los valores por horizonte? ¿Se miden sobre un test nuestro (80/20) o sobre un conjunto oculto suyo?
2. **¿Dónde se evalúa la ejecución?** ¿Los jueces corren la solución en nuestro workspace de Databricks Free Edition o la recrean en el suyo? ¿Qué entregable esperan (repo Git, bundle, export de notebooks)?
3. **Formato de la entrega y del Q&A:** ¿video, demo en vivo o ambos? ¿Hay duración máxima? ¿Cuál es la fecha límite?
4. **¿Hay puntos extra?** La rúbrica menciona una "calificación final ajustada (base + puntos extra)". ¿Qué los otorga?

## B. Datos (GCC)

5. **Escalado de las features.** Once columnas vienen en min-max [0, 1] y `Physical_Metric_B` en z-score, calculados sobre todo el dataset.
   - ¿Pueden compartir el mínimo y el máximo (o la media y la desviación estándar) originales de cada columna, o al menos sus unidades?
   - Sin eso no podemos verificar rangos físicos ni expresar el simulador en unidades de planta.
6. **Identidad de las variables.** Nuestras hipótesis (ver `variable_fingerprint.md`):
   - `Mineral_Phase_A` = B + C (exacto en los datos): ¿son silicatos totales (C₃S + C₂S) o alita total?
   - C = C₃S, B = C₂S, D = calcita o caliza, H = sulfato, G = fase menor (cal libre o periclasa).
   - `Physical_Metric_B` = Blaine; `Physical_Metric_A` = otra medida de finura.
   - `Chemical_Composition_A` = SO₃; `Chemical_Composition_B` = álcalis, MgO o cal libre.
   - ¿Pueden confirmar o corregir estas identidades? ¿Las fases son XRD/Rietveld o cálculo de Bogue?
7. **Unidades de los targets.** Son múltiplos de 10, con medianas R1 ≈ 2250, R7 ≈ 6000 y R28 ≈ 7500. ¿Son psi (ASTM C109) u otra unidad?
8. **Significado de `Type 1`–`Type 5`.** Nuestra lectura es:
   - Type 1 = uso general.
   - Type 2 = alta resistencia temprana (≈ Tipo III).
   - Type 3 = moderado o bajo calor, o resistente a sulfatos.
   - Type 4 = cemento de albañilería o de mortero.
   - Type 5 = no determinado (una sola fila).

   ¿Es correcta? ¿Qué norma aplica (ASTM C150/C595/C91 o NMX-C-414)?
9. **Límites de especificación.** ¿Cuál es la resistencia mínima (y máxima, si existe) a 1, 7 y 28 días por tipo, y qué margen interno usa la planta? Esto alimenta las alertas del dashboard y el modelo de ROI.
10. **Orden de las filas.** Las variables derivan de forma fuerte y monótona a lo largo del archivo, y `Physical_Metric_A` solo aparece desde la fila 586.
    - ¿Las filas están en orden cronológico?
    - ¿Hubo cambios de proceso, materia prima, molienda o separador, instrumento, laboratorio o método de ensayo durante el periodo? Dentro de Type 1 sube la caliza y bajan la finura y el SO₃, y aun así R28 sube y R1 baja.
    - ¿Pueden compartir la fecha o el periodo de cada registro?
11. **Registros con resistencias repetidas.** Hay 21 grupos de filas consecutivas con el mismo trío (R1, R7, R28) y fases distintas. ¿Son varias muestras de calidad del mismo lote o ensayo de resistencia? ¿Hay un identificador de lote?
12. **Registros sospechosos.**
    - La fila 619 tiene 7 fases en 0 y `Mineral_Phase_G` en el máximo: ¿es un error de captura?
    - Los ceros de `Mineral_Phase_H` en las filas 241–246: ¿son "no medido"?
    - La fila 265 (Type 4) tiene R1 = R7 = 3190: ¿el R7 se copió en el R1?
13. **Nulos.** ¿Los nulos de `Physical_Metric_A` (76%) significan "no se medía en ese periodo"? ¿Los bloques sin las 8 fases corresponden a días sin XRD?
14. **Repetibilidad del ensayo.** ¿Qué coeficiente de variación de repetibilidad tiene su laboratorio para los cubos de mortero a 1, 7 y 28 días? Lo usaríamos como piso de error.

## C. Negocio (GCC, para el ROI)

15. Costo por tonelada de clínker y de cada adición (caliza, yeso, aditivos); factor clínker por tipo; costo de energía de molienda (kWh/t y $/kWh).
16. Costo de un lote fuera de especificación (reproceso, descuento, reclamo) y tasa histórica.
17. Días promedio en silo antes de liberar el producto y su costo de almacenamiento.
18. Factor de emisión (t CO₂/t clínker) y precio de carbono de referencia que usa GCC.
