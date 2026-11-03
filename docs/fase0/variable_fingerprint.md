# Fase 0: huella de las variables anónimas

- **Fecha:** 2026-11-03 · **Estado:** revisado por química del cemento (aprobado con ajustes, ya incorporados). Falta la confirmación de GCC.
- **Fuente de la evidencia:** [`data_audit_results.md`](data_audit_results.md), generado con `uv run python -m gcc_strength.audit` (§5 estructura de fases, §8 perfil de features por tipo, §9 correlaciones). Toda la evidencia sale de la resolución de las columnas, de correlaciones **entre features** y de perfiles de features por `ProductType`. Los targets solo aparecen como medianas por tipo. No se usó la relación features → targets.

## Restricción que condiciona todo

Las 12 features vienen **pre-escaladas**: 11 en min-max [0, 1] y `Physical_Metric_B` en z-score. Por eso **no podemos comparar rangos absolutos** contra valores típicos. La identificación se basa en cuatro cosas:

1. **Resolución de cada columna.** Cada columna min-max es exactamente un entero dividido entre un divisor fijo (p. ej. A = k/7328, B = k/3947, C = k/5993, con error < 2·10⁻¹² en las 720 filas). El divisor es el número de pasos de redondeo del instrumento entre el mínimo y el máximo. Con una resolución supuesta (p. ej. 0.01%), eso da el rango original aproximado.
2. Estructura algebraica: dependencias exactas.
3. Signos de las correlaciones entre features. **Ojo:** las correlaciones globales están infladas por la deriva temporal y por la mezcla de tipos. Las que se usan abajo se recalcularon dentro de Type 1 y sin tendencia.
4. El perfil relativo por tipo de producto, comparado contra lo que la química predice para cada clase de cemento.

**Hasta que GCC lo confirme, toda identidad es una hipótesis.** Las features derivadas y las restricciones monótonas que dependan de ellas se activan solo si los datos de train las apoyan (Fase 2).

## Tabla de hipótesis

| Variable | Hipótesis de identidad | Evidencia | Confianza |
|---|---|---|---|
| `Mineral_Phase_A` | **Variable derivada: B + C** = silicatos totales (C₃S + C₂S). Alternativa: alita total (M1 + M3) | **Demostrado en conteos enteros:** con `k_A = 7328·A`, `k_B = 3947·B` y `k_C = 5993·C`, se cumple `k_A − k_B − k_C ∈ {−1, 0, +1}` en las 720 filas (540 / 91 / 89). Es decir, A es la suma redondeada de B y C a la misma resolución. Los coeficientes 0.5387 y 0.8178 son cocientes de rangos (3947/7328 y 5993/7328). B sube con G y con `Chemical_Composition_B`, y C baja, que es la firma de una mala cocción y apoya la lectura C₂S/C₃S. Se descartan C₃A cúbico + ortorrómbico y la suma de sulfatos | **Alta** que A = B + C; media–alta que son silicatos totales |
| `Mineral_Phase_C` | C₃S (alita) | Es el mayor componente de A (~74% de A en promedio). Máxima en Type 2 y mínima en Type 4 (diluido). Dentro de Type 1, sin tendencia, ρ(B, C) = −0.72 | Media–alta |
| `Mineral_Phase_B` | C₂S (belita) | Es el *trade-off* alita–belita: ρ(B, C) = −0.72 dentro de Type 1. Sube con G (cal libre), como en un clínker mal cocido | Media–alta |
| `Mineral_Phase_D` | **Calcita / caliza** (diluyente) | ≈ 0.95 en Type 4 y ≈ 0.06–0.08 en Types 2 y 3. Correlación −0.59 con A. Hay dos rutas independientes de las unidades (dilución de A y resolución de 0.01%) y coinciden: ≈ 40% de caliza en Type 4, ≈ 9–10% en Type 1 y ≈ 3% en Types 2/3 | **Alta–media** |
| `Mineral_Phase_F` | **C₃A** | Máxima en Type 2 (Tipo III) y mínima en Type 3. Tiene ceros reales (filas 312 y 491), imposibles para C₄AF en un cemento gris. Su CV implícito (34%) no corresponde a una fase principal estable | Baja–media |
| `Mineral_Phase_E` | **C₄AF** (ferrita) | Dentro de Type 1, sin tendencia, ρ(E, F) = −0.40, el *trade-off* aluminato–ferrita esperado por el módulo de alúmina | Baja–media |
| `Mineral_Phase_G` | **Cal libre** (más probable que periclasa) | Sube con B, como en un clínker mal cocido. El aspecto de "fase muy menor" (mediana de 0.025) es un artefacto: la fila 619 fija su máximo (el siguiente valor es 0.093) | Baja–media |
| `Mineral_Phase_H` | Fase de sulfato (yeso, basanita o anhidrita) | Sin tendencia y dentro de Type 1, su relación con la finura y con `Chemical_Composition_A` se debilita (de 0.52 a 0.18 y de 0.41 a 0.17). Alta en Type 2 y Type 5. Ceros en filas casi consecutivas (241–246): ¿no medido? | Baja |
| `Physical_Metric_B` | **Finura Blaine** | Su paso en z (0.02252) es consistente con un redondeo a 10 cm²/g y una σ ≈ 444 cm²/g. Si Type 1 ≈ 4500 cm²/g, quedan Type 2 ≈ 5600, Type 3 ≈ 4000 y Type 4 ≈ 7900, coherente con III > IL > I/II y con un cemento de albañilería muy fino | **Media–alta** |
| `Physical_Metric_A` | Segunda medida de finura (p. ej. % que pasa 45 µm o un percentil de granulometría láser); **no** residuo, porque correlaciona **positivo** con Blaine | Solo existe desde la fila 586 (instrumento o protocolo nuevo). Type 2 0.99, Type 3 0.33 | Baja |
| `Chemical_Composition_A` | **SO₃** | Con una resolución de 0.01% y el factor de dilución de Type 4: ≈ 3.3% (Type 1), ≈ 3.5% (Type 2) y ≈ 2.1% (Type 4). SiO₂, CaO y Al₂O₃ no cuadran con la magnitud de la dilución, y la LOI queda descartada | Media |
| `Chemical_Composition_B` | Indicador menor: residuo insoluble, álcalis (Na₂O eq), MgO o cal libre por titulación | Solo 35 valores únicos (resolución gruesa). Máxima en el único Type 5 (consistente con residuo insoluble en un cemento con puzolana) y ≈ 0.03 en Type 4. **Descarta LOI**. Se desconoce su mínimo original, así que no se puede usar en cocientes | Baja |

## Features físicas viables con datos escalados (Fase 2, solo train)

Con min-max, un cociente solo tiene sentido si el mínimo original es ≈ 0, y con Blaine en z-score un producto no tiene significado físico. Por lo tanto:

- **Viables (invariantes a escala):**
  - `log(C/B)` (C₃S/C₂S);
  - `C/A` (fracción de alita en los silicatos);
  - `log(D/A)` (diluyente frente a clínker);
  - `log(F/E)` (C₃A/C₄AF).

  Todas suponen que el mínimo original es 0, lo cual es probable porque la fila 619 tiene ceros, pero no está demostrado.
- **No viables:** SO₃/C₃A, porque no conocemos el mínimo de `Chemical_Composition_A`, y "finura × C₃S", porque Blaine está en z-score.
- **Restricciones monótonas:** ninguna por ahora. Las candidatas a evaluar en la Fase 2 son Blaine → R1 (+) y D → R28 (−).

## Pruebas discriminantes propuestas (Fase 2, solo con train)

1. **D = caliza:** si es diluyente, debe reducir sobre todo R28 dentro de Type 1. Se prueba con dependencias parciales.
2. **F = C₃A:** esperamos un efecto positivo en R1 y un posible óptimo en interacción con el sulfato (H/`Chemical_Composition_A`). E (C₄AF) debería ser casi neutro.
3. **`Physical_Metric_B` = Blaine:** esperamos un efecto positivo, mayor en R1 que en R28.
4. **Fila 265 (Type 4):** tiene R1 = R7 = 3190 psi, probablemente porque el R7 se copió en el R1. Ese R1 se marca como sospechoso (ver ADR 0002).

## Deriva temporal (lectura química)

No es una *transición* hacia cemento portland-caliza: el Type 1 ya tiene caliza de nivel IL (≈ 8–9%) desde el primer bloque. A lo largo del archivo hay **más caliza dentro del IL, menos Blaine y menos SO₃**. El aumento de R28 **no** se explica con esas tres variables y queda abierto. Hipótesis:

- un óptimo de sulfatos que depende de la edad;
- un cambio de molienda o de separador (distribución de tamaño de partícula);
- un cambio en el método o laboratorio de ensayo.

Se le pregunta a GCC.

## Mapeo probable de `ProductType`

**Advertencia:** la numeración `Type 1–5` del dataset **no es** la numeración romana de ASTM C150: "Type 2" se comporta como un Tipo III ASTM. Los perfiles de resistencia son medianas sobre el dataset completo (es el único análisis de targets por grupo permitido en la Fase 0), en **psi**. Las edades 1/7/28 días siguen la práctica ASTM. Las equivalencias NMX-C-414 no se usan, porque esa norma especifica 3 y 28 días con otro método de ensayo.

| Tipo | n | Resistencia (mediana) | Perfil de features | Hipótesis | Confianza |
|---|---|---|---|---|---|
| Type 1 | 680 | R1 2240, R7 5980, R28 7470 psi (≈ 15/41/52 MPa) | ≈ 9–10% de caliza, Blaine de referencia | **ASTM C595 Tipo IL** (portland-caliza; caliza 5–15%) | Alta–media |
| Type 2 | 38 | **R1 3670** (≈ 25 MPa), R28 8540 (≈ 59 MPa); R1/R28 = 0.43 | El más fino (Blaine ≈ 5600), más C₃S y C₃A, ≈ 3% de caliza | **ASTM C150 Tipo III** (alta resistencia temprana) | Alta–media |
| Type 3 | 66 | R1 2230 ≈ Type 1, **R28 7900** (> Type 1) | Más grueso (Blaine ≈ 4000), ≈ 3% de caliza, C₃A mínimo | **ASTM C150 no-III**: I/II, II(MH) o V (C₃A bajo, molienda más gruesa, mayor ganancia tardía) | Media |
| Type 4 | 6 | R7 2955, **R28 3485 psi** | Finura extrema (≈ 7900), ≈ 40% de caliza, SO₃ bajo | **Cemento de albañilería ASTM C91 Tipo M**. Mínimos de C91 a 7/28 días: N 500/900, S 1300/2100, M 1800/2900 psi | Media |
| Type 5 | 1 | R1 1710, R7 4240, R28 5580 psi (≈ 38 MPa) | H y `Chemical_Composition_B` en su máximo; sin Blaine | Hipótesis: **ASTM C595 Tipo IP** (portland-puzolana) | Baja |

**Implicaciones de modelado:**

- Type 4 (n = 6) y Type 5 (n = 1) **no tienen datos suficientes** para que el modelo aprenda su comportamiento. Se reporta así y el simulador avisa cuando se elige uno de esos tipos.
- `ProductType` se trata como categórico con interacción, no como un desplazamiento aditivo: Type 4 tiene un R28 absoluto 2–2.5 veces menor.
- **Cinética** `f(t) = f28·exp(s·(1 − √(28/t)))`:
  - El orden de los tipos es coherente con la química (s₇ ≈ 0.12 en Type 2, 0.22 en Type 1 y 0.28 en Type 3: Tipo III endurece más rápido).
  - Pero con la forma de raíz cuadrada fija, un solo `s` **sobreestima R1 entre 28% y 39%** en Types 1 y 2. Para el modelo cinético (c), el exponente debe quedar libre por tipo, y la curva no debe usarse para inferir R1.

## Pendiente

- Confirmación de GCC: ver `preguntas_organizadores.md`.
- Verificar las cifras de EN 196-1 en la norma (hoy solo se tiene una fuente secundaria).
