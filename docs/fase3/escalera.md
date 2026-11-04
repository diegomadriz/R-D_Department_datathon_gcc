# Fase 3 · Etapa 2: escalera de modelos

El protocolo y la selección se fijaron antes de calcular resultados en
[`conf/models.yaml`](../../conf/models.yaml) y el [ADR 0006](../adr/0006-protocolo-cv-y-seleccion.md).
Se extiende el framework de la Etapa 1 y se conserva su presupuesto L2/L3.
La única entrada de datos reales es `gcc_strength.eda.load_train`.

## Reproducción y artefactos

```bash
uv run python -m gcc_strength.ladder
```

El comando utiliza dos procesos de cálculo y limita los hilos numéricos dentro de cada
ajuste. Reanuda resultados completos solo si coinciden configuración, código, versiones
y huella de train. Para otro protocolo, usar `--output` con un directorio nuevo.

En `outputs/fase3/stage2/` quedan:

- `manifest.json`: protocolo, versiones y procedencia de la ejecución.
- `ladder.csv` y `ladder.md`: modelo × variante × target, medias/SD entre folds,
  ΔMAE contra L1 e IC95% pareado con 10 000 remuestreos de grupos.
- `finalists.csv`: tres familias × tres targets, elegidas con la regla del ADR 0006.
- `ensemble_gate.csv`: comparación de la mezcla fija contra L1 y los tres finalistas.
- `temporal_sensitivity.csv`: brecha entre CV agrupada y sensibilidad por orden.
- `candidates/`: OOF, métricas, parámetros y cobertura del experto de A por fold.

Los resultados temporales omiten el bloque de calentamiento. Sus poblaciones difieren
de las de CV aleatoria; la brecha no mide exclusivamente el efecto del esquema.
La SD entre folds no es un error estándar de observaciones independientes.

## Alcance

La escalera compara crudas/sustitución física y exclusión/indicador/experto de A en
ElasticNet, LightGBM, GPR y Huber. L0 y L1 son referencias comunes. Solo para R1 se
compara también LightGBM con restricción positiva de Blaine; se deriva la máscara de
los nombres del diseño ajustado en cada fold. ProductType se codifica nominalmente.

El GPR utiliza Matérn ARD + WhiteKernel, optimizando longitudes por eje dentro del fold
con presupuesto fijo. Huber estima su escala del target dentro del ajuste. Ni los
targets ni los metadatos de partición son inputs. Cada variante evalúa las mismas
etiquetas observadas y particiones para un target dado.

L6 es una mezcla de miembros y pesos fijados previamente, sin selección desde OOF.
Solo se admite para un target si mejora significativamente frente a cada referencia
según el gate declarado. Una fila de L6 en la escalera no implica aceptación.

## Límites

La elección de variantes sobre CV externa tiene optimismo de selección. Sus IC son
condicionales a los ajustes OOF; no validan de manera independiente todo el procedimiento
de selección ni corrigen multiplicidad. La sensibilidad temporal usa orden de archivo,
sin fechas confirmadas. Types 4/5 siguen sin soporte suficiente para conclusiones.
El presupuesto del optimizador GPR puede truncar la convergencia.

El ADR 0005 está aceptado tras la CV. Arquitecturas multi-salida, incertidumbre, pruebas de
cordura, explicaciones y tracking pertenecen a las siguientes etapas. La revisión
independiente sigue pendiente. El lockbox permanece cerrado y sus resultados no se
conocen. Las unidades originales de las resistencias tampoco están confirmadas.

## Resultados de la ejecución completa

| target     | family           | candidate                 | MAE CV ± sd    | ΔMAE vs L1 [IC95%]        |
|:-----------|:-----------------|:--------------------------|:---------------|:--------------------------|
| Strength1  | linear           | Huber_raw_submodel        | 114.81 ± 11.08 | -44.95 [-57.16, -30.96]   |
| Strength1  | lightgbm         | L4_raw_indicator_monotone | 121.02 ± 7.85  | -38.71 [-47.75, -29.57]   |
| Strength1  | gaussian_process | L5_raw_indicator          | 107.12 ± 6.30  | -52.62 [-62.44, -42.75]   |
| Strength28 | linear           | Huber_raw_submodel        | 326.56 ± 17.78 | -84.51 [-105.49, -63.40]  |
| Strength28 | lightgbm         | L4_physical_indicator     | 322.92 ± 26.29 | -88.37 [-116.67, -58.86]  |
| Strength28 | gaussian_process | L5_raw_indicator          | 299.50 ± 17.04 | -111.65 [-135.58, -86.78] |
| Strength7  | linear           | Huber_raw_indicator       | 274.97 ± 20.22 | -29.59 [-43.89, -14.72]   |
| Strength7  | lightgbm         | L4_raw_indicator          | 269.78 ± 16.49 | -34.78 [-51.39, -17.50]   |
| Strength7  | gaussian_process | L5_raw_indicator          | 256.38 ± 13.26 | -48.19 [-62.28, -33.85]   |

Se seleccionan candidatos por familia y target; no son resultados del lockbox.

- L6 Strength1: rechazado por el gate pareado.

- L6 Strength7: rechazado por el gate pareado.

- L6 Strength28: rechazado por el gate pareado.

La CV repetida activa 0 alarmas del piso de ruido entre candidatos/target.
La tabla completa y los diagnósticos temporales quedan en los artefactos locales.

Todos los finalistas tienen R² temporal negativo en R7 y R28. Para GPR, el MAE
temporal R1/R7/R28 es 161.91/412.06/470.91, frente a 107.12/256.38/299.50 en CV
agrupada: la deriva limita la estabilidad. El calentamiento cambia la población,
por lo que la brecha no aísla únicamente el efecto de validar por orden.
