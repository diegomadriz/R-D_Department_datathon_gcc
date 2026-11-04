# Etapa 3: multisalida, incertidumbre y cordura

Resultados sobre train exclusivamente. Arquitectura provisional, pendiente de revisión independiente y de la Etapa 4. El lockbox permanece cerrado.

Reproducción: `uv run python -m gcc_strength.stage3 --workers 3`. El manifiesto impide mezclar ejecuciones con código, configuración, finalistas, versiones o train distintos.

## Comparación de arquitecturas

Se congelan los tres finalistas por target de Etapa 2. Se recalculan folds comunes para todas las salidas sobre train; cada salida ajusta solo sus etiquetas disponibles. Estas OOF no son pares con las de Etapa 2, cuyos folds eran específicos por target. Ver [ADR 0007](../adr/0007-arquitectura-multisalida-e-incertidumbre.md) para el protocolo previo a resultados.

(a) independientes + proyección isotónica; (b) cadena con prefijos OOF anidados; (c) etiquetas log(R1/R28), log(R7/R28), R28 y reconstrucción coherente. Todas las arquitecturas cumplen monotonía al 100% en OOF. Unidades de resistencia sin confirmar.

| family           | architecture   | target     |   mae_mean |   mae_sd |   r2_mean |   mape_pct_mean |   rmse_mean |   r2_type1_mean |   mae_type1_mean |   overfit_gap_mean |
|:-----------------|:---------------|:-----------|-----------:|---------:|----------:|----------------:|------------:|----------------:|-----------------:|-------------------:|
| gaussian_process | a              | Strength1  |    106.692 |   7.5578 |    0.8316 |          4.7086 |     136.234 |          0.5641 |          103.868 |             0.0683 |
| gaussian_process | a              | Strength28 |    307.514 |  30.0064 |    0.5614 |          4.3491 |     430.873 |          0.4701 |          287.765 |             0.2033 |
| gaussian_process | a              | Strength7  |    259.651 |  23.3123 |    0.5798 |          4.4488 |     351.429 |          0.2781 |          248.184 |             0.1812 |
| gaussian_process | b              | Strength1  |    106.749 |   7.6219 |    0.8311 |          4.7105 |     136.384 |          0.5636 |          103.883 |             0.0692 |
| gaussian_process | b              | Strength28 |    311.774 |  32.6527 |    0.5432 |          4.4449 |     439.78  |          0.4677 |          289.793 |             0.1921 |
| gaussian_process | b              | Strength7  |    258.365 |  24.7613 |    0.5912 |          4.4131 |     346.511 |          0.2944 |          247.189 |             0.1595 |
| gaussian_process | c              | Strength1  |    110.782 |   8.6148 |    0.8174 |          4.8649 |     142.139 |          0.5435 |          106.278 |             0.08   |
| gaussian_process | c              | Strength28 |    307.135 |  30.7279 |    0.5627 |          4.3438 |     430.296 |          0.4719 |          287.419 |             0.2032 |
| gaussian_process | c              | Strength7  |    263.386 |  25.6587 |    0.5596 |          4.5401 |     360.326 |          0.2596 |          250.911 |             0.1982 |
| lightgbm         | a              | Strength1  |    120.675 |   7.2258 |    0.7893 |          5.3244 |     152.581 |          0.456  |          117.701 |             0.1328 |
| lightgbm         | a              | Strength28 |    321.594 |  26.61   |    0.512  |          4.6131 |     456.273 |          0.4553 |          290.89  |             0.2533 |
| lightgbm         | a              | Strength7  |    270.241 |  21.2316 |    0.5478 |          4.673  |     365.198 |          0.2355 |          256.371 |             0.2308 |
| lightgbm         | b              | Strength1  |    120.433 |   8.2254 |    0.7906 |          5.3143 |     152.134 |          0.4606 |          117.329 |             0.1277 |
| lightgbm         | b              | Strength28 |    324.293 |  26.983  |    0.5118 |          4.6433 |     456.278 |          0.4426 |          296.403 |             0.2186 |
| lightgbm         | b              | Strength7  |    270.375 |  21.329  |    0.5502 |          4.6752 |     364.333 |          0.2429 |          256.77  |             0.2083 |
| lightgbm         | c              | Strength1  |    125.068 |  10.0515 |    0.7705 |          5.4422 |     160.898 |          0.4359 |          119.691 |             0.124  |
| lightgbm         | c              | Strength28 |    322.036 |  26.717  |    0.5106 |          4.619  |     456.808 |          0.4504 |          291.372 |             0.2275 |
| lightgbm         | c              | Strength7  |    280.501 |  21.4698 |    0.4769 |          4.8867 |     393.358 |          0.1879 |          261.551 |             0.2521 |
| linear           | a              | Strength1  |    114.853 |  12.0393 |    0.6616 |          5.1276 |     175.257 |          0.5563 |          106.426 |             0.1989 |
| linear           | a              | Strength28 |    329.406 |  22.0479 |    0.5593 |          4.467  |     425.594 |          0.3661 |          322.578 |             0.058  |
| linear           | a              | Strength7  |    271.881 |  18.382  |    0.5355 |          4.5404 |     363.134 |          0.1615 |          268.27  |             0.094  |
| linear           | b              | Strength1  |    114.856 |  12.0611 |    0.6616 |          5.1285 |     175.298 |          0.5559 |          106.453 |             0.1989 |
| linear           | b              | Strength28 |    322.079 |  23.9927 |    0.5734 |          4.3824 |     419.028 |          0.3933 |          313.695 |             0.0413 |
| linear           | b              | Strength7  |    272.454 |  19.1175 |    0.5296 |          4.5495 |     364.942 |          0.1606 |          268.333 |             0.0897 |
| linear           | c              | Strength1  |    119.159 |  16.085  |    0.5116 |          5.3105 |     198     |          0.5522 |          107.261 |             0.3411 |
| linear           | c              | Strength28 |    327.337 |  20.5198 |    0.5699 |          4.43   |     420.579 |          0.3661 |          322.578 |             0.0474 |
| linear           | c              | Strength7  |    274.961 |  13.7728 |    0.551  |          4.5354 |     358.253 |          0.1473 |          269.284 |             0.0658 |

Selección provisional: **gaussian_process (a)**, MAE agregado OOF 224.8472. El IC pareado condiciona la comparación a las OOF disponibles; no representa una evaluación independiente posterior de la selección.

Las alternativas deben superar (a) con IC95% de ΔMAE agregado bajo cero; (c) debe tener también límite superior del IC de R1 ≤0. El agregado pareado pondera etiquetas observadas; la HPO minimiza la media por target de los MAE.

| family           | architecture   | target       | eligible   |   delta_mae |    ci_low |   ci_high | beats_l1   |   n_groups |   n_bootstrap |
|:-----------------|:---------------|:-------------|:-----------|------------:|----------:|----------:|:-----------|-----------:|--------------:|
| gaussian_process | b              | Strength1    | False      |     0.05869 |  -0.13169 |   0.24094 | False      |        571 |         10000 |
| gaussian_process | b              | Strength7    | False      |    -1.28113 |  -6.07319 |   3.54114 | False      |        593 |         10000 |
| gaussian_process | b              | Strength28   | False      |     4.21092 |  -0.48873 |   9.06131 | False      |        567 |         10000 |
| gaussian_process | b              | mean_targets | False      |     0.96107 |  -1.55733 |   3.52229 | False      |        600 |         10000 |
| gaussian_process | c              | Strength1    | False      |     4.08576 |   1.46234 |   6.7374  | False      |        571 |         10000 |
| gaussian_process | c              | Strength7    | False      |     3.74147 |   0.40835 |   7.19577 | False      |        593 |         10000 |
| gaussian_process | c              | Strength28   | False      |    -0.39603 |  -1.2316  |   0.45916 | False      |        567 |         10000 |
| gaussian_process | c              | mean_targets | False      |     2.49898 |   0.95782 |   4.04815 | False      |        600 |         10000 |
| lightgbm         | b              | Strength1    | False      |    -0.22859 |  -0.93472 |   0.47795 | False      |        571 |         10000 |
| lightgbm         | b              | Strength7    | False      |     0.11665 |  -2.75262 |   2.95507 | False      |        593 |         10000 |
| lightgbm         | b              | Strength28   | False      |     2.63501 |  -3.01978 |   8.31731 | False      |        567 |         10000 |
| lightgbm         | b              | mean_targets | False      |     0.82814 |  -1.34311 |   3.01256 | False      |        600 |         10000 |
| lightgbm         | c              | Strength1    | False      |     4.49659 |   0.01103 |   9.26872 | False      |        571 |         10000 |
| lightgbm         | c              | Strength7    | False      |    10.1809  |   1.31532 |  19.112   | False      |        593 |         10000 |
| lightgbm         | c              | Strength28   | False      |     0.40485 |  -1.99453 |   2.75395 | False      |        567 |         10000 |
| lightgbm         | c              | mean_targets | False      |     5.10094 |   1.40956 |   8.88137 | False      |        600 |         10000 |
| linear           | b              | Strength1    | True       |     0.00372 |  -0.0729  |   0.08275 | False      |        571 |         10000 |
| linear           | b              | Strength7    | True       |     0.57314 |  -0.25068 |   1.41034 | False      |        593 |         10000 |
| linear           | b              | Strength28   | True       |    -7.34804 | -12.8711  |  -1.56738 | True       |        567 |         10000 |
| linear           | b              | mean_targets | True       |    -2.211   |  -4.15842 |  -0.30278 | True       |        600 |         10000 |
| linear           | c              | Strength1    | False      |     4.30239 |   0.65438 |   9.35731 | False      |        571 |         10000 |
| linear           | c              | Strength7    | False      |     3.07273 |  -5.42873 |  11.3379  | False      |        593 |         10000 |
| linear           | c              | Strength28   | False      |    -2.06534 |  -6.24915 |   0       | False      |        567 |         10000 |
| linear           | c              | mean_targets | False      |     1.79446 |  -1.40453 |   4.93917 | False      |        600 |         10000 |

## Incertidumbre

CV+ nominal 90%, con cinco folds internos de calibración por cada fold externo, HPO anidada de cada modelo de calibración y residual máximo por grupo. Los intervalos solo reciben las features de validación externa; sus etiquetas se usan después para medir cobertura. Los límites no se proyectan por edad. La cobertura nominal no es una garantía exacta, especialmente con deriva, estratos dependientes de etiquetas y selección global previa de arquitectura.

| target     | scope   |   coverage |   group_coverage |   mean_width |   n_rows |   n_groups |   n_predictions | reliable   | coverage_alarm   |
|:-----------|:--------|-----------:|-----------------:|-------------:|---------:|-----------:|----------------:|:-----------|:-----------------|
| Strength1  | Todos   |     0.9111 |           0.9089 |      457.42  |      592 |        571 |            2960 | True       | False            |
| Strength1  | Type 1  |     0.9195 |           0.9175 |      456.158 |      517 |        497 |            2585 | True       | False            |
| Strength1  | Type 2  |     0.816  |           0.8083 |      476.326 |       25 |         24 |             125 | True       | True             |
| Strength1  | Type 3  |     0.8898 |           0.8898 |      461.181 |       49 |         49 |             245 | True       | False            |
| Strength1  | Type 5  |     0      |           0      |      452.62  |        1 |          1 |               5 | False      | True             |
| Strength28 | Todos   |     0.9075 |           0.9041 |     1378.55  |      588 |        567 |            2940 | True       | False            |
| Strength28 | Type 1  |     0.9198 |           0.9165 |     1369     |      511 |        491 |            2555 | True       | False            |
| Strength28 | Type 2  |     0.8296 |           0.8231 |     1384.84  |       27 |         26 |             135 | True       | True             |
| Strength28 | Type 3  |     0.9022 |           0.9022 |     1379.86  |       45 |         45 |             225 | True       | False            |
| Strength28 | Type 4  |     0.15   |           0.15   |     2544.95  |        4 |          4 |              20 | False      | True             |
| Strength28 | Type 5  |     0      |           0      |     1359.13  |        1 |          1 |               5 | False      | True             |
| Strength7  | Todos   |     0.9117 |           0.9086 |     1137.45  |      614 |        593 |            3070 | True       | False            |
| Strength7  | Type 1  |     0.9189 |           0.9157 |     1129.87  |      530 |        510 |            2650 | True       | False            |
| Strength7  | Type 2  |     0.8296 |           0.8231 |     1156.89  |       27 |         26 |             135 | True       | True             |
| Strength7  | Type 3  |     0.9462 |           0.9462 |     1139.3   |       52 |         52 |             260 | True       | False            |
| Strength7  | Type 4  |     0.3    |           0.3    |     1995.43  |        4 |          4 |              20 | False      | True             |
| Strength7  | Type 5  |     0      |           0      |     1099.29  |        1 |          1 |               5 | False      | True             |

Types 4/5 no permiten afirmar calibración estable; Type 4 no tiene R1 observado y por ello no tiene denominador para esa cobertura. Los tamaños n_rows/n_groups cuentan unidades únicas; n_predictions incluye las cinco repeticiones correlacionadas. Las alarmas por tipos escasos son descriptivas.

## Permutación y curvas de aprendizaje

Cordura en la primera repetición (cinco folds), presupuesto fijado en YAML. Diecinueve permutaciones conjuntas de etiquetas y nulos entre grupos del mismo tamaño; HPO repetida. Los modelos se mantienen fijos por arquitectura/familia: no se repite la selección completa de las etapas anteriores. La prueba es exploratoria y tiene resolución mínima p=0.05. Un R² nulo negativo puede reflejar sobreajuste; no se corrige a cero.

| target     |   observed_r2 |   shuffled_r2_mean |   shuffled_r2_sd |   p_value |   n_permutations | strongly_positive_null   |
|:-----------|--------------:|-------------------:|-----------------:|----------:|-----------------:|:-------------------------|
| Strength1  |       0.83166 |           -0.01961 |          0.01531 |      0.05 |               19 | False                    |
| Strength7  |       0.56211 |           -0.01978 |          0.01336 |      0.05 |               19 | False                    |
| Strength28 |       0.53063 |           -0.02477 |          0.01468 |      0.05 |               19 | False                    |

Las curvas usan subconjuntos anidados de grupos y validación idéntica para todas las fracciones. Media y SD de los cinco folds; la SD no es error estándar.

| target     |   fraction |   train_n |   train_mae |     mae |     r2 |   overfit_gap |
|:-----------|-----------:|----------:|------------:|--------:|-------:|--------------:|
| Strength1  |       0.25 |     118.8 |     79.7869 | 119.047 | 0.7888 |        0.1144 |
| Strength1  |       0.5  |     241.6 |     85.7281 | 111.972 | 0.8149 |        0.0793 |
| Strength1  |       0.75 |     360   |     89.6738 | 108.901 | 0.8242 |        0.0626 |
| Strength1  |       1    |     473.6 |     87.1409 | 106.535 | 0.8317 |        0.0696 |
| Strength28 |       0.25 |     115.8 |    161.23   | 362.749 | 0.3064 |        0.5465 |
| Strength28 |       0.5  |     237.2 |    225.251  | 326.05  | 0.4469 |        0.3005 |
| Strength28 |       0.75 |     354.4 |    240.851  | 319.689 | 0.4794 |        0.2167 |
| Strength28 |       1    |     470.4 |    241.712  | 310.958 | 0.5306 |        0.2339 |
| Strength7  |       0.25 |     123.8 |    180.61   | 305.185 | 0.367  |        0.4083 |
| Strength7  |       0.5  |     248.6 |    196.114  | 271.122 | 0.4711 |        0.2733 |
| Strength7  |       0.75 |     369.6 |    203.961  | 266.193 | 0.4911 |        0.2197 |
| Strength7  |       1    |     491.2 |    209.226  | 258.091 | 0.5621 |        0.1968 |

Figura: `outputs/fase3/stage3/learning_curves.png`.

## Sensibilidad temporal y alarmas

| target     |   mae_mean_cv |   mae_mean_temporal |   temporal_minus_cv_mae |
|:-----------|--------------:|--------------------:|------------------------:|
| Strength1  |       106.692 |             155.131 |                 48.4386 |
| Strength28 |       307.514 |             468.409 |                160.894  |
| Strength7  |       259.651 |             412.73  |                153.079  |

Los cinco cortes temporales preservan grupos; el primer bloque es calentamiento sin OOF. Cambian las poblaciones respecto de CV aleatoria y no hay fechas confirmadas.

Alarmas del piso de ruido (solo CV aleatoria): ninguna.

Ningún resultado autoriza promoción automática. Quedan pendientes explicación/MLflow de Etapa 4, revisión independiente de fuga y sobreajuste, unidades e identidades físicas y evaluación final autorizada. CV optimista por selección previa de finalistas/arquitectura, IC condicionales y multiplicidad sin corregir. Las bandas no expresan probabilidades calibradas de incumplimiento.

## Validación de entrega

229 pruebas en verde; `ruff check`, `ruff format` y `ruff format --check` en verde. La figura se revisó visualmente. `outputs/fase3/stage3/verification.json` verifica integridad de OOF, monotonía, intervalos, hashes de código/configuración y finalistas.
