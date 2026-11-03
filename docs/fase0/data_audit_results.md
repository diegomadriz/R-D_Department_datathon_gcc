# Resultados del data audit (generado)

> Archivo generado por `uv run python -m gcc_strength.audit`. No se edita a mano. La interpretación está en `data_audit.md`.

- Archivo: `data/datathon_data_gcc.csv` · SHA-256 `e410cbb15aa00d25c8d16b49d6a4486dbece185c5af77a3fdf3d72a50cc83b83`
- Filas: **791** · Columnas: **16**
- Columnas esperadas presentes y en orden: **True**

## 1. Esquema y tipos
| column                 | expected   | parsed_dtype   |   non_numeric_tokens |   n_unique |
|:-----------------------|:-----------|:---------------|---------------------:|-----------:|
| ProductType            | category   | object         |                    0 |          5 |
| Physical_Metric_A      | float      | float64        |                    0 |         97 |
| Physical_Metric_B      | float      | float64        |                    0 |        155 |
| Mineral_Phase_A        | float      | float64        |                    0 |        525 |
| Mineral_Phase_B        | float      | float64        |                    0 |        469 |
| Mineral_Phase_C        | float      | float64        |                    0 |        546 |
| Mineral_Phase_D        | float      | float64        |                    0 |        408 |
| Mineral_Phase_E        | float      | float64        |                    0 |        175 |
| Mineral_Phase_F        | float      | float64        |                    0 |        184 |
| Mineral_Phase_G        | float      | float64        |                    0 |         98 |
| Mineral_Phase_H        | float      | float64        |                    0 |        223 |
| Chemical_Composition_A | float      | float64        |                    0 |         89 |
| Chemical_Composition_B | float      | float64        |                    0 |         35 |
| Strength1              | float      | float64        |                    0 |        130 |
| Strength7              | float      | float64        |                    0 |        207 |
| Strength28             | float      | float64        |                    0 |        232 |

### Valores crudos de `ProductType`
| raw_value   |   n | canonical   |
|:------------|----:|:------------|
| 'Type 1'    | 680 | True        |
| 'Type 3'    |  66 | True        |
| 'Type 2'    |  38 | True        |
| 'Type 4'    |   6 | True        |
| 'Type 5'    |   1 | True        |

## 2. Nulos
### Por columna
|                        |   n_null |   pct_null |
|:-----------------------|---------:|-----------:|
| ProductType            |        0 |       0    |
| Physical_Metric_A      |      599 |      75.73 |
| Physical_Metric_B      |       22 |       2.78 |
| Mineral_Phase_A        |       71 |       8.98 |
| Mineral_Phase_B        |       71 |       8.98 |
| Mineral_Phase_C        |       71 |       8.98 |
| Mineral_Phase_D        |       71 |       8.98 |
| Mineral_Phase_E        |       72 |       9.1  |
| Mineral_Phase_F        |       71 |       8.98 |
| Mineral_Phase_G        |       71 |       8.98 |
| Mineral_Phase_H        |       71 |       8.98 |
| Chemical_Composition_A |       28 |       3.54 |
| Chemical_Composition_B |       29 |       3.67 |
| Strength1              |       51 |       6.45 |
| Strength7              |       22 |       2.78 |
| Strength28             |       59 |       7.46 |

### Por fila (número de nulos en la fila)
|   n_null_in_row |   n_rows |
|----------------:|---------:|
|               0 |      157 |
|               1 |      488 |
|               2 |       59 |
|               3 |        9 |
|               4 |        5 |
|               5 |        2 |
|               9 |       53 |
|              10 |        1 |
|              11 |        2 |
|              12 |        4 |
|              13 |        3 |
|              15 |        8 |

Filas vacías (solo `ProductType`): **8**, filas [128, 262, 264, 300, 368, 783, 789, 790].

### Patrón de nulos en targets
| pattern R1 R7 R28 (N=null)   |   n_rows |
|:-----------------------------|---------:|
| ooo                          |      693 |
| ooN                          |       38 |
| Noo                          |       36 |
| NNN                          |       13 |
| oNN                          |        6 |
| oNo                          |        3 |
| NoN                          |        2 |

### Co-ocurrencia de nulos (Jaccard de máscaras)
|                        |   Physical_Metric_A |   Physical_Metric_B |   Mineral_Phase_A |   Mineral_Phase_B |   Mineral_Phase_C |   Mineral_Phase_D |   Mineral_Phase_E |   Mineral_Phase_F |   Mineral_Phase_G |   Mineral_Phase_H |   Chemical_Composition_A |   Chemical_Composition_B |   Strength1 |   Strength7 |   Strength28 |
|:-----------------------|--------------------:|--------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|-------------------------:|-------------------------:|------------:|------------:|-------------:|
| Physical_Metric_A      |                1    |                0.04 |              0.12 |              0.12 |              0.12 |              0.12 |              0.12 |              0.12 |              0.12 |              0.12 |                     0.04 |                     0.04 |        0.08 |        0.03 |         0.04 |
| Physical_Metric_B      |                0.04 |                1    |              0.18 |              0.18 |              0.18 |              0.18 |              0.18 |              0.18 |              0.18 |              0.18 |                     0.35 |                     0.34 |        0.16 |        0.29 |         0.17 |
| Mineral_Phase_A        |                0.12 |                0.18 |              1    |              1    |              1    |              1    |              0.99 |              1    |              1    |              1    |                     0.19 |                     0.19 |        0.08 |        0.11 |         0.11 |
| Mineral_Phase_B        |                0.12 |                0.18 |              1    |              1    |              1    |              1    |              0.99 |              1    |              1    |              1    |                     0.19 |                     0.19 |        0.08 |        0.11 |         0.11 |
| Mineral_Phase_C        |                0.12 |                0.18 |              1    |              1    |              1    |              1    |              0.99 |              1    |              1    |              1    |                     0.19 |                     0.19 |        0.08 |        0.11 |         0.11 |
| Mineral_Phase_D        |                0.12 |                0.18 |              1    |              1    |              1    |              1    |              0.99 |              1    |              1    |              1    |                     0.19 |                     0.19 |        0.08 |        0.11 |         0.11 |
| Mineral_Phase_E        |                0.12 |                0.18 |              0.99 |              0.99 |              0.99 |              0.99 |              1    |              0.99 |              0.99 |              0.99 |                     0.19 |                     0.19 |        0.08 |        0.11 |         0.11 |
| Mineral_Phase_F        |                0.12 |                0.18 |              1    |              1    |              1    |              1    |              0.99 |              1    |              1    |              1    |                     0.19 |                     0.19 |        0.08 |        0.11 |         0.11 |
| Mineral_Phase_G        |                0.12 |                0.18 |              1    |              1    |              1    |              1    |              0.99 |              1    |              1    |              1    |                     0.19 |                     0.19 |        0.08 |        0.11 |         0.11 |
| Mineral_Phase_H        |                0.12 |                0.18 |              1    |              1    |              1    |              1    |              0.99 |              1    |              1    |              1    |                     0.19 |                     0.19 |        0.08 |        0.11 |         0.11 |
| Chemical_Composition_A |                0.04 |                0.35 |              0.19 |              0.19 |              0.19 |              0.19 |              0.19 |              0.19 |              0.19 |              0.19 |                     1    |                     0.97 |        0.14 |        0.22 |         0.16 |
| Chemical_Composition_B |                0.04 |                0.34 |              0.19 |              0.19 |              0.19 |              0.19 |              0.19 |              0.19 |              0.19 |              0.19 |                     0.97 |                     1    |        0.14 |        0.21 |         0.16 |
| Strength1              |                0.08 |                0.16 |              0.08 |              0.08 |              0.08 |              0.08 |              0.08 |              0.08 |              0.08 |              0.08 |                     0.14 |                     0.14 |        1    |        0.22 |         0.16 |
| Strength7              |                0.03 |                0.29 |              0.11 |              0.11 |              0.11 |              0.11 |              0.11 |              0.11 |              0.11 |              0.11 |                     0.22 |                     0.21 |        0.22 |        1    |         0.31 |
| Strength28             |                0.04 |                0.17 |              0.11 |              0.11 |              0.11 |              0.11 |              0.11 |              0.11 |              0.11 |              0.11 |                     0.16 |                     0.16 |        0.16 |        0.31 |         1    |

### Tasa de nulos (%) por `ProductType`
| ProductType   |   n_rows |   Physical_Metric_A |   Physical_Metric_B |   Mineral_Phase_A |   Mineral_Phase_B |   Mineral_Phase_C |   Mineral_Phase_D |   Mineral_Phase_E |   Mineral_Phase_F |   Mineral_Phase_G |   Mineral_Phase_H |   Chemical_Composition_A |   Chemical_Composition_B |   Strength1 |   Strength7 |   Strength28 |
|:--------------|---------:|--------------------:|--------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|-------------------------:|-------------------------:|------------:|------------:|-------------:|
| Type 1        |      680 |                77.2 |                 2.5 |               9.1 |               9.1 |               9.1 |               9.1 |               9.3 |               9.1 |               9.1 |               9.1 |                      2.9 |                      3.1 |         5.6 |         2.5 |          6   |
| Type 2        |       38 |                92.1 |                 5.3 |              13.2 |              13.2 |              13.2 |              13.2 |              13.2 |              13.2 |              13.2 |              13.2 |                      7.9 |                      7.9 |         7.9 |         2.6 |          7.9 |
| Type 3        |       66 |                48.5 |                 0   |               3   |               3   |               3   |               3   |               3   |               3   |               3   |               3   |                      1.5 |                      1.5 |         7.6 |         3   |         19.7 |
| Type 4        |        6 |               100   |                33.3 |              33.3 |              33.3 |              33.3 |              33.3 |              33.3 |              33.3 |              33.3 |              33.3 |                     66.7 |                     66.7 |        83.3 |        33.3 |         33.3 |
| Type 5        |        1 |               100   |               100   |               0   |               0   |               0   |               0   |               0   |               0   |               0   |               0   |                      0   |                      0   |         0   |         0   |          0   |

### Independencia nulo vs `ProductType` (chi²)
| column                 |   chi2 |   dof |   p_value |   min_expected |   pct_cells_expected_lt5 |
|:-----------------------|-------:|------:|----------:|---------------:|-------------------------:|
| Physical_Metric_A      |  35.25 |     4 |  4.14e-07 |           0.24 |                       40 |
| Physical_Metric_B      |  58.62 |     4 |  5.65e-12 |           0.03 |                       50 |
| Mineral_Phase_A        |   8.14 |     4 |  0.0865   |           0.09 |                       40 |
| Mineral_Phase_B        |   8.14 |     4 |  0.0865   |           0.09 |                       40 |
| Mineral_Phase_C        |   8.14 |     4 |  0.0865   |           0.09 |                       40 |
| Mineral_Phase_D        |   8.14 |     4 |  0.0865   |           0.09 |                       40 |
| Mineral_Phase_E        |   8.08 |     4 |  0.0888   |           0.09 |                       40 |
| Mineral_Phase_F        |   8.14 |     4 |  0.0865   |           0.09 |                       40 |
| Mineral_Phase_G        |   8.14 |     4 |  0.0865   |           0.09 |                       40 |
| Mineral_Phase_H        |   8.14 |     4 |  0.0865   |           0.09 |                       40 |
| Chemical_Composition_A |  73.68 |     4 |  3.79e-15 |           0.04 |                       50 |
| Chemical_Composition_B |  70.9  |     4 |  1.47e-14 |           0.04 |                       50 |
| Strength1              |  59.97 |     4 |  2.94e-12 |           0.06 |                       50 |
| Strength7              |  20.96 |     4 |  0.000323 |           0.03 |                       50 |
| Strength28             |  22.24 |     4 |  0.000179 |           0.07 |                       50 |

### `Physical_Metric_A` nulo vs presente: distribución de las demás features
| feature                |   median_when_null |   median_when_present |   p_value |
|:-----------------------|-------------------:|----------------------:|----------:|
| Physical_Metric_B      |              0.027 |                -0.66  |  6.16e-51 |
| Mineral_Phase_A        |              0.819 |                 0.802 |  0.011    |
| Mineral_Phase_B        |              0.395 |                 0.369 |  5.02e-09 |
| Mineral_Phase_C        |              0.737 |                 0.738 |  0.197    |
| Mineral_Phase_D        |              0.211 |                 0.237 |  1.02e-14 |
| Mineral_Phase_E        |              0.44  |                 0.412 |  0.0001   |
| Mineral_Phase_F        |              0.516 |                 0.416 |  9.76e-11 |
| Mineral_Phase_G        |              0.026 |                 0.024 |  8.33e-05 |
| Mineral_Phase_H        |              0.435 |                 0.317 |  1.48e-20 |
| Chemical_Composition_A |              0.678 |                 0.589 |  1.35e-39 |
| Chemical_Composition_B |              0.089 |                 0.101 |  5.28e-07 |

## 3. Duplicados y piso de ruido
- rows_without_any_feature_but_some_target: **[169, 263, 329, 549, 601]**
- rows_with_features_but_no_target: **[220, 314, 339, 519, 591]**
- exact_duplicate_rows (keep=first): **5**
- exact_duplicates_among_non_empty_rows: **0**
- empty_rows (all features and targets null): **8**
- feature_duplicates_non_empty (keep=False): **4**
- feature_duplicates_with_any_non_null_feature: **0**

### Grupos de filas con el mismo trío (R1, R7, R28) y features distintas
Regla: filas consecutivas que comparten ≥ 2 targets no nulos y todos los observados son iguales (`strength_groups`).
Grupos: **21**; filas involucradas: **44**.
Filas cuyo trío de targets es igual al de la fila anterior: **23**.

|    |   n_rows | product_types   | row_ids     |   max_row_gap |   n_features_identical_in_group |
|---:|---------:|:----------------|:------------|--------------:|--------------------------------:|
|  0 |        2 | Type 1          | 0,1         |             1 |                               4 |
|  1 |        2 | Type 1          | 71,72       |             1 |                              10 |
|  2 |        2 | Type 1          | 193,194     |             1 |                               4 |
|  3 |        2 | Type 1          | 236,237     |             1 |                               5 |
|  4 |        2 | Type 2          | 257,258     |             1 |                               4 |
|  5 |        2 | Type 1          | 315,316     |             1 |                              10 |
|  6 |        2 | Type 1          | 467,468     |             1 |                               4 |
|  7 |        2 | Type 1          | 478,479     |             1 |                               4 |
|  8 |        2 | Type 1          | 485,486     |             1 |                               4 |
|  9 |        2 | Type 1          | 490,491     |             1 |                               4 |
| 10 |        2 | Type 1          | 504,505     |             1 |                              10 |
| 11 |        2 | Type 1          | 520,521     |             1 |                               4 |
| 12 |        3 | Type 1          | 560,561,562 |             1 |                               4 |
| 13 |        2 | Type 1          | 563,564     |             1 |                               4 |
| 14 |        2 | Type 1          | 566,567     |             1 |                               4 |
| 15 |        2 | Type 1          | 569,570     |             1 |                               5 |
| 16 |        3 | Type 1          | 580,581,582 |             1 |                               4 |
| 17 |        2 | Type 1          | 619,620     |             1 |                               4 |
| 18 |        2 | Type 1          | 629,630     |             1 |                               4 |
| 19 |        2 | Type 1          | 657,658     |             1 |                               4 |
| 20 |        2 | Type 1          | 700,701     |             1 |                              10 |

### Casi-duplicados en features (mismo tipo, distancia de Chebyshev < tol)
Sin `Physical_Metric_A` (76% nulos); `Physical_Metric_B` (z-score) dividida entre 10. `implied_mape_pct` = 0.8365 × diferencia relativa mediana: es el MAPE de un predictor perfecto si toda la diferencia fuera ruido normal del ensayo (cota superior, no medición).
|   tol |   n_pairs |   Strength1_n |   Strength7_n |   Strength28_n |   Strength1_median_abs_diff |   Strength1_median_rel_diff_pct |   Strength1_implied_mape_pct |   Strength7_median_abs_diff |   Strength7_median_rel_diff_pct |   Strength7_implied_mape_pct |   Strength28_median_abs_diff |   Strength28_median_rel_diff_pct |   Strength28_implied_mape_pct |
|------:|----------:|--------------:|--------------:|---------------:|----------------------------:|--------------------------------:|-----------------------------:|----------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|---------------------------------:|------------------------------:|
|  0.01 |         0 |             0 |             0 |              0 |                         nan |                          nan    |                       nan    |                         nan |                          nan    |                       nan    |                          nan |                           nan    |                        nan    |
|  0.02 |         2 |             2 |             2 |              2 |                         125 |                            5.24 |                         4.38 |                         195 |                            3.43 |                         2.87 |                          140 |                             2.12 |                          1.78 |
|  0.03 |         8 |             8 |             8 |              8 |                          35 |                            1.75 |                         1.47 |                         340 |                            5.69 |                         4.76 |                          215 |                             3.13 |                          2.62 |
|  0.05 |       160 |           147 |           153 |            145 |                         120 |                            5.26 |                         4.4  |                         210 |                            3.43 |                         2.87 |                          270 |                             3.86 |                          3.23 |

## 4. Rangos, escalado y valores sospechosos
### Escalado inferido
| column                 | inferred_scaling   |   n_non_null |
|:-----------------------|:-------------------|-------------:|
| Physical_Metric_A      | min-max [0,1]      |          192 |
| Physical_Metric_B      | z-score (ddof=0)   |          769 |
| Mineral_Phase_A        | min-max [0,1]      |          720 |
| Mineral_Phase_B        | min-max [0,1]      |          720 |
| Mineral_Phase_C        | min-max [0,1]      |          720 |
| Mineral_Phase_D        | min-max [0,1]      |          720 |
| Mineral_Phase_E        | min-max [0,1]      |          719 |
| Mineral_Phase_F        | min-max [0,1]      |          720 |
| Mineral_Phase_G        | min-max [0,1]      |          720 |
| Mineral_Phase_H        | min-max [0,1]      |          720 |
| Chemical_Composition_A | min-max [0,1]      |          763 |
| Chemical_Composition_B | min-max [0,1]      |          762 |

### Rangos
| column                 |   n |      min |         p01 |     median |       p99 |     max |         mean |   std_ddof0 |   n_eq_0 |   n_eq_1 |             rows_at_min |   rows_at_max | all_multiple_of_10   |
|:-----------------------|----:|---------:|------------:|-----------:|----------:|--------:|-------------:|------------:|---------:|---------:|------------------------:|--------------:|:---------------------|
| Physical_Metric_A      | 192 |    0     |    0.1275   |    0.6877  |    0.9868 |    1    |    0.6438    |     0.1752  |        1 |        1 |                     752 |           596 |                      |
| Physical_Metric_B      | 769 |   -2.461 |   -1.861    |   -0.1646  |    3.168  |    8.64 |   -2.957e-16 |     1       |        0 |        0 |                     765 |           513 |                      |
| Mineral_Phase_A        | 720 |    0     |    0.6958   |    0.8103  |    0.9353 |    1    |    0.8107    |     0.05766 |        1 |        1 |                     619 |           597 |                      |
| Mineral_Phase_B        | 720 |    0     |    0.1791   |    0.386   |    0.5868 |    1    |    0.3872    |     0.07305 |        1 |        1 |                     619 |           491 |                      |
| Mineral_Phase_C        | 720 |    0     |    0.5528   |    0.7372  |    0.9203 |    1    |    0.7362    |     0.07664 |        1 |        1 |                     619 |           405 |                      |
| Mineral_Phase_D        | 720 |    0     |    0.04941  |    0.218   |    0.3051 |    1    |    0.2055    |     0.08155 |        1 |        1 |                     619 |           513 |                      |
| Mineral_Phase_E        | 719 |    0     |    0.04504  |    0.434   |    0.6206 |    1    |    0.4066    |     0.1314  |        1 |        1 |                     619 |           491 |                      |
| Mineral_Phase_F        | 720 |    0     |    0.1518   |    0.4903  |    0.909  |    1    |    0.4968    |     0.1697  |        3 |        1 |             312,491,619 |           220 |                      |
| Mineral_Phase_G        | 720 |    0     |    0.004768 |    0.0252  |    0.0748 |    1    |    0.02855   |     0.03856 |        1 |        1 |                     265 |           619 |                      |
| Mineral_Phase_H        | 720 |    0     |    0.01545  |    0.3929  |    0.8208 |    1    |    0.4053    |     0.1808  |        6 |        1 | 241,242,243,246,491,619 |           121 |                      |
| Chemical_Composition_A | 763 |    0     |    0.4151   |    0.6611  |    0.8584 |    1    |    0.6519    |     0.09185 |        1 |        1 |                     265 |           255 |                      |
| Chemical_Composition_B | 762 |    0     |    0.01786  |    0.09524 |    0.1786 |    1    |    0.09744   |     0.04588 |        2 |        1 |                  30,488 |           121 |                      |
| Strength1              | 740 | 1600     | 1808        | 2250       | 3787      | 4010    | 2308         |   360.3     |        0 |        0 |                     532 |           100 | True                 |
| Strength7              | 769 | 2490     | 5057        | 6000       | 7716      | 8210    | 6069         |   542.5     |        0 |        0 |                      54 |           285 | True                 |
| Strength28             | 732 | 2800     | 6132        | 7510       | 8787      | 9130    | 7480         |   639.2     |        0 |        0 |                      54 |           287 | True                 |

### Outliers (IQR 1.5× y z robusto > 3.5), asimetría y curtosis; dataset completo
| column                 |   iqr_outliers |   robust_z>3.5 |   skew |   kurtosis_excess |
|:-----------------------|---------------:|---------------:|-------:|------------------:|
| Physical_Metric_A      |             23 |             19 |  -1.37 |              1.9  |
| Physical_Metric_B      |             49 |             37 |   2.97 |             18.7  |
| Mineral_Phase_A        |             19 |              8 |  -4.43 |             56.78 |
| Mineral_Phase_B        |             52 |             35 |   0.19 |             10.2  |
| Mineral_Phase_C        |             36 |             10 |  -1.55 |             14.16 |
| Mineral_Phase_D        |            120 |             92 |   3.47 |             34.97 |
| Mineral_Phase_E        |             76 |             46 |  -1    |              1.34 |
| Mineral_Phase_F        |              5 |              0 |   0.17 |             -0.11 |
| Mineral_Phase_G        |             24 |             16 |  22.24 |            556.4  |
| Mineral_Phase_H        |              4 |              1 |   0.28 |             -0.12 |
| Chemical_Composition_A |             32 |             12 |  -1.13 |              7.29 |
| Chemical_Composition_B |             26 |              5 |  10.59 |            199.6  |
| Strength1              |             43 |             35 |   2.42 |              7.38 |
| Strength7              |             40 |             31 |  -0.16 |              7.09 |
| Strength28             |             12 |              5 |  -1.46 |              8.22 |

### Filas sospechosas
|     | ProductType   | phases_at_scaled_min_ge7   |   n_phases_at_0_or_1 | r1_gt_r7   | r7_gt_r28   | pmb_abs_z_gt_4   |
|----:|:--------------|:---------------------------|---------------------:|:-----------|:------------|:-----------------|
|  54 | Type 4        | False                      |                    0 | False      | False       | True             |
| 265 | Type 4        | False                      |                    1 | False      | False       | True             |
| 491 | Type 1        | False                      |                    4 | False      | False       | False            |
| 513 | Type 4        | False                      |                    1 | False      | False       | True             |
| 515 | Type 4        | False                      |                    0 | False      | False       | True             |
| 619 | Type 1        | True                       |                    8 | False      | False       | False            |

## 5. Estructura composicional de `Mineral_Phase_A..H`
### Suma por fila (unidades escaladas)
|                   |   count |   mean |    std |   min |   25% |   50% |   75% |   max |
|:------------------|--------:|-------:|-------:|------:|------:|------:|------:|------:|
| sum_scaled_phases |     719 |  3.476 | 0.3311 |     1 |  3.25 | 3.414 | 3.664 | 4.371 |

### R² de cada fase explicada por las otras siete (OLS + intercepto)
| phase           |   r2_from_other_phases |
|:----------------|-----------------------:|
| Mineral_Phase_A |                1       |
| Mineral_Phase_B |                1       |
| Mineral_Phase_C |                1       |
| Mineral_Phase_D |                0.6089  |
| Mineral_Phase_E |                0.1414  |
| Mineral_Phase_F |                0.2342  |
| Mineral_Phase_G |                0.5205  |
| Mineral_Phase_H |                0.07773 |

Valores singulares de la matriz de fases centrada: 5.241, 4.431, 3.412, 2.571, 2.145, 1.564, 0.6159, 0.001299

Relación casi exacta: `Mineral_Phase_A = -0.000016 + 0.538688·B + 0.817807·C` (n=720, R²=0.99999861, residuo máximo 1.54e-04; VIF implícito ≈ 7.19e+05).

Sin `Mineral_Phase_A`, valor singular mínimo de B..H: **0.5633** (sin otra dependencia lineal exacta, no hay cierre a constante).

## 6. Consistencia de targets
| group   |   n_R1_R7 |   pct_R1>R7 |   n_R7_R28 |   pct_R7>R28 |   pct_R1>R28 |
|:--------|----------:|------------:|-----------:|-------------:|-------------:|
| all     |       731 |           0 |        729 |            0 |            0 |
| Type 1  |       633 |           0 |        636 |            0 |            0 |
| Type 2  |        35 |           0 |         35 |            0 |            0 |
| Type 3  |        61 |           0 |         53 |            0 |            0 |
| Type 4  |         1 |           0 |          4 |            0 |            0 |
| Type 5  |         1 |           0 |          1 |            0 |            0 |

### Correlación entre targets (Pearson)
|            |   Strength1 |   Strength7 |   Strength28 |
|:-----------|------------:|------------:|-------------:|
| Strength1  |        1    |       0.7   |        0.23  |
| Strength7  |        0.7  |       1     |        0.766 |
| Strength28 |        0.23 |       0.766 |        1     |

### Baselines in-sample sobre el dataset completo (solo dimensionamiento)
> Usa también las filas que serán test: no se usa para decidir nada. El baseline oficial es L1 por CV sobre train (Fase 3).
|                                   |   n |    r2 |   mae |   rmse |   mape_pct |
|:----------------------------------|----:|------:|------:|-------:|-----------:|
| ('Strength1', 'L0_global_mean')   | 740 | 0     |   225 |    360 |       9.22 |
| ('Strength1', 'L1_mean_by_type')  | 740 | 0.657 |   161 |    211 |       7.14 |
| ('Strength7', 'L0_global_mean')   | 769 | 0     |   374 |    543 |       6.36 |
| ('Strength7', 'L1_mean_by_type')  | 769 | 0.496 |   302 |    385 |       4.98 |
| ('Strength28', 'L0_global_mean')  | 732 | 0     |   460 |    639 |       6.59 |
| ('Strength28', 'L1_mean_by_type') | 732 | 0.357 |   407 |    513 |       5.57 |

### Perfil de R1/R7/R28 por `ProductType`
| ProductType   |   Strength1_count |   Strength1_mean |   Strength1_std |   Strength1_median |   Strength7_count |   Strength7_mean |   Strength7_std |   Strength7_median |   Strength28_count |   Strength28_mean |   Strength28_std |   Strength28_median |   R1/R28_median |   R7/R28_median |
|:--------------|------------------:|-----------------:|----------------:|-------------------:|------------------:|-----------------:|----------------:|-------------------:|-------------------:|------------------:|-----------------:|--------------------:|----------------:|----------------:|
| Type 1        |               642 |             2243 |             204 |               2240 |               663 |             6019 |             383 |               5980 |                639 |              7429 |              522 |                7470 |           0.3   |           0.801 |
| Type 2        |                35 |             3606 |             338 |               3670 |                37 |             7415 |             464 |               7570 |                 35 |              8394 |              496 |                8540 |           0.43  |           0.886 |
| Type 3        |                61 |             2235 |             195 |               2230 |                64 |             6028 |             370 |               5990 |                 53 |              7833 |              432 |                7900 |           0.282 |           0.758 |
| Type 4        |                 1 |             3190 |             nan |               3190 |                 4 |             2898 |             311 |               2955 |                  4 |              3432 |              496 |                3485 |           0.915 |           0.848 |
| Type 5        |                 1 |             1710 |             nan |               1710 |                 1 |             4240 |             nan |               4240 |                  1 |              5580 |              nan |                5580 |           0.306 |           0.76  |

## 7. ¿El orden de filas es temporal? Deriva por bloques
### Posición de cada tipo en el archivo
| ProductType   |   n |   first_row |   last_row |   median_row |
|:--------------|----:|------------:|-----------:|-------------:|
| Type 1        | 680 |           0 |        790 |          398 |
| Type 2        |  38 |           6 |        598 |          284 |
| Type 3        |  66 |           9 |        781 |          650 |
| Type 4        |   6 |          54 |        515 |          264 |
| Type 5        |   1 |         121 |        121 |          121 |

### Mediana por bloque, solo Type 1 (8 bloques contiguos del archivo)
|   block | rows    |   n_type1 |   Physical_Metric_A |   Physical_Metric_B |   Mineral_Phase_A |   Mineral_Phase_B |   Mineral_Phase_C |   Mineral_Phase_D |   Mineral_Phase_E |   Mineral_Phase_F |   Mineral_Phase_G |   Mineral_Phase_H |   Chemical_Composition_A |   Chemical_Composition_B |   Strength1 |   Strength7 |   Strength28 |
|--------:|:--------|----------:|--------------------:|--------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|-------------------------:|-------------------------:|------------:|------------:|-------------:|
|       0 | 0-98    |        88 |             nan     |               0.714 |             0.84  |             0.387 |             0.771 |             0.198 |             0.463 |             0.537 |             0.027 |             0.487 |                    0.678 |                    0.077 |        2400 |        5810 |         6720 |
|       1 | 99-197  |        89 |             nan     |               0.376 |             0.835 |             0.385 |             0.761 |             0.216 |             0.449 |             0.626 |             0.022 |             0.656 |                    0.711 |                    0.089 |        2430 |        5895 |         6780 |
|       2 | 198-296 |        81 |             nan     |               0.038 |             0.822 |             0.408 |             0.735 |             0.197 |             0.484 |             0.549 |             0.026 |             0.321 |                    0.683 |                    0.089 |        2350 |        6410 |         7900 |
|       3 | 297-395 |        79 |             nan     |              -0.21  |             0.793 |             0.413 |             0.698 |             0.221 |             0.434 |             0.486 |             0.029 |             0.334 |                    0.622 |                    0.119 |        2220 |        6040 |         7480 |
|       4 | 396-494 |        92 |             nan     |              -0.345 |             0.787 |             0.405 |             0.697 |             0.222 |             0.446 |             0.436 |             0.022 |             0.36  |                    0.678 |                    0.077 |        2160 |        6175 |         7980 |
|       5 | 495-593 |        91 |               0.753 |              -0.198 |             0.758 |             0.377 |             0.678 |             0.24  |             0.384 |             0.34  |             0.025 |             0.424 |                    0.647 |                    0.101 |        2160 |        5990 |         7710 |
|       6 | 594-692 |        88 |               0.682 |              -0.592 |             0.79  |             0.365 |             0.719 |             0.247 |             0.223 |             0.455 |             0.021 |             0.325 |                    0.6   |                    0.101 |        2090 |        5775 |         7425 |
|       7 | 693-790 |        72 |               0.708 |              -0.615 |             0.804 |             0.368 |             0.734 |             0.24  |             0.49  |             0.385 |             0.02  |             0.308 |                    0.539 |                    0.101 |        2230 |        5965 |         7400 |

### Pruebas de deriva (Kruskal-Wallis entre bloques; Spearman vs número de fila)
|                                      |   kruskal_H |   p_value |   spearman_rho_vs_row |    p_rho |
|:-------------------------------------|------------:|----------:|----------------------:|---------:|
| ('all', 'Physical_Metric_A')         |         6.5 |   0.039   |                -0.167 | 0.02     |
| ('all', 'Physical_Metric_B')         |       408.7 |   3.2e-84 |                -0.711 | 3.1e-119 |
| ('all', 'Mineral_Phase_A')           |       274.4 |   1.8e-55 |                -0.419 | 6.4e-32  |
| ('all', 'Mineral_Phase_B')           |        64   |   2.4e-11 |                -0.158 | 2e-05    |
| ('all', 'Mineral_Phase_C')           |       188.8 |   2.7e-37 |                -0.261 | 1.1e-12  |
| ('all', 'Mineral_Phase_D')           |       177.1 |   7.9e-35 |                 0.415 | 2.9e-31  |
| ('all', 'Mineral_Phase_E')           |       151   |   2.5e-29 |                -0.219 | 2.8e-09  |
| ('all', 'Mineral_Phase_F')           |       206.1 |   5.8e-41 |                -0.463 | 1.8e-39  |
| ('all', 'Mineral_Phase_G')           |        40   |   1.3e-06 |                -0.106 | 0.0044   |
| ('all', 'Mineral_Phase_H')           |       251.7 |   1.2e-50 |                -0.42  | 3.7e-32  |
| ('all', 'Chemical_Composition_A')    |       271.6 |   7e-55   |                -0.504 | 2.4e-50  |
| ('all', 'Chemical_Composition_B')    |       178.1 |   4.9e-35 |                 0.231 | 1e-10    |
| ('all', 'Strength1')                 |       253.8 |   4.3e-51 |                -0.505 | 3.5e-49  |
| ('all', 'Strength7')                 |       161.8 |   1.3e-31 |                -0.146 | 4.8e-05  |
| ('all', 'Strength28')                |       321.6 |   1.5e-65 |                 0.297 | 2.5e-16  |
| ('Type 1', 'Physical_Metric_A')      |         6.6 |   0.037   |                -0.082 | 0.31     |
| ('Type 1', 'Physical_Metric_B')      |       421.5 |   5.9e-87 |                -0.762 | 5e-127   |
| ('Type 1', 'Mineral_Phase_A')        |       367.8 |   1.9e-75 |                -0.594 | 3.8e-60  |
| ('Type 1', 'Mineral_Phase_B')        |        68   |   3.8e-12 |                -0.186 | 3.2e-06  |
| ('Type 1', 'Mineral_Phase_C')        |       227.4 |   1.7e-45 |                -0.36  | 2.4e-20  |
| ('Type 1', 'Mineral_Phase_D')        |       270.9 |   9.8e-55 |                 0.621 | 3.4e-67  |
| ('Type 1', 'Mineral_Phase_E')        |       143.2 |   1.1e-27 |                -0.222 | 2.7e-08  |
| ('Type 1', 'Mineral_Phase_F')        |       159.9 |   3.4e-31 |                -0.427 | 9.3e-29  |
| ('Type 1', 'Mineral_Phase_G')        |        34.8 |   1.2e-05 |                -0.15  | 0.00019  |
| ('Type 1', 'Mineral_Phase_H')        |       254.8 |   2.7e-51 |                -0.419 | 1.3e-27  |
| ('Type 1', 'Chemical_Composition_A') |       267.4 |   5.5e-54 |                -0.525 | 5.3e-48  |
| ('Type 1', 'Chemical_Composition_B') |       143.2 |   1.1e-27 |                 0.216 | 2e-08    |
| ('Type 1', 'Strength1')              |       253   |   6.2e-51 |                -0.533 | 2.1e-48  |
| ('Type 1', 'Strength7')              |       159.8 |   3.5e-31 |                -0.093 | 0.017    |
| ('Type 1', 'Strength28')             |       339.8 |   1.8e-69 |                 0.353 | 3.9e-20  |

## 8. Perfil de features por `ProductType` (mediana)
|                        |   Type 1 |   Type 2 |   Type 3 |   Type 4 |   Type 5 |
|:-----------------------|---------:|---------:|---------:|---------:|---------:|
| Physical_Metric_A      |    0.699 |    0.992 |    0.334 |  nan     |  nan     |
| Physical_Metric_B      |   -0.165 |    2.47  |   -1.133 |    7.627 |  nan     |
| Mineral_Phase_A        |    0.805 |    0.897 |    0.868 |    0.513 |    0.632 |
| Mineral_Phase_B        |    0.385 |    0.401 |    0.397 |    0.288 |    0.331 |
| Mineral_Phase_C        |    0.727 |    0.843 |    0.812 |    0.436 |    0.554 |
| Mineral_Phase_D        |    0.222 |    0.062 |    0.082 |    0.952 |    0.191 |
| Mineral_Phase_E        |    0.44  |    0.381 |    0.403 |    0.306 |    0.469 |
| Mineral_Phase_F        |    0.494 |    0.615 |    0.434 |    0.399 |    0.463 |
| Mineral_Phase_G        |    0.025 |    0.033 |    0.031 |    0.009 |    0.01  |
| Mineral_Phase_H        |    0.393 |    0.627 |    0.307 |    0.323 |    1     |
| Chemical_Composition_A |    0.658 |    0.772 |    0.633 |    0.011 |    0.311 |
| Chemical_Composition_B |    0.095 |    0.095 |    0.119 |    0.03  |    1     |

## 9. Correlación entre features (Spearman; sin targets)
|                        |   Physical_Metric_A |   Physical_Metric_B |   Mineral_Phase_A |   Mineral_Phase_B |   Mineral_Phase_C |   Mineral_Phase_D |   Mineral_Phase_E |   Mineral_Phase_F |   Mineral_Phase_G |   Mineral_Phase_H |   Chemical_Composition_A |   Chemical_Composition_B |
|:-----------------------|--------------------:|--------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|------------------:|-------------------------:|-------------------------:|
| Physical_Metric_A      |                1    |                0.42 |             -0.11 |             -0.03 |             -0.07 |              0.11 |              0.03 |              0.08 |             -0.14 |             -0.06 |                     0.08 |                    -0.34 |
| Physical_Metric_B      |                0.42 |                1    |              0.25 |              0.03 |              0.19 |             -0.22 |              0.12 |              0.32 |              0.09 |              0.52 |                     0.5  |                    -0.28 |
| Mineral_Phase_A        |               -0.11 |                0.25 |              1    |              0.12 |              0.78 |             -0.59 |              0.11 |              0.35 |              0.13 |              0.23 |                     0.27 |                    -0.15 |
| Mineral_Phase_B        |               -0.03 |                0.03 |              0.12 |              1    |             -0.45 |             -0.17 |             -0.02 |              0.13 |              0.3  |              0.03 |                     0.04 |                     0.23 |
| Mineral_Phase_C        |               -0.07 |                0.19 |              0.78 |             -0.45 |              1    |             -0.39 |              0.09 |              0.26 |             -0.07 |              0.17 |                     0.21 |                    -0.25 |
| Mineral_Phase_D        |                0.11 |               -0.22 |             -0.59 |             -0.17 |             -0.39 |              1    |             -0.23 |             -0.15 |             -0.31 |             -0.08 |                    -0.33 |                    -0.04 |
| Mineral_Phase_E        |                0.03 |                0.12 |              0.11 |             -0.02 |              0.09 |             -0.23 |              1    |             -0.19 |             -0.01 |             -0    |                     0.05 |                    -0.23 |
| Mineral_Phase_F        |                0.08 |                0.32 |              0.35 |              0.13 |              0.26 |             -0.15 |             -0.19 |              1    |              0.04 |              0.18 |                     0.24 |                     0.03 |
| Mineral_Phase_G        |               -0.14 |                0.09 |              0.13 |              0.3  |             -0.07 |             -0.31 |             -0.01 |              0.04 |              1    |             -0.03 |                     0.07 |                     0.29 |
| Mineral_Phase_H        |               -0.06 |                0.52 |              0.23 |              0.03 |              0.17 |             -0.08 |             -0    |              0.18 |             -0.03 |              1    |                     0.41 |                    -0.13 |
| Chemical_Composition_A |                0.08 |                0.5  |              0.27 |              0.04 |              0.21 |             -0.33 |              0.05 |              0.24 |              0.07 |              0.41 |                     1    |                    -0.18 |
| Chemical_Composition_B |               -0.34 |               -0.28 |             -0.15 |              0.23 |             -0.25 |             -0.04 |             -0.23 |              0.03 |              0.29 |             -0.13 |                    -0.18 |                     1    |

