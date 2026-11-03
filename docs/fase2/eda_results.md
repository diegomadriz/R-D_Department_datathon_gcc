# Fase 2: resultados descriptivos de train


Generado con `uv run python -m gcc_strength.eda`. Población: **621 filas de train**.

No se usan valores del lockbox. Ninguna tabla es una métrica predictiva de CV.

Método: casos observados por variable/pareja; asimetría muestral y curtosis de exceso.

Spearman parcial: rangos de cada pareja residualizados contra intercepto + dummies de 8 bloques.

Tabla 05 (parciales) y 19b: bloques por cuantiles del rango de `_row_id` dentro de Type 1 y de los casos completos de cada prueba (8/24/48); controles adicionales también en rangos. No conservan grupos íntegros. La tabla 19b es sensibilidad puntual, sin IC ni ajuste por multiplicidad.

Bloques: anchura igual sobre min/max de `_row_id` en train; grupos íntegros asignados por su primera fila.

Física: IC percentil 95%, 2000 remuestreos de grupos dentro de bloque; semilla 42.

Se recalculan rangos en cada remuestreo; el contraste de edades usa casos completos comunes.

La tabla 19 ajusta solo por bloque; 19b añade los controles indicados. Ninguna asociación demuestra causalidad.

IC marginales exploratorios, sin corrección por multiplicidad; no resuelven dependencia serial residual.

Neutralidad E: IC completo dentro de [−0.10, +0.10]; cruzar cero no demuestra equivalencia.

VIF: solo Types 1–3, statsmodels, intercepto, casos completos por diseño; no imputación. `inf` indica singularidad.

Cinética: mínimos cuadrados en unidades del target, 3 edades; f28>0, 0≤s≤5; 0.05≤a≤1.5.

El exponente libre se comparte por tipo; tres inicios deterministas. Error **in-sample**.

Dos parámetros por muestra para tres puntos, más un exponente compartido: los MAPE libres bajo el piso de ruido de conf/thresholds.yaml son sobreajuste por construcción, no evidencia predictiva. Curva: interpolación.

MAD=0: valores distintos de la mediana reciben z robusto infinito. No se elimina ningún outlier.

Logs/cocientes positivos: operandos no positivos o faltantes → NaN + bandera. F/(E+F) admite E/F iguales a cero si E+F>0 y ambos son no negativos. Sin epsilon ni imputación. Tabla 17d: una candidata añadida a la vez.

Los ratios presuponen mínimos físicos nulos, todavía no confirmados; no hay cierre composicional.

Types 4 y 5: descriptivos no confiables. Un target sin observaciones no admite ajuste cinético.



## 01_descriptivos


| scope   | column                 |   n_rows |   n |      mean |      std |       min |       q25 |    median |       q75 |       max |     skew |   kurtosis_excess | reliable   |
|:--------|:-----------------------|---------:|----:|----------:|---------:|----------:|----------:|----------:|----------:|----------:|---------:|------------------:|:-----------|
| Todos   | Physical_Metric_A      |      621 | 154 |    0.6460 |   0.1767 |    0.0000 |    0.6000 |    0.6918 |    0.7534 |    1.0000 |  -1.3820 |            2.1410 | True       |
| Todos   | Physical_Metric_B      |      621 | 615 |    0.0051 |   1.0207 |   -2.3263 |   -0.4798 |   -0.1646 |    0.3759 |    8.6399 |   3.2628 |           21.1719 | True       |
| Todos   | Mineral_Phase_A        |      621 | 576 |    0.8110 |   0.0505 |    0.4760 |    0.7856 |    0.8100 |    0.8398 |    1.0000 |  -1.1659 |            8.4181 | True       |
| Todos   | Mineral_Phase_B        |      621 | 576 |    0.3873 |   0.0727 |    0.0345 |    0.3554 |    0.3851 |    0.4181 |    1.0000 |   0.7800 |           10.8974 | True       |
| Todos   | Mineral_Phase_C        |      621 | 576 |    0.7366 |   0.0731 |    0.2875 |    0.6959 |    0.7381 |    0.7732 |    1.0000 |  -0.5933 |            4.4647 | True       |
| Todos   | Mineral_Phase_D        |      621 | 576 |    0.2076 |   0.0854 |    0.0351 |    0.1939 |    0.2187 |    0.2397 |    1.0000 |   3.8690 |           36.1879 | True       |
| Todos   | Mineral_Phase_E        |      621 | 575 |    0.4081 |   0.1307 |    0.0059 |    0.3724 |    0.4340 |    0.4897 |    1.0000 |  -0.9647 |            1.5603 | True       |
| Todos   | Mineral_Phase_F        |      621 | 576 |    0.4966 |   0.1680 |    0.0000 |    0.3774 |    0.4903 |    0.6109 |    0.9416 |   0.1630 |           -0.2693 | True       |
| Todos   | Mineral_Phase_G        |      621 | 576 |    0.0273 |   0.0131 |    0.0000 |    0.0184 |    0.0259 |    0.0334 |    0.0933 |   1.2362 |            3.0001 | True       |
| Todos   | Mineral_Phase_H        |      621 | 576 |    0.4054 |   0.1823 |    0.0000 |    0.2857 |    0.3929 |    0.5097 |    1.0000 |   0.2869 |           -0.1013 | True       |
| Todos   | Chemical_Composition_A |      621 | 610 |    0.6514 |   0.0922 |    0.0000 |    0.6069 |    0.6611 |    0.7056 |    0.9889 |  -1.3104 |            8.3929 | True       |
| Todos   | Chemical_Composition_B |      621 | 609 |    0.0973 |   0.0467 |    0.0000 |    0.0774 |    0.0952 |    0.1131 |    1.0000 |  11.8502 |          229.3842 | True       |
| Todos   | Strength1              |      621 | 592 | 2300.9291 | 352.4403 | 1710.0000 | 2110.0000 | 2250.0000 | 2380.0000 | 4010.0000 |   2.5524 |            8.4340 | True       |
| Todos   | Strength7              |      621 | 614 | 6056.6938 | 553.5706 | 2490.0000 | 5752.5000 | 5990.0000 | 6290.0000 | 8210.0000 |  -0.3437 |            7.9513 | True       |
| Todos   | Strength28             |      621 | 588 | 7474.0816 | 660.7974 | 2800.0000 | 7190.0000 | 7505.0000 | 7900.0000 | 9130.0000 |  -1.6090 |            8.9656 | True       |
| Type 1  | Physical_Metric_A      |      537 | 124 |    0.6851 |   0.0980 |    0.3808 |    0.6329 |    0.7014 |    0.7534 |    0.9014 |  -0.7161 |            0.7694 | True       |
| Type 1  | Physical_Metric_B      |      537 | 532 |   -0.0597 |   0.5179 |   -1.2454 |   -0.4348 |   -0.1646 |    0.3308 |    1.3441 |   0.3594 |           -0.5570 | True       |
| Type 1  | Mineral_Phase_A        |      537 | 495 |    0.8027 |   0.0337 |    0.6870 |    0.7800 |    0.8047 |    0.8286 |    0.8796 |  -0.3902 |           -0.1615 | True       |
| Type 1  | Mineral_Phase_B        |      537 | 495 |    0.3865 |   0.0695 |    0.0345 |    0.3555 |    0.3841 |    0.4160 |    1.0000 |   1.0034 |           14.7060 | True       |
| Type 1  | Mineral_Phase_C        |      537 | 495 |    0.7269 |   0.0599 |    0.2875 |    0.6923 |    0.7283 |    0.7629 |    0.9191 |  -0.8707 |            6.2593 | True       |
| Type 1  | Mineral_Phase_D        |      537 | 495 |    0.2221 |   0.0351 |    0.0644 |    0.2050 |    0.2223 |    0.2414 |    0.3202 |  -0.7311 |            2.6070 | True       |
| Type 1  | Mineral_Phase_E        |      537 | 494 |    0.4146 |   0.1326 |    0.0059 |    0.3783 |    0.4399 |    0.4956 |    1.0000 |  -1.0235 |            1.7427 | True       |
| Type 1  | Mineral_Phase_F        |      537 | 495 |    0.4929 |   0.1659 |    0.0000 |    0.3755 |    0.4942 |    0.6051 |    0.9416 |   0.0812 |           -0.3090 | True       |
| Type 1  | Mineral_Phase_G        |      537 | 495 |    0.0266 |   0.0125 |    0.0020 |    0.0184 |    0.0245 |    0.0327 |    0.0933 |   1.3080 |            3.5138 | True       |
| Type 1  | Mineral_Phase_H        |      537 | 495 |    0.4019 |   0.1749 |    0.0000 |    0.2922 |    0.3929 |    0.4984 |    0.9351 |   0.1543 |           -0.1005 | True       |
| Type 1  | Chemical_Composition_A |      537 | 529 |    0.6481 |   0.0776 |    0.2611 |    0.6056 |    0.6611 |    0.7000 |    0.8944 |  -0.7879 |            1.9278 | True       |
| Type 1  | Chemical_Composition_B |      537 | 528 |    0.0938 |   0.0282 |    0.0000 |    0.0774 |    0.0952 |    0.1131 |    0.1786 |  -0.1230 |            0.7352 | True       |
| Type 1  | Strength1              |      537 | 517 | 2242.4178 | 201.6717 | 1730.0000 | 2110.0000 | 2240.0000 | 2360.0000 | 2870.0000 |   0.2825 |            0.1039 | True       |
| Type 1  | Strength7              |      537 | 530 | 6014.5094 | 379.9247 | 4980.0000 | 5750.0000 | 5980.0000 | 6240.0000 | 7280.0000 |   0.4978 |            0.3463 | True       |
| Type 1  | Strength28             |      537 | 511 | 7425.9687 | 524.8266 | 6070.0000 | 7135.0000 | 7460.0000 | 7785.0000 | 8790.0000 |  -0.2893 |           -0.2749 | True       |
| Type 2  | Physical_Metric_A      |       27 |   3 |    0.9927 |   0.0069 |    0.9863 |    0.9890 |    0.9918 |    0.9959 |    1.0000 |   0.5856 |          nan      | False      |
| Type 2  | Physical_Metric_B      |       27 |  27 |    2.4642 |   0.5054 |    1.2315 |    2.1548 |    2.4475 |    2.7402 |    3.5509 |  -0.1309 |            0.3573 | True       |
| Type 2  | Mineral_Phase_A        |       27 |  25 |    0.9045 |   0.0315 |    0.8484 |    0.8925 |    0.8996 |    0.9170 |    1.0000 |   0.9296 |            2.4126 | True       |
| Type 2  | Mineral_Phase_B        |       27 |  25 |    0.4066 |   0.0999 |    0.1773 |    0.3666 |    0.4006 |    0.4459 |    0.6364 |   0.2059 |            1.0826 | True       |
| Type 2  | Mineral_Phase_C        |       27 |  25 |    0.8382 |   0.0632 |    0.6921 |    0.8036 |    0.8432 |    0.8755 |    0.9338 |  -0.5738 |           -0.0901 | True       |
| Type 2  | Mineral_Phase_D        |       27 |  25 |    0.0649 |   0.0160 |    0.0427 |    0.0561 |    0.0600 |    0.0700 |    0.1238 |   2.1868 |            6.9438 | True       |
| Type 2  | Mineral_Phase_E        |       27 |  25 |    0.3689 |   0.1009 |    0.1496 |    0.2874 |    0.3959 |    0.4457 |    0.5367 |  -0.5206 |           -0.5024 | True       |
| Type 2  | Mineral_Phase_F        |       27 |  25 |    0.6219 |   0.1458 |    0.3619 |    0.5447 |    0.6070 |    0.7082 |    0.9339 |   0.1672 |            0.3065 | True       |
| Type 2  | Mineral_Phase_G        |       27 |  25 |    0.0307 |   0.0106 |    0.0041 |    0.0218 |    0.0300 |    0.0395 |    0.0456 |  -0.5519 |           -0.1621 | True       |
| Type 2  | Mineral_Phase_H        |       27 |  25 |    0.6160 |   0.1601 |    0.3149 |    0.5130 |    0.6364 |    0.7500 |    0.8734 |  -0.2551 |           -0.9365 | True       |
| Type 2  | Chemical_Composition_A |       27 |  27 |    0.7971 |   0.0847 |    0.6444 |    0.7528 |    0.7889 |    0.8278 |    0.9889 |   0.6422 |            0.6233 | True       |
| Type 2  | Chemical_Composition_B |       27 |  27 |    0.0983 |   0.0275 |    0.0357 |    0.0893 |    0.0893 |    0.1161 |    0.1667 |   0.0907 |            0.9108 | True       |
| Type 2  | Strength1              |       27 |  25 | 3673.2000 | 204.9943 | 3240.0000 | 3490.0000 | 3670.0000 | 3820.0000 | 4010.0000 |  -0.0916 |           -0.7257 | True       |
| Type 2  | Strength7              |       27 |  27 | 7494.0741 | 379.7900 | 6750.0000 | 7265.0000 | 7610.0000 | 7760.0000 | 8210.0000 |  -0.3676 |           -0.5831 | True       |
| Type 2  | Strength28             |       27 |  27 | 8428.1481 | 487.2137 | 7380.0000 | 8050.0000 | 8540.0000 | 8740.0000 | 9130.0000 |  -0.6937 |           -0.4522 | True       |
| Type 3  | Physical_Metric_A      |       52 |  27 |    0.4278 |   0.2601 |    0.0000 |    0.2068 |    0.3288 |    0.7068 |    0.8548 |   0.2152 |           -1.4936 | True       |
| Type 3  | Physical_Metric_B      |       52 |  52 |   -1.1930 |   0.5723 |   -2.3263 |   -1.6733 |   -1.1328 |   -0.8626 |    0.3984 |   0.1623 |           -0.2222 | True       |
| Type 3  | Mineral_Phase_A        |       52 |  51 |    0.8725 |   0.0319 |    0.7928 |    0.8519 |    0.8689 |    0.8938 |    0.9585 |   0.1876 |            0.5035 | True       |
| Type 3  | Mineral_Phase_B        |       52 |  51 |    0.3945 |   0.0849 |    0.1563 |    0.3585 |    0.3973 |    0.4312 |    0.5964 |  -0.3326 |            1.6049 | True       |
| Type 3  | Mineral_Phase_C        |       52 |  51 |    0.8070 |   0.0723 |    0.6291 |    0.7668 |    0.8039 |    0.8436 |    1.0000 |   0.0793 |            1.1897 | True       |
| Type 3  | Mineral_Phase_D        |       52 |  51 |    0.0803 |   0.0225 |    0.0351 |    0.0676 |    0.0809 |    0.0923 |    0.1252 |   0.0602 |           -0.3884 | True       |
| Type 3  | Mineral_Phase_E        |       52 |  51 |    0.3710 |   0.1181 |    0.0440 |    0.3284 |    0.4018 |    0.4443 |    0.5308 |  -1.2391 |            1.0892 | True       |
| Type 3  | Mineral_Phase_F        |       52 |  51 |    0.4763 |   0.1809 |    0.1518 |    0.3502 |    0.4358 |    0.5292 |    0.9105 |   0.9354 |            0.3952 | True       |
| Type 3  | Mineral_Phase_G        |       52 |  51 |    0.0343 |   0.0170 |    0.0054 |    0.0255 |    0.0313 |    0.0405 |    0.0838 |   0.9329 |            0.9077 | True       |
| Type 3  | Mineral_Phase_H        |       52 |  51 |    0.3316 |   0.1740 |    0.0682 |    0.2013 |    0.2955 |    0.3815 |    0.8344 |   1.2873 |            1.5587 | True       |
| Type 3  | Chemical_Composition_A |       52 |  51 |    0.6406 |   0.0764 |    0.4500 |    0.5917 |    0.6389 |    0.6972 |    0.8278 |   0.0725 |            0.2136 | True       |
| Type 3  | Chemical_Composition_B |       52 |  51 |    0.1171 |   0.0267 |    0.0417 |    0.1042 |    0.1190 |    0.1339 |    0.1845 |  -0.3337 |            1.0646 | True       |
| Type 3  | Strength1              |       52 |  49 | 2230.2041 | 202.7467 | 1780.0000 | 2110.0000 | 2220.0000 | 2310.0000 | 2970.0000 |   0.8204 |            2.6671 | True       |
| Type 3  | Strength7              |       52 |  52 | 6018.2692 | 381.5588 | 5120.0000 | 5745.0000 | 5975.0000 | 6325.0000 | 6790.0000 |  -0.1543 |           -0.4737 | True       |
| Type 3  | Strength28             |       52 |  45 | 7849.3333 | 423.9500 | 6950.0000 | 7520.0000 | 7920.0000 | 8140.0000 | 8720.0000 |  -0.1754 |           -0.6119 | True       |
| Type 4  | Physical_Metric_A      |        4 |   0 |  nan      | nan      |  nan      |  nan      |  nan      |  nan      |  nan      | nan      |          nan      | False      |
| Type 4  | Physical_Metric_B      |        4 |   4 |    7.5984 |   1.0237 |    6.5007 |    6.8554 |    7.6266 |    8.3697 |    8.6399 |  -0.0753 |           -4.4070 | False      |
| Type 4  | Mineral_Phase_A        |        4 |   4 |    0.5158 |   0.0440 |    0.4760 |    0.4791 |    0.5130 |    0.5497 |    0.5611 |   0.0958 |           -5.3660 | False      |
| Type 4  | Mineral_Phase_B        |        4 |   4 |    0.2846 |   0.0104 |    0.2693 |    0.2832 |    0.2881 |    0.2895 |    0.2929 |  -1.7045 |            3.2237 | False      |
| Type 4  | Mineral_Phase_C        |        4 |   4 |    0.4433 |   0.0576 |    0.3923 |    0.3962 |    0.4360 |    0.4831 |    0.5088 |   0.2890 |           -4.2819 | False      |
| Type 4  | Mineral_Phase_D        |        4 |   4 |    0.9319 |   0.0788 |    0.8237 |    0.8992 |    0.9519 |    0.9845 |    1.0000 |  -1.1456 |            0.5992 | False      |
| Type 4  | Mineral_Phase_E        |        4 |   4 |    0.3087 |   0.0633 |    0.2375 |    0.2705 |    0.3065 |    0.3446 |    0.3842 |   0.1651 |           -1.2534 | False      |
| Type 4  | Mineral_Phase_F        |        4 |   4 |    0.4387 |   0.1367 |    0.3346 |    0.3346 |    0.3988 |    0.5029 |    0.6226 |   1.0370 |           -0.3495 | False      |
| Type 4  | Mineral_Phase_G        |        4 |   4 |    0.0104 |   0.0102 |    0.0000 |    0.0036 |    0.0092 |    0.0160 |    0.0232 |   0.5121 |           -1.4533 | False      |
| Type 4  | Mineral_Phase_H        |        4 |   4 |    0.3157 |   0.1801 |    0.1169 |    0.1899 |    0.3231 |    0.4489 |    0.5000 |  -0.1245 |           -3.9737 | False      |
| Type 4  | Chemical_Composition_A |        4 |   2 |    0.0111 |   0.0157 |    0.0000 |    0.0056 |    0.0111 |    0.0167 |    0.0222 | nan      |          nan      | False      |
| Type 4  | Chemical_Composition_B |        4 |   2 |    0.0298 |   0.0084 |    0.0238 |    0.0268 |    0.0298 |    0.0327 |    0.0357 | nan      |          nan      | False      |
| Type 4  | Strength1              |        4 |   0 |  nan      | nan      |  nan      |  nan      |  nan      |  nan      |  nan      | nan      |          nan      | False      |
| Type 4  | Strength7              |        4 |   4 | 2897.5000 | 310.6311 | 2490.0000 | 2745.0000 | 2955.0000 | 3107.5000 | 3190.0000 |  -0.8201 |           -0.6027 | False      |
| Type 4  | Strength28             |        4 |   4 | 3432.5000 | 496.0763 | 2800.0000 | 3190.0000 | 3485.0000 | 3727.5000 | 3960.0000 |  -0.5318 |           -0.2974 | False      |
| Type 5  | Physical_Metric_A      |        1 |   0 |  nan      | nan      |  nan      |  nan      |  nan      |  nan      |  nan      | nan      |          nan      | False      |
| Type 5  | Physical_Metric_B      |        1 |   0 |  nan      | nan      |  nan      |  nan      |  nan      |  nan      |  nan      | nan      |          nan      | False      |
| Type 5  | Mineral_Phase_A        |        1 |   1 |    0.6317 | nan      |    0.6317 |    0.6317 |    0.6317 |    0.6317 |    0.6317 | nan      |          nan      | False      |
| Type 5  | Mineral_Phase_B        |        1 |   1 |    0.3311 | nan      |    0.3311 |    0.3311 |    0.3311 |    0.3311 |    0.3311 | nan      |          nan      | False      |
| Type 5  | Mineral_Phase_C        |        1 |   1 |    0.5541 | nan      |    0.5541 |    0.5541 |    0.5541 |    0.5541 |    0.5541 | nan      |          nan      | False      |
| Type 5  | Mineral_Phase_D        |        1 |   1 |    0.1911 | nan      |    0.1911 |    0.1911 |    0.1911 |    0.1911 |    0.1911 | nan      |          nan      | False      |
| Type 5  | Mineral_Phase_E        |        1 |   1 |    0.4692 | nan      |    0.4692 |    0.4692 |    0.4692 |    0.4692 |    0.4692 | nan      |          nan      | False      |
| Type 5  | Mineral_Phase_F        |        1 |   1 |    0.4630 | nan      |    0.4630 |    0.4630 |    0.4630 |    0.4630 |    0.4630 | nan      |          nan      | False      |
| Type 5  | Mineral_Phase_G        |        1 |   1 |    0.0095 | nan      |    0.0095 |    0.0095 |    0.0095 |    0.0095 |    0.0095 | nan      |          nan      | False      |
| Type 5  | Mineral_Phase_H        |        1 |   1 |    1.0000 | nan      |    1.0000 |    1.0000 |    1.0000 |    1.0000 |    1.0000 | nan      |          nan      | False      |
| Type 5  | Chemical_Composition_A |        1 |   1 |    0.3111 | nan      |    0.3111 |    0.3111 |    0.3111 |    0.3111 |    0.3111 | nan      |          nan      | False      |
| Type 5  | Chemical_Composition_B |        1 |   1 |    1.0000 | nan      |    1.0000 |    1.0000 |    1.0000 |    1.0000 |    1.0000 | nan      |          nan      | False      |
| Type 5  | Strength1              |        1 |   1 | 1710.0000 | nan      | 1710.0000 | 1710.0000 | 1710.0000 | 1710.0000 | 1710.0000 | nan      |          nan      | False      |
| Type 5  | Strength7              |        1 |   1 | 4240.0000 | nan      | 4240.0000 | 4240.0000 | 4240.0000 | 4240.0000 | 4240.0000 | nan      |          nan      | False      |
| Type 5  | Strength28             |        1 |   1 | 5580.0000 | nan      | 5580.0000 | 5580.0000 | 5580.0000 | 5580.0000 | 5580.0000 | nan      |          nan      | False      |




## 02_nulos


| column                    |   n_null |   pct_null |
|:--------------------------|---------:|-----------:|
| _row_id                   |        0 |     0.0000 |
| ProductType               |        0 |     0.0000 |
| Physical_Metric_A         |      467 |    75.2013 |
| Physical_Metric_B         |        6 |     0.9662 |
| Mineral_Phase_A           |       45 |     7.2464 |
| Mineral_Phase_B           |       45 |     7.2464 |
| Mineral_Phase_C           |       45 |     7.2464 |
| Mineral_Phase_D           |       45 |     7.2464 |
| Mineral_Phase_E           |       46 |     7.4074 |
| Mineral_Phase_F           |       45 |     7.2464 |
| Mineral_Phase_G           |       45 |     7.2464 |
| Mineral_Phase_H           |       45 |     7.2464 |
| Chemical_Composition_A    |       11 |     1.7713 |
| Chemical_Composition_B    |       12 |     1.9324 |
| Strength1                 |       29 |     4.6699 |
| Strength7                 |        7 |     1.1272 |
| Strength28                |       33 |     5.3140 |
| flag_phases_anomalous     |        0 |     0.0000 |
| flag_zero_Mineral_Phase_F |        0 |     0.0000 |
| flag_zero_Mineral_Phase_H |        0 |     0.0000 |
| flag_r1_suspect           |        0 |     0.0000 |
| strength_group_id         |        0 |     0.0000 |




## 03_outliers


| column                 |   n |   iqr_low |   iqr_high |   iqr_outliers |      mad | mad_zero   |   robust_z_outliers |
|:-----------------------|----:|----------:|-----------:|---------------:|---------:|:-----------|--------------------:|
| Physical_Metric_A      | 154 |    0.3699 |     0.9836 |             18 |   0.0699 | False      |                  14 |
| Physical_Metric_B      | 615 |   -1.7633 |     1.6594 |             42 |   0.4053 | False      |                  28 |
| Mineral_Phase_A        | 576 |    0.7044 |     0.9211 |             16 |   0.0287 | False      |                   6 |
| Mineral_Phase_B        | 576 |    0.2613 |     0.5122 |             46 |   0.0309 | False      |                  30 |
| Mineral_Phase_C        | 576 |    0.5801 |     0.8890 |             31 |   0.0386 | False      |                   8 |
| Mineral_Phase_D        | 576 |    0.1251 |     0.3084 |             95 |   0.0220 | False      |                  74 |
| Mineral_Phase_E        | 575 |    0.1965 |     0.6657 |             62 |   0.0587 | False      |                  44 |
| Mineral_Phase_F        | 576 |    0.0272 |     0.9611 |              1 |   0.1167 | False      |                   0 |
| Mineral_Phase_G        | 576 |   -0.0041 |     0.0559 |             20 |   0.0075 | False      |                  11 |
| Mineral_Phase_H        | 576 |   -0.0503 |     0.8458 |              3 |   0.1071 | False      |                   1 |
| Chemical_Composition_A | 610 |    0.4590 |     0.8535 |             16 |   0.0472 | False      |                   9 |
| Chemical_Composition_B | 609 |    0.0238 |     0.1667 |             17 |   0.0179 | False      |                   3 |
| Strength1              | 592 | 1705.0000 |  2785.0000 |             30 | 140.0000 | False      |                  25 |
| Strength7              | 614 | 4946.2500 |  7096.2500 |             32 | 260.0000 | False      |                  22 |
| Strength28             | 588 | 6125.0000 |  8965.0000 |             11 | 350.0000 | False      |                   5 |




## 04_correlaciones


| scope   | method           | feature                | target     |   n |     rho |
|:--------|:-----------------|:-----------------------|:-----------|----:|--------:|
| Todos   | pearson          | Physical_Metric_A      | Strength1  | 153 |  0.2644 |
| Todos   | spearman         | Physical_Metric_A      | Strength1  | 153 |  0.2062 |
| Todos   | pearson          | Physical_Metric_A      | Strength7  | 150 |  0.4340 |
| Todos   | spearman         | Physical_Metric_A      | Strength7  | 150 |  0.3798 |
| Todos   | pearson          | Physical_Metric_A      | Strength28 | 131 |  0.0060 |
| Todos   | spearman         | Physical_Metric_A      | Strength28 | 131 |  0.0648 |
| Todos   | pearson          | Physical_Metric_B      | Strength1  | 586 |  0.7230 |
| Todos   | spearman         | Physical_Metric_B      | Strength1  | 586 |  0.5136 |
| Todos   | pearson          | Physical_Metric_B      | Strength7  | 608 | -0.0039 |
| Todos   | spearman         | Physical_Metric_B      | Strength7  | 608 |  0.0815 |
| Todos   | pearson          | Physical_Metric_B      | Strength28 | 582 | -0.3747 |
| Todos   | spearman         | Physical_Metric_B      | Strength28 | 582 | -0.2893 |
| Todos   | pearson          | Mineral_Phase_A        | Strength1  | 547 |  0.5741 |
| Todos   | spearman         | Mineral_Phase_A        | Strength1  | 547 |  0.5031 |
| Todos   | pearson          | Mineral_Phase_A        | Strength7  | 569 |  0.5021 |
| Todos   | spearman         | Mineral_Phase_A        | Strength7  | 569 |  0.1778 |
| Todos   | pearson          | Mineral_Phase_A        | Strength28 | 544 |  0.2615 |
| Todos   | spearman         | Mineral_Phase_A        | Strength28 | 544 | -0.0911 |
| Todos   | pearson          | Mineral_Phase_B        | Strength1  | 547 |  0.0932 |
| Todos   | spearman         | Mineral_Phase_B        | Strength1  | 547 |  0.1565 |
| Todos   | pearson          | Mineral_Phase_B        | Strength7  | 569 |  0.2053 |
| Todos   | spearman         | Mineral_Phase_B        | Strength7  | 569 |  0.1918 |
| Todos   | pearson          | Mineral_Phase_B        | Strength28 | 544 |  0.1963 |
| Todos   | spearman         | Mineral_Phase_B        | Strength28 | 544 |  0.1186 |
| Todos   | pearson          | Mineral_Phase_C        | Strength1  | 547 |  0.3852 |
| Todos   | spearman         | Mineral_Phase_C        | Strength1  | 547 |  0.3504 |
| Todos   | pearson          | Mineral_Phase_C        | Strength7  | 569 |  0.2899 |
| Todos   | spearman         | Mineral_Phase_C        | Strength7  | 569 |  0.0571 |
| Todos   | pearson          | Mineral_Phase_C        | Strength28 | 544 |  0.0918 |
| Todos   | spearman         | Mineral_Phase_C        | Strength28 | 544 | -0.1376 |
| Todos   | pearson          | Mineral_Phase_D        | Strength1  | 547 | -0.4728 |
| Todos   | spearman         | Mineral_Phase_D        | Strength1  | 547 | -0.3270 |
| Todos   | pearson          | Mineral_Phase_D        | Strength7  | 569 | -0.5493 |
| Todos   | spearman         | Mineral_Phase_D        | Strength7  | 569 | -0.1758 |
| Todos   | pearson          | Mineral_Phase_D        | Strength28 | 544 | -0.5162 |
| Todos   | spearman         | Mineral_Phase_D        | Strength28 | 544 | -0.0760 |
| Todos   | pearson          | Mineral_Phase_E        | Strength1  | 546 |  0.0759 |
| Todos   | spearman         | Mineral_Phase_E        | Strength1  | 546 |  0.0923 |
| Todos   | pearson          | Mineral_Phase_E        | Strength7  | 568 |  0.0726 |
| Todos   | spearman         | Mineral_Phase_E        | Strength7  | 568 |  0.0577 |
| Todos   | pearson          | Mineral_Phase_E        | Strength28 | 543 | -0.0837 |
| Todos   | spearman         | Mineral_Phase_E        | Strength28 | 543 | -0.1523 |
| Todos   | pearson          | Mineral_Phase_F        | Strength1  | 547 |  0.2690 |
| Todos   | spearman         | Mineral_Phase_F        | Strength1  | 547 |  0.2838 |
| Todos   | pearson          | Mineral_Phase_F        | Strength7  | 569 |  0.1439 |
| Todos   | spearman         | Mineral_Phase_F        | Strength7  | 569 |  0.0949 |
| Todos   | pearson          | Mineral_Phase_F        | Strength28 | 544 | -0.1236 |
| Todos   | spearman         | Mineral_Phase_F        | Strength28 | 544 | -0.1650 |
| Todos   | pearson          | Mineral_Phase_G        | Strength1  | 547 |  0.0826 |
| Todos   | spearman         | Mineral_Phase_G        | Strength1  | 547 |  0.0350 |
| Todos   | pearson          | Mineral_Phase_G        | Strength7  | 569 |  0.0330 |
| Todos   | spearman         | Mineral_Phase_G        | Strength7  | 569 | -0.1000 |
| Todos   | pearson          | Mineral_Phase_G        | Strength28 | 544 |  0.0778 |
| Todos   | spearman         | Mineral_Phase_G        | Strength28 | 544 | -0.0079 |
| Todos   | pearson          | Mineral_Phase_H        | Strength1  | 547 |  0.4402 |
| Todos   | spearman         | Mineral_Phase_H        | Strength1  | 547 |  0.4830 |
| Todos   | pearson          | Mineral_Phase_H        | Strength7  | 569 |  0.1422 |
| Todos   | spearman         | Mineral_Phase_H        | Strength7  | 569 |  0.0770 |
| Todos   | pearson          | Mineral_Phase_H        | Strength28 | 544 | -0.1131 |
| Todos   | spearman         | Mineral_Phase_H        | Strength28 | 544 | -0.1330 |
| Todos   | pearson          | Chemical_Composition_A | Strength1  | 583 |  0.4077 |
| Todos   | spearman         | Chemical_Composition_A | Strength1  | 583 |  0.3197 |
| Todos   | pearson          | Chemical_Composition_A | Strength7  | 604 |  0.3928 |
| Todos   | spearman         | Chemical_Composition_A | Strength7  | 604 |  0.1792 |
| Todos   | pearson          | Chemical_Composition_A | Strength28 | 579 |  0.2423 |
| Todos   | spearman         | Chemical_Composition_A | Strength28 | 579 |  0.0453 |
| Todos   | pearson          | Chemical_Composition_B | Strength1  | 582 |  0.0034 |
| Todos   | spearman         | Chemical_Composition_B | Strength1  | 582 |  0.0532 |
| Todos   | pearson          | Chemical_Composition_B | Strength7  | 603 | -0.0395 |
| Todos   | spearman         | Chemical_Composition_B | Strength7  | 603 |  0.0668 |
| Todos   | pearson          | Chemical_Composition_B | Strength28 | 578 | -0.0165 |
| Todos   | spearman         | Chemical_Composition_B | Strength28 | 578 |  0.0914 |
| Type 1  | pearson          | Physical_Metric_A      | Strength1  | 124 |  0.2192 |
| Type 1  | spearman         | Physical_Metric_A      | Strength1  | 124 |  0.1928 |
| Type 1  | partial_spearman | Physical_Metric_A      | Strength1  | 124 |  0.1886 |
| Type 1  | pearson          | Physical_Metric_A      | Strength7  | 120 |  0.2639 |
| Type 1  | spearman         | Physical_Metric_A      | Strength7  | 120 |  0.2611 |
| Type 1  | partial_spearman | Physical_Metric_A      | Strength7  | 120 |  0.2107 |
| Type 1  | pearson          | Physical_Metric_A      | Strength28 | 106 |  0.0759 |
| Type 1  | spearman         | Physical_Metric_A      | Strength28 | 106 |  0.0754 |
| Type 1  | partial_spearman | Physical_Metric_A      | Strength28 | 106 |  0.0862 |
| Type 1  | pearson          | Physical_Metric_B      | Strength1  | 512 |  0.5156 |
| Type 1  | spearman         | Physical_Metric_B      | Strength1  | 512 |  0.5021 |
| Type 1  | partial_spearman | Physical_Metric_B      | Strength1  | 512 |  0.1461 |
| Type 1  | pearson          | Physical_Metric_B      | Strength7  | 525 | -0.0333 |
| Type 1  | spearman         | Physical_Metric_B      | Strength7  | 525 | -0.0367 |
| Type 1  | partial_spearman | Physical_Metric_B      | Strength7  | 525 | -0.0967 |
| Type 1  | pearson          | Physical_Metric_B      | Strength28 | 506 | -0.4772 |
| Type 1  | spearman         | Physical_Metric_B      | Strength28 | 506 | -0.4142 |
| Type 1  | partial_spearman | Physical_Metric_B      | Strength28 | 506 | -0.1877 |
| Type 1  | pearson          | Mineral_Phase_A        | Strength1  | 475 |  0.4847 |
| Type 1  | spearman         | Mineral_Phase_A        | Strength1  | 475 |  0.5121 |
| Type 1  | partial_spearman | Mineral_Phase_A        | Strength1  | 475 |  0.1477 |
| Type 1  | pearson          | Mineral_Phase_A        | Strength7  | 488 |  0.0698 |
| Type 1  | spearman         | Mineral_Phase_A        | Strength7  | 488 |  0.0295 |
| Type 1  | partial_spearman | Mineral_Phase_A        | Strength7  | 488 |  0.0197 |
| Type 1  | pearson          | Mineral_Phase_A        | Strength28 | 470 | -0.4129 |
| Type 1  | spearman         | Mineral_Phase_A        | Strength28 | 470 | -0.4188 |
| Type 1  | partial_spearman | Mineral_Phase_A        | Strength28 | 470 | -0.0738 |
| Type 1  | pearson          | Mineral_Phase_B        | Strength1  | 475 |  0.1178 |
| Type 1  | spearman         | Mineral_Phase_B        | Strength1  | 475 |  0.1516 |
| Type 1  | partial_spearman | Mineral_Phase_B        | Strength1  | 475 |  0.1202 |
| Type 1  | pearson          | Mineral_Phase_B        | Strength7  | 488 |  0.1752 |
| Type 1  | spearman         | Mineral_Phase_B        | Strength7  | 488 |  0.1478 |
| Type 1  | partial_spearman | Mineral_Phase_B        | Strength7  | 488 |  0.0332 |
| Type 1  | pearson          | Mineral_Phase_B        | Strength28 | 470 |  0.1329 |
| Type 1  | spearman         | Mineral_Phase_B        | Strength28 | 470 |  0.0605 |
| Type 1  | partial_spearman | Mineral_Phase_B        | Strength28 | 470 | -0.0043 |
| Type 1  | pearson          | Mineral_Phase_C        | Strength1  | 475 |  0.2434 |
| Type 1  | spearman         | Mineral_Phase_C        | Strength1  | 475 |  0.3199 |
| Type 1  | partial_spearman | Mineral_Phase_C        | Strength1  | 475 |  0.0074 |
| Type 1  | pearson          | Mineral_Phase_C        | Strength7  | 488 | -0.0858 |
| Type 1  | spearman         | Mineral_Phase_C        | Strength7  | 488 | -0.0757 |
| Type 1  | partial_spearman | Mineral_Phase_C        | Strength7  | 488 | -0.0188 |
| Type 1  | pearson          | Mineral_Phase_C        | Strength28 | 470 | -0.3868 |
| Type 1  | spearman         | Mineral_Phase_C        | Strength28 | 470 | -0.3879 |
| Type 1  | partial_spearman | Mineral_Phase_C        | Strength28 | 470 | -0.0561 |
| Type 1  | pearson          | Mineral_Phase_D        | Strength1  | 475 | -0.2724 |
| Type 1  | spearman         | Mineral_Phase_D        | Strength1  | 475 | -0.3095 |
| Type 1  | partial_spearman | Mineral_Phase_D        | Strength1  | 475 |  0.0729 |
| Type 1  | pearson          | Mineral_Phase_D        | Strength7  | 488 | -0.0942 |
| Type 1  | spearman         | Mineral_Phase_D        | Strength7  | 488 | -0.0382 |
| Type 1  | partial_spearman | Mineral_Phase_D        | Strength7  | 488 |  0.1430 |
| Type 1  | pearson          | Mineral_Phase_D        | Strength28 | 470 |  0.1343 |
| Type 1  | spearman         | Mineral_Phase_D        | Strength28 | 470 |  0.2036 |
| Type 1  | partial_spearman | Mineral_Phase_D        | Strength28 | 470 |  0.0428 |
| Type 1  | pearson          | Mineral_Phase_E        | Strength1  | 474 |  0.2351 |
| Type 1  | spearman         | Mineral_Phase_E        | Strength1  | 474 |  0.1392 |
| Type 1  | partial_spearman | Mineral_Phase_E        | Strength1  | 474 | -0.0864 |
| Type 1  | pearson          | Mineral_Phase_E        | Strength7  | 487 |  0.1431 |
| Type 1  | spearman         | Mineral_Phase_E        | Strength7  | 487 |  0.1024 |
| Type 1  | partial_spearman | Mineral_Phase_E        | Strength7  | 487 | -0.0176 |
| Type 1  | pearson          | Mineral_Phase_E        | Strength28 | 469 | -0.0885 |
| Type 1  | spearman         | Mineral_Phase_E        | Strength28 | 469 | -0.1125 |
| Type 1  | partial_spearman | Mineral_Phase_E        | Strength28 | 469 | -0.0710 |
| Type 1  | pearson          | Mineral_Phase_F        | Strength1  | 475 |  0.2467 |
| Type 1  | spearman         | Mineral_Phase_F        | Strength1  | 475 |  0.2466 |
| Type 1  | partial_spearman | Mineral_Phase_F        | Strength1  | 475 | -0.0391 |
| Type 1  | pearson          | Mineral_Phase_F        | Strength7  | 488 |  0.0381 |
| Type 1  | spearman         | Mineral_Phase_F        | Strength7  | 488 |  0.0015 |
| Type 1  | partial_spearman | Mineral_Phase_F        | Strength7  | 488 | -0.0194 |
| Type 1  | pearson          | Mineral_Phase_F        | Strength28 | 470 | -0.2702 |
| Type 1  | spearman         | Mineral_Phase_F        | Strength28 | 470 | -0.2621 |
| Type 1  | partial_spearman | Mineral_Phase_F        | Strength28 | 470 | -0.0498 |
| Type 1  | pearson          | Mineral_Phase_G        | Strength1  | 475 |  0.0281 |
| Type 1  | spearman         | Mineral_Phase_G        | Strength1  | 475 | -0.0097 |
| Type 1  | partial_spearman | Mineral_Phase_G        | Strength1  | 475 | -0.0225 |
| Type 1  | pearson          | Mineral_Phase_G        | Strength7  | 488 | -0.1075 |
| Type 1  | spearman         | Mineral_Phase_G        | Strength7  | 488 | -0.1861 |
| Type 1  | partial_spearman | Mineral_Phase_G        | Strength7  | 488 | -0.2871 |
| Type 1  | pearson          | Mineral_Phase_G        | Strength28 | 470 | -0.0692 |
| Type 1  | spearman         | Mineral_Phase_G        | Strength28 | 470 | -0.1221 |
| Type 1  | partial_spearman | Mineral_Phase_G        | Strength28 | 470 | -0.2405 |
| Type 1  | pearson          | Mineral_Phase_H        | Strength1  | 475 |  0.4882 |
| Type 1  | spearman         | Mineral_Phase_H        | Strength1  | 475 |  0.4846 |
| Type 1  | partial_spearman | Mineral_Phase_H        | Strength1  | 475 |  0.2958 |
| Type 1  | pearson          | Mineral_Phase_H        | Strength7  | 488 |  0.0070 |
| Type 1  | spearman         | Mineral_Phase_H        | Strength7  | 488 | -0.0015 |
| Type 1  | partial_spearman | Mineral_Phase_H        | Strength7  | 488 |  0.0859 |
| Type 1  | pearson          | Mineral_Phase_H        | Strength28 | 470 | -0.2649 |
| Type 1  | spearman         | Mineral_Phase_H        | Strength28 | 470 | -0.2210 |
| Type 1  | partial_spearman | Mineral_Phase_H        | Strength28 | 470 |  0.0883 |
| Type 1  | pearson          | Chemical_Composition_A | Strength1  | 509 |  0.2204 |
| Type 1  | spearman         | Chemical_Composition_A | Strength1  | 509 |  0.2761 |
| Type 1  | partial_spearman | Chemical_Composition_A | Strength1  | 509 | -0.0370 |
| Type 1  | pearson          | Chemical_Composition_A | Strength7  | 523 |  0.0689 |
| Type 1  | spearman         | Chemical_Composition_A | Strength7  | 523 |  0.0809 |
| Type 1  | partial_spearman | Chemical_Composition_A | Strength7  | 523 | -0.0450 |
| Type 1  | pearson          | Chemical_Composition_A | Strength28 | 505 | -0.0807 |
| Type 1  | spearman         | Chemical_Composition_A | Strength28 | 505 | -0.0648 |
| Type 1  | partial_spearman | Chemical_Composition_A | Strength28 | 505 |  0.0181 |
| Type 1  | pearson          | Chemical_Composition_B | Strength1  | 508 |  0.1360 |
| Type 1  | spearman         | Chemical_Composition_B | Strength1  | 508 |  0.0555 |
| Type 1  | partial_spearman | Chemical_Composition_B | Strength1  | 508 |  0.2926 |
| Type 1  | pearson          | Chemical_Composition_B | Strength7  | 522 |  0.0758 |
| Type 1  | spearman         | Chemical_Composition_B | Strength7  | 522 |  0.0566 |
| Type 1  | partial_spearman | Chemical_Composition_B | Strength7  | 522 |  0.1311 |
| Type 1  | pearson          | Chemical_Composition_B | Strength28 | 504 |  0.0347 |
| Type 1  | spearman         | Chemical_Composition_B | Strength28 | 504 |  0.0306 |
| Type 1  | partial_spearman | Chemical_Composition_B | Strength28 | 504 | -0.0085 |




## 05_correlacion_targets


| scope   | method                       |   n_blocks | left      | right      |   n |     rho |
|:--------|:-----------------------------|-----------:|:----------|:-----------|----:|--------:|
| Todos   | pearson                      |          0 | Strength1 | Strength7  | 585 |  0.7434 |
| Todos   | pearson                      |          0 | Strength1 | Strength28 | 561 |  0.2723 |
| Todos   | pearson                      |          0 | Strength7 | Strength28 | 586 |  0.7773 |
| Todos   | spearman                     |          0 | Strength1 | Strength7  | 585 |  0.5639 |
| Todos   | spearman                     |          0 | Strength1 | Strength28 | 561 |  0.0833 |
| Todos   | spearman                     |          0 | Strength7 | Strength28 | 586 |  0.6808 |
| Type 1  | pearson                      |          0 | Strength1 | Strength7  | 510 |  0.5181 |
| Type 1  | pearson                      |          0 | Strength1 | Strength28 | 493 | -0.0646 |
| Type 1  | pearson                      |          0 | Strength7 | Strength28 | 509 |  0.6677 |
| Type 1  | spearman                     |          0 | Strength1 | Strength7  | 510 |  0.4869 |
| Type 1  | spearman                     |          0 | Strength1 | Strength28 | 493 | -0.0496 |
| Type 1  | spearman                     |          0 | Strength7 | Strength28 | 509 |  0.6541 |
| Type 1  | partial_spearman_rank_blocks |          8 | Strength1 | Strength7  | 510 |  0.6316 |
| Type 1  | partial_spearman_rank_blocks |          8 | Strength1 | Strength28 | 493 |  0.4087 |
| Type 1  | partial_spearman_rank_blocks |          8 | Strength7 | Strength28 | 509 |  0.7005 |
| Type 1  | partial_spearman_rank_blocks |         24 | Strength1 | Strength7  | 510 |  0.6328 |
| Type 1  | partial_spearman_rank_blocks |         24 | Strength1 | Strength28 | 493 |  0.4323 |
| Type 1  | partial_spearman_rank_blocks |         24 | Strength7 | Strength28 | 509 |  0.6522 |
| Type 1  | partial_spearman_rank_blocks |         48 | Strength1 | Strength7  | 510 |  0.6350 |
| Type 1  | partial_spearman_rank_blocks |         48 | Strength1 | Strength28 | 493 |  0.4240 |
| Type 1  | partial_spearman_rank_blocks |         48 | Strength7 | Strength28 | 509 |  0.6292 |




## 06_vif_con_A


| feature                |   n_complete |         vif |
|:-----------------------|-------------:|------------:|
| Physical_Metric_B      |          558 |      1.7954 |
| Mineral_Phase_A        |          558 | 426760.6532 |
| Mineral_Phase_B        |          558 | 344483.0041 |
| Mineral_Phase_C        |          558 | 708044.0830 |
| Mineral_Phase_D        |          558 |      2.5412 |
| Mineral_Phase_E        |          558 |      1.2901 |
| Mineral_Phase_F        |          558 |      1.4060 |
| Mineral_Phase_G        |          558 |      1.2433 |
| Mineral_Phase_H        |          558 |      1.4405 |
| Chemical_Composition_A |          558 |      1.6149 |
| Chemical_Composition_B |          558 |      1.3823 |




## 07_vif_sin_A


| feature                |   n_complete |    vif |
|:-----------------------|-------------:|-------:|
| Physical_Metric_B      |          558 | 1.7912 |
| Mineral_Phase_B        |          558 | 2.8144 |
| Mineral_Phase_C        |          558 | 4.0268 |
| Mineral_Phase_D        |          558 | 2.5292 |
| Mineral_Phase_E        |          558 | 1.2875 |
| Mineral_Phase_F        |          558 | 1.4014 |
| Mineral_Phase_G        |          558 | 1.2399 |
| Mineral_Phase_H        |          558 | 1.4400 |
| Chemical_Composition_A |          558 | 1.6148 |
| Chemical_Composition_B |          558 | 1.3816 |




## 08_vif_crudo_propuesto


| feature                |   n_complete |    vif |
|:-----------------------|-------------:|-------:|
| Physical_Metric_B      |          558 | 1.7912 |
| Mineral_Phase_B        |          558 | 2.8144 |
| Mineral_Phase_C        |          558 | 4.0268 |
| Mineral_Phase_D        |          558 | 2.5292 |
| Mineral_Phase_E        |          558 | 1.2875 |
| Mineral_Phase_F        |          558 | 1.4014 |
| Mineral_Phase_G        |          558 | 1.2399 |
| Mineral_Phase_H        |          558 | 1.4400 |
| Chemical_Composition_A |          558 | 1.6148 |
| Chemical_Composition_B |          558 | 1.3816 |




## 09_vif_descartes_crudos


| removed   | vif   | n_complete   |
|-----------|-------|--------------|




## 10_top_features


| scope   | method   | feature           | target     |   n |     rho |   abs_rho |
|:--------|:---------|:------------------|:-----------|----:|--------:|----------:|
| Type 1  | spearman | Mineral_Phase_A   | Strength1  | 475 |  0.5121 |    0.5121 |
| Type 1  | spearman | Physical_Metric_B | Strength1  | 512 |  0.5021 |    0.5021 |
| Type 1  | spearman | Mineral_Phase_H   | Strength1  | 475 |  0.4846 |    0.4846 |
| Type 1  | spearman | Mineral_Phase_A   | Strength28 | 470 | -0.4188 |    0.4188 |
| Type 1  | spearman | Physical_Metric_B | Strength28 | 506 | -0.4142 |    0.4142 |
| Type 1  | spearman | Mineral_Phase_C   | Strength28 | 470 | -0.3879 |    0.3879 |
| Type 1  | spearman | Physical_Metric_A | Strength7  | 120 |  0.2611 |    0.2611 |
| Type 1  | spearman | Mineral_Phase_G   | Strength7  | 488 | -0.1861 |    0.1861 |
| Type 1  | spearman | Mineral_Phase_B   | Strength7  | 488 |  0.1478 |    0.1478 |




## 11_ratios_nulos


| feature         |   n_invalid |   pct_invalid |
|:----------------|------------:|--------------:|
| log_C_over_B    |          45 |        7.2464 |
| C_over_A        |          45 |        7.2464 |
| log_D_over_A    |          45 |        7.2464 |
| log_F_over_E    |          47 |        7.5684 |
| log_A           |          45 |        7.2464 |
| log_H_over_F    |          49 |        7.8905 |
| F_over_E_plus_F |          46 |        7.4074 |
| log_H_over_A    |          49 |        7.8905 |




## 12_correlaciones_ratios


| scope   | method           | feature         | target     |   n |     rho |
|:--------|:-----------------|:----------------|:-----------|----:|--------:|
| Todos   | pearson          | log_C_over_B    | Strength1  | 547 |  0.0629 |
| Todos   | spearman         | log_C_over_B    | Strength1  | 547 |  0.0389 |
| Todos   | pearson          | log_C_over_B    | Strength7  | 569 | -0.0209 |
| Todos   | spearman         | log_C_over_B    | Strength7  | 569 | -0.0841 |
| Todos   | pearson          | log_C_over_B    | Strength28 | 544 | -0.0721 |
| Todos   | spearman         | log_C_over_B    | Strength28 | 544 | -0.1200 |
| Todos   | pearson          | C_over_A        | Strength1  | 547 |  0.0669 |
| Todos   | spearman         | C_over_A        | Strength1  | 547 |  0.0387 |
| Todos   | pearson          | C_over_A        | Strength7  | 569 | -0.0270 |
| Todos   | spearman         | C_over_A        | Strength7  | 569 | -0.0842 |
| Todos   | pearson          | C_over_A        | Strength28 | 544 | -0.0956 |
| Todos   | spearman         | C_over_A        | Strength28 | 544 | -0.1201 |
| Todos   | pearson          | log_D_over_A    | Strength1  | 547 | -0.5208 |
| Todos   | spearman         | log_D_over_A    | Strength1  | 547 | -0.4074 |
| Todos   | pearson          | log_D_over_A    | Strength7  | 569 | -0.4889 |
| Todos   | spearman         | log_D_over_A    | Strength7  | 569 | -0.1938 |
| Todos   | pearson          | log_D_over_A    | Strength28 | 544 | -0.4077 |
| Todos   | spearman         | log_D_over_A    | Strength28 | 544 | -0.0273 |
| Todos   | pearson          | log_F_over_E    | Strength1  | 545 |  0.0369 |
| Todos   | spearman         | log_F_over_E    | Strength1  | 545 |  0.1561 |
| Todos   | pearson          | log_F_over_E    | Strength7  | 567 |  0.0158 |
| Todos   | spearman         | log_F_over_E    | Strength7  | 567 |  0.0398 |
| Todos   | pearson          | log_F_over_E    | Strength28 | 542 |  0.0027 |
| Todos   | spearman         | log_F_over_E    | Strength28 | 542 | -0.0373 |
| Todos   | pearson          | log_A           | Strength1  | 547 |  0.5627 |
| Todos   | spearman         | log_A           | Strength1  | 547 |  0.5031 |
| Todos   | pearson          | log_A           | Strength7  | 569 |  0.5201 |
| Todos   | spearman         | log_A           | Strength7  | 569 |  0.1778 |
| Todos   | pearson          | log_A           | Strength28 | 544 |  0.2990 |
| Todos   | spearman         | log_A           | Strength28 | 544 | -0.0911 |
| Todos   | pearson          | log_H_over_F    | Strength1  | 544 |  0.1680 |
| Todos   | spearman         | log_H_over_F    | Strength1  | 544 |  0.1727 |
| Todos   | pearson          | log_H_over_F    | Strength7  | 565 |  0.0409 |
| Todos   | spearman         | log_H_over_F    | Strength7  | 565 | -0.0044 |
| Todos   | pearson          | log_H_over_F    | Strength28 | 540 |  0.0085 |
| Todos   | spearman         | log_H_over_F    | Strength28 | 540 |  0.0044 |
| Todos   | pearson          | F_over_E_plus_F | Strength1  | 546 |  0.0915 |
| Todos   | spearman         | F_over_E_plus_F | Strength1  | 546 |  0.1557 |
| Todos   | pearson          | F_over_E_plus_F | Strength7  | 568 |  0.0359 |
| Todos   | spearman         | F_over_E_plus_F | Strength7  | 568 |  0.0361 |
| Todos   | pearson          | F_over_E_plus_F | Strength28 | 543 | -0.0161 |
| Todos   | spearman         | F_over_E_plus_F | Strength28 | 543 | -0.0418 |
| Todos   | pearson          | log_H_over_A    | Strength1  | 544 |  0.3191 |
| Todos   | spearman         | log_H_over_A    | Strength1  | 544 |  0.4289 |
| Todos   | pearson          | log_H_over_A    | Strength7  | 565 |  0.0867 |
| Todos   | spearman         | log_H_over_A    | Strength7  | 565 |  0.0586 |
| Todos   | pearson          | log_H_over_A    | Strength28 | 540 | -0.0966 |
| Todos   | spearman         | log_H_over_A    | Strength28 | 540 | -0.1281 |
| Type 1  | pearson          | log_C_over_B    | Strength1  | 475 | -0.0182 |
| Type 1  | spearman         | log_C_over_B    | Strength1  | 475 |  0.0189 |
| Type 1  | partial_spearman | log_C_over_B    | Strength1  | 475 | -0.0812 |
| Type 1  | pearson          | log_C_over_B    | Strength7  | 488 | -0.1435 |
| Type 1  | spearman         | log_C_over_B    | Strength7  | 488 | -0.1296 |
| Type 1  | partial_spearman | log_C_over_B    | Strength7  | 488 | -0.0212 |
| Type 1  | pearson          | log_C_over_B    | Strength28 | 470 | -0.1892 |
| Type 1  | spearman         | log_C_over_B    | Strength28 | 470 | -0.1925 |
| Type 1  | partial_spearman | log_C_over_B    | Strength28 | 470 | -0.0050 |
| Type 1  | pearson          | C_over_A        | Strength1  | 475 | -0.0064 |
| Type 1  | spearman         | C_over_A        | Strength1  | 475 |  0.0187 |
| Type 1  | partial_spearman | C_over_A        | Strength1  | 475 | -0.0815 |
| Type 1  | pearson          | C_over_A        | Strength7  | 488 | -0.1540 |
| Type 1  | spearman         | C_over_A        | Strength7  | 488 | -0.1298 |
| Type 1  | partial_spearman | C_over_A        | Strength7  | 488 | -0.0214 |
| Type 1  | pearson          | C_over_A        | Strength28 | 470 | -0.2217 |
| Type 1  | spearman         | C_over_A        | Strength28 | 470 | -0.1926 |
| Type 1  | partial_spearman | C_over_A        | Strength28 | 470 | -0.0051 |
| Type 1  | pearson          | log_D_over_A    | Strength1  | 475 | -0.3264 |
| Type 1  | spearman         | log_D_over_A    | Strength1  | 475 | -0.4107 |
| Type 1  | partial_spearman | log_D_over_A    | Strength1  | 475 |  0.0217 |
| Type 1  | pearson          | log_D_over_A    | Strength7  | 488 | -0.1063 |
| Type 1  | spearman         | log_D_over_A    | Strength7  | 488 | -0.0554 |
| Type 1  | partial_spearman | log_D_over_A    | Strength7  | 488 |  0.1254 |
| Type 1  | pearson          | log_D_over_A    | Strength28 | 470 |  0.1772 |
| Type 1  | spearman         | log_D_over_A    | Strength28 | 470 |  0.2807 |
| Type 1  | partial_spearman | log_D_over_A    | Strength28 | 470 |  0.0505 |
| Type 1  | pearson          | log_F_over_E    | Strength1  | 473 | -0.0674 |
| Type 1  | spearman         | log_F_over_E    | Strength1  | 473 |  0.1052 |
| Type 1  | partial_spearman | log_F_over_E    | Strength1  | 473 |  0.0172 |
| Type 1  | pearson          | log_F_over_E    | Strength7  | 486 | -0.0677 |
| Type 1  | spearman         | log_F_over_E    | Strength7  | 486 | -0.0472 |
| Type 1  | partial_spearman | log_F_over_E    | Strength7  | 486 | -0.0021 |
| Type 1  | pearson          | log_F_over_E    | Strength28 | 468 | -0.0526 |
| Type 1  | spearman         | log_F_over_E    | Strength28 | 468 | -0.1280 |
| Type 1  | partial_spearman | log_F_over_E    | Strength28 | 468 |  0.0029 |
| Type 1  | pearson          | log_A           | Strength1  | 475 |  0.4814 |
| Type 1  | spearman         | log_A           | Strength1  | 475 |  0.5121 |
| Type 1  | partial_spearman | log_A           | Strength1  | 475 |  0.1477 |
| Type 1  | pearson          | log_A           | Strength7  | 488 |  0.0746 |
| Type 1  | spearman         | log_A           | Strength7  | 488 |  0.0295 |
| Type 1  | partial_spearman | log_A           | Strength7  | 488 |  0.0197 |
| Type 1  | pearson          | log_A           | Strength28 | 470 | -0.4061 |
| Type 1  | spearman         | log_A           | Strength28 | 470 | -0.4188 |
| Type 1  | partial_spearman | log_A           | Strength28 | 470 | -0.0738 |
| Type 1  | pearson          | log_H_over_F    | Strength1  | 472 |  0.2162 |
| Type 1  | spearman         | log_H_over_F    | Strength1  | 472 |  0.1805 |
| Type 1  | partial_spearman | log_H_over_F    | Strength1  | 472 |  0.1886 |
| Type 1  | pearson          | log_H_over_F    | Strength7  | 484 |  0.0130 |
| Type 1  | spearman         | log_H_over_F    | Strength7  | 484 | -0.0105 |
| Type 1  | partial_spearman | log_H_over_F    | Strength7  | 484 |  0.0544 |
| Type 1  | pearson          | log_H_over_F    | Strength28 | 466 | -0.0109 |
| Type 1  | spearman         | log_H_over_F    | Strength28 | 466 | -0.0023 |
| Type 1  | partial_spearman | log_H_over_F    | Strength28 | 466 |  0.0660 |
| Type 1  | pearson          | F_over_E_plus_F | Strength1  | 474 | -0.0123 |
| Type 1  | spearman         | F_over_E_plus_F | Strength1  | 474 |  0.1048 |
| Type 1  | partial_spearman | F_over_E_plus_F | Strength1  | 474 |  0.0146 |
| Type 1  | pearson          | F_over_E_plus_F | Strength7  | 487 | -0.0604 |
| Type 1  | spearman         | F_over_E_plus_F | Strength7  | 487 | -0.0516 |
| Type 1  | partial_spearman | F_over_E_plus_F | Strength7  | 487 | -0.0047 |
| Type 1  | pearson          | F_over_E_plus_F | Strength28 | 469 | -0.0883 |
| Type 1  | spearman         | F_over_E_plus_F | Strength28 | 469 | -0.1332 |
| Type 1  | partial_spearman | F_over_E_plus_F | Strength28 | 469 |  0.0004 |
| Type 1  | pearson          | log_H_over_A    | Strength1  | 472 |  0.3976 |
| Type 1  | spearman         | log_H_over_A    | Strength1  | 472 |  0.4405 |
| Type 1  | partial_spearman | log_H_over_A    | Strength1  | 472 |  0.2783 |
| Type 1  | pearson          | log_H_over_A    | Strength7  | 484 |  0.0561 |
| Type 1  | spearman         | log_H_over_A    | Strength7  | 484 |  0.0086 |
| Type 1  | partial_spearman | log_H_over_A    | Strength7  | 484 |  0.0826 |
| Type 1  | pearson          | log_H_over_A    | Strength28 | 466 | -0.1373 |
| Type 1  | spearman         | log_H_over_A    | Strength28 | 466 | -0.1741 |
| Type 1  | partial_spearman | log_H_over_A    | Strength28 | 466 |  0.0835 |




## 13_vif_ratios


| feature         |   n_complete |     vif |
|:----------------|-------------:|--------:|
| log_C_over_B    |          566 | 17.5328 |
| C_over_A        |          566 | 17.6975 |
| log_D_over_A    |          566 |  2.2714 |
| log_F_over_E    |          566 | 25.6754 |
| log_A           |          566 |  2.4639 |
| log_H_over_F    |          566 | 10.5332 |
| F_over_E_plus_F |          566 | 32.5178 |
| log_H_over_A    |          566 |  8.2572 |




## 14_vif_crudas_ratios


| feature                |   n_complete |       vif |
|:-----------------------|-------------:|----------:|
| Physical_Metric_B      |          555 |    1.8976 |
| Mineral_Phase_B        |          555 |  764.7366 |
| Mineral_Phase_C        |          555 | 1514.4181 |
| Mineral_Phase_D        |          555 |   32.0331 |
| Mineral_Phase_E        |          555 |   30.6293 |
| Mineral_Phase_F        |          555 |   18.3148 |
| Mineral_Phase_G        |          555 |    1.2554 |
| Mineral_Phase_H        |          555 |    7.2369 |
| Chemical_Composition_A |          555 |    1.6693 |
| Chemical_Composition_B |          555 |    1.4085 |
| log_C_over_B           |          555 |   18.9410 |
| C_over_A               |          555 |  298.8673 |
| log_D_over_A           |          555 |   40.8936 |
| log_F_over_E           |          555 |   30.5474 |
| log_A                  |          555 |  890.8045 |
| log_H_over_F           |          555 |  161.4669 |
| F_over_E_plus_F        |          555 |  143.2094 |
| log_H_over_A           |          555 |  120.7550 |




## 15_vif_descartes_combinados


| removed         |       vif |   n_complete |
|:----------------|----------:|-------------:|
| Mineral_Phase_C | 1514.4181 |          555 |
| C_over_A        |  287.4464 |          555 |
| log_H_over_F    |  160.3805 |          555 |
| F_over_E_plus_F |   58.8221 |          555 |
| log_D_over_A    |   34.3258 |          555 |
| log_C_over_B    |   15.3218 |          555 |
| log_F_over_E    |   10.7783 |          555 |




## 16_vif_combinado_numerico


| feature                |   n_complete |    vif |
|:-----------------------|-------------:|-------:|
| Physical_Metric_B      |          555 | 1.7951 |
| Mineral_Phase_B        |          555 | 1.0809 |
| Mineral_Phase_D        |          555 | 2.4458 |
| Mineral_Phase_E        |          555 | 1.2581 |
| Mineral_Phase_F        |          555 | 1.4317 |
| Mineral_Phase_G        |          555 | 1.2244 |
| Mineral_Phase_H        |          555 | 6.9940 |
| Chemical_Composition_A |          555 | 1.6553 |
| Chemical_Composition_B |          555 | 1.3850 |
| log_A                  |          555 | 2.4287 |
| log_H_over_A           |          555 | 5.9190 |




## 17_vif_combinado_con_tipo


| feature                |   n_complete |    vif |
|:-----------------------|-------------:|-------:|
| Physical_Metric_B      |          555 | 4.4723 |
| Mineral_Phase_B        |          555 | 1.0913 |
| Mineral_Phase_D        |          555 | 6.5095 |
| Mineral_Phase_E        |          555 | 1.8235 |
| Mineral_Phase_F        |          555 | 1.6602 |
| Mineral_Phase_G        |          555 | 1.2889 |
| Mineral_Phase_H        |          555 | 7.0593 |
| Chemical_Composition_A |          555 | 1.6741 |
| Chemical_Composition_B |          555 | 1.4532 |
| log_A                  |          555 | 2.7167 |
| log_H_over_A           |          555 | 5.9688 |
| ProductType_Type 2     |          555 | 3.6604 |
| ProductType_Type 3     |          555 | 5.4219 |




## 17b_vif_crudo_con_tipo


| feature                |   n_complete |    vif |
|:-----------------------|-------------:|-------:|
| Physical_Metric_B      |          558 | 4.4123 |
| Mineral_Phase_B        |          558 | 3.0453 |
| Mineral_Phase_C        |          558 | 4.5036 |
| Mineral_Phase_D        |          558 | 6.1598 |
| Mineral_Phase_E        |          558 | 1.8095 |
| Mineral_Phase_F        |          558 | 1.6783 |
| Mineral_Phase_G        |          558 | 1.3009 |
| Mineral_Phase_H        |          558 | 1.5545 |
| Chemical_Composition_A |          558 | 1.6348 |
| Chemical_Composition_B |          558 | 1.4471 |
| ProductType_Type 2     |          558 | 3.6128 |
| ProductType_Type 3     |          558 | 5.1038 |




## 17c_vif_ratios_sustitucion


| feature                |   n_complete |    vif |
|:-----------------------|-------------:|-------:|
| Physical_Metric_B      |          557 | 4.3463 |
| Mineral_Phase_E        |          557 | 4.2628 |
| Mineral_Phase_G        |          557 | 1.2777 |
| Mineral_Phase_H        |          557 | 1.5695 |
| Chemical_Composition_A |          557 | 1.5892 |
| Chemical_Composition_B |          557 | 1.3141 |
| log_C_over_B           |          557 | 1.0925 |
| log_D_over_A           |          557 | 7.7949 |
| log_F_over_E           |          557 | 3.6124 |
| ProductType_Type 2     |          557 | 4.3276 |
| ProductType_Type 3     |          557 | 5.9099 |




## 17d_vif_nuevas_candidatas


| feature                |   n_complete |     vif | candidate       |
|:-----------------------|-------------:|--------:|:----------------|
| Physical_Metric_B      |          557 |  4.3821 | log_A           |
| Mineral_Phase_E        |          557 |  4.3803 | log_A           |
| Mineral_Phase_G        |          557 |  1.2846 | log_A           |
| Mineral_Phase_H        |          557 |  1.6092 | log_A           |
| Chemical_Composition_A |          557 |  1.6165 | log_A           |
| Chemical_Composition_B |          557 |  1.4173 | log_A           |
| log_C_over_B           |          557 |  1.0943 | log_A           |
| log_D_over_A           |          557 |  9.3714 | log_A           |
| log_F_over_E           |          557 |  3.8026 | log_A           |
| log_A                  |          557 |  2.7993 | log_A           |
| ProductType_Type 2     |          557 |  4.3346 | log_A           |
| ProductType_Type 3     |          557 |  5.9166 | log_A           |
| Physical_Metric_B      |          555 |  4.4146 | log_H_over_F    |
| Mineral_Phase_E        |          555 |  5.1410 | log_H_over_F    |
| Mineral_Phase_G        |          555 |  1.2801 | log_H_over_F    |
| Mineral_Phase_H        |          555 |  4.3840 | log_H_over_F    |
| Chemical_Composition_A |          555 |  1.6060 | log_H_over_F    |
| Chemical_Composition_B |          555 |  1.3215 | log_H_over_F    |
| log_C_over_B           |          555 |  1.0975 | log_H_over_F    |
| log_D_over_A           |          555 |  8.4373 | log_H_over_F    |
| log_F_over_E           |          555 |  6.5601 | log_H_over_F    |
| log_H_over_F           |          555 |  4.7690 | log_H_over_F    |
| ProductType_Type 2     |          555 |  4.5495 | log_H_over_F    |
| ProductType_Type 3     |          555 |  6.2467 | log_H_over_F    |
| Physical_Metric_B      |          557 |  4.3854 | F_over_E_plus_F |
| Mineral_Phase_E        |          557 |  4.3981 | F_over_E_plus_F |
| Mineral_Phase_G        |          557 |  1.2820 | F_over_E_plus_F |
| Mineral_Phase_H        |          557 |  1.5827 | F_over_E_plus_F |
| Chemical_Composition_A |          557 |  1.5892 | F_over_E_plus_F |
| Chemical_Composition_B |          557 |  1.3176 | F_over_E_plus_F |
| log_C_over_B           |          557 |  1.0926 | F_over_E_plus_F |
| log_D_over_A           |          557 |  8.2531 | F_over_E_plus_F |
| log_F_over_E           |          557 | 23.4727 | F_over_E_plus_F |
| F_over_E_plus_F        |          557 | 23.5473 | F_over_E_plus_F |
| ProductType_Type 2     |          557 |  4.4860 | F_over_E_plus_F |
| ProductType_Type 3     |          557 |  6.0611 | F_over_E_plus_F |
| Physical_Metric_B      |          555 |  4.3902 | log_H_over_A    |
| Mineral_Phase_E        |          555 |  4.2656 | log_H_over_A    |
| Mineral_Phase_G        |          555 |  1.2769 | log_H_over_A    |
| Mineral_Phase_H        |          555 |  6.5027 | log_H_over_A    |
| Chemical_Composition_A |          555 |  1.6094 | log_H_over_A    |
| Chemical_Composition_B |          555 |  1.3137 | log_H_over_A    |
| log_C_over_B           |          555 |  1.0941 | log_H_over_A    |
| log_D_over_A           |          555 |  8.2161 | log_H_over_A    |
| log_F_over_E           |          555 |  3.6276 | log_H_over_A    |
| log_H_over_A           |          555 |  5.6180 | log_H_over_A    |
| ProductType_Type 2     |          555 |  4.4264 | log_H_over_A    |
| ProductType_Type 3     |          555 |  6.1376 | log_H_over_A    |




## 18_cinetica


| ProductType   | model   |   n_complete |   exponent | exponent_at_bound   |   s_mean |    s_std |   s_min |   s_q25 |   s_median |   s_q75 |   s_max |   MAPE_Strength1 |   bias_Strength1 |   MAPE_Strength7 |   bias_Strength7 |   MAPE_Strength28 |   bias_Strength28 |   R1_overestimated_pct |
|:--------------|:--------|-------------:|-----------:|:--------------------|---------:|---------:|--------:|--------:|-----------:|--------:|--------:|-----------------:|-----------------:|-----------------:|-----------------:|------------------:|------------------:|-----------------------:|
| Type 1        | free    |          491 |     0.6947 | False               |   0.1318 |   0.0142 |  0.0965 |  0.1220 |     0.1336 |  0.1421 |  0.1645 |           2.3838 |          -0.1308 |           1.8717 |          -0.1162 |            1.0248 |            0.1505 |                49.8982 |
| Type 1        | sqrt    |          491 |     0.5000 | False               |   0.2707 |   0.0279 |  0.2004 |  0.2518 |     0.2750 |  0.2905 |  0.3336 |           5.9643 |           5.9199 |           4.0018 |          -3.9699 |            1.9688 |            1.9552 |                98.5743 |
| Type 2        | free    |           25 |     0.8501 | False               |   0.0520 |   0.0051 |  0.0403 |  0.0495 |     0.0529 |  0.0540 |  0.0617 |           0.6221 |          -0.0047 |           1.0340 |           0.0096 |            0.7074 |            0.0154 |                36.0000 |
| Type 2        | sqrt    |           25 |     0.5000 | False               |   0.1901 |   0.0176 |  0.1490 |  0.1815 |     0.1935 |  0.1975 |  0.2245 |           4.1899 |           4.1899 |           4.6256 |          -4.6256 |            2.6237 |            2.6237 |               100.0000 |
| Type 3        | free    |           42 |     0.5739 | False               |   0.2205 |   0.0192 |  0.1545 |  0.2091 |     0.2192 |  0.2325 |  0.2567 |           3.4014 |          -0.1857 |           2.2222 |           0.0344 |            1.0557 |            0.0690 |                42.8571 |
| Type 3        | sqrt    |           42 |     0.5000 | False               |   0.2917 |   0.0249 |  0.2054 |  0.2779 |     0.2900 |  0.3080 |  0.3378 |           3.5546 |           2.5737 |           2.2097 |          -1.6046 |            1.0319 |            0.7835 |                66.6667 |
| Type 5        | free    |            1 |     0.5026 | False               |   0.2726 | nan      |  0.2726 |  0.2726 |     0.2726 |  0.2726 |  0.2726 |           0.0000 |          -0.0000 |           0.0000 |           0.0000 |            0.0000 |            0.0000 |                 0.0000 |
| Type 5        | sqrt    |            1 |     0.5000 | False               |   0.2755 | nan      |  0.2755 |  0.2755 |     0.2755 |  0.2755 |  0.2755 |           0.0825 |           0.0825 |           0.0577 |          -0.0577 |            0.0255 |            0.0255 |               100.0000 |




## 18b_cinetica_anclada_R7


| ProductType   |   n_complete |   s7_median |   R1_bias_mean_pct |   R1_bias_median_pct |   R1_MAPE_pct |
|:--------------|-------------:|------------:|-------------------:|---------------------:|--------------:|
| Type 1        |          491 |      0.2180 |            35.9740 |              33.4477 |       36.1821 |
| Type 2        |           25 |      0.1242 |            39.6876 |              36.9218 |       39.6876 |
| Type 3        |           42 |      0.2868 |            15.0075 |               7.5349 |       19.1404 |
| Type 5        |            1 |      0.2746 |             0.4148 |               0.4148 |        0.4148 |




## 18c_cobertura_cinetica


| ProductType   |   n_rows |   n_complete |
|:--------------|---------:|-------------:|
| Type 1        |      537 |          491 |
| Type 2        |       27 |           25 |
| Type 3        |       52 |           42 |
| Type 4        |        4 |            0 |
| Type 5        |        1 |            1 |




## 19_fisica


| feature           | contrast               |   n |   n_groups |   effect |   ci_low |   ci_high |   valid_bootstraps | test                 | expected   | verdict        |
|:------------------|:-----------------------|----:|-----------:|---------:|---------:|----------:|-------------------:|:---------------------|:-----------|:---------------|
| Physical_Metric_B | Strength1              | 512 |        493 |   0.1461 |   0.0517 |    0.2428 |               2000 | Blaine R1            | +          | apoya          |
| Physical_Metric_B | Strength28             | 506 |        487 |  -0.1877 |  -0.2872 |   -0.0864 |               2000 | Blaine R28           | +          | no apoya       |
| Physical_Metric_B | Strength1 - Strength28 | 488 |        469 |   0.3244 |   0.2251 |    0.4256 |               2000 | Blaine R1 > R28      | +          | apoya          |
| Mineral_Phase_D   | Strength28             | 470 |        451 |   0.0428 |  -0.0521 |    0.1362 |               2000 | D caliza R28         | -          | no concluyente |
| Mineral_Phase_F   | Strength1              | 475 |        456 |  -0.0391 |  -0.1259 |    0.0479 |               2000 | F C3A R1             | +          | no concluyente |
| Mineral_Phase_E   | Strength1              | 474 |        455 |  -0.0864 |  -0.1761 |    0.0051 |               2000 | E ferrita Strength1  | neutral    | no concluyente |
| Mineral_Phase_E   | Strength7              | 487 |        468 |  -0.0176 |  -0.1045 |    0.0707 |               2000 | E ferrita Strength7  | neutral    | no concluyente |
| Mineral_Phase_E   | Strength28             | 469 |        450 |  -0.0710 |  -0.1601 |    0.0158 |               2000 | E ferrita Strength28 | neutral    | no concluyente |




## 19b_sensibilidad_blaine


|   n_blocks | controls               | target     |   n |     rho |
|-----------:|:-----------------------|:-----------|----:|--------:|
|          8 | ninguno                | Strength1  | 512 |  0.1786 |
|          8 | ninguno                | Strength28 | 506 | -0.1422 |
|          8 | D                      | Strength1  | 471 |  0.1552 |
|          8 | D                      | Strength28 | 466 | -0.2035 |
|          8 | D + Chem_A + C + G + H | Strength1  | 465 |  0.1322 |
|          8 | D + Chem_A + C + G + H | Strength28 | 461 | -0.1895 |
|         24 | ninguno                | Strength1  | 512 |  0.2347 |
|         24 | ninguno                | Strength28 | 506 | -0.0592 |
|         24 | D                      | Strength1  | 471 |  0.1644 |
|         24 | D                      | Strength28 | 466 | -0.1993 |
|         24 | D + Chem_A + C + G + H | Strength1  | 465 |  0.1028 |
|         24 | D + Chem_A + C + G + H | Strength28 | 461 | -0.2196 |
|         48 | ninguno                | Strength1  | 512 |  0.2491 |
|         48 | ninguno                | Strength28 | 506 | -0.1134 |
|         48 | D                      | Strength1  | 471 |  0.2121 |
|         48 | D                      | Strength28 | 466 | -0.1469 |
|         48 | D + Chem_A + C + G + H | Strength1  | 465 |  0.1435 |
|         48 | D + Chem_A + C + G + H | Strength28 | 461 | -0.1701 |




## 20_deriva


|   block |   row_id_min |   row_id_max | column            |   n |    median |
|--------:|-------------:|-------------:|:------------------|----:|----------:|
|       1 |            0 |           98 | Strength1         |  68 | 2400.0000 |
|       1 |            0 |           98 | Strength7         |  71 | 5830.0000 |
|       1 |            0 |           98 | Strength28        |  70 | 6730.0000 |
|       1 |            0 |           98 | Mineral_Phase_A   |  70 |    0.8400 |
|       1 |            0 |           98 | Physical_Metric_B |  72 |    0.7136 |
|       1 |            0 |           98 | Mineral_Phase_H   |  70 |    0.4724 |
|       1 |            0 |           98 | Mineral_Phase_C   |  70 |    0.7716 |
|       1 |            0 |           98 | Physical_Metric_A |   0 |  nan      |
|       1 |            0 |           98 | Mineral_Phase_G   |  70 |    0.0272 |
|       1 |            0 |           98 | Mineral_Phase_B   |  70 |    0.3855 |
|       2 |           99 |          197 | Strength1         |  64 | 2430.0000 |
|       2 |           99 |          197 | Strength7         |  71 | 5880.0000 |
|       2 |           99 |          197 | Strength28        |  69 | 6790.0000 |
|       2 |           99 |          197 | Mineral_Phase_A   |  71 |    0.8342 |
|       2 |           99 |          197 | Physical_Metric_B |  71 |    0.3759 |
|       2 |           99 |          197 | Mineral_Phase_H   |  71 |    0.6558 |
|       2 |           99 |          197 | Mineral_Phase_C   |  71 |    0.7612 |
|       2 |           99 |          197 | Physical_Metric_A |   0 |  nan      |
|       2 |           99 |          197 | Mineral_Phase_G   |  71 |    0.0218 |
|       2 |           99 |          197 | Mineral_Phase_B   |  71 |    0.3838 |
|       3 |          198 |          295 | Strength1         |  54 | 2335.0000 |
|       3 |          198 |          295 | Strength7         |  59 | 6380.0000 |
|       3 |          198 |          295 | Strength28        |  60 | 7830.0000 |
|       3 |          198 |          295 | Mineral_Phase_A   |  55 |    0.8199 |
|       3 |          198 |          295 | Physical_Metric_B |  60 |    0.0381 |
|       3 |          198 |          295 | Mineral_Phase_H   |  55 |    0.3149 |
|       3 |          198 |          295 | Mineral_Phase_C   |  55 |    0.7350 |
|       3 |          198 |          295 | Physical_Metric_A |   0 |  nan      |
|       3 |          198 |          295 | Mineral_Phase_G   |  55 |    0.0279 |
|       3 |          198 |          295 | Mineral_Phase_B   |  55 |    0.4077 |
|       4 |          296 |          393 | Strength1         |  59 | 2200.0000 |
|       4 |          296 |          393 | Strength7         |  61 | 6040.0000 |
|       4 |          296 |          393 | Strength28        |  60 | 7470.0000 |
|       4 |          296 |          393 | Mineral_Phase_A   |  61 |    0.7938 |
|       4 |          296 |          393 | Physical_Metric_B |  61 |   -0.2321 |
|       4 |          296 |          393 | Mineral_Phase_H   |  61 |    0.3344 |
|       4 |          296 |          393 | Mineral_Phase_C   |  61 |    0.6960 |
|       4 |          296 |          393 | Physical_Metric_A |   0 |  nan      |
|       4 |          296 |          393 | Mineral_Phase_G   |  61 |    0.0272 |
|       4 |          296 |          393 | Mineral_Phase_B   |  61 |    0.4125 |
|       5 |          395 |          493 | Strength1         |  75 | 2160.0000 |
|       5 |          395 |          493 | Strength7         |  75 | 6160.0000 |
|       5 |          395 |          493 | Strength28        |  75 | 7940.0000 |
|       5 |          395 |          493 | Mineral_Phase_A   |  42 |    0.7868 |
|       5 |          395 |          493 | Physical_Metric_B |  75 |   -0.3447 |
|       5 |          395 |          493 | Mineral_Phase_H   |  42 |    0.3604 |
|       5 |          395 |          493 | Mineral_Phase_C   |  42 |    0.6958 |
|       5 |          395 |          493 | Physical_Metric_A |   0 |  nan      |
|       5 |          395 |          493 | Mineral_Phase_G   |  42 |    0.0235 |
|       5 |          395 |          493 | Mineral_Phase_B   |  42 |    0.3944 |
|       6 |          494 |          590 | Strength1         |  70 | 2160.0000 |
|       6 |          494 |          590 | Strength7         |  70 | 5975.0000 |
|       6 |          494 |          590 | Strength28        |  69 | 7710.0000 |
|       6 |          494 |          590 | Mineral_Phase_A   |  70 |    0.7582 |
|       6 |          494 |          590 | Physical_Metric_B |  69 |   -0.1646 |
|       6 |          494 |          590 | Mineral_Phase_H   |  70 |    0.4367 |
|       6 |          494 |          590 | Mineral_Phase_C   |  70 |    0.6817 |
|       6 |          494 |          590 | Physical_Metric_A |   2 |    0.7644 |
|       6 |          494 |          590 | Mineral_Phase_G   |  70 |    0.0245 |
|       6 |          494 |          590 | Mineral_Phase_B   |  70 |    0.3738 |
|       7 |          592 |          690 | Strength1         |  70 | 2085.0000 |
|       7 |          592 |          690 | Strength7         |  70 | 5745.0000 |
|       7 |          592 |          690 | Strength28        |  70 | 7390.0000 |
|       7 |          592 |          690 | Mineral_Phase_A   |  69 |    0.7900 |
|       7 |          592 |          690 | Physical_Metric_B |  68 |   -0.5924 |
|       7 |          592 |          690 | Mineral_Phase_H   |  69 |    0.3279 |
|       7 |          592 |          690 | Mineral_Phase_C   |  69 |    0.7193 |
|       7 |          592 |          690 | Physical_Metric_A |  67 |    0.6904 |
|       7 |          592 |          690 | Mineral_Phase_G   |  69 |    0.0232 |
|       7 |          592 |          690 | Mineral_Phase_B   |  69 |    0.3643 |
|       8 |          691 |          788 | Strength1         |  57 | 2230.0000 |
|       8 |          691 |          788 | Strength7         |  53 | 5950.0000 |
|       8 |          691 |          788 | Strength28        |  38 | 7400.0000 |
|       8 |          691 |          788 | Mineral_Phase_A   |  57 |    0.8014 |
|       8 |          691 |          788 | Physical_Metric_B |  56 |   -0.5924 |
|       8 |          691 |          788 | Mineral_Phase_H   |  57 |    0.3247 |
|       8 |          691 |          788 | Mineral_Phase_C   |  57 |    0.7325 |
|       8 |          691 |          788 | Physical_Metric_A |  55 |    0.7041 |
|       8 |          691 |          788 | Mineral_Phase_G   |  57 |    0.0211 |
|       8 |          691 |          788 | Mineral_Phase_B   |  57 |    0.3719 |




## 21_tipos


| ProductType   |   n | reliable   |
|:--------------|----:|:-----------|
| Type 1        | 537 | True       |
| Type 2        |  27 | True       |
| Type 3        |  52 | True       |
| Type 4        |   4 | False      |
| Type 5        |   1 | False      |




## 22_figuras


- [distribution_Physical_Metric_A](figures/distribution_Physical_Metric_A.png)

- [distribution_Physical_Metric_B](figures/distribution_Physical_Metric_B.png)

- [distribution_Mineral_Phase_A](figures/distribution_Mineral_Phase_A.png)

- [distribution_Mineral_Phase_B](figures/distribution_Mineral_Phase_B.png)

- [distribution_Mineral_Phase_C](figures/distribution_Mineral_Phase_C.png)

- [distribution_Mineral_Phase_D](figures/distribution_Mineral_Phase_D.png)

- [distribution_Mineral_Phase_E](figures/distribution_Mineral_Phase_E.png)

- [distribution_Mineral_Phase_F](figures/distribution_Mineral_Phase_F.png)

- [distribution_Mineral_Phase_G](figures/distribution_Mineral_Phase_G.png)

- [distribution_Mineral_Phase_H](figures/distribution_Mineral_Phase_H.png)

- [distribution_Chemical_Composition_A](figures/distribution_Chemical_Composition_A.png)

- [distribution_Chemical_Composition_B](figures/distribution_Chemical_Composition_B.png)

- [distribution_Strength1](figures/distribution_Strength1.png)

- [distribution_Strength7](figures/distribution_Strength7.png)

- [distribution_Strength28](figures/distribution_Strength28.png)

- [scatter_Strength1](figures/scatter_Strength1.png)

- [scatter_Strength7](figures/scatter_Strength7.png)

- [scatter_Strength28](figures/scatter_Strength28.png)

- [correlations_spearman](figures/correlations_spearman.png)

- [correlations_partial_spearman](figures/correlations_partial_spearman.png)

- [drift](figures/drift.png)

- [kinetics_fit_error](figures/kinetics_fit_error.png)
