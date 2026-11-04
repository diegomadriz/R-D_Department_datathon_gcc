# ADR 0006: Protocolo de CV y selección de modelos

- **Estado:** Aceptado para el framework y la escalera; resultados en outputs/fase3/stage2
- **Fecha:** 2026-11-03
- **Relacionados:** ADR 0002, 0003 y 0005

## Decisión previa a resultados

1. Solo se carga train mediante `gcc_strength.eda.load_train`. No se cambia Gold ni
   su split. No existe ejecutor de evaluación del lockbox en este framework.
2. Cada target usa todas sus etiquetas observadas; no se exigen los otros targets.
   Las tres variantes de `Physical_Metric_A` evalúan las mismas filas y folds para
   cada target. Los estratos se recalculan con `split.assign_strata` sobre esa población;
   las etiquetas de otros horizontes intervienen únicamente en la estratificación.
3. CV externa: `StratifiedGroupKFold`, 5 folds × 5 repeticiones. Semilla base 42;
   las repeticiones usan 42 + índice (42…46), única derivación prevista por el protocolo.
   Ningún grupo de ensayo cruza una partición. La reducción de folds/repeticiones
   sirve exclusivamente para pruebas del framework, sin seleccionar modelos finales.
4. HPO anidada: tres folds internos agrupados/estratificados; se recalculan estratos
   dentro del entrenamiento externo. `conf/models.yaml` fija nueve configuraciones
   por modelo y target en cada fold externo (675 ajustes internos por modelo/target).
   Se minimiza la media de MAE de los folds internos. Empates: primera configuración
   en el orden determinista del muestreador con semilla 42. L0/L1 no tienen búsqueda.
   La escasez de tipos raros puede impedir balance interno; se conserva la separación
   de grupos y se mantiene visible el aviso de sklearn, sin duplicar ni borrar filas.
5. L0 estima la media del entrenamiento del fold. L1 estima medias nominales por
   tipo; un tipo ausente usa la media global del fold. L2 usa crudas sin A mineral.
   L3 usa sustitución física de ADR 0005: log(C/B), log(D/A), log(F/E), E/G/H,
   Blaine y ambas químicas. Los otros ratios e interacciones siguen como ablations,
   no se apilan automáticamente. F/(E+F) reemplaza log(F/E).
6. Imputación mediana, escalado, one-hot con categoría de referencia, centrado y
   banderas IQR se ajustan dentro del Pipeline por fold. Columnas totalmente ausentes
   se conservan e imputan con cero; no aportan variación aprendida. Indicadores de
   ratios y ausencia de A se calculan dentro del Pipeline; no se importan banderas de Gold.
7. Variante A: exclusión como referencia; indicador + imputación como candidata;
   experto con A observada y fallback sin A como tercera candidata. El experto exige
   20 etiquetas observadas en entrenamiento; si no llega, todo usa fallback. Se informa
   tamaño del experto y cobertura observada/efectiva en cada fold. Experto y fallback
   comparten hiperparámetros candidatos para no ampliar el presupuesto de búsqueda.
8. Sensibilidad temporal: seis bloques contiguos de grupos en orden `_row_id` y cinco
   cortes forward-chaining, siempre en fronteras de grupo. El primer bloque es calentamiento
   sin predicciones OOF. HPO interna solo usa el pasado disponible de cada corte externo.
   Esta sensibilidad no demuestra validación cronológica: no hay fechas confirmadas.
9. Métricas: R², MAE, MAPE porcentual, RMSE, R²/MAE dentro de Type 1 y tamaños.
   MAPE omite targets cero e informa denominador; R² constante o con n<2 es indefinido.
   Se guardan todas las predicciones OOF por repetición, sin promediarlas antes de medir.
10. Superioridad frente a L1: bootstrap pareado de ΔMAE = modelo − L1, 10 000
    remuestreos de grupos íntegros, semilla 42, IC percentil 95% completamente bajo cero.
    Se mantienen juntas las repeticiones del mismo grupo; no son muestras independientes.
    Se suman pérdidas y conteos para preservar MAE por fila con tamaños de grupo distintos.
    El IC describe incertidumbre sobre grupos condicionada a los ajustes CV obtenidos;
    no captura toda dependencia serial ni la incertidumbre de repetir la selección completa.

## Pendientes

El ADR 0005 está aceptado para las decisiones de features tomadas por CV. La ampliación
de la Etapa 2 compara variantes y selecciona tres familias con la regla siguiente;
la comprobación VIF dentro del fold y la auditoría independiente de fuga y sobreajuste
siguen pendientes. Las arquitecturas y la incertidumbre se documentan en el ADR 0007.
La regla prevista para seleccionar finales es menor MAE de CV por familia y target;
empate numérico (tolerancia 1e-9 en unidades del target): menor número de columnas del
diseño y luego menor complejidad del estimador. No se decide con el lockbox.

## Ampliación previa a resultados de la Etapa 2

11. Se conserva el presupuesto L2/L3 (9 candidatos). L4 usa LightGBM 4.6.0,
    100 árboles, learning_rate 0.05, min_child_samples 15, regularización L2=1,
    dos candidatos de hojas (7/15), un hilo y modo determinista. Se comparan crudas
    y sustitución física con las tres variantes de A. Para R1 se repiten las seis
    combinaciones con restricción positiva únicamente en Blaine. La máscara se crea
    desde los nombres posteriores al one-hot de cada fold; los demás ejes quedan libres.
    Se excluyen interacciones de esta candidata para conservar su garantía monótona.
12. L5 es GPR exacto, Matérn ν=1.5 ARD + WhiteKernel, features estandarizadas y
    normalización del target aprendidas por fold. Presupuesto: cuatro perfiles, producto
    de escalas iniciales de calidad 1/3 y ruido 0.1/0.4; ejes nominales inicializados en 1.
    ARD es un vector por eje tras el encoding: se optimizan sus longitudes individualmente
    por verosimilitud marginal dentro de cada ajuste, con L-BFGS-B limitado a 8 iteraciones,
    16 evaluaciones y ftol=1e-6, sin reinicios. Límites de longitud [0.1,100], ruido [0.01,2].
    El presupuesto fijo puede truncar la convergencia; no es una optimización exhaustiva.
    El WhiteKernel expresa ruido en el target normalizado.
13. Alternativa robusta Huber: ε=1.35/1.75 (dos candidatos), sin borrar filas ni
    inspeccionar residuales globales. Escalado de features y del target se ajusta dentro
    del Pipeline por fold. También se comparan sus dos diseños y tres variantes de A.
14. Las tres familias finales son lineal (L2/L3/Huber), boosting (L4) y GPR (L5).
    Por familia **y target** gana el menor MAE medio entre folds externos. Dentro de
    1e-9 del mínimo gana menor número medio de columnas ajustadas (experto + fallback),
    luego el orden de complejidad fijado en YAML y finalmente el ID alfabético.
    La sensibilidad temporal no cambia esta selección; informa su fragilidad.
    Elegir variantes sobre CV externa introduce optimismo de selección: sus métricas
    e IC no equivalen a una evaluación independiente posterior de toda esta selección.
15. L6 es la media aritmética **fijada previamente** de L3 físico sin A, L4 crudo sin A
    sin restricción y L5 físico sin A. Cada miembro tiene su propia HPO anidada sobre
    las mismas particiones. No se estiman pesos ni se eligen miembros con OOF.
    Se admite solo por target si el IC95% de ΔMAE queda bajo cero frente a L1 **y cada
    uno de los tres finalistas**. Si falla, se guarda como candidata rechazada y no
    se incorpora a la selección. Estos contrastes son condicionales y exploratorios,
    sin corrección por multiplicidad ni validación independiente de la selección.
16. Se ejecutan ambos esquemas para toda la escalera. Artefactos: OOF, métricas,
    parámetros/cobertura por fold, tabla con media ± sd e IC pareado contra L1,
    comparación de variantes y sensibilidad temporal, finalistas y gate L6.
    Un manifiesto de configuración, código y huella de train permite reanudar sin
    reutilizar resultados de otro protocolo. Los resultados reducidos de tests nunca
    se usan como selección final. El lockbox sigue cerrado; no se añade ejecutor.

La verificación VIF de todas las ablations no se ha incorporado en esta etapa;
la penalización lineal y los diseños del ADR 0005 se conservan.

## Cierre de Etapa 4: configuración congelada y ejecución

17. `conf/models.yaml:final` conserva las nueve variantes ganadoras de Etapa 2.
    Para cada una se copia el perfil de hiperparámetros más frecuente entre sus
    25 selecciones internas anidadas; empate por JSON ordenado lexicográficamente.
    Se conservan la ruta de procedencia y la frecuencia. No se vuelve a seleccionar
    con los resultados de Etapa 4. Huber ganó con representación cruda; no se fuerza
    una representación física que perdió la comparación anterior.
18. El orquestador final reajusta estos perfiles congelados en CV agrupada 5×5
    sobre folds comunes para las tres salidas, con proyección isotónica. No repite
    HPO: esta ya determinó las elecciones de Etapa 2. La CV final es diagnóstica y
    condicional a esa selección sobre el mismo train, con optimismo potencial;
    no constituye una nueva estimación independiente de selección anidada.
19. CV+ de las tres familias calibra dentro de cada entrenamiento externo, con cinco
    folds agrupados y perfiles congelados; no recibe etiquetas de validación externa
    en esos ajustes. La elección previa del perfil sobre train limita la interpretación
    de cobertura. MAE e IC se calculan con todas las OOF de cada repetición, y los
    IC percentiles (10 000 remuestreos de grupos) mantienen juntas sus repeticiones.
    El IC de MAE estima la pérdida media por fila, que puede diferir de la media no
    ponderada de los 25 MAE de fold; no es un IC construido desde la SD de folds.
20. La prueba de permutación del predictor GPR (a) seleccionado usa B=200 sobre
    los cinco folds de la primera repetición, con perfiles congelados. No repite
    selección de variantes ni HPO previa: contrasta señal condicional al predictor
    elegido, con p=(1+#R²_perm≥R²_real)/201 y resolución mínima 1/201.
21. MLflow local registra un padre, nueve hijos por familia/salida y un hijo multisalida.
    Cada salida registrada incluye la proyección del vector completo; sus SHAP
    explican el componente antes de proyectar. Las firmas solo admiten calidad.
    `train_size=621` y `test_size=152` son los tamaños del protocolo; `fit_size`
    distingue el subconjunto del modo rápido. No se consulta el lockbox para obtenerlos.
    `mlflow_audit` verifica integridad de artefactos, no sustituye revisión independiente
    de fuga, sobreajuste ni autorización de evaluación final.
