# ADR 0007: Arquitectura multisalida e incertidumbre

- **Estado:** Borrador; protocolo fijado antes de ejecutar la Etapa 3
- **Fecha:** 2026-11-03
- **Relacionados:** ADR 0002, 0005 y 0006

## Protocolo

1. Se reutilizan las nueve elecciones de `outputs/fase3/stage2/finalists.csv`:
   tres familias por tres targets, conservando representación, variante de A y nivel.
   No se reejecuta la escalera. La elección previa sobre CV puede introducir optimismo.
2. Las arquitecturas comparten folds sobre toda la población de train; cada salida
   ajusta solo sus etiquetas disponibles. Esto cambia las particiones específicas
   por target de Etapa 2, por lo que no se mezclan sus OOF ni se comparan como pares.
   Se mantienen 5×5 folds agrupados/estratificados y cinco cortes temporales.
3. (a): modelos independientes y proyección isotónica euclídea por fila.
   (b): cadena R1→R7→R28 con `cross_val_predict` por grupo. Para entrenar R28,
   el prefijo R1→R7 vuelve a generar sus propios inputs OOF dentro de cada fold.
   No se admite un OOF global reutilizado entre niveles. Inferencia recibe calidad
   solamente y produce las features auxiliares desde predicciones del prefijo.
4. (c): etiquetas log(R1/R28), log(R7/R28) y R28, nunca inputs medidos.
   Cada etiqueta log requiere únicamente su pareja observada; R28 conserva todas
   sus etiquetas. Los logs predichos se proyectan crecientes y no positivos;
   se reconstruyen con R28 predicho positivo. No se ajustan exponentes cinéticos.
   La restricción Blaine→R1 de LightGBM se desactiva para el log-ratio: su signo
   no se deduce del signo del target original. A/B conservan esa candidata fijada.
5. HPO conjunta anidada con tres folds internos: hasta cuatro perfiles acoplados
   por índice determinista de los candidatos del ADR 0006 (dos para Huber/L4,
   cuatro para GPR). No hay producto cartesiano de hiperparámetros entre salidas.
   Se minimiza la media de los MAE de los tres targets, con sus unidades originales.
   En cada candidato, los prefijos de cadena tienen parámetros fijos; la selección
   del perfil completo se hace exclusivamente dentro del entrenamiento externo.
   Este OOF interno es condicional al perfil, pero toda comparación externa queda
   separada de la búsqueda. La misma búsqueda se repite en cada tamaño/permutación.
6. (a) es la referencia. Una alternativa requiere IC95% de ΔMAE agregado bajo 0
   con 10 000 remuestreos pareados de grupos. Para admitir (c), además el IC95%
   de ΔMAE R1 debe tener límite superior ≤0: criterio conservador de no empeorar,
   sin inventar un margen de no inferioridad después de ver resultados.
   Se elige la menor pérdida OOF entre alternativas admisibles, luego entre familias.
   El agregado pareado pondera todas las etiquetas observadas; no imputa targets.
   Empates exactos se resuelven por ID estable. IC condicionales, sin ajuste múltiple.
7. CV+ nominal 90%: en cada fold externo se particiona solo su entrenamiento en
   cinco folds de calibración agrupados. Cada modelo de calibración repite su HPO
   sin recibir las etiquetas de su calibración ni de la validación externa.
   Un grupo aporta su máximo residual absoluto observado por target. Los candidatos
   de intervalo usan la predicción de ese mismo modelo ± residual del grupo.
   Se toman órdenes finitos floor((G+1)α) y ceil((G+1)(1−α)), α=0.10; si el orden
   cae fuera de rango, el extremo es infinito. No se proyectan extremos por edad.
   La cobertura nominal es empírica, no garantía exacta de 90%: CV+ tiene una cota
   general aproximada de 1−2α. Agrupamiento, estratificación dependiente de etiquetas,
   deriva y selección de arquitectura limitan las garantías de intercambiabilidad.
8. Cobertura OOF externa por target y tipo: denominadores de predicciones, filas
   y grupos únicos, cobertura de grupo, ancho y alarma fuera de [0.85,0.95].
   Types 4/5 y menos de veinte grupos se etiquetan sin soporte fiable. La selección
   global previa de arquitectura/familia hace esta cobertura diagnóstica y condicional.
9. Cordura con los cinco folds de la primera repetición, declarados antes de resultados:
   19 permutaciones conjuntas de etiquetas y nulos entre grupos del mismo tamaño,
   semilla derivada determinísticamente de 42; HPO repetida por permutación/fold.
   p=(1+#R²_perm≥R²_real)/20, resolución 0.05. Prueba exploratoria condicional a
   finalistas/arquitectura fijados; no repite la selección de las etapas previas.
   La deriva limita intercambiabilidad. R² barajado puede ser negativo por sobreajuste;
   no se fuerza a cero. Una media >0.1 genera alarma de señal espuria.
10. Curvas al 25/50/75/100% de grupos del entrenamiento externo, subconjuntos
    anidados por permutación fija de grupos con semilla 42. Nunca se fraccionan grupos
    ni cambia la validación entre tamaños; HPO y preprocesamiento se reajustan.
    Se guardan MAE de ajuste y CV, R² y overfit_gap=R²_train−R²_CV.
11. Se reportan alarmas si MAPE medio de CV cae bajo los pisos de ruido de YAML.
    Las alarmas requieren revisión independiente antes de promover el predictor;
    no son evidencia automática de fuga. No se incorpora ejecutor de evaluación final.
12. Manifiesto de código, configuración, versiones, finalistas y train para reanudación.
    Artefactos bajo `outputs/fase3/stage3/`; sin cambios en Gold, datos o infraestructura.

## Resultados

Ejecución completa en `outputs/fase3/stage3/`; detalle en
[Etapa 3](../fase3/multisalida.md). Se mantiene el estado de borrador hasta la revisión
independiente, sin promoción del predictor.

- Gana provisionalmente GPR con (a): MAE R1/R7/R28 = 106.69/259.65/307.51;
  MAPE = 4.709/4.449/4.349%, ninguno bajo el piso de ruido.
- En GPR, (b) tiene ΔMAE agregado +0.961, IC95% [−1.557,3.522].
  (c) empeora R1 +4.086, IC95% [1.462,6.737], por lo que falla también su gate.
  Huber admite (b) dentro de su familia, sin superar el GPR de referencia.
- Monotonía OOF 100%. CV+ global R1/R7/R28 = 91.1/91.2/90.7%.
  Type 2 queda en 81.6/83.0/83.0% y requiere atención antes de promover bandas.
  Types 4/5 no tienen soporte fiable; Type 4 carece de R1 observado.
- Con 19 permutaciones, R² barajado medio = −0.020/−0.020/−0.025 y p=0.05
  en las tres salidas, resolución mínima del presupuesto; sin nulidad positiva espuria.
- Las curvas mejoran al ampliar grupos. overfit_gap R1/R7/R28 =
  0.0683/0.1812/0.2033. La sensibilidad temporal empeora MAE en
  48.4/153.1/160.9: la deriva sigue siendo una limitación material.
- Se completaron 270 ajustes externos de arquitectura, 25 evaluaciones externas
  de CV+ y 115 tareas de cordura; 229 pruebas unitarias/rúbrica en verde,
  lint y formato en verde. La reproducibilidad exacta se prueba en una configuración
  sintética reducida; la reanudación de artefactos valida hashes y configuración.

Se calcularon explicaciones SHAP descriptivas para los nueve componentes de la
arquitectura (a), antes de proyección; no explican la cadena (b) ni son OOF.
Quedan pendientes la revisión independiente, el registro MLflow de Etapa 4 y cualquier
conclusión de evaluación final autorizada. La cobertura por tipo y la deriva impiden
presentar este resultado como un predictor ya aprobado.

Referencia del método: Barber, Candès, Ramdas y Tibshirani,
[Predictive inference with the jackknife+](https://arxiv.org/abs/1905.02928).
La cota finita de CV+ incluye además un término dependiente del número de folds;
no se presenta la tasa nominal como garantía bajo la deriva de este conjunto.

## Cierre de Etapa 4

La comparación final usa (a) para las tres familias, según el alcance de cierre.
Se mantiene GPR (a) como predictor provisional seleccionado en Etapa 3. Huber (b)
sigue siendo un resultado de aquella comparación, pero no se registra como (a).
Los perfiles por target se congelan desde Etapa 2 como define el ADR 0006 §17–21.
CV+ se ejecuta de nuevo para **las tres familias**, con calibración por grupo dentro
de cada fold externo; no se reutiliza la cobertura previa exclusiva de GPR.
Los diagnósticos de permutación se amplían a B=200 con perfiles congelados, sin
repetir HPO; las curvas de aprendizaje usan 25/50/75/100% de grupos del entrenamiento
externo y validación fija. Se conserva la advertencia sobre selección previa,
intercambiabilidad, tipos raros y diferencias de población en sensibilidad temporal.
Las cifras y las dos ejecuciones reproducidas del cierre se documentan en
[Modelado](../fase3/modelado.md). El estado sigue pendiente de revisión independiente.
