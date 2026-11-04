# ADR 0010: app de resultados en Vercel y dashboard mínimo en Databricks

- **Estado:** Aceptado
- **Fecha:** 2026-11-03
- **Relacionados:** ADR 0002, 0005, 0007, 0008 y 0009.

## Contexto

El criterio 5 de la rúbrica califica la experiencia: simulación, curvas y alertas
visuales dinámicas. El enunciado (§3.4) pide seis vistas y sugiere un AI/BI
Dashboard. Las Databricks Apps de Free Edition se apagan a las 24 horas y el
cómputo diario es limitado, así que una app de consulta en vivo no es confiable
para una evaluación.

Cuatro hechos del proyecto condicionan el diseño:

- El CSV no trae identificador de lote ni fecha, y sus columnas son anónimas. Sus
  identidades químicas son hipótesis de la Fase 0 (`variable_fingerprint.md`). Las
  unidades siguen sin confirmar.
- La predicción del GPR es `k(x, X_train)·α`. Su archivo lleva las filas de train y
  con el kernel y α se reconstruyen los targets: publicar el modelo equivale a
  publicar los datos.
- El CV+ evaluado no es `ŷ ± q`: combina los centros de cinco modelos de calibración
  por edad con los residuos máximos por grupo. El modelo `@champion` solo entrega la
  predicción puntual. El ancho medio casi no varía entre tipos (GPR a 28 días:
  1378–1389 psi en el lockbox), de modo que el CV+ no se ensancha por sí solo fuera
  de la región con datos.
- `gold_predictions` contiene predicciones OOF de train, fila a fila, y
  `gold_train_ranges` solo publica mínimo, máximo, p05 y p95 por tipo.

## Decisión

1. **Dos superficies.** La app principal es web (Next.js en Vercel) y un AI/BI
   Dashboard mínimo en Databricks cubre las seis vistas de §3.4 sobre las mismas
   tablas Gold, exportado a `dashboards/*.lvdash.json`.
2. **Datos por snapshot.** Un paso del job exporta las tablas Gold de consumo y los
   modelos a JSON en un volumen. La app nunca consulta el warehouse en vivo. El
   snapshot se lee solo en el servidor; nunca va en `public/` ni en el bundle del
   cliente, y no se versiona en git. Cualquier cuenta lo regenera con el job.
3. **Solo agregados en la app.** La app cubre también las seis vistas de §3.4, para que
   la evaluación no dependa de abrir Databricks: distribuciones, comparación de R², MAE
   y MAPE de los 3 modelos, importancia global por edad, calidad contra resistencia,
   predicho contra real y filtro por `ProductType`. Los scatters se muestran como
   hexbin o densidad y se ocultan las celdas con pocos registros, con un mínimo fijado
   en `conf/`. Usan solo train y predicciones OOF, con ese rótulo; las filas de test no
   se leen. Las vistas fila a fila quedan en el AI/BI, dentro de Databricks.
4. **Servicio del modelo en TypeScript, en el servidor.** Un Route Handler en Node
   implementa el preprocesamiento del `Pipeline` ya ajustado, el GPR (Matérn ν=1.5 con
   ARD + WhiteKernel y `normalize_y`) y la proyección isotónica. Un test de paridad
   exige reproducir el pyfunc `@champion` con tolerancia 1e-6 sobre entradas generadas
   por Python. La API rechaza cualquier tipo distinto de Types 1–3, porque un tipo no
   visto se codificaría como Type 1.
5. **Intervalos: CV+ exacto.** El snapshot incluye los 15 modelos de calibración
   (5 por edad, ajustados solo con train) y los residuos por grupo, y la paridad se
   comprueba también contra `cvplus_bounds`. Si el tamaño o la latencia lo impiden, una
   aproximación `ŷ ± q` requiere enmendar este ADR y reportar su cobertura OOF en train.
6. **P(no cumplir).** Se calcula con la distribución empírica de los residuos con signo
   OOF de train a 28 días: `P = fracción de residuos r con ŷ + r < meta`. No asume
   normalidad. Se muestra solo si pasa una prueba de calibración sobre OOF de train,
   global y por tipo, con un umbral fijado en `conf/` antes de calcularla. Si no pasa,
   la app muestra el veredicto sin porcentaje. Nunca se calibra con test.
7. **Veredicto de tres estados**, con `ŷ` y `L` como predicción y límite inferior a
   28 días:
   - *Cumple:* `L ≥ meta`.
   - *En vigilancia:* `ŷ ≥ meta > L`.
   - *Bajo meta:* `ŷ < meta`.

   El rojo se reserva para el último. Las metas viven en `conf/specs.yaml` y se rotulan
   como supuesto hasta que GCC las confirme.
8. **Fuera de la región con datos.** La región es p1–p99 por tipo y variable, solo de
   train; `gold_train_ranges` debe ampliarse con esos percentiles y una rejilla para
   mostrar cada variable en percentil. Es una comprobación marginal, no multivariada.
   Fuera de la región se muestra un aviso, se oculta P(no cumplir) y la banda se rotula
   "cobertura no validada". Esto reemplaza al requisito previo de ensanchar la banda:
   solo se ensanchará con la varianza del GPR si una validación en train, con pliegues
   temporales como aproximación a régimen nuevo, muestra que mejora la cobertura sin
   sobrecubrir dentro de la región. No se usan multiplicadores fijos.
9. **Evaluación de lotes por CSV.** El usuario sube un CSV con las columnas originales;
   se procesa en el servidor, no se guarda y no se registra su contenido. Las filas
   incompletas o de tipos no admitidos se devuelven como "sin predicción" con su causa.
   Lo histórico se resume en conteos por tipo y estado, rotulados como OOF de train. La
   app no presenta el histórico como producción reciente, porque no hay fechas ni lotes.
10. **Diseño inverso.** Cada palanca se mueve solo en su dirección física: la finura
    sube, la caliza baja y el SO₃ va hacia el óptimo. Nunca se propone bajar la finura
    ni el C₃S para subir R28. Si el modelo no premia esa dirección, la fila dice "Sin
    recomendación". La búsqueda se limita a la región con datos del tipo.
11. **Honestidad de las etiquetas.** Las identidades químicas se presentan como
    interpretación probable, con la columna original visible y su nivel de confianza.
    Type 2 lleva marca de baja confianza; Types 4/5 no se simulan.
12. **Acceso.** Contraseña en el middleware, guardada en una variable de entorno de
    Vercel, que se entrega a los jueces junto con el enlace. Ningún token de Databricks
    va en el repo.
13. **Captura de valores.** El simulador acepta sliders y captura numérica por variable,
    como pide el criterio 5. Las columnas están escaladas, así que la captura usa la
    escala de la columna original y se muestra también su percentil.
14. **Ejecución de punta a punta.** La exigencia de correr sin intervención manual se
    cumple con el job y el AI/BI en Databricks, que se regeneran solos desde cero. La app
    es una publicación: conserva su último snapshot y actualizarla es un paso aparte y
    documentado, que no forma parte de la corrida evaluada.
15. **Valor de negocio.** El ahorro se desglosa en merma, silos, aditivos y clínker, con
    Monte Carlo, tornado y curva de valor contra MAE. Cada supuesto lleva fuente o la marca
    "por validar con GCC". La narrativa física visible se presenta como interpretación
    probable, conforme al punto 11.

## Alternativas descartadas

- **Databricks App como principal:** se apaga a las 24 horas en Free Edition.
- **GPR en el navegador:** expone las filas de train.
- **Función Python en Vercel:** reutiliza el código, pero con un bundle pesado y
  arranque en frío.
- **Modelo sustituto en el cliente:** no lleva filas, pero pierde exactitud y agrega un
  error que habría que reportar.
- **Tabla de lotes con el histórico:** mostraría datos fila a fila e inventaría una
  noción de "recientes" que el dato no sostiene.
- **P(no cumplir) con distribución normal y ensanchamiento fijo:** son supuestos sin
  respaldo en el modelo evaluado.

## Consecuencias

- **Datos en un tercero.** El servidor de Vercel almacena el JSON del modelo y, con él,
  filas de train. "No va al navegador" no significa "no sale de Databricks". El permiso
  de GCC sigue sin confirmar (`docs/fase0/preguntas_organizadores.md`) y debe resolverse
  antes de desplegar el modelo real. Sin ese permiso, la simulación queda dentro de
  Databricks: `gold_sim_grid` publica en el AI/BI predicciones reales del modelo,
  precalculadas sobre una rejilla por tipo y palanca, sin filas de datos, y la app web
  se publica sin el modelo. No se usa un modelo sintético, porque el simulador mostraría
  números falsos en el componente que más pesa en el criterio 5.
- La rejilla solo cubre combinaciones precalculadas: es un respaldo con menos libertad
  que el simulador de la app, y agrega una tabla y cómputo al job.
- El benchmark del criterio 4 no está definido: la app compara contra la referencia propia
  L1 y no declara ese criterio como cumplido hasta recibirlo.
- El job debe crecer: exportar modelos de calibración, residuos y percentiles. Se hace en
  módulos nuevos, porque los componentes congelados del protocolo del lockbox no cambian.
- La app depende de una corrida completa del job en Databricks, todavía no verificada.
  Hasta entonces usa un snapshot de prueba rotulado.
- Hay dos implementaciones de la misma matemática. El test de paridad es obligatorio y
  debe correr cada vez que cambie el snapshot.
- `app/` deja de ser el placeholder de la Databricks App.

## Enmienda 1: versión de evaluación

**Aceptada el 2026-11-03**, por falta de tiempo antes de la entrega. Sustituye para esta
versión los puntos 2, 4, 5 y 9. Los demás puntos siguen vigentes. Los puntos sustituidos
quedan como la ruta hacia una versión de producción.

- **Modelo:** la app usa una aproximación aditiva del GPR congelado: la predicción con
  las medianas del tipo más el efecto de cada variable por separado, leído de una rejilla
  de percentiles. No lleva filas de datos ni parámetros del modelo, de modo que puede
  servirse al navegador. Difiere del GPR completo en 36 / 120 / 125 psi de MAE a 1 / 7 / 28
  días sobre train; no capta interacciones y la app lo declara a la vista.
- **Intervalos:** semiancho medio del CV+ OOF por tipo y edad. Su cobertura medida
  corresponde al GPR, no a la aproximación.
- **P(no cumplir):** se muestra solo si la desviación máxima de calibración sobre OOF de
  train es de 0.05 o menos. Con ese umbral se muestra en Type 1 (0.012) y se oculta en
  Type 2 (0.10) y Type 3 (0.073). El umbral se fijó después de ver estos valores, por lo
  que es una decisión de presentación, no una validación preregistrada.
- **Datos:** `app/scripts/export_snapshot.py` genera `app/public/snapshot.js` en local,
  solo con train. Como contiene únicamente agregados, se versiona y se entrega al
  navegador. Las celdas y barras con menos de 5 registros se omiten.
- **Lotes:** la carga de CSV valida encabezados pero no predice, porque esta versión no
  tiene servidor de modelo. Se ofrece un ejemplo sintético rotulado.
- **Sin cambio:** Lotes de ejemplo y ROI siguen siendo ilustrativos, rotulados como tales.
  `gold_sim_grid`, el AI/BI y la exportación desde el job quedan pendientes.

## Enmienda 2: acceso abierto

**Aceptada el 2026-11-04.** Sustituye el punto 12: por decisión del equipo se retira la
contraseña y la app queda abierta para cualquiera con el enlace, sin una protección de
acceso. Se publican `index.html`, `snapshot.js` (agregados de train, sin filas individuales,
con celdas de al menos 5 observaciones) y el video de demo, que muestra métricas de test.

El permiso de GCC y el uso de su marca siguen sin confirmar, y esta decisión no los
resuelve. El encabezado `X-Robots-Tag: noindex, nofollow` pide a los buscadores no indexar
la app, pero no limita el acceso. Si GCC lo objeta, la app se retira de Vercel y se
restaura el proxy desde el historial de Git.

## Sin verificar

- La latencia del SHAP exacto no está medida. Con 4,096 coaliciones, 16 centroides y 621
  filas son cientos de millones de operaciones de kernel por edad: se espera del orden de
  segundos, no milisegundos, por lo que se sirve en una ruta aparte y bajo demanda.
- El tamaño del snapshot con 18 GPR y sus límites en una función de Vercel.
- Si el plan Hobby de Vercel admite este uso (sus términos restringen el uso comercial).
- Unidades (psi), metas por tipo, identidades químicas y uso de la marca GCC.
