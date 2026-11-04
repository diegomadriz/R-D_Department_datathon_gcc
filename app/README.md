# `app/`

App web de resultados (Next.js en Vercel): simulador con curva de 0 a 28 días y banda de incertidumbre, alertas de especificación, diseño inverso, evaluación de lotes por CSV, resumen de resistencias, comparación de modelos y valor de negocio. Lee un snapshot de las tablas Gold solo en el servidor. Si no se autoriza publicar el modelo fuera de Databricks, el respaldo de la simulación es `gold_sim_grid` en el dashboard. La decisión y sus límites están en el [ADR 0010](../docs/adr/0010-app-vercel-y-dashboard-minimo.md).
