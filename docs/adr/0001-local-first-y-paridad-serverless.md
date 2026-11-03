# ADR 0001: Desarrollo local-first con paridad con Databricks serverless

- **Estado:** Aceptado
- **Fecha:** 2026-11-03

## Contexto

- Databricks Free Edition tiene un límite diario de cómputo y solo ofrece serverless.
- La solución debe poder recrearse desde cero en cualquier cuenta y correr de punta a punta sin intervención manual.
- Lo que funciona en local tiene que funcionar igual en el runtime serverless.

## Decisión

1. **Local-first.** Toda la lógica se desarrolla y prueba en local: pandas y sklearn en la Fase 0, y PySpark y MLflow locales a partir de la Fase 1. Databricks se reserva para las ejecuciones de integración y la final.
2. **Python 3.12.3**, fijado con `uv` (`.python-version` y `requires-python`). Es la versión del entorno serverless de Databricks, *environment version* 6. Fuente: https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/
3. **Dependencias alineadas con el runtime.** Las librerías que incluye el entorno serverless 6 se fijan a su versión exacta: pandas 2.3.3, numpy 2.3.4, scipy 1.16.3, scikit-learn 1.7.2, statsmodels 0.14.5, matplotlib 3.10.6, pyarrow 21.0.0, ipykernel 6.31.0 y pytest 8.4.2. Las demás quedan fijadas en `uv.lock`. pyspark, mlflow, lightgbm y shap entrarán cuando se validen contra el runtime.
4. **El dato crudo se versiona.** `data/datathon_data_gcc.csv` (~177 KB) va en Git, y su SHA-256 queda en `data/CHECKSUMS.txt`. `tests/unit/test_smoke.py` verifica que el archivo no cambie.
5. **Despliegue multi-cuenta.** El despliegue será con Databricks Asset Bundles, con un perfil del CLI por cuenta. Los perfiles y tokens viven en `~/.databrickscfg` y nunca en el repositorio.

## Alternativas consideradas

- **Desarrollar directamente en Databricks:** se descartó porque consume la cuota diaria y dificulta las pruebas unitarias.
- **Usar la versión de Python del sistema** (3.13 o posterior): se descartó porque no coincide con el runtime serverless y rompería la paridad.

## Consecuencias

- Cuando Databricks publique un *environment version* nuevo, se revisan las versiones fijadas.
- PySpark local necesita Java 17 o 21. El proyecto fijará `JAVA_HOME` a la 21 cuando se agregue PySpark.
