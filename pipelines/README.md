# `pipelines/`

Capas del medallón Bronze → Silver → Gold. Si Free Edition lo permite, se definen de forma declarativa; si no, como notebooks o scripts con Delta y checks propios. Cada capa debe poder correr desde un esquema vacío. Silver solo aplica transformaciones deterministas: las imputaciones y escalados estadísticos van dentro del `Pipeline` de sklearn.
