# `pipelines/`

Reservado para definiciones de pipelines declarativos. Hoy el medallón Bronze → Silver → Gold corre como un job de notebooks (`notebooks/00`–`04`), con la lógica en `src/gcc_strength/`, y no como pipeline declarativo; las razones están en el ADR 0003. Silver solo aplica transformaciones deterministas: las imputaciones y escalados estadísticos van dentro del `Pipeline` de sklearn.
