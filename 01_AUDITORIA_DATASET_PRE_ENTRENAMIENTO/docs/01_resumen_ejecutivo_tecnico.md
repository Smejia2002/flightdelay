# Resumen Ejecutivo Técnico

## Hallazgos clave
- Tamaño: 35,669,175 filas y 25 columnas.
- Tipos: mezcla de numéricos (int/float), fechas y categóricas.
- Nulos: 0.0 en todas las columnas según df.isna(); nulos encubiertos (''/NA/NULL) no detectados en columnas object (STATION_KEY = 0.0).
- Rango temporal: FL_DATE entre 2020-01-01 y 2025-09-30. Continuidad diaria confirmada (0 días faltantes).
- Target: DEP_DEL15 con desbalance moderado (0=28,934,579; 1=6,734,596; ~18.9% retraso).

## Riesgos / Calidad
- Fuga de información: DEP_DELAY es una variable posterior al evento y filtra la etiqueta. Debe excluirse de features.
- Valores fuera de dominio:
  - CRS_DEP_TIME max 2400 (debería ser 0-2359).
  - DEP_HOUR max 24 (debería ser 0-23).
  - sched_minute_of_day max 1440 (debería ser 0-1439).
  - PRECIP_1H min -0.1 (precipitación negativa).
  - LATITUDE min -14.33 y LONGITUDE max 145.73 (fuera de rango esperado si el dominio es sólo EEUU).

## Conclusión
El dataset es usable para un MVP, pero requiere limpieza de dominios y exclusión de variables con fuga. Confirmada la continuidad temporal; falta definir reglas de dominio geográficas.
