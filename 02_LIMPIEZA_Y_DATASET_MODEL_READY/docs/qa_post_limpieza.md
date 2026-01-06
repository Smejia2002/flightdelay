# QA Post Limpieza

## Dataset verificado
- Archivo: outputs/flights_model_ready.parquet
- Filas: 35,669,175
- Columnas: 15

## Tipos de datos
- datetime: FL_DATE
- category: OP_UNIQUE_CARRIER, ORIGIN, DEST
- numericos: DAY_OF_WEEK, DEP_DEL15, DISTANCE, sched_minute_of_day, LATITUDE, LONGITUDE, TEMP, WIND_SPD, PRECIP_1H, CLIMATE_SEVERITY_IDX
- object: STATION_KEY

## Nulos
- Proporcion de nulos: 0.0 en todas las columnas (sin nulos detectados)

## Conclusión
El dataset model-ready cumple con el esquema esperado y no presenta nulos tras la limpieza.
