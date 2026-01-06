# QA Post Feature Engineering

## Dataset verificado
- Archivo: outputs/flights_features.parquet
- Filas: 35,669,175
- Columnas: 12

## Tipos de datos
- category: aerolinea, origen, destino
- numericos: distancia_km (float32), dep_hour (int8), dep_dow (int8), dep_month (int8), temp (float32), wind_spd (float32), precip_1h (float32), climate_severity_idx (int8), DEP_DEL15 (int8)

## Nulos
- Proporcion de nulos: 0.0 en todas las columnas (sin nulos detectados)

## Conclusión
El dataset de features es consistente, incluye clima via API externa y esta listo para modelado.
