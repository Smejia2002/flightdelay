# Diccionario de Features

## Features para entrenamiento (reconstruibles desde API)

| Feature | Tipo | Descripcion | Fuente / Derivacion |
| --- | --- | --- | --- |
| aerolinea | category | Codigo de aerolinea operadora | OP_UNIQUE_CARRIER |
| origen | category | Aeropuerto de origen | ORIGIN |
| destino | category | Aeropuerto de destino | DEST |
| distancia_km | float32 | Distancia del vuelo en km | DISTANCE * 1.60934 |
| dep_hour | int8 | Hora de salida programada (0-23) | fecha_partida.hour (o sched_minute_of_day // 60) |
| dep_dow | int8 | Dia de la semana (1-7) | fecha_partida.dayofweek + 1 |
| dep_month | int8 | Mes (1-12) | fecha_partida.month |
| temp | float32 | Temperatura | TEMP |
| wind_spd | float32 | Velocidad del viento | WIND_SPD |
| precip_1h | float32 | Precipitacion 1H | PRECIP_1H |
| climate_severity_idx | int8 | Indice de severidad climatica | CLIMATE_SEVERITY_IDX |

## Target
| Variable | Tipo | Descripcion |
| --- | --- | --- |
| DEP_DEL15 | int8 | 1 si retraso >= 15 min, 0 si puntual |

## Restricciones
- No se incluyen variables no disponibles en inferencia.
- No se usan variables posteriores al despegue.

## Schema final (parquet)
- aerolinea: category
- origen: category
- destino: category
- distancia_km: float32
- dep_hour: int8
- dep_dow: int8
- dep_month: int8
- temp: float32
- wind_spd: float32
- precip_1h: float32
- climate_severity_idx: int8
- DEP_DEL15: int8
