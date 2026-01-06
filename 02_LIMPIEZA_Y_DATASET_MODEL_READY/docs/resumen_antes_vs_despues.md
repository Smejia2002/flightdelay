# Resumen Antes vs Despues

## Filas y columnas
- Filas iniciales: 35669175
- Filas finales: 35669175
- Columnas iniciales: 25
- Columnas finales: 15

## Columnas eliminadas
- Fuga: ['DEP_DELAY']
- Redundancias: ['ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'YEAR', 'MONTH', 'DAY_OF_MONTH', 'DEP_HOUR', 'CRS_DEP_TIME', 'distance_bin', 'DIST_MET_KM']

## Correcciones de dominio
- sched_minute_of_day 1440->0: 2 filas
- PRECIP_1H negativa->0: 1935275 filas

## Outliers (winsorizacion p01/p99)
- Umbrales: {
  "TEMP": {
    "p01": -8.899999618530273,
    "p99": 35.599998474121094
  },
  "WIND_SPD": {
    "p01": 0.0,
    "p99": 10.300000190734863
  },
  "PRECIP_1H": {
    "p01": 0.0,
    "p99": 2.299999952316284
  },
  "DISTANCE": {
    "p01": 100.0,
    "p99": 2611.0
  }
}
- Valores recortados: {
  "TEMP": {
    "below": 353793,
    "above": 288717
  },
  "WIND_SPD": {
    "below": 0,
    "above": 305485
  },
  "PRECIP_1H": {
    "below": 0,
    "above": 331151
  },
  "DISTANCE": {
    "below": 348227,
    "above": 319498
  }
}
