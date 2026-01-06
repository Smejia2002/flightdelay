# Reglas de limpieza

1. Eliminar variables con fuga de informacion: DEP_DELAY.
2. Eliminar redundancias: ORIGIN_CITY_NAME, DEST_CITY_NAME, YEAR, MONTH, DAY_OF_MONTH, DEP_HOUR, CRS_DEP_TIME, distance_bin, DIST_MET_KM.
3. Corregir dominios:
   - sched_minute_of_day: 1440 -> 0 (medianoche).
   - PRECIP_1H negativa -> 0.
4. Tratar outliers (winsorizacion 1%/99%) en TEMP, WIND_SPD, PRECIP_1H, DISTANCE para robustez del MVP.

Justificacion de outliers:
- En un MVP explicable se busca estabilidad frente a valores extremos no representativos sin eliminar registros.
