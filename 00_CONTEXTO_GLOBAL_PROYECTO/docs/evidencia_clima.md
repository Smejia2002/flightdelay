# Evidencia de uso de clima

## Decisiones confirmadas
- Se incorporan variables climaticas: TEMP, WIND_SPD, PRECIP_1H, CLIMATE_SEVERITY_IDX.
- Fuente de clima en inferencia: API externa usando latitud/longitud y fecha_partida.
- Momento de prediccion: 2 horas antes de la salida (se usa pronostico, no observacion posterior).

## Implicaciones
- El dataset de features incluye variables de clima derivadas del servicio externo.
- El modelo entrenado consume dichas variables y requiere enriquecimiento en inferencia.
