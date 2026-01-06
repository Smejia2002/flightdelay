# Pipeline de Inferencia

1. Recibir JSON de entrada (contrato fijo).
2. Enriquecer con clima via API externa (lat/lon + fecha_partida).
3. Construir features:
   - aerolinea, origen, destino
   - distancia_km
   - dep_hour, dep_dow, dep_month (desde fecha_partida)
   - temp, wind_spd, precip_1h, climate_severity_idx
4. Cargar modelo joblib y predecir probabilidad.
5. Mapear salida a {prevision, probabilidad}.
