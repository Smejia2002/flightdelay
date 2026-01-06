# Reporte de Tamaño del Modelo

- Modelo: model.joblib
- Total de features (post-encoding): 795

## Desglose de features

- Categóricas (one-hot): 787
  - aerolinea: 18 categorías
  - origen: 384 categorías
  - destino: 385 categorías
- Numéricas: 8
  - **Base (4)**: distancia_km, dep_hour, dep_dow, dep_month
  - **Clima (4)**: temp, wind_spd, precip_1h, climate_severity_idx

## Nota sobre features de clima
Las features climáticas se obtienen vía API externa usando pronóstico disponible 2 horas antes de la salida (T-2h), garantizando ausencia de data leakage.
