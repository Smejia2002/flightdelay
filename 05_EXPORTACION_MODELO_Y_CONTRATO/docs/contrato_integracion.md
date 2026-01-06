# Contrato de Integracion

## Request
```json
{
  "aerolinea": "AZ",
  "origen": "GIG",
  "destino": "GRU",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}
```

## Response
```json
{
  "prevision": "Puntual|Retrasado",
  "probabilidad": 0.00
}
```

## Enriquecimiento de clima
- Se consulta un servicio externo con latitud/longitud y fecha_partida.
- Se usa el pronostico disponible 2 horas antes de la salida.
- Variables: TEMP, WIND_SPD, PRECIP_1H, CLIMATE_SEVERITY_IDX.
