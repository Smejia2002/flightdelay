# README Final - FlightOnTime MVP

## Que hace el MVP
- Predice si un vuelo saldra puntual o con retraso (>= 15 minutos).
- Usa variables disponibles antes del vuelo y clima via API externa.
- Ofrece una respuesta simple: prevision y probabilidad.

## Que NO hace el MVP
- No optimiza al maximo el modelo ni garantiza el mejor rendimiento posible.
- No incluye manejo completo de cambios operativos o eventos extraordinarios.
- No ejecuta monitoreo ni retraining automatico.

## Flujo general
1. Datos historicos -> limpieza -> dataset model-ready.
2. Feature engineering con variables reconstruibles desde API.
3. Entrenamiento Logistic Regression (baseline explicable).
4. API /predict con enriquecimiento de clima.

## Ejemplo de uso
```json
{
  "aerolinea": "AZ",
  "origen": "GIG",
  "destino": "GRU",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}
```
Respuesta esperada:
```json
{
  "prevision": "Puntual",
  "probabilidad": 0.65
}
```

## Limitaciones
Ver `docs/limitaciones_y_mejoras.md`.
