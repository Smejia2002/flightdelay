# FlightOnTime - Contexto Global

## Objetivo del MVP
Construir un modelo de clasificacion binaria, simple y explicable, que anticipe si un vuelo despegara con retraso para informar a un publico no tecnico.

## Definicion operativa de retraso
Un vuelo se considera retrasado si despega 15 minutos o mas tarde respecto a su horario programado.
Target: DEP_DEL15 (0 = Puntual, 1 = Retrasado).

## Momento de prediccion (obligatorio)
La prediccion se realiza 2 horas antes de la salida programada, al cierre de check-in. En este momento solo se usan variables disponibles antes del vuelo.

## Contrato fijo del API
Entrada:
{
  "aerolinea": "AZ",
  "origen": "GIG",
  "destino": "GRU",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}
Salida:
{
  "prevision": "Puntual|Retrasado",
  "probabilidad": 0.00-1.00
}

## Enriquecimiento de clima
Las variables de clima se obtienen via un servicio externo usando latitud/longitud y fecha_partida.
Se utiliza el pronostico disponible 2 horas antes de la salida para evitar fuga de informacion.
Variables de clima: TEMP, WIND_SPD, PRECIP_1H, CLIMATE_SEVERITY_IDX.
El contrato de entrada al API del usuario no cambia; el clima se agrega en un paso interno de enriquecimiento.

## Reglas del proyecto (fuente de verdad)
- Prohibido usar informacion futura (fuga de informacion).
- Solo variables disponibles antes del vuelo.
- Modelo simple, explicable y reproducible.
- Cada fase tiene su carpeta.
- No mezclar archivos entre fases.
- Alcance: MVP funcional (sin optimizacion extrema).

## Control de tiempo (hackathon)
| Fase | Prioridad | Tiempo estimado | Si falta tiempo, simplificar |
| --- | --- | --- | --- |
| Contexto y definiciones | Alta | 1-2 h | Limitarse a README y notebook base |
| Preparacion de datos | Alta | 4-6 h | Usar subset de variables clave |
| Modelado baseline | Alta | 3-4 h | Usar un modelo lineal interpretable |
| Evaluacion y explicabilidad | Media | 2-3 h | Reportar metricas basicas |
| API y demo | Media | 2-3 h | Mock de API y ejemplo de inferencia |
| Documentacion final | Baja | 1-2 h | Resumen ejecutivo de 1 pagina |
