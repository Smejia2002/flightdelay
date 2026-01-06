# Fase 03 - Feature Engineering

## Objetivo
Crear features simples, interpretables y reconstruibles desde el API, incluyendo clima via enriquecimiento externo.

## Inputs
- ../02_LIMPIEZA_Y_DATASET_MODEL_READY/outputs/flights_model_ready.parquet

## Outputs
- outputs/flights_features.parquet

## Docs
- docs/diccionario_features.md
- docs/qa_post_features.md

## Reglas
- No crear features no disponibles en inferencia.
- Clima se obtiene via API externa usando lat/lon + fecha_partida.
