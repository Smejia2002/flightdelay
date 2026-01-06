# Fase 04 - Entrenamiento y Evaluacion

## Objetivo
Entrenar un modelo baseline mejor que "siempre Puntual" y documentar metricas y confusion.

## Inputs
- ../03_FEATURE_ENGINEERING/outputs/flights_features.parquet

## Outputs
- outputs/modelo_entrenado.pkl
- MODELO_ENTRENADO_V1/
  - model.joblib
  - metadata.json
  - label_encoders.pkl
  - scaler.pkl

## Docs
- docs/reporte_metricas.md
- docs/comparacion_baseline.md
- docs/matriz_confusion.md

## Notas
- El modelo usa variables de clima enriquecidas via API externa.
- Se mantiene class_weight='balanced' para desbalance.
