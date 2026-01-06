# Auditoria de Dataset Pre-Entrenamiento - FlightOnTime

## Objetivo
Evaluar si el dataset es confiable y usable antes de entrenar. No se entrena ningun modelo en esta fase.

## Alcance
- Cargar el Parquet original
- Validar filas, columnas y tipos
- Analizar nulos y nulos encubiertos
- Revisar rangos y valores fuera de dominio
- Verificar continuidad temporal
- Identificar posibles fugas de informacion
- Analizar distribucion del target (desbalance)

## Dataset
Origen sugerido: ../DATASET ORIGINAL/DATASET_FINAL_HACKATHON_2026.parquet

## Entregables
Ver `docs/`:
- 01_resumen_ejecutivo_tecnico.md
- 02_acciones_obligatorias.md
- 03_checklist_auditoria.md
- 04_informe_no_tecnico.md

## Regla
No entrenar ningun modelo en esta fase.
