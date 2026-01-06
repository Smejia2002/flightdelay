# Indice General del Proyecto - FlightOnTime

## 00_CONTEXTO_GLOBAL_PROYECTO
- README: ../00_CONTEXTO_GLOBAL_PROYECTO/README.md
- Notebook: ../00_CONTEXTO_GLOBAL_PROYECTO/00_Contexto_Global.ipynb
- Evidencia clima: ../00_CONTEXTO_GLOBAL_PROYECTO/docs/evidencia_clima.md

## 01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO
- README: ../01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO/README.md
- Notebook: ../01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO/01_Auditoria_Dataset.ipynb
- Docs:
  - ../01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO/docs/01_resumen_ejecutivo_tecnico.md
  - ../01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO/docs/02_acciones_obligatorias.md
  - ../01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO/docs/03_checklist_auditoria.md
  - ../01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO/docs/04_informe_no_tecnico.md

## 02_LIMPIEZA_Y_DATASET_MODEL_READY
- README: ../02_LIMPIEZA_Y_DATASET_MODEL_READY/README.md
- Notebook: ../02_LIMPIEZA_Y_DATASET_MODEL_READY/02_Limpieza_y_Model_Ready.ipynb
- Output: ../02_LIMPIEZA_Y_DATASET_MODEL_READY/outputs/flights_model_ready.parquet
- Docs:
  - ../02_LIMPIEZA_Y_DATASET_MODEL_READY/docs/reglas_limpieza.md
  - ../02_LIMPIEZA_Y_DATASET_MODEL_READY/docs/resumen_antes_vs_despues.md
  - ../02_LIMPIEZA_Y_DATASET_MODEL_READY/docs/trazabilidad_datos.md
  - ../02_LIMPIEZA_Y_DATASET_MODEL_READY/docs/qa_post_limpieza.md

## 03_FEATURE_ENGINEERING
- README: ../03_FEATURE_ENGINEERING/README.md
- Notebook: ../03_FEATURE_ENGINEERING/03_Feature_Engineering.ipynb
- Output: ../03_FEATURE_ENGINEERING/outputs/flights_features.parquet
- Docs:
  - ../03_FEATURE_ENGINEERING/docs/diccionario_features.md
  - ../03_FEATURE_ENGINEERING/docs/qa_post_features.md

## 04_ENTRENAMIENTO_Y_EVALUACION
- README: ../04_ENTRENAMIENTO_Y_EVALUACION/README.md
- Notebook: ../04_ENTRENAMIENTO_Y_EVALUACION/04_Entrenamiento_y_Evaluacion.ipynb
- Output: ../04_ENTRENAMIENTO_Y_EVALUACION/outputs/modelo_entrenado.pkl
- Modelo final: ../04_ENTRENAMIENTO_Y_EVALUACION/MODELO_ENTRENADO_V1/
- Docs:
  - ../04_ENTRENAMIENTO_Y_EVALUACION/docs/reporte_metricas.md
  - ../04_ENTRENAMIENTO_Y_EVALUACION/docs/comparacion_baseline.md
  - ../04_ENTRENAMIENTO_Y_EVALUACION/docs/matriz_confusion.md
  - ../04_ENTRENAMIENTO_Y_EVALUACION/docs/reporte_tamano_modelo.md

## 05_EXPORTACION_MODELO_Y_CONTRATO
- README: ../05_EXPORTACION_MODELO_Y_CONTRATO/README.md
- Notebook: ../05_EXPORTACION_MODELO_Y_CONTRATO/05_Exportacion_Modelo_y_Contrato.ipynb
- Outputs:
  - ../05_EXPORTACION_MODELO_Y_CONTRATO/outputs/model.joblib
  - ../05_EXPORTACION_MODELO_Y_CONTRATO/outputs/metadata.json
  - ../05_EXPORTACION_MODELO_Y_CONTRATO/outputs/example_request.json
  - ../05_EXPORTACION_MODELO_Y_CONTRATO/outputs/example_response.json
- Docs:
  - ../05_EXPORTACION_MODELO_Y_CONTRATO/docs/contrato_integracion.md
  - ../05_EXPORTACION_MODELO_Y_CONTRATO/docs/pipeline_inferencia.md

## 06_BACKEND_API
- Notebook: ../06_BACKEND_API/06_Backend_API_Validacion.ipynb
- Outputs (pendientes de URL remota):
  - ../06_BACKEND_API/outputs/respuesta_puntual.json
  - ../06_BACKEND_API/outputs/respuesta_retrasado.json
  - ../06_BACKEND_API/outputs/respuesta_error.json

## 07_DOCUMENTACION_Y_DEMO
- Notebook: ../07_DOCUMENTACION_Y_DEMO/07_Documentacion_y_Demo.ipynb
- Docs:
  - ../07_DOCUMENTACION_Y_DEMO/docs/README_final.md
  - ../07_DOCUMENTACION_Y_DEMO/docs/guion_demo.md
  - ../07_DOCUMENTACION_Y_DEMO/docs/limitaciones_y_mejoras.md

## DATASET ORIGINAL
- ../DATASET ORIGINAL/DATASET_FINAL_HACKATHON_2026.parquet
