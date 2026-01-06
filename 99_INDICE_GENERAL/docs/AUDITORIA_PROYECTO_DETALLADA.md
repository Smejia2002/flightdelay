# Auditoria Completa del Proyecto FlightOnTime

## Alcance y fuentes revisadas
Se revisaron documentos Markdown, notebooks y scripts en las fases 00 a 07, mas el indice general. Fuentes clave:
- 00_CONTEXTO_GLOBAL_PROYECTO (README, notebook, evidencia clima)
- 01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO (notebook y docs)
- 02_LIMPIEZA_Y_DATASET_MODEL_READY (notebook, script, docs)
- 03_FEATURE_ENGINEERING (notebook y docs)
- 04_ENTRENAMIENTO_Y_EVALUACION (notebook y docs)
- 05_EXPORTACION_MODELO_Y_CONTRATO (notebook y docs)
- 06_BACKEND_API (notebook pendiente de ejecucion)
- 07_DOCUMENTACION_Y_DEMO (notebook y docs)
- 99_INDICE_GENERAL (indice)

## Resumen ejecutivo
El MVP esta estructurado por fases y mantiene trazabilidad desde el dataset original hasta el modelo exportado y el contrato de integracion. Se incorporo clima via API externa (pronostico T-2h) sin cambiar el contrato del usuario. El modelo baseline (Logistic Regression con class_weight) supera al baseline naive en F1 y recall, pero queda un warning de convergencia. La validacion del endpoint /predict esta pendiente por falta de URL remota.

## Paso a paso del proceso (lenguaje natural)

### 00. Contexto global
1. Se definio el problema: predecir si un vuelo despega con retraso (>=15 min).
2. Se fijo el momento de prediccion en T-2h (cierre de check-in).
3. Se establecio el contrato de entrada/salida del API.
4. Se documento el enriquecimiento de clima via API externa usando lat/lon y fecha_partida.
5. Se dejaron reglas no negociables (sin fuga, variables pre-vuelo, modelo explicable).
Evidencia: 00_CONTEXTO_GLOBAL_PROYECTO/README.md, 00_CONTEXTO_GLOBAL_PROYECTO/00_Contexto_Global.ipynb, 00_CONTEXTO_GLOBAL_PROYECTO/docs/evidencia_clima.md.

### 01. Auditoria del dataset
1. Se cargo el parquet original y se revisaron filas, columnas y tipos.
2. Se confirmo rango temporal 2020-01-01 a 2025-09-30 con continuidad diaria (0 dias faltantes).
3. Se revisaron nulos y nulos encubiertos en columnas object.
4. Se detectaron riesgos: fuga por DEP_DELAY y dominios fuera de rango en tiempos y clima.
5. Se confirmo desbalance moderado en DEP_DEL15 (~18.9% retraso).
Evidencia: 01_AUDITORIA_DATASET_PRE_ENTRENAMIENTO/01_Auditoria_Dataset.ipynb y docs/.

### 02. Limpieza y dataset model-ready
1. Se eliminaron columnas con fuga (DEP_DELAY) y redundantes (ORIGIN_CITY_NAME, DEST_CITY_NAME, YEAR, MONTH, DAY_OF_MONTH, DEP_HOUR, CRS_DEP_TIME, distance_bin, DIST_MET_KM).
2. Se corrigieron dominios: sched_minute_of_day 1440->0 y PRECIP_1H negativa -> 0.
3. Se aplico winsorizacion p01/p99 en TEMP, WIND_SPD, PRECIP_1H y DISTANCE.
4. Se guardo flights_model_ready.parquet con 15 columnas.
5. Se genero trazabilidad con porcentaje de filas afectadas por regla.
Evidencia: 02_LIMPIEZA_Y_DATASET_MODEL_READY/02_Limpieza_y_Model_Ready.ipynb, scripts/clean_dataset.py, docs/*.

### 03. Feature engineering (con clima)
1. Se partio de flights_model_ready.parquet.
2. Se derivaron dep_hour, dep_dow y dep_month desde fecha_partida (sched_minute_of_day y FL_DATE).
3. Se mantuvieron categoricas (aerolinea, origen, destino).
4. Se incluyeron variables de clima: temp, wind_spd, precip_1h, climate_severity_idx.
5. Se guardo flights_features.parquet con 12 columnas.
6. Se actualizo el diccionario de features y QA.
Evidencia: 03_FEATURE_ENGINEERING/03_Feature_Engineering.ipynb, docs/diccionario_features.md, docs/qa_post_features.md.

### 04. Entrenamiento y evaluacion
1. Se entreno Logistic Regression con class_weight='balanced'.
2. Se comparo contra baseline naive (siempre Puntual).
3. Se evaluaron metricas: Accuracy 0.6301, Precision 0.2767, Recall 0.5943, F1 0.3776.
4. Se genero matriz de confusion y comparacion con baseline.
5. Se exporto modelo y artefactos (MODELO_ENTRENADO_V1).
Nota: existe warning de convergencia (max_iter=200) reportado por sklearn.
Evidencia: 04_ENTRENAMIENTO_Y_EVALUACION/04_Entrenamiento_y_Evaluacion.ipynb y docs/.

### 05. Exportacion y contrato de integracion
1. Se exporto el modelo a outputs/model.joblib.
2. Se creo metadata.json con features y notas de clima.
3. Se generaron ejemplos request/response.
4. Se documento pipeline de inferencia JSON -> features -> modelo -> salida.
Evidencia: 05_EXPORTACION_MODELO_Y_CONTRATO/05_Exportacion_Modelo_y_Contrato.ipynb y docs/.

### 06. Backend API (validacion pendiente)
1. Se preparo notebook para probar POST /predict con tres casos.
2. La ejecucion esta pendiente por falta de URL remota.
Evidencia: 06_BACKEND_API/06_Backend_API_Validacion.ipynb.

### 07. Documentacion y demo
1. Se redacto README final con que hace y que no hace el MVP.
2. Se preparo guion de demo (60-120s) en lenguaje simple.
3. Se listaron limitaciones y mejoras futuras.
Evidencia: 07_DOCUMENTACION_Y_DEMO/07_Documentacion_y_Demo.ipynb y docs/.

### 99. Indice general
1. Se creo un indice completo con enlaces a todas las fases y artefactos.
Evidencia: 99_INDICE_GENERAL/docs/README_indice_general.md.

## Observaciones y riesgos detectados
- **[CORREGIDO]** Errores de codificación UTF-8 en archivos markdown han sido corregidos.
- **[CORREGIDO]** Documentación de features numéricas actualizada a 8 (4 base + 4 clima).
- El entrenamiento presenta warning de convergencia; podría mejorar con mayor max_iter o escalado.
- La validación real del endpoint /predict está pendiente por URL remota no definida.
- El enriquecimiento de clima requiere un proveedor externo estable y políticas de cache/fallback.

## Esquema de presentaciones

### Presentacion tecnica (10-12 min)
1. Contexto, target y momento de prediccion (T-2h).
2. Auditoria del dataset: tamanos, calidad, fuga y dominios.
3. Limpieza y trazabilidad (reglas y porcentaje afectado).
4. Feature engineering y contrato de inferencia (incluye clima).
5. Modelado: baseline vs Logistic Regression, metricas y confusion.
6. Artefactos de modelo y contrato de integracion.
7. Limitaciones tecnicas y siguientes pasos.

### Presentacion no tecnica (6-8 min)
1. Problema en una frase: anticipar retrasos para mejorar experiencia.
2. Entrada simple: aerolinea, origen, destino, fecha y distancia.
3. Enriquecimiento automatico con clima.
4. Salida simple: "Puntual" o "Retrasado" con probabilidad.
5. Resultado del MVP: mejora frente a adivinar siempre puntual.
6. Que falta para version productiva (monitoreo, mejoras del modelo).
