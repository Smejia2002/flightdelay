# 📑 ÍNDICE MAESTRO - FlightOnTime v2.0

**Guía completa de navegación del proyecto**  
**Versión**: 2.0.0  
**Fecha**: 2026-01-13

---

## 📖 DOCUMENTACIÓN

### 🎯 **Para Empezar**
| Documento                              | Descripción                          | Tiempo de lectura |
| -------------------------------------- | ------------------------------------ | ----------------- |
| **[README.md](README.md)**             | Documentación principal del proyecto | 10 min            |
| **[GUIA_RAPIDA.md](GUIA_RAPIDA.md)** ⭐ | Guía de 5 minutos para presentación  | 5 min             |
| **[CHANGELOG.md](CHANGELOG.md)**       | Registro de cambios v1.0 → v2.0      | 3 min             |

### 📊 **Documentación Técnica**
| Documento                                                              | Descripción                    | Audiencia |
| ---------------------------------------------------------------------- | ------------------------------ | --------- |
| **[THRESHOLD_DECISION.md](THRESHOLD_DECISION.md)**                     | Justificación threshold 0.5200 | Técnica   |
| **[VISUALIZACIONES_INTERACTIVAS.md](VISUALIZACIONES_INTERACTIVAS.md)** | Guía completa de Plotly        | Técnica   |
| **[data/data_dictionary.md](data/data_dictionary.md)**                 | Diccionario de datos           | Técnica   |

---

## 💻 SCRIPTS EJECUTABLES

### 🚀 **Principales** (Ya Pre-ejecutados)
| Script                        | Función                           | Estado       | Tiempo  |
| ----------------------------- | --------------------------------- | ------------ | ------- |
| `train_model.py`              | Entrenar modelo con 15M registros | ✅ Completado | ~50 min |
| `generate_interactive_viz.py` | Generar visualizaciones Plotly    | ✅ Completado | ~30 seg |

### 🎯 **Para Demostración** (Ejecutar en presentación)
| Script                  | Función                   | Uso                            |
| ----------------------- | ------------------------- | ------------------------------ |
| **`predict.py`** ⭐      | Predicción en tiempo real | `python predict.py`            |
| `optimize_threshold.py` | Optimizar umbral          | `python optimize_threshold.py` |

---

## 🎨 VISUALIZACIONES

### 🌐 **Interactivas (Plotly)** ⭐ NUEVO

**Portal Principal:**
```
outputs/figures/index.html
```

**Visualizaciones Individuales:**
1. `confusion_matrix_xgboost_interactive.html` - Matriz de confusión
2. `roc_curve_xgboost_interactive.html` - Curva ROC
3. `pr_curve_xgboost_interactive.html` - Precision-Recall
4. `feature_importance_xgboost_interactive.html` - Feature importance
5. `threshold_analysis_xgboost_interactive.html` - Análisis threshold
6. `models_comparison_interactive.html` - Comparación modelos

### 📊 **Estáticas (PNG)** - Backup
1. `confusion_matrix_xgboost.png`
2. `roc_curve_xgboost.png`
3. `pr_curve_xgboost.png`
4. `feature_importance_xgboost.png`
5. `threshold_analysis_xgboost.png`
6. `models_comparison.png`
7. `threshold_optimization.png`

---

## 🧠 MODELO

### 📁 **Archivos del Modelo**
| Archivo                          | Descripción                     | Tamaño |
| -------------------------------- | ------------------------------- | ------ |
| `models/model.joblib`            | XGBoost entrenado               | 502 KB |
| `models/metadata.json`           | Metadatos (threshold, features) | 2 KB   |
| `models/feature_engineer.joblib` | Transformador features          | 35 KB  |
| `models/training_info.json`      | Info entrenamiento 15M          | 2 KB   |

### 📊 **Métricas y Resultados**
| Archivo                                       | Descripción                 |
| --------------------------------------------- | --------------------------- |
| `outputs/metrics/evaluation_report.md`        | Reporte evaluación completo |
| `outputs/metrics/evaluation_results.json`     | Resultados 4 modelos (JSON) |
| `outputs/metrics/threshold_optimization.json` | Análisis 85 thresholds      |

---

## 💡 CÓDIGO FUENTE

### 📂 **Módulos Python** (`src/`)
| Módulo                     | Función                    | Líneas |
| -------------------------- | -------------------------- | ------ |
| `config.py`                | Configuración centralizada | ~100   |
| `features.py`              | Feature engineering        | ~220   |
| `modeling.py`              | Entrenamiento modelos      | ~330   |
| `evaluation.py`            | Evaluación (matplotlib)    | ~365   |
| **`interactive_viz.py`** ⭐ | Visualizaciones Plotly     | ~730   |

### 📊 **Notebooks**
| Notebook                    | Descripción                    |
| --------------------------- | ------------------------------ |
| `notebooks/EDA_final.ipynb` | Análisis exploratorio completo |

---

## 📁 ESTRUCTURA COMPLETA

```
PRUEBA ESPECIAL FINAL VUELOS 2.0/
│
├── 📄 README.md                        # ⭐ PRINCIPAL - Lee esto primero
├── 📄 GUIA_RAPIDA.md                   # ⭐ PARA PRESENTACIÓN
├── 📄 CHANGELOG.md                     # Cambios v1.0 → v2.0
├── 📄 THRESHOLD_DECISION.md            # Decisión técnica threshold
├── 📄 VISUALIZACIONES_INTERACTIVAS.md  # Guía Plotly
├── 📄 INDICE_MAESTRO.md                # Este archivo
├── 📄 requirements.txt                 # Dependencias Python
│
├── 📂 0.0. DATASET ORIGINAL/
│   └── dataset_prepared.parquet        # 35.6M vuelos
│
├── 📂 data/
│   └── data_dictionary.md
│
├── 📂 src/                             # CÓDIGO MODULAR
│   ├── __init__.py
│   ├── config.py
│   ├── features.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── interactive_viz.py              # ⭐ NUEVO
│
├── 📂 models/                          # MODELOS ENTRENADOS
│   ├── model.joblib
│   ├── metadata.json
│   ├── feature_engineer.joblib
│   └── training_info.json              # ⭐ NUEVO
│
├── 📂 outputs/
│   ├── figures/
│   │   ├── index.html                  # ⭐ DASHBOARD
│   │   ├── *_interactive.html          # ⭐ 6 gráficos Plotly
│   │   └── *.png                       # Gráficos PNG
│   └── metrics/
│       ├── evaluation_report.md
│       ├── evaluation_results.json
│       └── threshold_optimization.json # ⭐ NUEVO
│
├── 📂 notebooks/
│   └── EDA_final.ipynb
│
├── 📂 optional_helpers/
│
├── 🐍 train_model.py                   # Entrenamiento 15M
├── 🐍 predict.py                       # ⭐ DEMO EN VIVO
├── 🐍 optimize_threshold.py            # ⭐ Optimizador
└── 🐍 generate_interactive_viz.py      # ⭐ Generador Plotly
```

⭐ = Nuevo en v2.0

---

## 🎯 RUTAS DE NAVEGACIÓN SUGERIDAS

### Para **Jueces del Hackathon** 👨‍⚖️
1. `GUIA_RAPIDA.md` (5 min)
2. `outputs/figures/index.html` (explorar visualizaciones)
3. Demo: `python predict.py` (opción 1)
4. `README.md` (si requieren más detalles)

### Para **Desarrolladores** 👨‍💻
1. `README.md` (completo)
2. `src/` (revisar código modular)
3. `train_model.py` (pipeline de entrenamiento)
4. `CHANGELOG.md` (entender evolución)

### Para **Data Scientists** 👨‍🔬
1. `README.md` (sección Modelo)
2. `THRESHOLD_DECISION.md` (decisiones técnicas)
3. `outputs/metrics/` (métricas detalladas)
4. `notebooks/EDA_final.ipynb` (análisis exploratorio)
5. `VISUALIZACIONES_INTERACTIVAS.md` (implementación Plotly)

### Para **Stakeholders de Negocio** 👔
1. `GUIA_RAPIDA.md` (sección Impacto)
2. `README.md` (sección Beneficiarios)
3. Demo visual: `outputs/figures/index.html`

---

## 🔍 BÚSQUEDA RÁPIDA

### "¿Dónde encuentro...?"

| ¿Qué buscas?                   | Dónde está                                |
| ------------------------------ | ----------------------------------------- |
| **Métricas del modelo**        | `README.md` líneas 40-75                  |
| **Dashboard interactivo**      | `outputs/figures/index.html`              |
| **Threshold actual**           | `models/metadata.json` línea 3            |
| **Features usadas**            | `models/metadata.json` líneas 4-22        |
| **Justificación threshold**    | `THRESHOLD_DECISION.md`                   |
| **Cómo hacer predicciones**    | `GUIA_RAPIDA.md` o `predict.py`           |
| **Visualizaciones PNG**        | `outputs/figures/*.png`                   |
| **Código feature engineering** | `src/features.py`                         |
| **Código entrenamiento**       | `src/modeling.py` + `train_model.py`      |
| **Comparación modelos**        | `outputs/metrics/evaluation_results.json` |

---

## 📊 DATOS CLAVE (MEMORIZAR)

### Modelo
- **Algoritmo**: XGBoost
- **Accuracy**: 72.46%
- **Recall**: 53.51%
- **ROC-AUC**: 0.7172
- **Threshold**: 0.5200

### Dataset
- **Total**: 35.6M vuelos
- **Entrenamiento**: 15M (42%)
- **Features**: 17
- **Período**: 2020-2024

### Impacto
- **Retrasos detectados extra**: +227,305
- **Mejora recall**: +7.8%
- **Mejora accuracy**: +6.66%

---

## 💻 COMANDOS CLAVE

```bash
# Ver dashboard
start outputs\figures\index.html

# Hacer predicción
python predict.py

# Optimizar threshold
python optimize_threshold.py

# Re-generar visualizaciones
python generate_interactive_viz.py

# Ver métricas
cat outputs\metrics\evaluation_report.md
```

---

## ✅ ESTADO DEL PROYECTO

| Componente      | Estado        | Versión                 |
| --------------- | ------------- | ----------------------- |
| Documentación   | ✅ Completa    | 2.0                     |
| Modelo          | ✅ Entrenado   | 15M registros           |
| Visualizaciones | ✅ Generadas   | 6 Plotly + 7 PNG        |
| Scripts         | ✅ Funcionales | predict.py, optimize.py |
| Threshold       | ✅ Optimizado  | 0.5200                  |
| Dashboard       | ✅ Operativo   | index.html              |

**ESTADO GENERAL**: 🟢 **LISTO PARA HACKATHON**

---

## 📞 AYUDA RÁPIDA

### Problema común ¿Solución?
- "No encuentro X" → Busca aquí primero
- "¿Qué archivo abro?" → `GUIA_RAPIDA.md`
- "¿Cómo presento?" → `GUIA_RAPIDA.md` → Script
- "¿Qué es nuevo?" → `CHANGELOG.md`
- "¿Decisiones técnicas?" → `THRESHOLD_DECISION.md`

---

**Este índice maestro es tu mapa del proyecto. Guárdalo a mano.** 🗺️

---

*Última actualización: 2026-01-13*  
*FlightOnTime v2.0 - Hackathon Edition*
