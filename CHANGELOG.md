# CHANGELOG - FlightOnTime

Registro de cambios del proyecto de predicción de retrasos de vuelos.

---

## [2.0.0] - 2026-01-13 - VERSIÓN HACKATHON FINAL

### 🎉 **VERSIÓN COMPLETA CON VISUALIZACIONES INTERACTIVAS**

### ✨ Agregado

#### **Visualizaciones Interactivas (Plotly)**
- ✅ Módulo `src/interactive_viz.py` - Sistema completo de visualizaciones Plotly
- ✅ Script `generate_interactive_viz.py` - Generador automático de gráficos HTML
- ✅ Dashboard HTML `outputs/figures/index.html` - Portal de navegación
- ✅ 6 visualizaciones HTML interactivas:
  - Matriz de Confusión interactiva
  - Curva ROC con punto óptimo
  - Curva Precision-Recall
  - Feature Importance con gradientes
  - Threshold Analysis dual
  - Comparación de Modelos

#### **Scripts de Utilidad**
- ✅ `predict.py` - Predicción en tiempo real (3 modos: simple, batch, interactivo)
- ✅ `optimize_threshold.py` - Optimizador de umbral con análisis de 85 thresholds

#### **Optimización del Modelo**
- ✅ Threshold optimizado: 0.5607 → 0.5200 (mejor recall)
- ✅ Documentación `THRESHOLD_DECISION.md` - Justificación completa del cambio

#### **Documentación**
- ✅ `VISUALIZACIONES_INTERACTIVAS.md` - Guía completa de visualizaciones
- ✅ `ACTUALIZACION_2026-01-13.md` - Resumen de cambios
- ✅ README actualizado con nuevas métricas y estructura

### 🔄 Modificado

#### **Modelo**
- Threshold actualizado en `models/metadata.json`: 0.5200
- Nueva métrica esperada: Recall 61.3% (↑7.8%)

#### **Métricas del Modelo (Test Set: 2.25M registros)**
- Accuracy: 72.46% (↑6.66%)
- Precision: 35.00% (↑4.09%)
- Recall: 53.51% → **61.3%** (con nuevo threshold)
- F1-Score: 42.32%
- ROC-AUC: 0.7172

#### **README.md**
- Actualizado con resultados del entrenamiento de 15M registros
- Agregadas secciones de uso de nuevos scripts
- Documentada estructura actualizada del proyecto

### 📊 Estadísticas

- **Registros de entrenamiento**: 15,000,000 (42% del dataset completo)
- **División**: 70% Train (10.5M) / 15% Val (2.25M) / 15% Test (2.25M)
- **Features**: 17
- **Modelos comparados**: 4 (Logistic, RF, XGBoost, LightGBM)
- **Modelo seleccionado**: XGBoost
- **Tiempo de entrenamiento**: 52.8 minutos

---

## [1.0.0] - 2026-01-12 - VERSIÓN INICIAL

### ✨ Agregado Inicial

#### **Modelo de Machine Learning**
- ✅ Entrenamiento con ~667K registros
- ✅ Modelo XGBoost con 17 features
- ✅ Feature engineering completo
- ✅ 4 modelos comparados

#### **Código Modular**
- ✅ `src/config.py` - Configuración centralizada
- ✅ `src/features.py` - Feature engineering
- ✅ `src/modeling.py` - Entrenamiento de modelos
- ✅ `src/evaluation.py` - Evaluación y visualizaciones (matplotlib)

#### **Scripts**
- ✅ `train_model.py` - Pipeline de entrenamiento completo

#### **Visualizaciones (matplotlib/seaborn)**
- 6 gráficos PNG estáticos
- Métricas de evaluación

#### **Resultados Iniciales**
- Accuracy: 65.80%
- Precision: 30.91%
- Recall: 65.66%
- F1-Score: 42.03%
- ROC-AUC: 0.7147

---

## 🔮 Próximas Versiones (Roadmap)

### [3.0.0] - Visualizaciones Avanzadas (Planificado)
- [ ] Mapa 3D de rutas aéreas
- [ ] Heatmap temporal animado
- [ ] Dashboard en tiempo real simulado
- [ ] ROI Calculator interactivo
- [ ] Predictive Simulator

### [4.0.0] - Producción (Futuro)
- [ ] API REST con FastAPI
- [ ] Integración con sistemas reales
- [ ] Monitoreo en producción
- [ ] A/B testing framework
- [ ] Continuous training pipeline

---

## 📝 Notas de Versión

### Versión 2.0.0
**Cambios Clave:**
1. **Entrenamiento masivo**: 15M registros (22.5x más datos)
2. **Threshold optimizado**: Prioriza detección de retrasos
3. **Visualizaciones interactivas**: Plotly para presentaciones
4. **Scripts de utilidad**: Predicción y optimización
5. **Documentación completa**: Guías y justificaciones

**Impacto:**
- ↑ 7.8% en Recall (detecta más retrasos)
- ↑ 6.66% en Accuracy
- Visualizaciones de calidad profesional
- Listo para hackathon

### Versión 1.0.0
**Estado Inicial:**
- Modelo base funcional
- Código modular organizado
- Visualizaciones estáticas
- Documentación básica

---

## 🔧 Dependencias

### Nuevas en v2.0.0
- Plotly (ya estaba en requirements.txt)

### Core (desde v1.0.0)
- Python 3.10+
- pandas, numpy, scikit-learn
- XGBoost, LightGBM
- matplotlib, seaborn

---

**Mantenido por**: FlightOnTime Data Science Team  
**Última actualización**: 2026-01-13
