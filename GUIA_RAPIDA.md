# 📚 GUÍA RÁPIDA - FlightOnTime v2.0

**Versión**: 2.0.0 - Hackathon Ready  
**Fecha**: 2026-01-13  
**Estado**: ✅ Listo para Presentación

---

## ⚡ INICIO RÁPIDO (5 minutos)

### 1. Ver el Dashboard Interactivo
```bash
# Abrir en navegador
outputs/figures/index.html
```
**¡Esto es lo primero que debes mostrar a los jueces!** 🎯

### 2. Hacer una Predicción
```bash
python predict.py
# Selecciona opción 1 (Ejemplo simple)
```

### 3. Ver Métricas del Modelo
```bash
cat outputs/metrics/evaluation_report.md
```

---

## 📊 CIFRAS CLAVE PARA LA PRESENTACIÓN

### Modelo
- **Algoritmo**: XGBoost
- **Datos de entrenamiento**: 15,000,000 registros
- **Accuracy**: 72.46%
- **Recall**: 53.51% (detecta 53% de retrasos)
- **ROC-AUC**: 0.7172
- **Threshold**: 0.5200 (optimizado)

### Dataset
- **Total**: 35.6M vuelos (2020-2024)
- **Features**: 17
- **División**: 70% Train / 15% Val / 15% Test

### Impacto
- **Retrasos detectados**: 280,622 (test)
- **Falsos negativos**: 144,201 (test)
- **Beneficiarios**: Pasajeros, aerolíneas, aeropuertos

---

## 🎨 VISUALIZACIONES (Para la Demo)

### Dashboard Principal
```
outputs/figures/index.html
```
**Incluye 6 visualizaciones interactivas:**

1. 📊 **Matriz de Confusión** - Resultados del modelo
2. 📈 **Curva ROC** - AUC = 0.7172
3. 📉 **Precision-Recall** - Trade-off visual
4. ⭐ **Feature Importance** - Top features
5. 🎚️ **Threshold Analysis** - Optimización
6. 🏆 **Comparación Modelos** - XGBoost vs otros

**Todas son interactivas**: Zoom, hover, exportar

---

## 🎯 ESTRUCTURA DE LA PRESENTACIÓN

### 1. Problema (30 seg)
> "Los retrasos de vuelos afectan a millones de pasajeros. Necesitamos predecirlos con 24h de anticipación."

**Mostrar**: Estadística de dataset (35.6M vuelos)

### 2. Solución (45 seg)
> "Modelo XGBoost entrenado con 15 millones de registros, 17 features predictivas."

**Mostrar**: `outputs/figures/feature_importance_xgboost_interactive.html`

### 3. Resultados (60 seg)
> "72.46% accuracy, detectamos 53% de retrasos antes de que sucedan."

**Mostrar**: 
- `outputs/figures/confusion_matrix_xgboost_interactive.html`
- `outputs/figures/roc_curve_xgboost_interactive.html`

### 4. Demo en Vivo (60 seg)
```bash
python predict.py
# Selecciona opción 3 (Interactivo)
```
**Deja que un juez ingrese datos**

### 5. Impacto (30 seg)
> "227,305 retrasos más detectados. Ahorro en costos para aerolíneas y mejor experiencia para pasajeros."

**Mostrar**: Números del dashboard

### 6. Valor Técnico (30 seg)
> "Threshold optimizado, visualizaciones interactivas, código modular, listo para producción."

**Mostrar**: Dashboard completo

**Total**: ~4 minutos + Q&A

---

## 🗂️ ARCHIVOS IMPORTANTES

### Para la Presentación
```
📁 outputs/figures/index.html          # Dashboard (abre esto)
📄 README.md                            # Documentación completa
📄 CHANGELOG.md                         # Qué es nuevo
```

### Para Demostración
```
🐍 predict.py                           # Demo en vivo
🐍 optimize_threshold.py                # Análisis técnico
```

### Para Jueces Técnicos
```
📁 src/                                 # Código fuente modular
📁 outputs/metrics/                     # Métricas detalladas
📝 THRESHOLD_DECISION.md                # Decisiones técnicas
```

---

## 💬 SCRIPT DE ELEVATOR PITCH (30 segundos)

> "FlightOnTime predice retrasos de vuelos 24 horas antes usando Machine Learning. Entrenamos XGBoost con 15 millones de registros, alcanzando 72.46% de accuracy y detectando 53% de los retrasos. El modelo está optimizado para minimizar sorpresas desagradables en el aeropuerto, beneficiando a pasajeros, aerolíneas y aeropuertos. Todo con visualizaciones interactivas profesionales y listo para producción."

---

## 🎤 PREGUNTAS FRECUENTES DE JUECES

### Q: "¿Cuántos datos usaron?"
**A**: 15 millones de registros para entrenamiento, de un dataset de 35.6M vuelos entre 2020-2024.

### Q: "¿Qué accuracy tienen?"
**A**: 72.46% de accuracy general. Más importante, detectamos 53% de los retrasos (recall), que es nuestra prioridad.

### Q: "¿Por qué XGBoost?"
**A**: Comparamos 4 algoritmos. XGBoost tuvo el mejor balance de métricas: accuracy 72.46%, ROC-AUC 0.7172.

### Q: "¿Cómo evitan data leakage?"
**A**: Solo usamos información disponible 24h antes del vuelo. Excluimos datos de demora real y relacionados.

### Q: "¿El modelo está en producción?"
**A**: Código modular Python, modelo serializado (joblib), listo para API REST. Ver `predict.py` para demo.

### Q: "¿Cómo optimizaron el threshold?"
**A**: Analizamos 85 thresholds diferentes, seleccionamos 0.52 para maximizar recall manteniendo precision aceptable. Ver `THRESHOLD_DECISION.md`.

### Q: "¿Qué features son más importantes?"
**A**: Top 3: Minuto del día, año, severidad climática. Ver visualización interactiva de feature importance.

---

## 🚨 TROUBLESHOOTING RÁPIDO

### Problema: "No puedo abrir el dashboard"
**Solución**:
```bash
cd outputs/figures
start index.html
```

### Problema: "predict.py da error"
**Solución**: Verifica que existan:
- `models/model.joblib`
- `models/metadata.json`
- `models/feature_engineer.joblib`

### Problema: "Visualizaciones no cargan"
**Solución**: Los archivos HTML son grandes (hasta 96MB). Dale unos segundos para cargar.

---

## ✅ CHECKLIST PRE-PRESENTACIÓN

- [ ] Dashboard abre correctamente (`outputs/figures/index.html`)
- [ ] `predict.py` funciona (prueba opción 1)
- [ ] Laptop conectado al proyector
- [ ] Navegador abierto con pestañas preparadas
- [ ] Script de presentación memorizado
- [ ] Números clave memorizados (72.46%, 53%, 15M)
- [ ] Demo preparada (predict.py opción 3)
- [ ] Backup de README.md impreso

---

## 🎯 PUNTOS FUERTES A DESTACAR

1. ✅ **Escala masiva**: 15M registros (no muchos equipos lograrán esto)
2. ✅ **Visualizaciones profesionales**: Plotly interactivo (destaca visualmente)
3. ✅ **Optimización técnica**: Threshold ajustado con análisis riguroso
4. ✅ **Código limpio**: Modular, documentado, production-ready
5. ✅ **Foco en negocio**: Prioriza recall (detectar retrasos) sobre precision
6. ✅ **Demo en vivo**: Funcional, no solo slides

---

## 🎁 BONUS: Cosas para Mencionar si Sobra Tiempo

- "Código en GitHub listo para compartir"
- "17 features cuidadosamente seleccionadas sin data leakage"
- "4 modelos comparados sistemáticamente"
- "Documentación completa con 5 documentos técnicos"
- "Threshold optimization con 85 valores analizados"
- "Matriz de confusión: 227K retrasos detectados correctamente"

---

## 📞 CONTACTO RÁPIDO

**Proyecto**: FlightOnTime  
**Versión**: 2.0.0  
**Team**: Data Science Team  
**Hackathon**: Aviación Civil 2026

---

**¡BUENA SUERTE EN LA PRESENTACIÓN!** 🚀✈️

---

*Este documento es tu guía de 5 minutos para dominar la presentación.*  
*Para detalles completos, ver README.md*
