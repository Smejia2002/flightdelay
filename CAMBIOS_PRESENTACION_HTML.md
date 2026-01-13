# ✅ ACTUALIZACIÓN - Visualizaciones Reales para Presentación

**Fecha**: 2026-01-13  
**Cambio**: Usar implementación real (HTML + API) en vez de dashboard Streamlit

---

## 🎯 **CAMBIOS REALIZADOS EN GUION_PRESENTACION.md**

### **ANTES** (Dashboard Streamlit):
```
- Streamlit corriendo en localhost:8501
- 4 páginas: Home, ROI, Simulator, Mapa 3D
- Navegación por sidebar
```

### **AHORA** (Implementación Real):
```
✅ Visualizaciones HTML Plotly (outputs/figures/)
✅ 6 visualizaciones interactivas
✅ API REST FastAPI (opcional para demo)
✅ Archivos locales - NO requieren internet
```

---

## 📂 **ARCHIVOS A USAR EN PRESENTACIÓN**

### **Obligatorios**:
1. `outputs/figures/index.html` - Portal principal ⭐
2. `outputs/figures/confusion_matrix_xgboost_interactive.html` - Matriz
3. `outputs/figures/feature_importance_xgboost_interactive.html` - Features

### **Opcionales** (según tiempo):
4. `outputs/figures/threshold_analysis_xgboost_interactive.html` - Threshold
5. `outputs/figures/roc_curve_xgboost_interactive.html` - ROC
6. `http://localhost:8000/docs` - API Swagger (si inician backend)

---

## 🎬 **SECUENCIA DE DEMO ACTUALIZADA**

```
[02:10-02:50] Portal index.html
   └─ Mostrar 6 visualizaciones disponibles

[02:50-03:30] Matriz de Confusión
   ├─ Click en index.html → abre nueva tab
   ├─ Hover sobre celdas
   └─ Números interactivos

[03:30-04:10] Feature Importance  
   ├─ Click desde index.html
   ├─ Hover sobre barras
   ├─ Zoom interactivo (arrastrar)
   └─ Reset axes

[04:10-04:30] Threshold / API (elegir uno)
   └─ Threshold Analysis O API demo
```

---

## ✅ **VENTAJAS DE USAR HTML REAL**

1. ✅ **Sin dependencias** - No requiere Streamlit npm corriendo
2. ✅ **Offline** - Archivos locales, sin internet
3. ✅ **Rápido** - Carga instantánea
4. ✅ **Plotly puro** - Profesional, usado por empresas Fortune 500
5. ✅ **Producción real** - Es lo que entregarían a cliente

---

## ⚙️ **PREPARACIÓN TÉCNICA**

### **10 MIN ANTES**:

```bash
# 1. Abrir navegador (Chrome/Firefox)

# 2. Abrir tabs en este orden:

# Tab 1 - Portal
file:///d:/VUELOS%20HACKATON/PRUEBA%20ESPECIAL%20FINAL%20VUELOS%202.0/outputs/figures/index.html

# Tab 2 - Confusion Matrix  
file:///d:/VUELOS%20HACKATON/PRUEBA%20ESPECIAL%20FINAL%20VUELOS%202.0/outputs/figures/confusion_matrix_xgboost_interactive.html

# Tab 3 - Feature Importance
file:///d:/VUELOS%20HACKATON/PRUEBA%20ESPECIAL%20FINAL%20VUELOS%202.0/outputs/figures/feature_importance_xgboost_interactive.html

# Tab 4 - Threshold Analysis
file:///d:/VUELOS%20HACKATON/PRUEBA%20ESPECIAL%20FINAL%20VUELOS%202.0/outputs/figures/threshold_analysis_xgboost_interactive.html

# SI VAN A DEMOS API (opcional):
# Terminal - Iniciar API
cd backend
python main.py

# Tab 5 - Swagger
http://localhost:8000/docs
```

---

## 💡 **TIPS PARA LA DEMO**

### **Interactividad Plotly**:
- ✅ **Hover** sobre gráficos → Info aparece
- ✅ **Zoom** → Click y arrastrar área
- ✅ **Pan** → Shift + Click y arrastrar
- ✅ **Reset** → Botón arriba-derecha
- ✅ **Export** → Cámara arriba-derecha (PNG/SVG)

### **Navegación**:
- Use Ctrl+Tab para cambiar entre tabs rápido
- O simplemente click en la tab
- index.html tiene links a todas las visualizaciones

---

## 🎤 **NARRACIÓN ACTUALIZADA**

### **Intro Demo** [02:10]:
> "Este es nuestro portal de **visualizaciones profesionales**, desarrollado con **Plotly** - la misma tecnología que usa Uber, Airbnb y Tesla."

### **Matriz** [02:50]:
> "Esta es nuestra **matriz de confusión** del test set con 2.25 millones de vuelos. **Completamente interactiva** - hover para ver números exactos."

### **Features** [03:30]:
> "Ahora, qué hace que un vuelo se retrase. Pueden ver que **'sched_minute_of_day'** es lo más importante. Y miren - puedo hacer **zoom interactivo**."

### **Cierre Demo** [04:30]:
> "Todo esto es **código production-ready** en HTML y JavaScript. Listo para integrar en cualquier sistema mañana."

---

## ⚠️ **IMPORTANTE**

### **NO necesitan**:
- ❌ Streamlit corriendo
- ❌ Python server
- ❌ Internet
- ❌ Instalaciones adicionales

### **SÍ necesitan**:
- ✅ Navegador moderno (Chrome/Firefox)
- ✅ Archivos HTML en outputs/figures/
- ✅ Mouse para interactividad
- ✅ (Opcional) API corriendo si la demostrarán

---

## 📋 **CHECKLIST PRE-DEMO**

- [ ] Todos los archivos HTML abren correctamente
- [ ] Interactividad funciona (hover, zoom)
- [ ] Navegador en modo fullscreen (F11)
- [ ] Zoom al 100% (Ctrl+0)  
- [ ] Proyector conectado
- [ ] Practicaron secuencia de tabs

---

**VENTAJA CLAVE**: Es la implementación REAL que entregarían. No es demo, es el producto final. ✅

---

*Documento creado: 2026-01-13*  
*Para más detalles ver: GUION_PRESENTACION.md (actualizado)*
