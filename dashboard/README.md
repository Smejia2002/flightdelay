# 🎨 Dashboard Interactivo - FlightOnTime

**Versión**: 2.0  
**Framework**: Streamlit  
**Puerto**: 8501

---

## 🚀 **Inicio Rápido**

### 1. Instalar dependencias
```bash
cd dashboard
pip install -r requirements.txt
```

### 2. Iniciar dashboard
```bash
# Opción 1: Script automático (Windows)
start_dashboard.bat

# Opción 2: Manual
streamlit run app.py
```

### 3. Abrir en navegador
```
http://localhost:8501
```

---

## 📊 **Páginas Disponibles**

### **🏠 Dashboard Principal** (`app.py`)
- Overview del proyecto
- Métricas clave (Accuracy, Recall, ROC-AUC)
- Comparación de 4 modelos
- Matriz de confusión
- Top features importantes
- Información técnica del modelo

### **🥉 ROI Calculator** (Página 1)
- Calculadora interactiva de retorno de inversión
- Sliders para ajustar parámetros
- Proyección a 5 años
- Comparación con/sin modelo
- Desglose de ahorros por aerolíneas y pasajeros
- Gráficos de payback y ROI

### **🥈 Predictive Simulator** (Página 2)
- Simulador de predicciones en tiempo real
- Form para ingresar datos de vuelo
- Predicción instantánea
- Gauge chart de probabilidad
- Explicabilidad (factores influyentes)
- Comparación con datos históricos
- Integración con modelo real (si está disponible)

### **🥇 Mapa 3D de Rutas** (Página 3)
- Visualización de red de rutas aéreas
- Globo 3D interactivo y rotable
- Colores por probabilidad de retraso:
  - 🟢 Verde: < 50% (bajo riesgo)
  - 🟠 Naranja: 50-65% (medio riesgo)
  - 🔴 Rojo: > 65% (alto riesgo)
- Filtros por probabilidad y volumen
- Estadísticas de rutas
- Top 10 rutas críticas

---

## 🎨 **Características**

### Interactividad
- ✅ Sliders y controles en tiempo real
- ✅ Formularios dinámicos
- ✅ Gráficos Plotly interactivos
- ✅ Filtros y búsqueda
- ✅ Navegación por pestañas

### Visualización
- ✅ Gráficos de barras, líneas, pastel
- ✅ Heatmaps y matrices
- ✅ Gauge charts
- ✅ Globo 3D con proyección geográfica
- ✅ Dashboards multi-tab

### Responsive
- ✅ Se adapta a cualquier pantalla
- ✅ Layout optimizado
- ✅ Sidebar colapsable

---

## 💡 **Cómo Usar**

### **Para Presentaciones**

1. **Iniciar dashboard**: `start_dashboard.bat`
2. **Navegar**: Usar sidebar para cambiar de página
3. **Interactuar**: Ajustar sliders y ver resultados en tiempo real
4. **Demostrar**: Usar Predictive Simulator para demo en vivo

### **Para Jueces**

- **ROI Calculator**: Justifica el valor del proyecto
- **Predictive Simulator**: Demo interactiva del modelo
- **Mapa 3D**: Impacto visual inmediato

---

## 🎯 **Casos de Uso**

### 1. **Calcular ROI** (5 min)
```
1. Abrir ROI Calculator
2. Ajustar parámetros (vuelos/mes, costos)
3. Ver proyección a  años
4. Mostrar payback period
```

### 2. **Demo en Vivo** (5 min)
```
1. Abrir Predictive Simulator
2. Ingresar datos de vuelo real
3. Obtener predicción instantánea
4. Explicar factores influyentes
```

### 3. **Mostrar Alcance** (3 min)
```
1. Abrir Mapa 3D
2. Rotar globo
3. Filtrar por probabilidad
4. Mostrar estadísticas
```

---

## 📋 **Estructura**

```
dashboard/
├── app.py                          # Dashboard principal
├── pages/
│   ├── 1_🥉_ROI_Calculator.py     # Calculadora ROI
│   ├── 2_🥈_Predictive_Simulator.py # Simulador
│   └── 3_🥇_3D_Routes_Map.py      # Mapa 3D
├── requirements.txt                # Dependencias
├── start_dashboard.bat             # Script inicio
└── README.md                       # Este archivo
```

---

## 🐛 **Troubleshooting**

### Puerto 8501 en uso
```bash
streamlit run app.py --server.port 8502
```

### Modelo no carga
- El Predictive Simulator funciona en modo simulación si el modelo no está disponible
- Para predicciones reales, asegúrate de tener `../models/model.joblib`

### Dependencias faltantes
```bash
pip install -r requirements.txt --upgrade
```

---

## 🎨 **Personalización**

### Cambiar colores
Edita el CSS en cada archivo `.py`:
```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(...);
    }
</style>
""", unsafe_allow_html=True)
```

### Agregar nuevas páginas
1. Crear archivo en `pages/` con formato `N_😀_Nombre.py`
2. Streamlit lo detectará automáticamente

---

## 📊 **Datos del Dashboard**

- **Métricas**: Basadas en entrenamiento real con 15M registros
- **Rutas**: Simulación basada en datos históricos
- **ROI**: Cálculos con parámetros realistas del sector

---

## ✅ **Checklist**

- [x] Dashboard principal funcional
- [x] ROI Calculator con cálculos dinámicos
- [x] Predictive Simulator con form interactivo
- [x] Mapa 3D con rutas y filtros
- [x] Gráficos interactivos Plotly
- [x] Navegación por sidebar
- [x] Responsive design
- [x] Script de inicio
- [x] Documentación completa

---

**¡Dashboard listo para impresionar en el hackathon!** 🚀

---

*Última actualización: 2026-01-13*  
*FlightOnTime v2.0 - Dashboard Edition*
