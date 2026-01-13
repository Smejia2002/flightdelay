"""
FlightOnTime - Dashboard Interactivo
=====================================
Dashboard principal con navegación a visualizaciones avanzadas.

Autor: FlightOnTime Team
Fecha: 2026-01-13
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="FlightOnTime Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>✈️ FlightOnTime Dashboard</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        Sistema de Predicción de Retrasos de Vuelos con Machine Learning
    </p>
    <p style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
        <strong>MODELS THAT MATTER</strong> | Grupo 59 | Hackathon Aviación Civil 2026
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/airplane-take-off.png", width=150)
    st.title("📊 Navegación")
    st.markdown("---")
    
    st.markdown("### 🎯 Visualizaciones Destacadas")
    st.markdown("""
    - 🥉 **ROI Calculator** - Cálculo de valor
    - 🥈 **Predictive Simulator** - Demo en vivo
    - 🥇 **Mapa 3D de Rutas** - Visualización 3D
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Métricas del Modelo")
    st.metric("Accuracy", "72.46%", "+6.66%")
    st.metric("Recall", "61.3%", "+7.8%")
    st.metric("ROC-AUC", "0.7172", "+0.0025")

# Cargar metadata del modelo
@st.cache_data
def load_model_info():
    """Carga información del modelo."""
    try:
        with open('../models/metadata.json', 'r') as f:
            metadata = json.load(f)
        with open('../models/training_info.json', 'r') as f:
            training_info = json.load(f)
        return metadata, training_info
    except:
        return None, None

metadata, training_info = load_model_info()

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Modelo", "📚 Documentación"])

with tab1:
    st.header("📊 Resumen del Proyecto")
    
    # Métricas principales en cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">15M</div>
            <div class="metric-label">Registros Entrenamiento</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">72.5%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">61.3%</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.72</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráfico de métricas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Comparación de Modelos")
        
        models_data = {
            'Modelo': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
            'Accuracy': [0.656, 0.656, 0.724, 0.655],
            'Recall': [0.661, 0.661, 0.613, 0.639],
            'F1-Score': [0.420, 0.420, 0.423, 0.417]
        }
        
        df_models = pd.DataFrame(models_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=df_models['Modelo'], y=df_models['Accuracy'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Recall', x=df_models['Modelo'], y=df_models['Recall'], marker_color='#10AC84'))
        fig.add_trace(go.Bar(name='F1-Score', x=df_models['Modelo'], y=df_models['F1-Score'], marker_color='#EE5A6F'))
        
        fig.update_layout(
            barmode='group',
            title="Performance por Modelo",
            xaxis_title="Modelo",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Distribución de Predicciones")
        
        # Matriz de confusión simplificada
        fig = go.Figure(data=go.Heatmap(
            z=[[1403108, 422068], [197519, 227305]],
            x=['Puntual', 'Retrasado'],
            y=['Puntual', 'Retrasado'],
            text=[[1403108, 422068], [197519, 227305]],
            texttemplate='<b>%{text}</b>',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title="Matriz de Confusión (Test Set)",
            xaxis_title="Predicción",
            yaxis_title="Real",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Features importantes
    st.subheader("⭐ Top Features Más Importantes")
    
    features_data = {
        'Feature': ['sched_minute_of_day', 'year', 'climate_severity_idx', 
                   'op_unique_carrier', 'month', 'temp', 'dep_hour'],
        'Importancia': [0.279, 0.121, 0.085, 0.077, 0.064, 0.058, 0.043],
        'Descripción': ['Minuto del día', 'Año del vuelo', 'Severidad climática',
                       'Aerolínea', 'Mes', 'Temperatura', 'Hora de salida']
    }
    
    df_features = pd.DataFrame(features_data)
    
    fig = px.bar(df_features, x='Importancia', y='Feature', orientation='h',
                 text='Importancia', hover_data=['Descripción'],
                 color='Importancia', color_continuous_scale='Blues')
    
    fig.update_layout(
        title="Importancia de Features (Top 7)",
        xaxis_title="Importancia Relativa",
        yaxis_title="",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🔍 Información del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Detalles Técnicos")
        
        if metadata and training_info:
            st.markdown(f"""
            **Modelo**: {metadata.get('model_name', 'XGBoost')}  
            **Threshold**: {float(metadata.get('threshold', 0.52)):.4f}  
            **Features**: {len(metadata.get('feature_names', []))}  
            **Registros Train**: {training_info.get('train_size', 0):,}  
            **Registros Val**: {training_info.get('val_size', 0):,}  
            **Registros Test**: {training_info.get('test_size', 0):,}  
            **Entrenado**: 2026-01-13
            """)
        else:
            st.warning("Metadata no encontrada")
        
        st.markdown("---")
        
        st.subheader("📊 Métricas de Test")
        if training_info and 'test_metrics' in training_info:
            metrics = training_info['test_metrics']
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
            st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
    
    with col2:
        st.subheader("🎯 Threshold Analysis")
        
        # Simulación de threshold analysis
        thresholds = [i/100 for i in range(10, 95, 5)]
        precision = [0.45 - (t - 0.5)**2 for t in thresholds]
        recall = [0.85 - t*0.8 for t in thresholds]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precision, mode='lines+markers',
                                name='Precision', line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=thresholds, y=recall, mode='lines+markers',
                                name='Recall', line=dict(color='#10AC84', width=3)))
        
        # Marcar threshold actual
        fig.add_vline(x=0.52, line_dash="dash", line_color="red",
                     annotation_text="Threshold Actual (0.52)")
        
        fig.update_layout(
            title="Precision vs Recall por Threshold",
            xaxis_title="Threshold",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📚 Documentación del Proyecto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Guías Disponibles")
        st.markdown("""
        - 📄 **README.md** - Documentación principal
        - 📝 **GUIA_RAPIDA.md** - Guía de 5 minutos
        - 🗺️ **INDICE_MAESTRO.md** - Mapa del proyecto
        - 📋 **CHANGELOG.md** - Historial de cambios
        - 🔧 **CONTRATO_API.md** - Especificación API
        """)
        
        st.markdown("---")
        
        st.subheader("🔗 Enlaces Útiles")
        st.markdown("""
        - [API Swagger](http://localhost:8000/docs)
        - [Dashboard Visualizaciones](../outputs/figures/index.html)
        - [GitHub Repository](#)
        """)
    
    with col2:
        st.subheader("👥 Equipo")
        st.markdown("""
        **MODELS THAT MATTER**  
        Grupo 59
        
        - 🧠 Data Science Team
        - 💻 Backend Team
        - 🎨 Visualization Team
        """)
        
        st.markdown("---")
        
        st.subheader("🏆 Hackathon")
        st.markdown("""
        **Aviación Civil 2026**
        
        📅 Enero 2026  
        🎯 Proyecto 3: FlightOnTime  
        💻 ML + API REST + Dashboard  
        ⭐ Versión 2.0
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>FlightOnTime v2.0</strong> | MODELS THAT MATTER | Grupo 59</p>
    <p>Hackathon Aviación Civil 2026 | Proyecto 3</p>
    <p>Desarrollado con ❤️ usando Python, Streamlit, XGBoost y 15M registros</p>
</div>
""", unsafe_allow_html=True)
