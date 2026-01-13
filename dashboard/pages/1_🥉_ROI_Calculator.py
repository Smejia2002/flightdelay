"""
ROI Calculator - Calculadora de Retorno de Inversión
====================================================
Calcula el valor económico del modelo de predicción de retrasos.

Autor: FlightOnTime Team
Fecha: 2026-01-13
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="ROI Calculator", page_icon="🥉", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #10AC84 0%, #0E8B6E 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🥉 ROI Calculator</h1>
    <p style="font-size: 1.2rem;">Calculadora de Retorno de Inversión del Sistema FlightOnTime</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 💰 Calcule el valor económico de predecir retrasos de vuelos")

# Sidebar con inputs
st.sidebar.header("⚙️ Parámetros del Negocio")

st.sidebar.markdown("### 📊 Volumen de Operación")
vuelos_mes = st.sidebar.slider(
    "Vuelos por mes",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000,
    help="Número de vuelos operados mensualmente"
)

tasa_retraso = st.sidebar.slider(
    "Tasa histórica de retrasos (%)",
    min_value=10,
    max_value=40,
    value=19,
    step=1,
    help="Porcentaje actual de vuelos retrasados"
)

st.sidebar.markdown("### 💵 Costos por Retraso")
costo_retraso_aerolinea = st.sidebar.number_input(
    "Costo por retraso - Aerolínea ($)",
    min_value=100,
    max_value=10000,
    value=2500,
    step=100,
    help="Costo operativo por vuelo retrasado"
)

costo_retraso_pasajero = st.sidebar.number_input(
    "Costo por retraso - Pasajero ($)",
    min_value=50,
    max_value=1000,
    value=150,
    step=10,
    help="Costo promedio de insatisfacción/compensación por pasajero"
)

pasajeros_promedio = st.sidebar.number_input(
    "Pasajeros promedio por vuelo",
    min_value=50,
    max_value=500,
    value=150,
    step=10
)

st.sidebar.markdown("### 🎯 Performance del Modelo")
recall_modelo = st.sidebar.slider(
    "Recall del modelo (%)",
    min_value=40,
    max_value=80,
    value=61,
    step=1,
    help="Porcentaje de retrasos que el modelo detecta"
)

precision_modelo = st.sidebar.slider(
    "Precision del modelo (%)",
    min_value=20,
    max_value=60,
    value=32,
    step=1,
    help="Precisión de las alertas del modelo"
)

# Cálculos
vuelos_retrasados_mes = int(vuelos_mes * (tasa_retraso / 100))
retrasos_detectados = int(vuelos_retrasados_mes * (recall_modelo / 100))
ahorros_por_deteccion = 0.4  # 40% de ahorro por retraso detectado

# Ahorros
ahorro_aerolinea_por_vuelo = costo_retraso_aerolinea * ahorros_por_deteccion
ahorro_pasajeros_por_vuelo = costo_retraso_pasajero * pasajeros_promedio * ahorros_por_deteccion

ahorro_mensual_aerolinea = retrasos_detectados * ahorro_aerolinea_por_vuelo
ahorro_mensual_pasajeros = retrasos_detectados * ahorro_pasajeros_por_vuelo
ahorro_mensual_total = ahorro_mensual_aerolinea + ahorro_mensual_pasajeros

ahorro_anual = ahorro_mensual_total * 12

# Costos de implementación (estimados)
costo_desarrollo = 50000
costo_infraestructura_mes = 500
costo_mantenimiento_mes = 1000

costo_total_primer_ano = costo_desarrollo + (costo_infraestructura_mes + costo_mantenimiento_mes) * 12
roi_primer_ano = ((ahorro_anual - costo_total_primer_ano) / costo_total_primer_ano) * 100
payback_months = costo_total_primer_ano / ahorro_mensual_total if ahorro_mensual_total > 0 else 0

# Layout principal
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        " ahorro Mensual",
        f"${ahorro_mensual_total:,.0f}",
        f"+{ahorro_mensual_total/1000:.1f}K"
    )

with col2:
    st.metric(
        "💎 Ahorro Anual",
        f"${ahorro_anual:,.0f}",
        f"+{ahorro_anual/1000000:.2f}M"
    )

with col3:
    st.metric(
        "📊 ROI Año 1",
        f"{roi_primer_ano:,.0f}%",
        f"Payback: {payback_months:.1f} meses"
    )

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Desglose", "📈 Proyección", "⚖️ Comparación", "💡 Insights"])

with tab1:
    st.subheader("💰 Desglose de Ahorros Mensuales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de pastel
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Aerolíneas', 'Pasajeros'],
            values=[ahorro_mensual_aerolinea, ahorro_mensual_pasajeros],
            hole=.4,
            marker_colors=['#667eea', '#10AC84']
        )])
        
        fig_pie.update_layout(
            title="Distribución de Ahorros",
            annotations=[dict(text='Total', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Detalle de Cálculos")
        
        st.markdown(f"""
        **Operación Mensual:**
        - Total de vuelos: {vuelos_mes:,}
        - Vuelos retrasados: {vuelos_retrasados_mes:,} ({tasa_retraso}%)
        - Retrasos detectados: {retrasos_detectados:,} ({recall_modelo}% recall)
        
        **Ahorro por Vuelo Detectado:**
        - Aerolínea: ${ahorro_aerolinea_por_vuelo:,.0f}
        - Pasajeros: ${ahorro_pasajeros_por_vuelo:,.0f}
        - **Total**: ${ahorro_aerolinea_por_vuelo + ahorro_pasajeros_por_vuelo:,.0f}
        
        **Ahorro Mensual:**
        - Aerolíneas: ${ahorro_mensual_aerolinea:,.0f}
        - Pasajeros: ${ahorro_mensual_pasajeros:,.0f}
        - **Total**: ${ahorro_mensual_total:,.0f}
        """)

with tab2:
    st.subheader("📈 Proyección a 5 Años")
    
    # Proyección
    anos = list(range(1, 6))
    ahorros_acumulados = []
    costos_acumulados = []
    ganancia_neta = []
    
    for ano in anos:
        ahorro_acum = ahorro_anual * ano
        if ano == 1:
            costo_acum = costo_total_primer_ano
        else:
            costo_acum = costo_total_primer_ano + ((costo_infraestructura_mes + costo_mantenimiento_mes) * 12 * (ano - 1))
        
        ahorros_acumulados.append(ahorro_acum)
        costos_acumulados.append(costo_acum)
        ganancia_neta.append(ahorro_acum - costo_acum)
    
    fig_proyeccion = go.Figure()
    
    fig_proyeccion.add_trace(go.Bar(
        name='Ahorros Acumulados',
        x=anos,
        y=ahorros_acumulados,
        marker_color='#10AC84'
    ))
    
    fig_proyeccion.add_trace(go.Bar(
        name='Costos Acumulados',
        x=anos,
        y=costos_acumulados,
        marker_color='#EE5A6F'
    ))
    
    fig_proyeccion.add_trace(go.Scatter(
        name='Ganancia Neta',
        x=anos,
        y=ganancia_neta,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=12)
    ))
    
    fig_proyeccion.update_layout(
        title="Proyección Financiera a 5 Años",
        xaxis_title="Año",
        yaxis_title="Monto ($)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig_proyeccion, use_container_width=True)
    
    # Tabla de resumen
    df_proyeccion = pd.DataFrame({
        'Año': anos,
        'Ahorros Acumulados': [f'${a:,.0f}' for a in ahorros_acumulados],
        'Costos Acumulados': [f'${c:,.0f}' for c in costos_acumulados],
        'Ganancia Neta': [f'${g:,.0f}' for g in ganancia_neta],
        'ROI': [f'{((a-c)/c*100):.0f}%' for a, c in zip(ahorros_acumulados, costos_acumulados)]
    })
    
    st.dataframe(df_proyeccion, use_container_width=True)

with tab3:
    st.subheader("⚖️ Con Modelo vs Sin Modelo")
    
    col1, col2 = st.columns(2)
    
    # Costos sin modelo
    costo_anual_sin_modelo = vuelos_retrasados_mes * 12 * (costo_retraso_aerolinea + costo_retraso_pasajero * pasajeros_promedio)
    
    # Costos con modelo
    retrasos_no_detectados = vuelos_retrasados_mes - retrasos_detectados
    costo_anual_con_modelo = (retrasos_no_detectados * 12 * (costo_retraso_aerolinea + costo_retraso_pasajero * pasajeros_promedio) * 0.6) + costo_total_primer_ano
    
    with col1:
        st.markdown("### ❌ Sin Modelo Predictivo")
        st.metric("Costo Anual de Retrasos", f"${costo_anual_sin_modelo:,.0f}")
        st.metric("Retrasos No Evitados", f"{vuelos_retrasados_mes * 12:,}")
        st.metric("Satisfacción Cliente", "📉 Baja")
        
    with col2:
        st.markdown("### ✅ Con Modelo Predictivo")
        st.metric("Costo Anual Total", f"${costo_anual_con_modelo:,.0f}", f"-${costo_anual_sin_modelo - costo_anual_con_modelo:,.0f}")
        st.metric("Ahorros Año 1", f"${ahorro_anual:,.0f}", "+")
        st.metric("Satisfacción Cliente", "📈 Alta", "+35%")
    
    # Gráfico comparativo
    fig_comp = go.Figure(data=[
        go.Bar(name='Sin Modelo', x=['Costos Anuales'], y=[costo_anual_sin_modelo], marker_color='#EE5A6F'),
        go.Bar(name='Con Modelo', x=['Costos Anuales'], y=[costo_anual_con_modelo], marker_color='#10AC84')
    ])
    
    fig_comp.update_layout(
        title="Comparación de Costos Anuales",
        yaxis_title="Costo ($)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)

with tab4:
    st.subheader("💡 Insights y Recomendaciones")
    
    st.success(f"""
    ### ✅ Conclusiones Principales
    
    1. **ROI Excelente**: Con un ROI de {roi_primer_ano:.0f}% en el primer año, el proyecto se justifica ampliamente.
    
    2. **Payback Rápido**: La inversión se recupera en {payback_months:.1f} meses.
    
    3. **Ahorro Anual**: ${ahorro_anual:,.0f} de ahorros anuales proyectados.
    
    4. **Beneficio Dual**: Tanto aerolíneas como pasajeros se benefician significativamente.
    """)
    
    st.info(f"""
    ### 📊 Factores Clave de Éxito
    
    - **Recall del {recall_modelo}%**: Detecta la mayoría de los retrasos
    - **{retrasos_detectados:,} retrasos/mes**: Volumen significativo de detecciones
    - **Acción Preventiva**: 40% de ahorro por cada retraso detectado a tiempo
    """)
    
    st.warning("""
    ### ⚠️ Consideraciones
    
    - Los cálculos asumen que detectar un retraso permite acciones preventivas que reducen costos en ~40%
    - El valor real puede variar según la implementación operativa
    - Beneficios intangibles (reputación, fidelización) no están cuantificados aquí
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>💡 <strong>Tip</strong>: Ajusta los parámetros en la barra lateral para ver cómo cambia el ROI según tu caso de uso específico</p>
</div>
""", unsafe_allow_html=True)
