"""
3D Routes Map - Mapa 3D de Rutas Aéreas
========================================
Visualización espectacular de rutas con globo 3D interactivo.

Autor: FlightOnTime Team
Fecha: 2026-01-13
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Mapa 3D de Rutas", page_icon="🥇", layout="wide")

# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #F79F1F 0%, #EE5A6F 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>🥇 Mapa 3D de Rutas Aéreas</h1>
    <p style="font-size: 1.2rem;">Visualización interactiva de probabilidades de retraso por ruta</p>
</div>
""", unsafe_allow_html=True)

# Datos de aeropuertos principales (coordenadas)
aeropuertos = {
    'JFK': {'lat': 40.6413, 'lon': -73.7781, 'nombre': 'New York JFK'},
    'LAX': {'lat': 33.9416, 'lon': -118.4085, 'nombre': 'Los Angeles'},
    'ORD': {'lat': 41.9742, 'lon': -87.9073, 'nombre': 'Chicago O´ Hare'},
    'ATL': {'lat': 33.6407, 'lon': -84.4277, 'nombre': 'Atlanta'},
    'DFW': {'lat': 32.8998, 'lon': -97.0403, 'nombre': 'Dallas Fort Worth'},
    'SFO': {'lat': 37.6213, 'lon': -122.3790, 'nombre': 'San Francisco'},
    'MIA': {'lat': 25.7959, 'lon': -80.2870, 'nombre': 'Miami'},
    'SEA': {'lat': 47.4502, 'lon': -122.3088, 'nombre': 'Seattle'},
    'LAS': {'lat': 36.0840, 'lon': -115.1537, 'nombre': 'Las Vegas'},
    'BOS': {'lat': 42.3656, 'lon': -71.0096, 'nombre': 'Boston'}
}

# Rutas principales con probabilidades simuladas
rutas = [
    ('JFK', 'LAX', 0.68, 25000),
    ('JFK', 'SFO', 0.72, 18000),
    ('ATL', 'LAX', 0.45, 22000),
    ('ATL', 'ORD', 0.38, 15000),
    ('ORD', 'LAX', 0.52, 20000),
    ('DFW', 'JFK', 0.61, 17000),
    ('SFO', 'JFK', 0.74, 19000),
    ('LAX', 'MIA', 0.55, 12000),
    ('SEA', 'JFK', 0.68, 14000),
    ('BOS', 'SFO', 0.64, 13000),
    ('ATL', 'SFO', 0.49, 16000),
    ('ORD', 'SFO', 0.58, 18000),
]

# Sidebar con filtros
with st.sidebar:
    st.header("🎛️ Controles")
    
    mostrar_aeropuertos = st.checkbox("Mostrar aeropuertos", value=True)
    mostrar_rutas = st.checkbox("Mostrar rutas", value=True)
    
    st.markdown("### 🎨 Filtros")
    prob_min = st.slider(
        "Probabilidad mínima de retraso",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filtrar rutas por probabilidad"
    )
    
    volumen_min = st.slider(
        "Volumen mínimo de vuelos/mes",
        min_value=0,
        max_value=30000,
        value=0,
        step=1000
    )
    
    st.markdown("### 🌍 Vista")
    proyeccion = st.selectbox(
        "Proyección",
        options=["orthographic", "natural earth"],
        index=0
    )

# Filtrar rutas
rutas_filtradas = [(o, d, p, v) for o, d, p, v in rutas 
                   if p >= prob_min and v >= volumen_min]

# Crear figura 3D
fig = go.Figure()

# Añadir rutas
if mostrar_rutas and rutas_filtradas:
    for origen, destino, prob, volumen in rutas_filtradas:
        lat_o, lon_o = aeropuertos[origen]['lat'], aeropuertos[origen]['lon']
        lat_d, lon_d = aeropuertos[destino]['lat'], aeropuertos[destino]['lon']
        
        # Color según probabilidad
        if prob >= 0.65:
            color = '#EE5A6F'  # Rojo (alto riesgo)
        elif prob >= 0.50:
            color = '#F79F1F'  # Naranja (medio)
        else:
            color = '#10AC84'  # Verde (bajo)
        
        # Ancho según volumen
        width = 1 + (volumen / 30000) * 4
        
        fig.add_trace(go.Scattergeo(
            lon=[lon_o, lon_d],
            lat=[lat_o, lat_d],
            mode='lines',
            line=dict(width=width, color=color),
            opacity=0.7,
            name=f'{origen}→{destino}',
            hovertemplate=(
                f'<b>{origen} → {destino}</b><br>'
                f'Probabilidad retraso: {prob:.0%}<br>'
                f'Vuelos/mes: {volumen:,}<br>'
                '<extra></extra>'
            )
        ))

# Añadir aeropuertos
if mostrar_aeropuertos:
    lats = [info['lat'] for info in aeropuertos.values()]
    lons = [info['lon'] for info in aeropuertos.values()]
    nombres = [f"{code} - {info['nombre']}" for code, info in aeropuertos.items()]
    
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        text=nombres,
        mode='markers+text',
        marker=dict(
            size=12,
            color='#667eea',
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        textposition="top center",
        textfont=dict(size=10, color='#2C3E50'),
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Aeropuertos'
    ))

# Layout del mapa
fig.update_layout(
    title=dict(
        text='<b>Red de Rutas Aéreas - Probabilidad de Retrasos</b>',
        x=0.5,
        xanchor='center',
        font=dict(size=20)
    ),
    showlegend=False,
    geo=dict(
        projection_type=proyeccion,
        showland=True,
        landcolor='rgb(243, 243, 243)',
        coastlinecolor='rgb(204, 204, 204)',
        countrycolor='rgb(204, 204, 204)',
        showcountries=True,
        showocean=True,
        oceancolor='rgb(230, 245, 255)',
        showlakes=True,
        lakecolor='rgb(230, 245, 255)',
        resolution=50,
        center=dict(lat=39, lon=-98),  # Centro en USA
        scope='north america'
    ),
    height=700,
    margin=dict(l=0, r=0, t=60, b=0)
)

# Tabs
tab1, tab2, tab3 = st.tabs(["🌍 Mapa Interactivo", "📊 Estadísticas", "💡 Insights"])

with tab1:
    st.plotly_chart(fig, use_container_width=True)
    
    # Leyenda
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #10AC84; color: white; padding: 1rem; border-radius: 5px; text-align: center;">
            <strong>🟢 Bajo Riesgo</strong><br>
            < 50% probabilidad
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #F79F1F; color: white; padding: 1rem; border-radius: 5px; text-align: center;">
            <strong>🟠 Riesgo Medio</strong><br>
            50-65% probabilidad
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #EE5A6F; color: white; padding: 1rem; border-radius: 5px; text-align: center;">
            <strong>🔴 Alto Riesgo</strong><br>
            > 65% probabilidad
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("📊 Top Rutas por Probabilidad de Retraso")
    
    # Crear DataFrame
    df_rutas = pd.DataFrame(rutas_filtradas, columns=['Origen', 'Destino', 'Probabilidad', 'Vuelos/Mes'])
    df_rutas['Ruta'] = df_rutas['Origen'] + ' → ' + df_rutas['Destino']
    df_rutas['Probabilidad %'] = (df_rutas['Probabilidad'] * 100).round(1)
    df_rutas = df_rutas.sort_values('Probabilidad', ascending=False)
    
    #Top 10
    st.dataframe(
        df_rutas[['Ruta', 'Probabilidad %', 'Vuelos/Mes']].head(10),
        use_container_width=True,
        hide_index=True
    )
    
    # Gráfico de barras
    fig_bar = go.Figure(go.Bar(
        x=df_rutas['Ruta'].head(10),
        y=df_rutas['Probabilidad'].head(10),
        marker_color=df_rutas['Probabilidad'].head(10),
        marker_colorscale='RdYlGn_r',
        text=[f"{p:.0%}" for p in df_rutas['Probabilidad'].head(10)],
        textposition='outside'
    ))
    
    fig_bar.update_layout(
        title="Top 10 Rutas con Mayor Probabilidad de Retraso",
        xaxis_title="Ruta",
        yaxis_title="Probabilidad de Retraso",
        height=400,
        yaxis_tickformat='.0%'
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.subheader("💡 Insights Clave")
    
    # Calcular estadísticas
    prob_promedio = np.mean([p for _, _, p, _ in rutas])
    ruta_max = max(rutas, key=lambda x: x[2])
    ruta_min = min(rutas, key=lambda x: x[2])
    volumen_total = sum([v for _, _, _, v in rutas])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        ### 📊 Estadísticas Generales
        
        - **Probabilidad promedio**: {prob_promedio:.0%}
        - **Volumen total mensual**: {volumen_total:,} vuelos
        - **Rutas analizadas**: {len(rutas)}
        - **Rutas de alto riesgo**: {len([r for r in rutas if r[2] >= 0.65])}
        """)
    
    with col2:
        st.warning(f"""
        ### ⚠️ Rutas Críticas
        
        **Mayor riesgo:**
        - {ruta_max[0]} → {ruta_max[1]}: {ruta_max[2]:.0%}
        
        **Menor riesgo:**
        - {ruta_min[0]} → {ruta_min[1]}: {ruta_min[2]:.0%}
        """)
    
    st.success("""
    ### ✅ Recomendaciones
    
    1. **Priorizar rutas rojas** para acciones preventivas
    2. **Monitorear rutas naranjas** en horarios pico
    3. **Optimizar recursos** en rutas de alto volumen
    4. **Comunicación proactiva** en rutas de alto riesgo
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>💡 <strong>Tip</strong>: Rota el globo arrastrando, usa scroll para zoom</p>
    <p>🎨 Ajusta los filtros en la barra lateral para explorar diferentes escenarios</p>
</div>
""", unsafe_allow_html=True)
