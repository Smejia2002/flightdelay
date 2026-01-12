import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

def calculate_sched_minute_of_day(df, time_col='CRS_DEP_TIME'):
    """
    Convierte la hora programada (HHMM) a minutos del día (0-1439).
    Ejemplo: 1330 -> 13*60 + 30 = 810 minutos.
    """
    # Aseguramos que sea entero
    times = df[time_col].fillna(0).astype(int)
    
    # División entera para obtener horas y módulo para minutos
    hours = times // 100
    minutes = times % 100
    
    return (hours * 60) + minutes

def clean_climate_data(df):
    """
    Aplica reglas de negocio a los datos climáticos.
    Regla 1: Precipitación negativa (-1) significa trazas -> convertir a 0.
    """
    if 'PRECIP_1H' in df.columns:
        # Reemplazar valores negativos con 0
        df['PRECIP_1H'] = df['PRECIP_1H'].apply(lambda x: 0 if x < 0 else x)
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia del gran círculo entre dos puntos en la tierra (especificado en grados decimales).
    """
    # Convertir grados decimales a radianes
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Fórmula de Haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radio de la Tierra en km
    return c * r

def process_features(df):
    """
    PIPELINE MAESTRO: Ejecuta todas las transformaciones necesarias.
    Este es la función que llamará el modelo antes de predecir.
    """
    df = df.copy()
    
    # 1. Transformación Temporal
    if 'CRS_DEP_TIME' in df.columns:
        df['sched_minute_of_day'] = calculate_sched_minute_of_day(df)
    
    # 2. Limpieza Climática
    df = clean_climate_data(df)
    
    # 3. Casteo de Categorías (Optimización de Memoria)
    cat_cols = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df
