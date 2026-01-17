"""
FlightOnTime - Feature Engineering (Versión Fusionada)
======================================================
Combina la optimización matemática (NumPy/Vectorización) con la
estructura orientada a objetos para producción (MLOps).

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. FUNCIONES AUXILIARES OPTIMIZADAS ( Lógica Matemática) 🧠
# =============================================================================

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """
    Calcula distancia en Kilómetros usando NumPy (Vectorizado).
    Miles de veces más rápido que iterar filas.
    """
    R = 6371  # Radio de la Tierra en km
    
    # Asegurar que sean arrays de numpy y convertir a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def calculate_sched_minute_of_day(df, time_col='CRS_DEP_TIME'):
    """Convierte HHMM a minutos del día (0-1439) de forma vectorizada."""
    # Rellenar nulos y asegurar enteros
    times = df[time_col].fillna(0).astype(int)
    # Cálculo directo
    return (times // 100) * 60 + (times % 100)

# =============================================================================
# 2. CLASE PRINCIPAL (La Estructura) 🏗️
# =============================================================================

class FlightFeatureEngineer:
    """
    Pipeline de Ingeniería de Features compatible con Scikit-Learn.
    Maneja el entrenamiento (fit) y la transformación (transform) para evitar Data Leakage.
    """
    
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False
        
        # Definición de columnas esperadas
        self.cat_cols = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
        self.num_cols = ['MONTH', 'DAY_OF_WEEK', 'sched_minute_of_day', 
                         'DISTANCE', 'TEMP', 'WIND_SPD', 'PRECIP_1H', 
                         'CLIMATE_SEVERITY_IDX']

    def fit(self, df: pd.DataFrame):
        """
        Aprende los códigos de las categorías (Aerolíneas, Origen, Destino).
        Se ejecuta SOLO con el set de entrenamiento.
        """
        print("⚙️ Entrenando encoders (LabelEncoding)...")
        for col in self.cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Convertimos a string para evitar errores de tipos mixtos
                le.fit(df[col].astype(str))
                self.encoders[col] = le
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training=False) -> pd.DataFrame:
        """
        Aplica TODAS las transformaciones matemáticas y de limpieza.
        """
        df = df.copy()
        
        # --- A. LÓGICA DE NEGOCIO (Tu código) ---
        
        # 1. Tiempo: Convertir HHMM a minutos
        if 'CRS_DEP_TIME' in df.columns:
            df['sched_minute_of_day'] = calculate_sched_minute_of_day(df)
            
        # 2. Clima: Limpieza y Severidad
        if 'PRECIP_1H' in df.columns:
            # Regla: Lluvia negativa es 0
            df['PRECIP_1H'] = df['PRECIP_1H'].mask(df['PRECIP_1H'] < 0, 0)
            
        if 'WIND_SPD' in df.columns and 'PRECIP_1H' in df.columns:
            # Feature Sintético: Severidad = Viento + (Lluvia * Factor)
            df['CLIMATE_SEVERITY_IDX'] = df['WIND_SPD'] + (df['PRECIP_1H'] * 5)
            
        # 3. Distancia: Calcular si falta (Vectorizado)
        required_coords = ['LATITUDE_ORIGIN', 'LONGITUDE_ORIGIN', 'LATITUDE_DEST', 'LONGITUDE_DEST']
        if 'DISTANCE' not in df.columns and all(c in df.columns for c in required_coords):
            # Calculamos km
            dist_km = haversine_distance_vectorized(
                df['LATITUDE_ORIGIN'], df['LONGITUDE_ORIGIN'],
                df['LATITUDE_DEST'], df['LONGITUDE_DEST']
            )
            # Convertimos a Millas (Estándar en aviación US)
            df['DISTANCE'] = dist_km * 0.621371

        # --- B. CODIFICACIÓN (Código del compañero) ---
        
        for col in self.cat_cols:
            if col in df.columns and col in self.encoders:
                le = self.encoders[col]
                # Manejo de categorías nuevas no vistas (para que no rompa en producción)
                # Si es training, transformamos directo. Si es test/prod, usamos map seguro.
                df[col] = df[col].astype(str)
                
                if is_training:
                    df[col] = le.transform(df[col])
                else:
                    # Truco Pro: Las categorías desconocidas se van a -1 o a la moda
                    # Aquí usamos un método seguro con map
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    df[col] = df[col].map(mapping).fillna(-1).astype(int)

        # 4. Selección Final
        # Devolvemos solo las columnas que el modelo necesita, si existen
        cols_to_return = [c for c in (self.cat_cols + self.num_cols) if c in df.columns]
        
        # Si tenemos el target, lo dejamos pasar
        if 'DEP_DEL15' in df.columns:
            cols_to_return.append('DEP_DEL15')
            
        return df[cols_to_return]

    def save_encoders(self, path='models/encoders.pkl'):
        """Guarda el cerebro de los encoders para usarlo en la API."""
        joblib.dump(self.encoders, path)
        print(f"💾 Encoders guardados en {path}")

# =============================================================================
# 3. FUNCIÓN WRAPPER (Para compatibilidad con notebooks) 🔄
# =============================================================================

def process_features(df):
    """
    Función simple para llamar desde tus notebooks antiguos.
    Crea una instancia, fitea y transforma en un solo paso.
    """
    engineer = FlightFeatureEngineer()
    # Asumimos que si llamas a esto es entrenamiento o EDA, así que fit_transform
    return engineer.fit(df).transform(df, is_training=True)astype('category')
            
    return df

