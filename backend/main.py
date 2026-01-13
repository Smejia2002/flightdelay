"""
FlightOnTime - API REST para Predicción de Retrasos
====================================================
API desarrollada con FastAPI para el Hackathon de Aviación Civil 2026.

Equipo: MODELS THAT MATTER - Grupo 59
Proyecto 3: FlightOnTime ✈️ — Predicción de Retrasos de Vuelos

Endpoints:
    POST /predict - Predice si un vuelo será puntual o retrasado
    GET /health - Verifica estado de la API
    GET /model-info - Información del modelo

Autor: MODELS THAT MATTER
Fecha: 2026-01-13
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Inicializar FastAPI
app = FastAPI(
    title="FlightOnTime API",
    description="API de predicción de retrasos de vuelos usando Machine Learning - MODELS THAT MATTER (Grupo 59)",
    version="2.0.0",
    contact={
        "name": "MODELS THAT MATTER - Grupo 59",
        "email": "team@flightontime.com"
    }
)

# CORS - Permitir acceso desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
FEATURE_ENGINEER_PATH = BASE_DIR / "models" / "feature_engineer.joblib"

# Variables globales para modelo
model = None
metadata = None
feature_engineer = None


# ============================================================================
# MODELOS PYDANTIC (VALIDACIÓN DE DATOS)
# ============================================================================

class FlightRequest(BaseModel):
    """Modelo de entrada para predicción de vuelo."""
    
    aerolinea: str = Field(..., description="Código de aerolínea (ej: AA, DL, UA)", min_length=2, max_length=3)
    origen: str = Field(..., description="Código IATA aeropuerto origen (ej: JFK, GRU)", min_length=3, max_length=3)
    destino: str = Field(..., description="Código IATA aeropuerto destino (ej: LAX, GIG)", min_length=3, max_length=3)
    fecha_partida: str = Field(..., description="Fecha/hora de partida ISO 8601 (ej: 2025-11-10T14:30:00)")
    distancia_km: float = Field(..., description="Distancia del vuelo en kilómetros", gt=0)
    
    # Campos opcionales (si están disponibles mejoran la predicción)
    temperatura: Optional[float] = Field(None, description="Temperatura en °C", ge=-50, le=60)
    velocidad_viento: Optional[float] = Field(None, description="Velocidad del viento en km/h", ge=0)
    precipitacion: Optional[float] = Field(None, description="Precipitación en mm", ge=0)
    
    @validator('fecha_partida')
    def validar_fecha(cls, v):
        """Valida que la fecha esté en formato ISO 8601."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Fecha debe estar en formato ISO 8601 (YYYY-MM-DDTHH:MM:SS)')
    
    @validator('aerolinea', 'origen', 'destino')
    def validar_codigos(cls, v):
        """Valida que los códigos estén en mayúsculas."""
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
                "aerolinea": "AA",
                "origen": "JFK",
                "destino": "LAX",
                "fecha_partida": "2025-11-10T14:30:00",
                "distancia_km": 3983,
                "temperatura": 25.5,
                "velocidad_viento": 15.3,
                "precipitacion": 0.0
            }
        }


class FlightResponse(BaseModel):
    """Modelo de salida para predicción de vuelo."""
    
    prevision: str = Field(..., description="Previsión: 'Puntual' o 'Retrasado'")
    probabilidad: float = Field(..., description="Probabilidad de la previsión (0.0 a 1.0)", ge=0.0, le=1.0)
    confianza: Optional[str] = Field(None, description="Nivel de confianza: Alta, Media, Baja")
    detalles: Optional[dict] = Field(None, description="Información adicional sobre la predicción")
    
    class Config:
        schema_extra = {
            "example": {
                "prevision": "Retrasado",
                "probabilidad": 0.78,
                "confianza": "Alta",
                "detalles": {
                    "umbral_usado": 0.52,
                    "probabilidad_puntual": 0.22,
                    "probabilidad_retrasado": 0.78
                }
            }
        }


class ModelInfo(BaseModel):
    """Información del modelo."""
    
    nombre: str
    version: str
    accuracy: float
    recall: float
    roc_auc: float
    threshold: float
    features: int
    registros_entrenamiento: int


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    
    status: str
    modelo_cargado: bool
    version_api: str
    timestamp: str


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def cargar_modelo():
    """Carga el modelo, metadata y feature engineer."""
    global model, metadata, feature_engineer
    
    try:
        # Cargar modelo
        model = joblib.load(MODEL_PATH)
        
        # Cargar metadata
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        # Cargar feature engineer
        feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
        
        print("✅ Modelo cargado exitosamente")
        print(f"   - Modelo: {metadata['model_name']}")
        print(f"   - Threshold: {metadata['threshold']}")
        print(f"   - Features: {len(metadata['feature_names'])}")
        
        return True
    except Exception as e:
        print(f"❌ Error cargando modelo: {str(e)}")
        return False


def preparar_features(request: FlightRequest) -> pd.DataFrame:
    """
    Prepara las features para el modelo a partir de la request.
    
    Args:
        request: Datos del vuelo
    
    Returns:
        DataFrame con features preparadas
    """
    # Parsear fecha
    fecha = datetime.fromisoformat(request.fecha_partida.replace('Z', '+00:00'))
    
    # Convertir distancia de km a millas (el modelo espera millas)
    distance_miles = request.distancia_km * 0.621371
    
    # Crear diccionario con features base
    features = {
        'year': fecha.year,
        'month': fecha.month,
        'day_of_month': fecha.day,
        'day_of_week': fecha.weekday() + 1,  # 1=Lun, 7=Dom
        'dep_hour': fecha.hour,
        'sched_minute_of_day': fecha.hour * 60 + fecha.minute,
        'op_unique_carrier': request.aerolinea,
        'origin': request.origen,
        'dest': request.destino,
        'distance': distance_miles,
        'latitude': 0.0,  # Valor por defecto (idealmente buscar en DB)
        'longitude': 0.0,  # Valor por defecto
        'dist_met_km': 10.0,  # Valor por defecto
        'temp': request.temperatura if request.temperatura is not None else 20.0,
        'wind_spd': request.velocidad_viento if request.velocidad_viento is not None else 10.0,
        'precip_1h': request.precipitacion if request.precipitacion is not None else 0.0,
        'climate_severity_idx': 0.3  # Calculado basado en clima (simplificado)
    }
    
    # Crear DataFrame
    df = pd.DataFrame([features])
    
    # Transformar categóricas si es necesario
    if hasattr(feature_engineer, 'transform_categorical'):
        try:
            df = feature_engineer.transform_categorical(df)
        except:
            # Si falla, usar encoding manual simple
            if 'op_unique_carrier' in df.columns:
                df['op_unique_carrier_encoded'] = hash(df['op_unique_carrier'].iloc[0]) % 100
            if 'origin' in df.columns:
                df['origin_encoded'] = hash(df['origin'].iloc[0]) % 500
            if 'dest' in df.columns:
                df['dest_encoded'] = hash(df['dest'].iloc[0]) % 500
    
    # Asegurarse de que tenemos todas las features del modelo
    for feature_name in metadata['feature_names']:
        if feature_name not in df.columns:
            # Si falta alguna feature, asignar valor por defecto
            if '_encoded' in feature_name:
                df[feature_name] = 0  # Valor por defecto para codificadas
            else:
                df[feature_name] = 0.0
    
    # Seleccionar solo las features del modelo en el orden correcto
    df = df[metadata['feature_names']]
    
    return df


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Ejecuta al iniciar la API."""
    print("\n" + "="*70)
    print("🚀 INICIANDO FLIGHTONTIME API")
    print("="*70)
    cargar_modelo()
    print("="*70 + "\n")


@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz - Información básica de la API."""
    return {
        "nombre": "FlightOnTime API",
        "version": "2.0.0",
        "descripcion": "API de predicción de retrasos de vuelos",
        "documentacion": "/docs",
        "endpoints": {
            "prediccion": "POST /predict",
            "salud": "GET /health",
            "info_modelo": "GET /model-info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verifica el estado de la API y si el modelo está cargado."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "modelo_cargado": model is not None,
        "version_api": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", response_model=ModelInfo, tags=["General"])
async def get_model_info():
    """Retorna información sobre el modelo de predicción."""
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "nombre": metadata['model_name'],
        "version": "2.0.0",
        "accuracy": metadata['metrics']['accuracy'],
        "recall": metadata['metrics']['recall'],
        "roc_auc": metadata['metrics']['roc_auc'],
        "threshold": float(metadata['threshold']),
        "features": len(metadata['feature_names']),
        "registros_entrenamiento": 15_000_000
    }


@app.post("/predict", response_model=FlightResponse, tags=["Predicción"])
async def predict_flight_delay(request: FlightRequest):
    """
    Predice si un vuelo será puntual o retrasado.
    
    **Entrada**:
    - aerolinea: Código de 2-3 letras (AA, DL, UA, etc.)
    - origen: Código IATA de 3 letras del aeropuerto origen (JFK, GRU, etc.)
    - destino: Código IATA de 3 letras del aeropuerto destino (LAX, GIG, etc.)
    - fecha_partida: Fecha/hora en formato ISO 8601 (2025-11-10T14:30:00)
    - distancia_km: Distancia del vuelo en kilómetros
    - temperatura (opcional): Temperatura en °C
    - velocidad_viento (opcional): Velocidad del viento en km/h
    - precipitacion (opcional): Precipitación en mm
    
    **Salida**:
    - prevision: "Puntual" o "Retrasado"
    - probabilidad: Probabilidad de la previsión (0.0 a 1.0)
    - confianza: Nivel de confianza (Alta, Media, Baja)
    - detalles: Información adicional
    """
    # Verificar que el modelo esté cargado
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Intente más tarde."
        )
    
    try:
        # Preparar features
        X = preparar_features(request)
        
        # Hacer predicción
        proba = model.predict_proba(X)[0, 1]  # Probabilidad de retraso
        
        # Usar threshold optimizado
        threshold = float(metadata['threshold'])
        prediction = 1 if proba >= threshold else 0
        
        # Determinar previsión y nivel de confianza
        prevision = "Retrasado" if prediction == 1 else "Puntual"
        
        # Calcular confianza basado en qué tan lejos está de 0.5
        distancia_decision = abs(proba - 0.5)
        if distancia_decision > 0.3:
            confianza = "Alta"
        elif distancia_decision > 0.15:
            confianza = "Media"
        else:
            confianza = "Baja"
        
        # Preparar respuesta
        response = {
            "prevision": prevision,
            "probabilidad": round(float(proba), 4),
            "confianza": confianza,
            "detalles": {
                "umbral_usado": threshold,
                "probabilidad_puntual": round(float(1 - proba), 4),
                "probabilidad_retrasado": round(float(proba), 4),
                "fecha_consulta": datetime.now().isoformat()
            }
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción: {str(e)}"
        )


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 FLIGHTONTIME API - Modo Desarrollo")
    print("="*70)
    print("\n📡 Servidor corriendo en: http://localhost:8000")
    print("📚 Documentación Swagger: http://localhost:8000/docs")
    print("📋 Documentación ReDoc: http://localhost:8000/redoc")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
