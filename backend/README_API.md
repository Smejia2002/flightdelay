# 🔌 API REST - FlightOnTime

**Versión**: 2.0.0  
**Framework**: FastAPI  
**Puerto**: 8000  
**Base URL**: `http://localhost:8000`

---

## 🚀 **Inicio Rápido**

### 1. Instalar Dependencias
```bash
cd backend
pip install -r requirements.txt
```

### 2. Ejecutar API
```bash
python main.py
```

### 3. Ver Documentación
```
http://localhost:8000/docs     # Swagger UI
http://localhost:8000/redoc    # ReDoc
```

---

## 📡 **ENDPOINTS**

### **POST /predict** - Predicción de Retraso

Predice si un vuelo será puntual o retrasado.

**URL**: `/predict`  
**Método**: `POST`  
**Content-Type**: `application/json`

#### **REQUEST**

```json
{
  "aerolinea": "AA",
  "origen": "JFK",
  "destino": "LAX",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 3983,
  "temperatura": 25.5,
  "velocidad_viento": 15.3,
  "precipitacion": 0.0
}
```

**Campos Obligatorios**:
| Campo           | Tipo   | Descripción                    | Ejemplo               |
| --------------- | ------ | ------------------------------ | --------------------- |
| `aerolinea`     | string | Código aerolínea (2-3 letras)  | "AA", "DL", "UA"      |
| `origen`        | string | Código IATA origen (3 letras)  | "JFK", "GRU"          |
| `destino`       | string | Código IATA destino (3 letras) | "LAX", "GIG"          |
| `fecha_partida` | string | ISO 8601                       | "2025-11-10T14:30:00" |
| `distancia_km`  | float  | Distancia en km                | 3983                  |

**Campos Opcionales** (mejoran predicción):
| Campo              | Tipo  | Descripción       |
| ------------------ | ----- | ----------------- |
| `temperatura`      | float | Temperatura en °C |
| `velocidad_viento` | float | Viento en km/h    |
| `precipitacion`    | float | Precipitación mm  |

#### **RESPONSE**

```json
{
  "prevision": "Retrasado",
  "probabilidad": 0.7834,
  "confianza": "Alta",
  "detalles": {
    "umbral_usado": 0.52,
    "probabilidad_puntual": 0.2166,
    "probabilidad_retrasado": 0.7834,
    "fecha_consulta": "2026-01-13T08:40:00"
  }
}
```

**Respuesta**:
| Campo          | Tipo   | Descripción              |
| -------------- | ------ | ------------------------ |
| `prevision`    | string | "Puntual" o "Retrasado"  |
| `probabilidad` | float  | Probabilidad (0.0 - 1.0) |
| `confianza`    | string | "Alta", "Media", "Baja"  |
| `detalles`     | object | Información adicional    |

---

### **GET /health** - Estado de la API

Verifica si la API y el modelo están funcionando.

**URL**: `/health`  
**Método**: `GET`

#### **RESPONSE**

```json
{
  "status": "healthy",
  "modelo_cargado": true,
  "version_api": "2.0.0",
  "timestamp": "2026-01-13T08:40:00"
}
```

---

### **GET /model-info** - Información del Modelo

Retorna métricas y detalles del modelo ML.

**URL**: `/model-info`  
**Método**: `GET`

#### **RESPONSE**

```json
{
  "nombre": "XGBoost",
  "version": "2.0.0",
  "accuracy": 0.7246,
  "recall": 0.6130,
  "roc_auc": 0.7172,
  "threshold": 0.52,
  "features": 17,
  "registros_entrenamiento": 15000000
}
```

---

## 💻 **EJEMPLOS DE USO**

### **cURL - Caso Puntual**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "aerolinea": "DL",
    "origen": "ATL",
    "destino": "ORD",
    "fecha_partida": "2025-06-15T08:00:00",
    "distancia_km": 975,
    "temperatura": 22.0,
    "velocidad_viento": 8.0,
    "precipitacion": 0.0
  }'
```

**Respuesta esperada**:
```json
{
  "prevision": "Puntual",
  "probabilidad": 0.32
}
```

---

### **cURL - Caso Retrasado**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "aerolinea": "UA",
    "origen": "SFO",
    "destino": "JFK",
    "fecha_partida": "2025-12-20T18:00:00",
    "distancia_km": 4150,
    "temperatura": 8.0,
    "velocidad_viento": 32.0,
    "precipitacion": 8.5
  }'
```

**Respuesta esperada**:
```json
{
  "prevision": "Retrasado",
  "probabilidad": 0.82
}
```

---

### **cURL - Caso Error (Campos Faltantes)**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "aerolinea": "AA",
    "origen": "JFK"
  }'
```

**Respuesta esperada**:
```json
{
  "detail": [
    {
      "loc": ["body", "destino"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

### **Python - Requests**

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "aerolinea": "AA",
    "origen": "JFK",
    "destino": "LAX",
    "fecha_partida": "2025-11-10T14:30:00",
    "distancia_km": 3983
}

response = requests.post(url, json=data)
result = response.json()

print(f"Previsión: {result['prevision']}")
print(f"Probabilidad: {result['probabilidad']:.2%}")
```

---

### **JavaScript - Fetch**

```javascript
const url = "http://localhost:8000/predict";
const data = {
  aerolinea: "AA",
  origen: "JFK",
  destino: "LAX",
  fecha_partida: "2025-11-10T14:30:00",
  distancia_km: 3983
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
})
.then(res => res.json())
.then(result => {
  console.log(`Previsión: ${result.prevision}`);
  console.log(`Probabilidad: ${result.probabilidad}`);
});
```

---

## 🔐 **CÓDIGOS DE ESTADO HTTP**

| Código | Descripción                                |
| ------ | ------------------------------------------ |
| 200    | Éxito - Predicción realizada               |
| 422    | Error de validación - Request inválida     |
| 500    | Error interno del servidor                 |
| 503    | Servicio no disponible - Modelo no cargado |

---

## 🧪 **TESTING**

### Probar Health Check
```bash
curl http://localhost:8000/health
```

### Probar Info Modelo
```bash
curl http://localhost:8000/model-info
```

### Probar Predicción
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @ejemplo_vuelo.json
```

Donde `ejemplo_vuelo.json`:
```json
{
  "aerolinea": "AA",
  "origen": "JFK",
  "destino": "LAX",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 3983
}
```

---

## 📝 **NOTAS TÉCNICAS**

### Validaciones Automáticas
- ✅ Códigos de aeropuerto/aerolínea se convierten a mayúsculas
- ✅ Fecha debe estar en formato ISO 8601
- ✅ Distancia debe ser > 0
- ✅ Temperatura entre -50°C y 60°C
- ✅ Viento y precipitación >= 0

### Conversiones Automáticas
- Distancia km → millas (modelo espera millas)
- Códigos IATA → encodings numéricos
- Fecha → features temporales (año, mes, día, hora, etc.)

### Valores por Defecto
Si no se proveen campos opcionales:
- `temperatura`: 20°C
- `velocidad_viento`: 10 km/h
- `precipitacion`: 0 mm

---

## 🎯 **SWAGGER UI**

La API tiene documentación interactiva en:

```
http://localhost:8000/docs
```

**Características**:
- 📝 Documentación completa de endpoints
- 🧪 Interfaz de prueba integrada
- 📊 Modelos de datos (schemas)
- ✅ Validación en tiempo real

---

## 🐛 **TROUBLESHOOTING**

### Problema: "Modelo no cargado"
**Solución**:
```bash
# Verificar que existan:
ls models/model.joblib
ls models/metadata.json
ls models/feature_engineer.joblib
```

### Problema: "Puerto 8000 en uso"
**Solución**:
```bash
# Cambiar puerto en main.py línea final:
uvicorn.run("main:app", host="0.0.0.0", port=8001)
```

### Problema: "ModuleNotFoundError"
**Solución**:
```bash
cd backend
pip install -r requirements.txt
```

---

**Documentación completa**: Ver `/docs` cuando la API esté corriendo  
**Última actualización**: 2026-01-13
