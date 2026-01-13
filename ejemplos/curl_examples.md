# 🧪 EJEMPLOS cURL - FlightOnTime API

Ejemplos de uso de la API con cURL para testing y demostración.

---

## 🏥 **1. HEALTH CHECK**

Verificar que la API está corriendo:

```bash
curl -X GET "http://localhost:8000/health"
```

**Respuesta esperada**:
```json
{
  "status": "healthy",
  "modelo_cargado": true,
  "version_api": "2.0.0",
  "timestamp": "2026-01-13T08:40:00"
}
```

---

## 📊 **2. MODEL INFO**

Obtener información del modelo:

```bash
curl -X GET "http://localhost:8000/model-info"
```

**Respuesta esperada**:
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

## ✅ **3. CASO PUNTUAL** (Baja probabilidad de retraso)

Vuelo: Delta Airlines ATL → ORD  
Condiciones: Clima bueno, horario favorable

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
  "probabilidad": 0.32,
  "confianza": "Media",
  "detalles": {
    "umbral_usado": 0.52,
    "probabilidad_puntual": 0.68,
    "probabilidad_retrasado": 0.32
  }
}
```

---

## ⚠️ **4. CASO RETRASADO** (Alta probabilidad de retraso)

Vuelo: United Airlines SFO → JFK  
Condiciones: Clima malo, vuelo largo, horario congestionado

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
  "probabilidad": 0.82,
  "confianza": "Alta",
  "detalles": {
    "umbral_usado": 0.52,
    "probabilidad_puntual": 0.18,
    "probabilidad_retrasado": 0.82
  }
}
```

---

## 🎯 **5. EJEMPLO OFICIAL DEL HACKATHON**

Vuelo: GIG → GRU (Ejemplo de la descripción oficial)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "aerolinea": "AZ",
    "origen": "GIG",
    "destino": "GRU",
    "fecha_partida": "2025-11-10T14:30:00",
    "distancia_km": 350
  }'
```

**Respuesta esperada**:
```json
{
  "prevision": "Puntual" | "Retrasado",
  "probabilidad": 0.XX,
  "confianza": "Media"
}
```

---

## ❌ **6. ERROR - CAMPOS FALTANTES**

Request inválida (faltan campos obligatorios):

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "aerolinea": "AA",
    "origen": "JFK"
  }'
```

**Respuesta esperada** (HTTP 422):
```json
{
  "detail": [
    {
      "loc": ["body", "destino"],
      "msg": "field required",
      "type": "value_error.missing"
    },
    {
      "loc": ["body", "fecha_partida"],
      "msg": "field required",
      "type": "value_error.missing"
    },
    {
      "loc": ["body", "distancia_km"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## ❌ **7. ERROR - FORMATO DE FECHA INVÁLIDO**

Fecha en formato incorrecto:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "aerolinea": "AA",
    "origen": "JFK",
    "destino": "LAX",
    "fecha_partida": "2025/11/10 14:30",
    "distancia_km": 3983
  }'
```

**Respuesta esperada** (HTTP 422):
```json
{
  "detail": [
    {
      "loc": ["body", "fecha_partida"],
      "msg": "Fecha debe estar en formato ISO 8601 (YYYY-MM-DDTHH:MM:SS)",
      "type": "value_error"
    }
  ]
}
```

---

## ✈️ **8. VUELO DE DEMOSTRACIÓN AA JFK-LAX**

Vuelo usado en las demos del proyecto:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "aerolinea": "AA",
    "origen": "JFK",
    "destino": "LAX",
    "fecha_partida": "2025-03-15T14:30:00",
    "distancia_km": 3983,
    "temperatura": 25.5,
    "velocidad_viento": 15.3,
    "precipitacion": 0.0
  }'
```

---

## 💾 **GUARDAR EJEMPLOS COMO ARCHIVOS**

### Crear archivo JSON de ejemplo:

```bash
cat > ejemplo_puntual.json << 'EOF'
{
  "aerolinea": "DL",
  "origen": "ATL",
  "destino": "ORD",
  "fecha_partida": "2025-06-15T08:00:00",
  "distancia_km": 975,
  "temperatura": 22.0,
  "velocidad_viento": 8.0,
  "precipitacion": 0.0
}
EOF
```

### Usar el archivo:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @ejemplo_puntual.json
```

---

## 🔍 **TESTING COMPLETO**

Script para probar todos los endpoints:

```bash
#!/bin/bash

echo "=== TESTING FLIGHTONTIME API ==="

echo "\n1. Health Check..."
curl -s http://localhost:8000/health | jq

echo "\n2. Model Info..."
curl -s http://localhost:8000/model-info | jq

echo "\n3. Predicción Puntual..."
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "aerolinea": "DL",
    "origen": "ATL",
    "destino": "ORD",
    "fecha_partida": "2025-06-15T08:00:00",
    "distancia_km": 975
  }' | jq

echo "\n4. Predicción Retrasado..."
curl -s -X POST http://localhost:8000/predict \
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
  }' | jq

echo "\n=== TESTING COMPLETADO ==="
```

Guardar como `test_api.sh` y ejecutar:
```bash
chmod +x test_api.sh
./test_api.sh
```

---

## 📱 **FORMATOS ALTERNATIVOS**

### Con pretty-print (jq):
```bash
curl http://localhost:8000/health | jq
```

### Con headers verbosos:
```bash
curl -v -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Guardar respuesta:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{...}' > respuesta.json
```

---

**Última actualización**: 2026-01-13  
**Base URL**: `http://localhost:8000`
