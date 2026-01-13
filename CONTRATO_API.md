# 📋 CONTRATO DE INTEGRACIÓN API - FlightOnTime

**Versión**: 2.0.0  
**Fecha**: 2026-01-13  
**Status**: OFICIAL - Hackathon Aviación Civil 2026

---

## 📡 **ESPECIFICACIÓN TÉCNICA**

### **Endpoint Principal**
```
POST /predict
```

**Base URL**: `http://localhost:8000`  
**Content-Type**: `application/json`  
**Charset**: UTF-8

---

## 📥 **FORMATO DE ENTRADA (REQUEST)**

### **Schema JSON**

```json
{
  "aerolinea": "string",
  "origen": "string",
  "destino": "string",
  "fecha_partida": "string (ISO 8601)",
  "distancia_km": number,
  "temperatura": number (opcional),
  "velocidad_viento": number (opcional),
  "precipitacion": number (opcional)
}
```

### **Campos Obligatorios**

| Campo           | Tipo   | Restricciones             | Descripción                       | Ejemplo               |
| --------------- | ------ | ------------------------- | --------------------------------- | --------------------- |
| `aerolinea`     | string | 2-3 caracteres, uppercase | Código IATA/ICAO de aerolínea     | "AA", "DL", "UA"      |
| `origen`        | string | 3 caracteres, uppercase   | Código IATA aeropuerto origen     | "JFK", "GRU", "GIG"   |
| `destino`       | string | 3 caracteres, uppercase   | Código IATA aeropuerto destino    | "LAX", "ORD", "ATL"   |
| `fecha_partida` | string | ISO 8601                  | Fecha/hora de salida programada   | "2025-11-10T14:30:00" |
| `distancia_km`  | number | > 0                       | Distancia del vuelo en kilómetros | 350, 3983             |

**Nota**: `distancia_km` se convierte a millas antes de la inferencia para mantener compatibilidad con el modelo.

### **Campos Opcionales** (Mejoran la predicción)

| Campo              | Tipo   | Rango    | Descripción                 | Valor por defecto |
| ------------------ | ------ | -------- | --------------------------- | ----------------- |
| `temperatura`      | number | -50 a 60 | Temperatura en °C en origen | 20.0              |
| `velocidad_viento` | number | >= 0     | Velocidad viento en km/h    | 10.0              |
| `precipitacion`    | number | >= 0     | Precipitación en mm         | 0.0               |

### **Ejemplo Completo (Oficial Hackathon)**

```json
{
  "aerolinea": "AZ",
  "origen": "GIG",
  "destino": "GRU",
  "fecha_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}
```

### **Ejemplo con Campos Opcionales**

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

---

## 📤 **FORMATO DE SALIDA (RESPONSE)**

### **Schema JSON (Éxito - HTTP 200)**

```json
{
  "prevision": "string",
  "probabilidad": number,
  "confianza": "string",
  "detalles": {
    "umbral_usado": number,
    "probabilidad_puntual": number,
    "probabilidad_retrasado": number,
    "fecha_consulta": "string"
  }
}
```

### **Campos de Respuesta**

| Campo          | Tipo   | Valores Posibles        | Descripción                                 |
| -------------- | ------ | ----------------------- | ------------------------------------------- |
| `prevision`    | string | "Puntual", "Retrasado"  | Predicción final del modelo                 |
| `probabilidad` | number | 0.0 - 1.0               | Probabilidad de la predicción (4 decimales) |
| `confianza`    | string | "Alta", "Media", "Baja" | Nivel de confianza en la predicción         |
| `detalles`     | object | -                       | Información adicional                       |

### **Objeto `detalles`**

| Campo                    | Tipo   | Descripción                         |
| ------------------------ | ------ | ----------------------------------- |
| `umbral_usado`           | number | Threshold usado para clasificación  |
| `probabilidad_puntual`   | number | Prob. de que el vuelo sea puntual   |
| `probabilidad_retrasado` | number | Prob. de que el vuelo se retrase    |
| `fecha_consulta`         | string | Timestamp de la consulta (ISO 8601) |

### **Ejemplo Respuesta PUNTUAL**

```json
{
  "prevision": "Puntual",
  "probabilidad": 0.2234,
  "confianza": "Media",
  "detalles": {
    "umbral_usado": 0.52,
    "probabilidad_puntual": 0.7766,
    "probabilidad_retrasado": 0.2234,
    "fecha_consulta": "2026-01-13T08:40:00"
  }
}
```

### **Ejemplo Respuesta RETRASADO**

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

---

## ⚠️ **CÓDIGOS DE ERROR**

### **HTTP 422 - Unprocessable Entity**

Error de validación en la entrada.

**Estructura**:
```json
{
  "detail": [
    {
      "loc": ["body", "campo"],
      "msg": "mensaje de error",
      "type": "tipo_error"
    }
  ]
}
```

**Ejemplo - Campo faltante**:
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

**Ejemplo - Validación fallida**:
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

### **HTTP 500 - Internal Server Error**

Error interno del servidor.

```json
{
  "detail": "Error en predicción: [mensaje detallado]"
}
```

### **HTTP 503 - Service Unavailable**

Modelo no está cargado.

```json
{
  "detail": "Modelo no disponible. Intente más tarde."
}
```

---

## 🔐 **VALIDACIONES AUTOMÁTICAS**

La API aplica las siguientes validaciones:

1. **Códigos aeroportuarios**: Convertidos automáticamente a MAYÚSCULAS
2. **Códigode aerolínea**: Convertido automáticamente a MAYÚSCULAS
3. **Fecha**: Debe ser ISO 8601 (`YYYY-MM-DDTHH:MM:SS`)
4. **Distancia**: Debe ser > 0
5. **Temperatura**: Entre -50°C y 60°C
6. **Viento**: >= 0
7. **Precipitación**: >= 0

---

## 🔄 **CONVERSIONES INTERNAS**

La API realiza estas conversiones transparentes:

| Dato de Entrada  | Conversión        | Uso Interno                  |
| ---------------- | ----------------- | ---------------------------- |
| `distancia_km`   | km → millas       | Modelo espera millas         |
| `fecha_partida`  | Fecha → Features  | year, month, day, hour, etc. |
| `aerolinea`      | String → Encoding | Encoding numérico            |
| `origen/destino` | String → Encoding | Encoding numérico            |

---

## 📊 **LÓGICA DE DECISION**

```
SI probabilidad_retrasado >= threshold (0.52):
    prevision = "Retrasado"
SINO:
    prevision = "Puntual"

Confianza:
    SI |probabilidad - 0.5| > 0.3: "Alta"
    SI |probabilidad - 0.5| > 0.15: "Media"
    SINO: "Baja"
```

---

## 🧪 **CASOS DE PRUEBA OBLIGATORIOS**

### **Test 1: Vuelo Puntual**
```json
INPUT: {
  "aerolinea": "DL",
  "origen": "ATL",
  "destino": "ORD",
  "fecha_partida": "2025-06-15T08:00:00",
  "distancia_km": 975
}
EXPECTED: prevision = "Puntual", probabilidad < 0.5
```

### **Test 2: Vuelo Retrasado**
```json
INPUT: {
  "aerolinea": "UA",
  "origen": "SFO",
  "destino": "JFK",
  "fecha_partida": "2025-12-20T18:00:00",
  "distancia_km": 4150,
  "temperatura": 8.0,
  "velocidad_viento": 32.0,
  "precipitacion": 8.5
}
EXPECTED: prevision = "Retrasado", probabilidad > 0.7
```

### **Test 3: Error - Campos Faltantes**
```json
INPUT: {
  "aerolinea": "AA",
  "origen": "JFK"
}
EXPECTED: HTTP 422, detail con campos faltantes
```

---

## 📝 **NOTAS DE IMPLEMENTACIÓN**

### **Backend (Java/Spring Boot)**
```java
// Ejemplo de clase Request
public class FlightRequest {
    @NotNull
    @Size(min=2, max=3)
    private String aerolinea;
    
    @NotNull
    @Size(min=3, max=3)
    private String origen;
    
    @NotNull
    @Size(min=3, max=3)
    private String destino;
    
    @NotNull
    @Pattern(regexp="\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}")
    private String fecha_partida;
    
    @NotNull
    @Positive
    private Double distancia_km;
    
    // Campos opcionales
    private Double temperatura;
    private Double velocidad_viento;
    private Double precipitacion;
}
```

### **Backend (Python/FastAPI)**
```python
from pydantic import BaseModel, Field

class FlightRequest(BaseModel):
    aerolinea: str = Field(..., min_length=2, max_length=3)
    origen: str = Field(..., min_length=3, max_length=3)
    destino: str = Field(..., min_length=3, max_length=3)
    fecha_partida: str
    distancia_km: float = Field(..., gt=0)
    temperatura: Optional[float] = None
    velocidad_viento: Optional[float] = None
    precipitacion: Optional[float] = None
```

---

## ✅ **CHECKLIST DE CUMPLIMIENTO**

API debe cumplir:

- [ ] Endpoint POST /predict funcional
- [ ] Acepta JSON con 5 campos obligatorios
- [ ] Retorna JSON con prevision + probabilidad
- [ ] Validación de entrada (HTTP 422 si falla)
- [ ] Manejo de errores (HTTP 500, 503)
- [ ] Respuestas en formato estándar
- [ ] Documentación Swagger/OpenAPI
- [ ] 3 ejemplos de uso (Postman/cURL)
- [ ] README con instrucciones de ejecución

---

## 🔗 **REFERENCIAS**

- **Documentación Swagger**: `http://localhost:8000/docs`
- **Documentación ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `GET http://localhost:8000/health`
- **Model Info**: `GET http://localhost:8000/model-info`

---

**Firmado**: FlightOnTime Data Science & Backend Teams  
**Fecha**: 2026-01-13  
**Versión Contrato**: 2.0.0  
**Estado**: ✅ APROBADO
