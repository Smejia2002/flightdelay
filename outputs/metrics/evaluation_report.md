# 📊 Reporte de Evaluación de Modelos

**Mejor Modelo:** XGBoost

## Comparación de Modelos

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| XGBoost | 0.7232 | 0.3501 | 0.5430 | 0.4257 | 0.7194 |

## Modelo Seleccionado: XGBoost

### Métricas Detalladas

- **Accuracy:** 0.7232
- **Precision:** 0.3501
- **Recall:** 0.5430
- **F1-Score:** 0.4257
- **ROC-AUC:** 0.7194
- **PR-AUC:** 0.3874

### Matriz de Confusión

```
                  Predicción
                  Puntual  Retrasado
Real  Puntual    3319108  1018723
      Retrasado   461820   548682
```

### Interpretación

- **Verdaderos Negativos (Puntuales correctos):** 3,319,108
- **Falsos Positivos (Alertas falsas):** 1,018,723
- **Falsos Negativos (Retrasos no detectados):** 461,820
- **Verdaderos Positivos (Retrasos detectados):** 548,682
