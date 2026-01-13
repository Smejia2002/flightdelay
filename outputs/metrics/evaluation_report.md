# 📊 Reporte de Evaluación de Modelos

**Mejor Modelo:** XGBoost

## Comparación de Modelos

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| LogisticRegression | 0.6077 | 0.2633 | 0.5995 | 0.3659 | 0.6423 |
| RandomForest | 0.6183 | 0.2851 | 0.6779 | 0.4014 | 0.6967 |
| XGBoost | 0.6560 | 0.3083 | 0.6606 | 0.4204 | 0.7167 |
| LightGBM | 0.6547 | 0.3067 | 0.6576 | 0.4183 | 0.7140 |

## Modelo Seleccionado: XGBoost

### Métricas Detalladas

- **Accuracy:** 0.6560
- **Precision:** 0.3083
- **Recall:** 0.6606
- **F1-Score:** 0.4204
- **ROC-AUC:** 0.7167
- **PR-AUC:** 0.3828

### Matriz de Confusión

```
                  Predicción
                  Puntual  Retrasado
Real  Puntual    1195455   629722
      Retrasado   144201   280622
```

### Interpretación

- **Verdaderos Negativos (Puntuales correctos):** 1,195,455
- **Falsos Positivos (Alertas falsas):** 629,722
- **Falsos Negativos (Retrasos no detectados):** 144,201
- **Verdaderos Positivos (Retrasos detectados):** 280,622
