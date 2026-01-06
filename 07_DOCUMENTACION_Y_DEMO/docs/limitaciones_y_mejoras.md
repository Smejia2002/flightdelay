# Limitaciones y Mejoras

## Limitaciones del MVP
- Modelo baseline (Logistic Regression) sin optimizacion avanzada.
- Dependencia de un servicio externo de clima.
- Posible degradacion por cambios operativos o estacionales.
- Convergencia limitada (max_iter=200).

## Mejoras futuras
- Probar modelos mas robustos y tuning de hiperparametros.
- Calibrar probabilidades y definir umbrales de negocio.
- Monitoreo de drift y retraining programado.
- Cache de clima y tolerancia a fallos del proveedor.
- Evaluar variables adicionales disponibles antes del vuelo.
