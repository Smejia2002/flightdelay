# Acciones Obligatorias

1. Eliminar variables con fuga de informacion (DEP_DELAY y cualquier variable basada en tiempos reales de salida/llegada).
2. Normalizar campos de tiempo:
   - CRS_DEP_TIME dentro de 0-2359.
   - DEP_HOUR dentro de 0-23.
   - sched_minute_of_day dentro de 0-1439.
   Definir tratamiento para valores 2400/1440.
3. Corregir PRECIP_1H negativa (set a 0 o marcar como missing).
4. Validar dominio geografico: LAT/LON fuera de rango esperado deben revisarse (filtrar, corregir o documentar).
5. Revisar nulos encubiertos tambien en columnas categoricas (category) y estandarizar codigos de missing.
6. Confirmar disponibilidad de variables climaticas en el momento de prediccion (evitar fuga temporal).
