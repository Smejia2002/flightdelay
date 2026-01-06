# Informe No Tecnico

## Que encontramos
- El dataset tiene muchos vuelos (35.7 millones) y variables basicas de aerolinea, origen, destino, hora y clima.
- La mayoria de vuelos son puntuales: ~81% a tiempo y ~19% con retraso.
- No se observan valores faltantes obvios, pero hay algunos valores raros que requieren limpieza.

## Aspectos a corregir antes de entrenar
- Hay una columna con el retraso real (DEP_DELAY). Usarla daria una ventaja injusta y haria el modelo poco realista.
- Algunas horas aparecen como 24:00 y algunos minutos como 1440, valores que no deberian existir.
- Hay valores de precipitacion negativos y coordenadas fuera del rango esperado.
- Se confirmo que todos los dias del periodo estan presentes (sin huecos).

## Que sigue
Aplicar limpieza basica, definir reglas de dominio y volver a validar antes de entrenar el modelo.
