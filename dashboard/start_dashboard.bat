@echo off
REM ========================================
REM FlightOnTime Dashboard - Script Inicio
REM ========================================

echo.
echo ======================================
echo  FLIGHTONTIME DASHBOARD - Iniciando...
echo ======================================
echo.

echo [1/3] Verificando dependencias...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Instalando dependencias...
    pip install -r requirements.txt
)

echo [2/3] Verificando estructura...
if not exist "app.py" (
    echo.
    echo ERROR: app.py no encontrado
    pause
    exit /b 1
)

echo [3/3] Iniciando dashboard...
echo.
echo ======================================
echo  Dashboard corriendo en:
echo  http://localhost:8501
echo.
echo  Navegacion:
echo  - Dashboard Principal
echo  - ROI Calculator
echo  - Predictive Simulator
echo  - 3D Routes Map
echo ======================================
echo.
echo Presiona Ctrl+C para detener
echo.

streamlit run app.py
