@echo off
REM ========================================
REM FlightOnTime API - Script de Inicio
REM ========================================

echo.
echo ======================================
echo  FLIGHTONTIME API - Iniciando...
echo ======================================
echo.

REM Verificar que estamos en el directorio correcto
if not exist "main.py" (
    echo ERROR: No se encuentra main.py
    echo Asegurate de estar en la carpeta backend/
    pause
    exit /b 1
)

echo [1/3] Verificando dependencias...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Instalando dependencias...
    pip install -r requirements.txt
)

echo [2/3] Verificando modelo...
if not exist "..\models\model.joblib" (
    echo.
    echo ERROR: Modelo no encontrado
    echo Ejecuta primero: python ..\train_model.py
    pause
    exit /b 1
)

echo [3/3] Iniciando API...
echo.
echo ======================================
echo  API corriendo en:
echo  http://localhost:8000
echo.
echo  Documentacion:
echo  http://localhost:8000/docs
echo ======================================
echo.
echo Presiona Ctrl+C para detener
echo.

python main.py
