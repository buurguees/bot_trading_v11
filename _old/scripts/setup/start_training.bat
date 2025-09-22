@echo off
echo ========================================
echo    SISTEMA DE ENTRENAMIENTO AUTOMATIZADO
echo ========================================
echo.
echo Selecciona una opcion:
echo.
echo 1. Entrenamiento estandar (con auto-backfill)
echo 2. Entrenamiento sin auto-backfill (recomendado)
echo 3. Entrenamiento optimizado (backtests historicos)
echo 4. Solo monitoreo
echo 5. Salir
echo.
set /p choice="Ingresa tu opcion (1-5): "

if "%choice%"=="1" (
    echo.
    echo Iniciando entrenamiento estandar...
    call core\ml\training\_runs\start_training.bat
) else if "%choice%"=="2" (
    echo.
    echo Iniciando entrenamiento sin auto-backfill...
    call core\ml\training\_runs\start_training_no_backfill.bat
) else if "%choice%"=="3" (
    echo.
    echo Iniciando entrenamiento optimizado...
    call core\ml\training\_runs\start_training_optimized.bat
) else if "%choice%"=="4" (
    echo.
    echo Iniciando solo monitoreo...
    python core\ml\monitoring\monitor_emergency.py
    pause
) else if "%choice%"=="5" (
    echo.
    echo Saliendo...
    exit
) else (
    echo.
    echo Opcion invalida. Saliendo...
    pause
    exit
)
