@echo off
echo ========================================
echo       MONITOREO DEL SISTEMA
echo ========================================
echo.
echo Selecciona una opcion de monitoreo:
echo.
echo 1. Monitoreo de emergencia
echo 2. Monitoreo de backtests historicos
echo 3. Verificar actividad reciente
echo 4. Verificar duplicados
echo 5. Verificar candidatos a promocion
echo 6. Verificar cambios en PnL
echo 7. Salir
echo.
set /p choice="Ingresa tu opcion (1-7): "

if "%choice%"=="1" (
    echo.
    echo Ejecutando monitoreo de emergencia...
    python core\ml\monitoring\monitor_emergency.py
) else if "%choice%"=="2" (
    echo.
    echo Ejecutando monitoreo de backtests historicos...
    python core\ml\monitoring\monitor_historical_backtests.py
) else if "%choice%"=="3" (
    echo.
    echo Verificando actividad reciente...
    python core\ml\monitoring\check_recent_activity.py
) else if "%choice%"=="4" (
    echo.
    echo Verificando duplicados...
    python core\ml\monitoring\check_duplicates.py
) else if "%choice%"=="5" (
    echo.
    echo Verificando candidatos a promocion...
    python core\ml\monitoring\check_promotion_candidates.py
) else if "%choice%"=="6" (
    echo.
    echo Verificando cambios en PnL...
    python core\ml\monitoring\check_pnl_changes.py
) else if "%choice%"=="7" (
    echo.
    echo Saliendo...
    exit
) else (
    echo.
    echo Opcion invalida. Saliendo...
    pause
    exit
)

echo.
pause
