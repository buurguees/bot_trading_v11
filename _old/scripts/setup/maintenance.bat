@echo off
echo ========================================
echo       MANTENIMIENTO DEL SISTEMA
echo ========================================
echo.
echo Selecciona una opcion de mantenimiento:
echo.
echo 1. Limpiar duplicados de predicciones
echo 2. Aplicar correcciones de emergencia
echo 3. Configurar datos completos
echo 4. Verificar configuracion de datos
echo 5. Salir
echo.
set /p choice="Ingresa tu opcion (1-5): "

if "%choice%"=="1" (
    echo.
    echo Limpiando duplicados de predicciones...
    python scripts\maintenance\fix_prediction_duplicates.py
) else if "%choice%"=="2" (
    echo.
    echo Aplicando correcciones de emergencia...
    python scripts\maintenance\emergency_fix.py
) else if "%choice%"=="3" (
    echo.
    echo Configurando datos completos...
    python scripts\setup\setup_full_data.py
) else if "%choice%"=="4" (
    echo.
    echo Verificando configuracion de datos...
    python scripts\setup\verify_data_setup.py
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

echo.
pause
