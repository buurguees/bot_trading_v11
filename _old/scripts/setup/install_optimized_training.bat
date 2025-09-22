@echo off
echo ========================================
echo   INSTALACION DE ENTRENAMIENTO OPTIMIZADO
echo ========================================
echo.

echo 1. Instalando dependencias basicas...
pip install psutil>=5.9.0
if %errorlevel% neq 0 (
    echo Error instalando psutil
    pause
    exit /b 1
)

echo.
echo 2. Verificando requisitos del sistema...
python core/ml/training/configure_optimized_training.py --check-requirements
if %errorlevel% neq 0 (
    echo Error en verificacion de requisitos
    pause
    exit /b 1
)

echo.
echo 3. Creando configuracion...
python core/ml/training/configure_optimized_training.py --create-logging
if %errorlevel% neq 0 (
    echo Error creando configuracion
    pause
    exit /b 1
)

echo.
echo 4. Ejecutando prueba de sistema...
python test_optimized_training.py --max-bars 1000
if %errorlevel% neq 0 (
    echo Error en prueba del sistema
    pause
    exit /b 1
)

echo.
echo ========================================
echo   INSTALACION COMPLETADA EXITOSAMENTE
echo ========================================
echo.
echo Proximos pasos:
echo 1. Ejecutar entrenamiento: python -m core.ml.training.train_direction --help
echo 2. Monitorear progreso: python core/ml/monitoring/monitor_training_progress.py --help
echo 3. Ver documentacion: ENTRENAMIENTO_OPTIMIZADO.md
echo.
pause
