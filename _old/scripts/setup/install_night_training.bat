@echo off
echo ========================================
echo   INSTALACION DE ENTRENAMIENTO NOCTURNO
echo ========================================
echo.

echo 1. Instalando dependencias adicionales...
pip install psutil>=5.9.0
pip install flask>=2.0.0
pip install requests>=2.25.0
if %errorlevel% neq 0 (
    echo Error instalando dependencias
    pause
    exit /b 1
)

echo.
echo 2. Creando directorios necesarios...
if not exist "logs" mkdir logs
if not exist "artifacts\direction" mkdir artifacts\direction
if not exist "config\ml" mkdir config\ml

echo.
echo 3. Configurando entrenamiento nocturno...
python core\ml\training\night_train\configure_night_training.py --create
if %errorlevel% neq 0 (
    echo Error creando configuracion
    pause
    exit /b 1
)

echo.
echo 4. Optimizando configuracion para el sistema...
python core\ml\training\night_train\configure_night_training.py --optimize
if %errorlevel% neq 0 (
    echo Error optimizando configuracion
    pause
    exit /b 1
)

echo.
echo 5. Validando configuracion...
python core\ml\training\night_train\configure_night_training.py --validate
if %errorlevel% neq 0 (
    echo Error en validacion
    pause
    exit /b 1
)

echo.
echo 6. Estimando tiempo de ejecucion...
python core\ml\training\night_train\configure_night_training.py --estimate

echo.
echo ========================================
echo   INSTALACION COMPLETADA EXITOSAMENTE
echo ========================================
echo.
echo Proximos pasos:
echo 1. Iniciar entrenamiento: python core\ml\training\night_train\start_night_training.py
echo 2. Monitorear: python core\ml\monitoring\monitor_night_training.py
echo 3. Dashboard: http://localhost:5000
echo 4. Configurar: python core\ml\training\night_train\configure_night_training.py --help
echo.
pause
