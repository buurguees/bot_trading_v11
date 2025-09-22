@echo off
echo ========================================
echo    INSTALADOR DEL PROCESADOR DE SEÑALES
echo ========================================
echo.

REM Verificar que estamos en el directorio correcto
if not exist "core\ml\signals\signal_processor.py" (
    echo ERROR: No se encuentra signal_processor.py
    echo Asegurate de ejecutar este script desde el directorio raiz del proyecto
    pause
    exit /b 1
)

echo [1/4] Creando directorios necesarios...
if not exist "logs" mkdir logs
if not exist "config\signals" mkdir config\signals
if not exist "core\ml\signals" mkdir core\ml\signals

echo [2/4] Verificando archivos de configuracion...
if not exist "config\signals\signal_processing.yaml" (
    echo ERROR: No se encuentra signal_processing.yaml
    echo Ejecuta primero la creacion de archivos
    pause
    exit /b 1
)

echo [3/4] Verificando dependencias...
python -c "import yaml, pandas, numpy, sqlalchemy" 2>nul
if errorlevel 1 (
    echo ERROR: Faltan dependencias de Python
    echo Instala con: pip install pyyaml pandas numpy sqlalchemy
    pause
    exit /b 1
)

echo [4/4] Probando el procesador...
python -c "from core.ml.signals.signal_processor import create_signal_processor; print('Procesador importado correctamente')"
if errorlevel 1 (
    echo ERROR: No se puede importar el procesador
    echo Verifica la configuracion de la base de datos
    pause
    exit /b 1
)

echo.
echo ========================================
echo    INSTALACION COMPLETADA EXITOSAMENTE
echo ========================================
echo.
echo Para usar el procesador:
echo.
echo 1. Ejecucion una vez:
echo    python scripts\run_signal_processor.py --mode realtime
echo.
echo 2. Ejecucion continua (daemon):
echo    python scripts\start_signal_processor_daemon.py
echo.
echo 3. Procesamiento en lote:
echo    python scripts\run_signal_processor.py --mode batch --start-time "2024-01-01 00:00:00" --end-time "2024-01-01 23:59:59"
echo.
echo Presiona cualquier tecla para continuar...
pause >nul
