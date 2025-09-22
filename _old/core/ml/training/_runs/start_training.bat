@echo off
echo Iniciando sistema de entrenamiento automatizado...
echo.

REM Activar entorno virtual
call venv\Scripts\activate.bat

REM Crear directorio de logs si no existe
if not exist "logs" mkdir logs

REM Iniciar monitoreo en segundo plano
start "Monitor" cmd /c "python -m core.ml.training.daily_train.monitor"

REM Esperar un poco para que el monitor se inicie
timeout /t 5 /nobreak >nul

REM Iniciar entrenamiento principal
echo Iniciando entrenamiento principal...
python -m core.ml.training.daily_train.runner

pause
