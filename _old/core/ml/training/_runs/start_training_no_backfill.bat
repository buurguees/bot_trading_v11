@echo off
echo Iniciando entrenamiento sin auto-backfill de features...
echo.
echo Este script omite la generacion automatica de features
echo ya que tienes un sistema de features en tiempo real funcionando.
echo.

REM Activar entorno virtual
call venv\Scripts\activate.bat

REM Ejecutar runner con --skip-backfill
python -m core.ml.training.daily_train.runner --skip-backfill

pause
