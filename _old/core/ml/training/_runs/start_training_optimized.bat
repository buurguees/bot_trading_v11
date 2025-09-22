@echo off
echo Iniciando entrenamiento optimizado con backtests historicos...
echo.
echo Configuracion:
echo - Entrenamiento cada 60 minutos
echo - Backtests con TODO el historial (365+ dias)
echo - Sin auto-backfill de features
echo - Umbrales ajustados para objetivos realistas
echo.

REM Activar entorno virtual
call venv\Scripts\activate.bat

REM Ejecutar runner optimizado
python -m core.ml.training.daily_train.runner --skip-backfill

pause
