@echo off
REM ============================================================================
REM SCRIPT DE MANTENIMIENTO DE BASE DE DATOS
REM Ejecuta optimizaciones con índices BRIN y VACUUM/ANALYZE
REM ============================================================================

echo.
echo ============================================================================
echo MANTENIMIENTO DE BASE DE DATOS - TRADING BOT
echo ============================================================================
echo.

REM Verificar que psql esté disponible
where psql >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: psql no encontrado. Asegúrate de que PostgreSQL esté instalado.
    echo.
    pause
    exit /b 1
)

REM Configurar variables de entorno
set PGHOST=192.168.10.109
set PGPORT=5432
set PGDATABASE=trading_db
set PGUSER=trading_user
set PGPASSWORD=160501

echo Iniciando mantenimiento de base de datos...
echo Host: %PGHOST%
echo Database: %PGDATABASE%
echo Usuario: %PGUSER%
echo.

REM Crear directorio de logs si no existe
if not exist "logs" mkdir logs

REM Generar nombre de archivo de log con timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "timestamp=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"

set "LOG_FILE=logs\db_maintenance_%timestamp%.log"

echo Ejecutando script de mantenimiento...
echo Log guardado en: %LOG_FILE%
echo.

REM Ejecutar script SQL y guardar log
psql -h %PGHOST% -p %PGPORT% -d %PGDATABASE% -U %PGUSER% -f "db_maintenance.sql" > "%LOG_FILE%" 2>&1

REM Verificar resultado
if %errorlevel% equ 0 (
    echo.
    echo ============================================================================
    echo MANTENIMIENTO COMPLETADO EXITOSAMENTE
    echo ============================================================================
    echo.
    echo Log guardado en: %LOG_FILE%
    echo.
    echo Resumen de la ejecución:
    echo - Índices BRIN creados para tablas grandes
    echo - VACUUM ANALYZE ejecutado en todas las tablas
    echo - Estadísticas actualizadas
    echo - Datos verificados
    echo.
) else (
    echo.
    echo ============================================================================
    echo ERROR EN EL MANTENIMIENTO
    echo ============================================================================
    echo.
    echo Código de error: %errorlevel%
    echo Revisa el log en: %LOG_FILE%
    echo.
)

REM Mostrar últimas líneas del log
echo Últimas líneas del log:
echo ----------------------------------------------------------------------------
tail -n 20 "%LOG_FILE%"
echo ----------------------------------------------------------------------------
echo.

REM Preguntar si abrir el log completo
set /p "open_log=¿Abrir el log completo? (s/n): "
if /i "%open_log%"=="s" (
    notepad "%LOG_FILE%"
)

echo.
echo Mantenimiento finalizado.
pause
