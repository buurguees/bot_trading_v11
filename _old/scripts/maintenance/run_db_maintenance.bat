@echo off
REM =====================================================
REM MANTENIMIENTO DE BASE DE DATOS - BOT TRADING V11
REM =====================================================
REM Script para ejecutar mantenimiento de BD en Windows
REM Incluye índices BRIN y VACUUM/ANALYZE
REM =====================================================

setlocal enabledelayedexpansion

echo.
echo =====================================================
echo MANTENIMIENTO DE BASE DE DATOS - BOT TRADING V11
echo =====================================================
echo.

REM Verificar si psql está disponible
where psql >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: psql no encontrado en el PATH
    echo Por favor instala PostgreSQL o agrega psql al PATH
    echo.
    pause
    exit /b 1
)

REM Configuración de conexión (ajustar según tu entorno)
set DB_HOST=192.168.10.109
set DB_PORT=5432
set DB_NAME=trading_db
set DB_USER=trading_user
set DB_PASSWORD=160501

REM Archivo SQL de mantenimiento
set SQL_FILE=%~dp0db_maintenance.sql

REM Verificar que el archivo SQL existe
if not exist "%SQL_FILE%" (
    echo ERROR: No se encontró el archivo %SQL_FILE%
    echo.
    pause
    exit /b 1
)

echo Configuración de conexión:
echo - Host: %DB_HOST%
echo - Puerto: %DB_PORT%
echo - Base de datos: %DB_NAME%
echo - Usuario: %DB_USER%
echo - Archivo SQL: %SQL_FILE%
echo.

REM Preguntar confirmación
set /p CONFIRM="¿Continuar con el mantenimiento? (s/N): "
if /i not "%CONFIRM%"=="s" (
    echo Operación cancelada.
    pause
    exit /b 0
)

echo.
echo Iniciando mantenimiento de base de datos...
echo =====================================================

REM Ejecutar el script SQL
echo Ejecutando script de mantenimiento...
psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -f "%SQL_FILE%"

if %errorlevel% equ 0 (
    echo.
    echo =====================================================
    echo MANTENIMIENTO COMPLETADO EXITOSAMENTE
    echo =====================================================
    echo.
    echo Los siguientes procesos se ejecutaron:
    echo - Creación de índices BRIN para tablas temporales
    echo - VACUUM ANALYZE en todas las tablas principales
    echo - Generación de estadísticas de mantenimiento
    echo.
) else (
    echo.
    echo =====================================================
    echo ERROR EN EL MANTENIMIENTO
    echo =====================================================
    echo.
    echo Hubo un error durante la ejecución del mantenimiento.
    echo Revisa los mensajes de error anteriores.
    echo.
)

echo Presiona cualquier tecla para continuar...
pause >nul

REM Opcional: Abrir log de resultados
set /p OPEN_LOG="¿Abrir archivo de log? (s/N): "
if /i "%OPEN_LOG%"=="s" (
    if exist "%~dp0db_maintenance.log" (
        notepad "%~dp0db_maintenance.log"
    ) else (
        echo No se encontró archivo de log.
    )
)

endlocal
