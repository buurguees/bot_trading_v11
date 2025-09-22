#!/bin/bash
# ============================================================================
# SCRIPT DE MANTENIMIENTO DE BASE DE DATOS
# Ejecuta optimizaciones con índices BRIN y VACUUM/ANALYZE
# ============================================================================

set -e  # Salir si hay algún error

echo ""
echo "============================================================================"
echo "MANTENIMIENTO DE BASE DE DATOS - TRADING BOT"
echo "============================================================================"
echo ""

# Verificar que psql esté disponible
if ! command -v psql &> /dev/null; then
    echo "ERROR: psql no encontrado. Asegúrate de que PostgreSQL esté instalado."
    echo ""
    exit 1
fi

# Configurar variables de entorno
export PGHOST=192.168.10.109
export PGPORT=5432
export PGDATABASE=trading_db
export PGUSER=trading_user
export PGPASSWORD=160501

echo "Iniciando mantenimiento de base de datos..."
echo "Host: $PGHOST"
echo "Database: $PGDATABASE"
echo "Usuario: $PGUSER"
echo ""

# Crear directorio de logs si no existe
mkdir -p logs

# Generar nombre de archivo de log con timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/db_maintenance_${timestamp}.log"

echo "Ejecutando script de mantenimiento..."
echo "Log guardado en: $LOG_FILE"
echo ""

# Ejecutar script SQL y guardar log
if psql -h "$PGHOST" -p "$PGPORT" -d "$PGDATABASE" -U "$PGUSER" -f "db_maintenance.sql" > "$LOG_FILE" 2>&1; then
    echo ""
    echo "============================================================================"
    echo "MANTENIMIENTO COMPLETADO EXITOSAMENTE"
    echo "============================================================================"
    echo ""
    echo "Log guardado en: $LOG_FILE"
    echo ""
    echo "Resumen de la ejecución:"
    echo "- Índices BRIN creados para tablas grandes"
    echo "- VACUUM ANALYZE ejecutado en todas las tablas"
    echo "- Estadísticas actualizadas"
    echo "- Datos verificados"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "ERROR EN EL MANTENIMIENTO"
    echo "============================================================================"
    echo ""
    echo "Código de error: $?"
    echo "Revisa el log en: $LOG_FILE"
    echo ""
fi

# Mostrar últimas líneas del log
echo "Últimas líneas del log:"
echo "----------------------------------------------------------------------------"
tail -n 20 "$LOG_FILE"
echo "----------------------------------------------------------------------------"
echo ""

# Preguntar si abrir el log completo
read -p "¿Abrir el log completo? (s/n): " open_log
if [[ "$open_log" =~ ^[Ss]$ ]]; then
    if command -v nano &> /dev/null; then
        nano "$LOG_FILE"
    elif command -v vim &> /dev/null; then
        vim "$LOG_FILE"
    else
        cat "$LOG_FILE"
    fi
fi

echo ""
echo "Mantenimiento finalizado."
