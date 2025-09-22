#!/bin/bash
# =====================================================
# MANTENIMIENTO DE BASE DE DATOS - BOT TRADING V11
# =====================================================
# Script para ejecutar mantenimiento de BD en Linux/Mac
# Incluye índices BRIN y VACUUM/ANALYZE
# =====================================================

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con colores
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo
    echo "====================================================="
    echo "MANTENIMIENTO DE BASE DE DATOS - BOT TRADING V11"
    echo "====================================================="
    echo
}

# Configuración de conexión (ajustar según tu entorno)
DB_HOST="${DB_HOST:-192.168.10.109}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-trading_db}"
DB_USER="${DB_USER:-trading_user}"
DB_PASSWORD="${DB_PASSWORD:-160501}"

# Archivo SQL de mantenimiento
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_FILE="${SCRIPT_DIR}/db_maintenance.sql"
LOG_FILE="${SCRIPT_DIR}/db_maintenance_$(date +%Y%m%d_%H%M%S).log"

print_header

# Verificar si psql está disponible
if ! command -v psql &> /dev/null; then
    print_error "psql no encontrado en el PATH"
    print_info "Por favor instala PostgreSQL o agrega psql al PATH"
    exit 1
fi

# Verificar que el archivo SQL existe
if [[ ! -f "$SQL_FILE" ]]; then
    print_error "No se encontró el archivo $SQL_FILE"
    exit 1
fi

print_info "Configuración de conexión:"
print_info "- Host: $DB_HOST"
print_info "- Puerto: $DB_PORT"
print_info "- Base de datos: $DB_NAME"
print_info "- Usuario: $DB_USER"
print_info "- Archivo SQL: $SQL_FILE"
print_info "- Archivo de log: $LOG_FILE"
echo

# Preguntar confirmación
read -p "¿Continuar con el mantenimiento? (s/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    print_info "Operación cancelada."
    exit 0
fi

print_info "Iniciando mantenimiento de base de datos..."
echo "====================================================="

# Función para ejecutar comandos con logging
run_with_log() {
    local cmd="$1"
    local description="$2"
    
    print_info "Ejecutando: $description"
    echo "--- $description ---" >> "$LOG_FILE"
    echo "Comando: $cmd" >> "$LOG_FILE"
    echo "Timestamp: $(date)" >> "$LOG_FILE"
    echo >> "$LOG_FILE"
    
    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        print_success "$description completado"
    else
        print_error "Error en: $description"
        print_info "Revisa el log: $LOG_FILE"
        return 1
    fi
}

# Ejecutar el script SQL con logging
run_with_log \
    "PGPASSWORD='$DB_PASSWORD' psql -h '$DB_HOST' -p '$DB_PORT' -U '$DB_USER' -d '$DB_NAME' -f '$SQL_FILE'" \
    "Script de mantenimiento de base de datos"

if [[ $? -eq 0 ]]; then
    echo
    echo "====================================================="
    print_success "MANTENIMIENTO COMPLETADO EXITOSAMENTE"
    echo "====================================================="
    echo
    print_info "Los siguientes procesos se ejecutaron:"
    print_info "- Creación de índices BRIN para tablas temporales"
    print_info "- VACUUM ANALYZE en todas las tablas principales"
    print_info "- Generación de estadísticas de mantenimiento"
    echo
    print_info "Log guardado en: $LOG_FILE"
    echo
else
    echo
    echo "====================================================="
    print_error "ERROR EN EL MANTENIMIENTO"
    echo "====================================================="
    echo
    print_error "Hubo un error durante la ejecución del mantenimiento."
    print_info "Revisa el log: $LOG_FILE"
    echo
    exit 1
fi

# Opcional: Mostrar resumen del log
if [[ -f "$LOG_FILE" ]]; then
    print_info "Resumen del log:"
    echo "--- Últimas 20 líneas del log ---"
    tail -n 20 "$LOG_FILE"
    echo "--- Fin del resumen ---"
    echo
fi

# Opcional: Abrir log de resultados
read -p "¿Abrir archivo de log completo? (s/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    if command -v less &> /dev/null; then
        less "$LOG_FILE"
    elif command -v cat &> /dev/null; then
        cat "$LOG_FILE"
    else
        print_warning "No se encontró un visor de texto disponible"
    fi
fi

print_success "Script de mantenimiento finalizado"
