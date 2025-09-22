#!/usr/bin/env python3
"""
Script de mantenimiento de base de datos
Ejecuta optimizaciones con índices BRIN y VACUUM/ANALYZE
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configurar logging
def setup_logging():
    """Configurar el sistema de logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"db_maintenance_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def check_psql():
    """Verificar que psql esté disponible"""
    try:
        result = subprocess.run(['psql', '--version'], 
                              capture_output=True, text=True, check=True)
        logging.info(f"PostgreSQL encontrado: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("ERROR: psql no encontrado. Asegúrate de que PostgreSQL esté instalado.")
        return False

def run_maintenance():
    """Ejecutar el script de mantenimiento"""
    # Configurar variables de entorno
    env = os.environ.copy()
    env.update({
        'PGHOST': '192.168.10.109',
        'PGPORT': '5432',
        'PGDATABASE': 'trading_db',
        'PGUSER': 'trading_user',
        'PGPASSWORD': '160501'
    })
    
    # Ruta al script SQL
    script_path = Path(__file__).parent / "db_maintenance.sql"
    
    if not script_path.exists():
        logging.error(f"ERROR: No se encontró el script {script_path}")
        return False
    
    logging.info("Iniciando mantenimiento de base de datos...")
    logging.info(f"Host: {env['PGHOST']}")
    logging.info(f"Database: {env['PGDATABASE']}")
    logging.info(f"Usuario: {env['PGUSER']}")
    
    try:
        # Ejecutar psql con el script
        result = subprocess.run([
            'psql',
            '-h', env['PGHOST'],
            '-p', env['PGPORT'],
            '-d', env['PGDATABASE'],
            '-U', env['PGUSER'],
            '-f', str(script_path)
        ], env=env, capture_output=True, text=True, check=True)
        
        logging.info("MANTENIMIENTO COMPLETADO EXITOSAMENTE")
        logging.info("Resumen de la ejecución:")
        logging.info("- Índices BRIN creados para tablas grandes")
        logging.info("- VACUUM ANALYZE ejecutado en todas las tablas")
        logging.info("- Estadísticas actualizadas")
        logging.info("- Datos verificados")
        
        # Mostrar salida de psql
        if result.stdout:
            logging.info("Salida de psql:")
            logging.info(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR EN EL MANTENIMIENTO: {e}")
        logging.error(f"Código de error: {e.returncode}")
        if e.stderr:
            logging.error(f"Error: {e.stderr}")
        if e.stdout:
            logging.error(f"Salida: {e.stdout}")
        return False

def main():
    """Función principal"""
    print("=" * 80)
    print("MANTENIMIENTO DE BASE DE DATOS - TRADING BOT")
    print("=" * 80)
    print()
    
    # Configurar logging
    log_file = setup_logging()
    logging.info(f"Log guardado en: {log_file}")
    
    # Verificar psql
    if not check_psql():
        sys.exit(1)
    
    # Ejecutar mantenimiento
    if run_maintenance():
        print("\n" + "=" * 80)
        print("MANTENIMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print(f"Log guardado en: {log_file}")
        print()
    else:
        print("\n" + "=" * 80)
        print("ERROR EN EL MANTENIMIENTO")
        print("=" * 80)
        print(f"Revisa el log en: {log_file}")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()
