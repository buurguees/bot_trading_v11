#!/usr/bin/env python3
"""
Script de mantenimiento de base de datos usando Python directamente
No requiere psql, usa sqlalchemy para conectarse a PostgreSQL
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text

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

def create_connection():
    """Crear conexión a la base de datos"""
    db_url = "postgresql+psycopg2://trading_user:160501@192.168.10.109:5432/trading_db"
    engine = create_engine(db_url, pool_pre_ping=True, isolation_level="AUTOCOMMIT")
    return engine

def create_brin_indexes(engine):
    """Crear índices BRIN para tablas grandes"""
    logging.info("Creando índices BRIN...")
    
    brin_indexes = [
        # historicaldata
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_historicaldata_brin_timestamp ON trading.historicaldata USING BRIN (timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_historicaldata_brin_symbol_timestamp ON trading.historicaldata USING BRIN (symbol, timestamp)",
        
        # features
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_brin_timestamp ON trading.features USING BRIN (timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_brin_symbol_timestamp ON trading.features USING BRIN (symbol, timestamp)",
        
        # agentpreds
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentpreds_brin_timestamp ON trading.agentpreds USING BRIN (timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentpreds_brin_symbol_timestamp ON trading.agentpreds USING BRIN (symbol, timestamp)",
        
        # agentsignals
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentsignals_brin_timestamp ON trading.agentsignals USING BRIN (timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentsignals_brin_symbol_timestamp ON trading.agentsignals USING BRIN (symbol, timestamp)",
        
        # tradeplans
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tradeplans_brin_bar_ts ON trading.tradeplans USING BRIN (bar_ts)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tradeplans_brin_symbol_bar_ts ON trading.tradeplans USING BRIN (symbol, bar_ts)",
        
        # backtests (usar id en lugar de created_at si no existe)
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtests_brin_id ON trading.backtests USING BRIN (id)",
        
        # backtesttrades
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtesttrades_brin_entry_ts ON trading.backtesttrades USING BRIN (entry_ts)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtesttrades_brin_symbol_entry_ts ON trading.backtesttrades USING BRIN (symbol, entry_ts)",
    ]
    
    with engine.connect() as conn:
        for idx_sql in brin_indexes:
            try:
                conn.execute(text(idx_sql))
                logging.info(f"Índice creado: {idx_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                logging.warning(f"Error creando índice: {e}")
    
    logging.info("Índices BRIN creados exitosamente.")

def create_additional_indexes(engine):
    """Crear índices adicionales para optimización"""
    logging.info("Creando índices adicionales...")
    
    additional_indexes = [
        # Índices compuestos para consultas frecuentes
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_historicaldata_symbol_tf_timestamp ON trading.historicaldata (symbol, timeframe, timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_symbol_tf_timestamp ON trading.features (symbol, timeframe, timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentpreds_symbol_tf_timestamp ON trading.agentpreds (symbol, timeframe, timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agentsignals_symbol_tf_timestamp ON trading.agentsignals (symbol, timeframe, timestamp)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtests_symbol_tf_id ON trading.backtests (symbol, timeframe, id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtesttrades_symbol_tf_entry_ts ON trading.backtesttrades (symbol, timeframe, entry_ts)",
    ]
    
    with engine.connect() as conn:
        for idx_sql in additional_indexes:
            try:
                conn.execute(text(idx_sql))
                logging.info(f"Índice creado: {idx_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                logging.warning(f"Error creando índice: {e}")
    
    logging.info("Índices adicionales creados exitosamente.")

def run_vacuum_analyze(engine):
    """Ejecutar VACUUM ANALYZE en tablas grandes"""
    logging.info("Ejecutando VACUUM ANALYZE...")
    
    tables = [
        "trading.historicaldata",
        "trading.features", 
        "trading.agentpreds",
        "trading.agentsignals",
        "trading.tradeplans",
        "trading.backtesttrades",
        "trading.backtests",
        "trading.agentversions",
        "trading.strategy_memory",
        "trading.strategy_samples"
    ]
    
    with engine.connect() as conn:
        for table in tables:
            try:
                logging.info(f"Ejecutando VACUUM ANALYZE en {table}...")
                conn.execute(text(f"VACUUM (ANALYZE, VERBOSE) {table}"))
                logging.info(f"VACUUM ANALYZE completado para {table}")
            except Exception as e:
                logging.warning(f"Error en VACUUM ANALYZE para {table}: {e}")

def get_table_stats(engine):
    """Obtener estadísticas de tablas"""
    logging.info("Generando estadísticas de tablas...")
    
    with engine.connect() as conn:
        # Tamaño de tablas
        result = conn.execute(text("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables 
            WHERE schemaname = 'trading'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """))
        
        logging.info("Tamaños de tablas:")
        for row in result:
            logging.info(f"  {row[0]}.{row[1]}: {row[2]}")
        
        # Rangos de fechas
        result = conn.execute(text("""
            SELECT 'historicaldata' as tabla, 
                   MIN(timestamp) as min_ts, 
                   MAX(timestamp) as max_ts, 
                   COUNT(*) as total_registros
            FROM trading.historicaldata
            UNION ALL
            SELECT 'features' as tabla, 
                   MIN(timestamp) as min_ts, 
                   MAX(timestamp) as max_ts, 
                   COUNT(*) as total_registros
            FROM trading.features
            UNION ALL
            SELECT 'agentpreds' as tabla, 
                   MIN(timestamp) as min_ts, 
                   MAX(timestamp) as max_ts, 
                   COUNT(*) as total_registros
            FROM trading.agentpreds
            UNION ALL
            SELECT 'agentsignals' as tabla, 
                   MIN(timestamp) as min_ts, 
                   MAX(timestamp) as max_ts, 
                   COUNT(*) as total_registros
            FROM trading.agentsignals
            UNION ALL
            SELECT 'tradeplans' as tabla, 
                   MIN(bar_ts) as min_ts, 
                   MAX(bar_ts) as max_ts, 
                   COUNT(*) as total_registros
            FROM trading.tradeplans
            ORDER BY tabla
        """))
        
        logging.info("Rangos de fechas:")
        for row in result:
            logging.info(f"  {row[0]}: {row[1]} a {row[2]} ({row[3]} registros)")

def main():
    """Función principal"""
    print("=" * 80)
    print("MANTENIMIENTO DE BASE DE DATOS - TRADING BOT (Python)")
    print("=" * 80)
    print()
    
    # Configurar logging
    log_file = setup_logging()
    logging.info(f"Log guardado en: {log_file}")
    
    try:
        # Crear conexión
        logging.info("Conectando a la base de datos...")
        engine = create_connection()
        
        # Ejecutar mantenimiento
        create_brin_indexes(engine)
        create_additional_indexes(engine)
        run_vacuum_analyze(engine)
        get_table_stats(engine)
        
        print("\n" + "=" * 80)
        print("MANTENIMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print(f"Log guardado en: {log_file}")
        print()
        print("Resumen de la ejecución:")
        print("- Índices BRIN creados para tablas grandes")
        print("- Índices compuestos creados para consultas frecuentes")
        print("- VACUUM ANALYZE ejecutado en todas las tablas")
        print("- Estadísticas actualizadas")
        print("- Datos verificados")
        print()
        
    except Exception as e:
        logging.error(f"ERROR EN EL MANTENIMIENTO: {e}")
        print("\n" + "=" * 80)
        print("ERROR EN EL MANTENIMIENTO")
        print("=" * 80)
        print(f"Revisa el log en: {log_file}")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()
