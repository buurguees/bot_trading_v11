#!/usr/bin/env python3
"""
Verificar Estructura de Tablas
==============================

Script para verificar la estructura de las tablas de la base de datos.

Uso:
    python scripts/check_table_structure.py
"""

import os
import sys
import logging

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("CheckTableStructure")

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

def check_table_structure():
    """Verifica la estructura de las tablas"""
    
    if not DB_URL:
        logger.error("❌ No se encontró DB_URL en config/.env")
        return False
    
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    
    try:
        with engine.begin() as conn:
            # Verificar estructura de ml.agent_preds
            logger.info("🔍 Verificando estructura de ml.agent_preds...")
            
            columns_query = text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'ml' AND table_name = 'agent_preds'
                ORDER BY ordinal_position
            """)
            columns = conn.execute(columns_query).mappings().all()
            
            logger.info("📊 Columnas de ml.agent_preds:")
            for col in columns:
                logger.info(f"  {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})")
            
            # Verificar estructura de trading.trade_plans
            logger.info("\n🔍 Verificando estructura de trading.trade_plans...")
            
            plans_columns_query = text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'trading' AND table_name = 'trade_plans'
                ORDER BY ordinal_position
            """)
            plans_columns = conn.execute(plans_columns_query).mappings().all()
            
            logger.info("📊 Columnas de trading.trade_plans:")
            for col in plans_columns:
                logger.info(f"  {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})")
            
            # Verificar estructura de ml.strategies
            logger.info("\n🔍 Verificando estructura de ml.strategies...")
            
            strategies_columns_query = text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'ml' AND table_name = 'strategies'
                ORDER BY ordinal_position
            """)
            strategies_columns = conn.execute(strategies_columns_query).mappings().all()
            
            logger.info("📊 Columnas de ml.strategies:")
            for col in strategies_columns:
                logger.info(f"  {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})")
            
            logger.info("\n✅ Verificación de estructura completada")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error verificando estructura: {e}")
        return False

if __name__ == "__main__":
    success = check_table_structure()
    sys.exit(0 if success else 1)
