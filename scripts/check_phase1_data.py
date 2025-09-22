#!/usr/bin/env python3
"""
Verificar Datos Phase 1
=======================

Script para verificar que hay predicciones de agentes en la base de datos.

Uso:
    python scripts/check_phase1_data.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("CheckPhase1Data")

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

def check_phase1_data():
    """Verifica los datos de Phase 1 en la base de datos"""
    
    if not DB_URL:
        logger.error("❌ No se encontró DB_URL en config/.env")
        return False
    
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    
    try:
        with engine.begin() as conn:
            # Verificar predicciones de agentes
            logger.info("🔍 Verificando predicciones de agentes...")
            
            # Contar total de predicciones
            total_query = text("SELECT COUNT(*) as total FROM ml.agent_preds")
            total_result = conn.execute(total_query).scalar()
            logger.info(f"📊 Total predicciones: {total_result}")
            
            if total_result == 0:
                logger.warning("⚠️ No hay predicciones en ml.agent_preds")
                return False
            
            # Predicciones por agente
            agent_query = text("""
                SELECT task, COUNT(*) as count, 
                       MIN(created_at) as first_pred,
                       MAX(created_at) as last_pred
                FROM ml.agent_preds 
                GROUP BY task
                ORDER BY task
            """)
            agent_results = conn.execute(agent_query).mappings().all()
            
            logger.info("📈 Predicciones por agente:")
            for row in agent_results:
                logger.info(f"  {row['task']}: {row['count']} predicciones")
                logger.info(f"    Primera: {row['first_pred']}")
                logger.info(f"    Última: {row['last_pred']}")
            
            # Predicciones recientes (últimas 24h)
            recent_query = text("""
                SELECT COUNT(*) as recent_count
                FROM ml.agent_preds
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            recent_count = conn.execute(recent_query).scalar()
            logger.info(f"🕐 Predicciones últimas 24h: {recent_count}")
            
            # Predicciones por símbolo
            symbol_query = text("""
                SELECT symbol, COUNT(*) as count
                FROM ml.agent_preds
                GROUP BY symbol
                ORDER BY count DESC
                LIMIT 10
            """)
            symbol_results = conn.execute(symbol_query).mappings().all()
            
            logger.info("🏷️ Top 10 símbolos por predicciones:")
            for row in symbol_results:
                logger.info(f"  {row['symbol']}: {row['count']} predicciones")
            
            # Verificar trade_plans
            logger.info("\n🔍 Verificando trade_plans...")
            plans_query = text("SELECT COUNT(*) as total FROM trading.trade_plans")
            plans_count = conn.execute(plans_query).scalar()
            logger.info(f"📋 Total trade_plans: {plans_count}")
            
            if plans_count > 0:
                plans_recent_query = text("""
                    SELECT COUNT(*) as recent_plans
                    FROM trading.trade_plans
                    WHERE ts >= NOW() - INTERVAL '24 hours'
                """)
                recent_plans = conn.execute(plans_recent_query).scalar()
                logger.info(f"🕐 Trade_plans últimas 24h: {recent_plans}")
            
            # Verificar estrategias
            logger.info("\n🔍 Verificando estrategias...")
            strategies_query = text("SELECT COUNT(*) as total FROM ml.strategies")
            strategies_count = conn.execute(strategies_query).scalar()
            logger.info(f"🎯 Total estrategias: {strategies_count}")
            
            if strategies_count > 0:
                strategies_status_query = text("""
                    SELECT status, COUNT(*) as count
                    FROM ml.strategies
                    GROUP BY status
                    ORDER BY count DESC
                """)
                status_results = conn.execute(strategies_status_query).mappings().all()
                
                logger.info("📊 Estrategias por estado:")
                for row in status_results:
                    logger.info(f"  {row['status']}: {row['count']} estrategias")
            
            logger.info("\n✅ Verificación completada")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error verificando datos: {e}")
        return False

if __name__ == "__main__":
    success = check_phase1_data()
    sys.exit(0 if success else 1)
