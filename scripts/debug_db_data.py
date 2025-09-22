#!/usr/bin/env python3
"""
Debug Database Data
==================

Script para verificar qué datos hay en la base de datos.

Uso:
    python scripts/debug_db_data.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.data.database import create_engine
from sqlalchemy import text

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("DebugDB")

def debug_db_data():
    """Verifica los datos en la base de datos"""
    
    logger.info("🔍 Verificando datos en la base de datos...")
    
    try:
        engine = create_engine(os.getenv("DB_URL"), pool_pre_ping=True, future=True)
        
        with engine.begin() as conn:
            # 1. Verificar historical_data
            logger.info("=" * 50)
            logger.info("📊 HISTORICAL DATA")
            logger.info("=" * 50)
            
            hist_query = """
            SELECT 
                symbol, 
                timeframe, 
                COUNT(*) as count,
                MIN(ts) as first_ts,
                MAX(ts) as last_ts
            FROM market.historical_data 
            GROUP BY symbol, timeframe 
            ORDER BY symbol, timeframe
            """
            
            hist_results = conn.execute(text(hist_query)).mappings().all()
            for row in hist_results:
                logger.info(f"  {row['symbol']} {row['timeframe']}: {row['count']} registros ({row['first_ts']} → {row['last_ts']})")
            
            # 2. Verificar features
            logger.info("=" * 50)
            logger.info("🔧 FEATURES")
            logger.info("=" * 50)
            
            feat_query = """
            SELECT 
                symbol, 
                timeframe, 
                COUNT(*) as count,
                MIN(ts) as first_ts,
                MAX(ts) as last_ts
            FROM market.features 
            GROUP BY symbol, timeframe 
            ORDER BY symbol, timeframe
            """
            
            feat_results = conn.execute(text(feat_query)).mappings().all()
            for row in feat_results:
                logger.info(f"  {row['symbol']} {row['timeframe']}: {row['count']} registros ({row['first_ts']} → {row['last_ts']})")
            
            # 3. Verificar agent_preds
            logger.info("=" * 50)
            logger.info("🤖 AGENT PREDS")
            logger.info("=" * 50)
            
            pred_query = """
            SELECT 
                symbol, 
                timeframe, 
                task,
                COUNT(*) as count,
                MIN(created_at) as first_ts,
                MAX(created_at) as last_ts
            FROM ml.agent_preds 
            GROUP BY symbol, timeframe, task
            ORDER BY symbol, timeframe, task
            """
            
            pred_results = conn.execute(text(pred_query)).mappings().all()
            if pred_results:
                for row in pred_results:
                    logger.info(f"  {row['symbol']} {row['timeframe']} {row['task']}: {row['count']} predicciones ({row['first_ts']} → {row['last_ts']})")
            else:
                logger.info("  ❌ No hay predicciones de agentes")
            
            # 4. Verificar trade_plans
            logger.info("=" * 50)
            logger.info("📋 TRADE PLANS")
            logger.info("=" * 50)
            
            plan_query = """
            SELECT 
                symbol, 
                timeframe, 
                COUNT(*) as count,
                MIN(ts) as first_ts,
                MAX(ts) as last_ts
            FROM trading.trade_plans 
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
            """
            
            plan_results = conn.execute(text(plan_query)).mappings().all()
            if plan_results:
                for row in plan_results:
                    logger.info(f"  {row['symbol']} {row['timeframe']}: {row['count']} planes ({row['first_ts']} → {row['last_ts']})")
            else:
                logger.info("  ❌ No hay trade plans")
            
            # 5. Verificar strategies
            logger.info("=" * 50)
            logger.info("🎯 STRATEGIES")
            logger.info("=" * 50)
            
            strat_query = """
            SELECT 
                symbol, 
                timeframe, 
                status,
                COUNT(*) as count
            FROM ml.strategies 
            GROUP BY symbol, timeframe, status
            ORDER BY symbol, timeframe, status
            """
            
            strat_results = conn.execute(text(strat_query)).mappings().all()
            if strat_results:
                for row in strat_results:
                    logger.info(f"  {row['symbol']} {row['timeframe']} {row['status']}: {row['count']} estrategias")
            else:
                logger.info("  ❌ No hay estrategias")
        
        logger.info("✅ Verificación completada")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verificando BD: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_db_data()
    sys.exit(0 if success else 1)
