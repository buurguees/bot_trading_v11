#!/usr/bin/env python3
"""
Generate Historical Trade Plans
===============================

Script para generar trade_plans históricos usando las predicciones existentes.

Uso:
    python scripts/generate_historical_plans.py
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.trading.planner_backfill import backfill as planner_backfill
from core.research.strategy_miner import mine_candidates

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("GenerateHistoricalPlans")

def generate_historical_plans():
    """Genera trade_plans históricos usando predicciones existentes"""
    
    logger.info("🚀 Generando trade_plans históricos...")
    
    # 1. Planner Backfill (usar predicciones existentes)
    logger.info("=" * 60)
    logger.info("📅 GENERANDO TRADE PLANS HISTÓRICOS")
    logger.info("=" * 60)
    
    start_time = time.time()
    try:
        # Limitar a 30 días para evitar sobrecarga
        total_plans = planner_backfill(window_days=30)
        planner_time = time.time() - start_time
        logger.info(f"✅ Planner backfill completado: {total_plans} planes en {planner_time:.1f}s")
    except Exception as e:
        logger.error(f"❌ Error en Planner backfill: {e}")
        return False
    
    # 2. Strategy Mining
    logger.info("=" * 60)
    logger.info("🔍 MINERÍA DE ESTRATEGIAS")
    logger.info("=" * 60)
    
    start_time = time.time()
    try:
        total_strategies = mine_candidates()
        mining_time = time.time() - start_time
        logger.info(f"✅ Minería completada: {total_strategies} estrategias en {mining_time:.1f}s")
    except Exception as e:
        logger.error(f"❌ Error en minería: {e}")
        return False
    
    # Resumen final
    total_time = planner_time + mining_time
    logger.info("=" * 60)
    logger.info("📊 RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info(f"✅ Trade Plans: {total_plans} planes en {planner_time:.1f}s")
    logger.info(f"✅ Estrategias: {total_strategies} estrategias en {mining_time:.1f}s")
    logger.info(f"⏱️  Tiempo total: {total_time:.1f}s")
    logger.info("🎉 Proceso completado exitosamente!")
    
    return True

if __name__ == "__main__":
    success = generate_historical_plans()
    sys.exit(0 if success else 1)
