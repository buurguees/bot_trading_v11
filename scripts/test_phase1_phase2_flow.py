#!/usr/bin/env python3
"""
Test Phase 1 → Phase 2 Flow
===========================

Script para probar el flujo completo:
1. Phase 1: Backfill de agentes históricos
2. Phase 2: Backfill de planner + minería de estrategias

Uso:
    python scripts/test_phase1_phase2_flow.py
"""

import os
import sys
import time
import logging
from datetime import datetime

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.data.database import create_engine
from core.ml.agents.agents_backfill import backfill_all_agents_all_tfs
from core.trading.planner_backfill import backfill as planner_backfill
from core.research.strategy_miner import mine_candidates

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("TestPhaseFlow")

def test_phase1_phase2_flow():
    """Prueba el flujo completo Phase 1 → Phase 2"""
    
    logger.info("🚀 Iniciando test de flujo Phase 1 → Phase 2")
    
    # 1. Phase 1: Backfill de agentes históricos
    logger.info("=" * 60)
    logger.info("📊 PHASE 1: Backfill de Agentes Históricos")
    logger.info("=" * 60)
    
    start_time = time.time()
    try:
        total_predictions = backfill_all_agents_all_tfs()
        phase1_time = time.time() - start_time
        logger.info(f"✅ Phase 1 completada: {total_predictions} predicciones en {phase1_time:.1f}s")
    except Exception as e:
        logger.error(f"❌ Error en Phase 1: {e}")
        return False
    
    # 2. Phase 2: Backfill de planner
    logger.info("=" * 60)
    logger.info("📅 PHASE 2A: Backfill de Planner Histórico")
    logger.info("=" * 60)
    
    start_time = time.time()
    try:
        total_plans = planner_backfill()
        planner_time = time.time() - start_time
        logger.info(f"✅ Planner backfill completado: {total_plans} planes en {planner_time:.1f}s")
    except Exception as e:
        logger.error(f"❌ Error en Planner backfill: {e}")
        return False
    
    # 3. Phase 2: Minería de estrategias
    logger.info("=" * 60)
    logger.info("🔍 PHASE 2B: Minería de Estrategias Históricas")
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
    total_time = phase1_time + planner_time + mining_time
    logger.info("=" * 60)
    logger.info("📊 RESUMEN FINAL")
    logger.info("=" * 60)
    logger.info(f"✅ Phase 1 (Agentes): {total_predictions} predicciones en {phase1_time:.1f}s")
    logger.info(f"✅ Phase 2A (Planner): {total_plans} planes en {planner_time:.1f}s")
    logger.info(f"✅ Phase 2B (Minería): {total_strategies} estrategias en {mining_time:.1f}s")
    logger.info(f"⏱️  Tiempo total: {total_time:.1f}s")
    logger.info("🎉 Flujo completo ejecutado exitosamente!")
    
    return True

if __name__ == "__main__":
    success = test_phase1_phase2_flow()
    sys.exit(0 if success else 1)
