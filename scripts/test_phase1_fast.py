#!/usr/bin/env python3
"""
Test Phase 1 Rápido
===================

Versión rápida del backfill de agentes para testing.
Solo procesa los últimos 7 días de datos.

Uso:
    python scripts/test_phase1_fast.py
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.ml.agents.agents_backfill import backfill_agent

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("TestPhase1Fast")

def test_phase1_fast():
    """Prueba rápida del backfill de agentes (solo 7 días)"""
    
    logger.info("🚀 Iniciando test rápido de Phase 1 (7 días)")
    
    # Configurar variables de entorno para ventana pequeña
    os.environ["AGENTS_BACKFILL_WINDOW_DAYS"] = "7"
    os.environ["AGENTS_BACKFILL_CHUNK_DAYS"] = "1"
    os.environ["AGENTS_BACKFILL_TIMEFRAME"] = "1m"
    
    start_time = time.time()
    
    try:
        # Solo direction para testing rápido
        logger.info("🔄 Procesando direction agent...")
        direction_count = backfill_agent("direction", window_days=7)
        logger.info(f"✅ Direction: {direction_count} predicciones")
        
        # Solo regime para testing rápido
        logger.info("🔄 Procesando regime agent...")
        regime_count = backfill_agent("regime", window_days=7)
        logger.info(f"✅ Regime: {regime_count} predicciones")
        
        # Solo smc para testing rápido
        logger.info("🔄 Procesando smc agent...")
        smc_count = backfill_agent("smc", window_days=7)
        logger.info(f"✅ SMC: {smc_count} predicciones")
        
        total_time = time.time() - start_time
        total_predictions = direction_count + regime_count + smc_count
        
        logger.info("=" * 60)
        logger.info("📊 RESUMEN RÁPIDO")
        logger.info("=" * 60)
        logger.info(f"✅ Direction: {direction_count} predicciones")
        logger.info(f"✅ Regime: {regime_count} predicciones")
        logger.info(f"✅ SMC: {smc_count} predicciones")
        logger.info(f"📊 Total: {total_predictions} predicciones")
        logger.info(f"⏱️  Tiempo: {total_time:.1f}s")
        logger.info("🎉 Test rápido completado!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en test rápido: {e}")
        return False

if __name__ == "__main__":
    success = test_phase1_fast()
    sys.exit(0 if success else 1)
