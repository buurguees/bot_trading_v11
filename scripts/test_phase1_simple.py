#!/usr/bin/env python3
"""
Test Phase 1 Simple
===================

Script de prueba simple para Phase 1 con un solo símbolo y timeframe.

Uso:
    python scripts/test_phase1_simple.py
"""

import os
import sys
import time
import logging
from datetime import datetime

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.ml.agents.agents_backfill import backfill_agent

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("TestPhase1Simple")

def test_phase1_simple():
    """Prueba simple de Phase 1 con un solo agente"""
    
    logger.info("🚀 Iniciando test simple de Phase 1")
    
    # Probar solo direction en 1m
    logger.info("=" * 60)
    logger.info("📊 PHASE 1 SIMPLE: Direction Agent (1m)")
    logger.info("=" * 60)
    
    start_time = time.time()
    try:
        # Probar solo direction con ventana pequeña
        total_predictions = backfill_agent("direction", window_days=7)  # Solo 7 días
        phase1_time = time.time() - start_time
        logger.info(f"✅ Phase 1 simple completada: {total_predictions} predicciones en {phase1_time:.1f}s")
        return True
    except Exception as e:
        logger.error(f"❌ Error en Phase 1 simple: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase1_simple()
    sys.exit(0 if success else 1)
