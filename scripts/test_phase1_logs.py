#!/usr/bin/env python3
"""
Test Phase 1 Logs
=================

Script para probar la funcionalidad de logs de Phase 1.

Uso:
    python scripts/test_phase1_logs.py
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
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("TestPhase1Logs")

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

def test_phase1_logs():
    """Prueba la funcionalidad de logs de Phase 1"""
    
    if not DB_URL:
        logger.error("❌ No se encontró DB_URL en config/.env")
        return False
    
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    
    try:
        with engine.begin() as conn:
            # Simular la consulta del GUI
            logger.info("🔍 Probando consulta de logs de Phase 1...")
            
            logs_query = """
            SELECT 
                symbol,
                timeframe,
                task,
                pred_label,
                pred_conf,
                created_at,
                probs
            FROM ml.agent_preds
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
            LIMIT 1000
            """
            logs_df = pd.read_sql_query(logs_query, conn)
            
            logger.info(f"📊 Logs encontrados: {len(logs_df)}")
            
            if logs_df.empty:
                logger.warning("⚠️ No hay logs recientes")
                return False
            
            # Mostrar algunos ejemplos
            logger.info("📝 Ejemplos de logs:")
            for i, (_, row) in enumerate(logs_df.head(10).iterrows()):
                timestamp = pd.to_datetime(row['created_at']).strftime("%H:%M:%S")
                agent = row['task']
                symbol = row['symbol']
                tf = row['timeframe']
                label = row['pred_label']
                conf = float(row['pred_conf']) if pd.notna(row['pred_conf']) else 0.0
                probs = row.get('probs', '{}')
                
                # Determinar nivel de log basado en confianza
                if conf < 0.3:
                    level = "WARNING"
                elif conf < 0.1:
                    level = "ERROR"
                else:
                    level = "INFO"
                
                logger.info(f"  [{timestamp}] [{agent.upper()}] {symbol} {tf}: {label} (conf: {conf:.3f}) - {level}")
            
            # Estadísticas
            total_preds = len(logs_df)
            direction_preds = len(logs_df[logs_df['task'] == 'direction'])
            regime_preds = len(logs_df[logs_df['task'] == 'regime'])
            smc_preds = len(logs_df[logs_df['task'] == 'smc'])
            
            logger.info("\n📊 Estadísticas:")
            logger.info(f"  Total predicciones: {total_preds}")
            logger.info(f"  Direction: {direction_preds}")
            logger.info(f"  Regime: {regime_preds}")
            logger.info(f"  SMC: {smc_preds}")
            
            # Última actividad
            if not logs_df.empty:
                last_activity = pd.to_datetime(logs_df['created_at'].iloc[0]).strftime("%H:%M:%S")
                logger.info(f"  Última actividad: {last_activity}")
            
            logger.info("✅ Test de logs completado exitosamente")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error probando logs: {e}")
        return False

if __name__ == "__main__":
    success = test_phase1_logs()
    sys.exit(0 if success else 1)
