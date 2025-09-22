#!/usr/bin/env python3
"""
Test Trend Detection
===================

Script para probar la detección de tendencias en los logs.

Uso:
    python scripts/test_trend_detection.py
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("TestTrendDetection")

def test_trend_detection():
    """Prueba la detección de tendencias"""
    
    try:
        # Importar la clase GUI
        from scripts.data.gui_training_monitor import EnhancedTrainingMonitorGUI
        
        logger.info("🔍 Creando instancia del GUI...")
        
        # Crear instancia del GUI (sin mostrar ventana)
        gui = EnhancedTrainingMonitorGUI(refresh_sec=5)
        
        logger.info("✅ GUI creado exitosamente")
        
        # Obtener datos de logs
        logger.info("🔍 Obteniendo datos de logs...")
        logs_data = gui.fetch_phase1_logs()
        
        if logs_data.get("logs") is not None and not logs_data["logs"].empty:
            logs_df = logs_data["logs"]
            logger.info(f"✅ Obtenidos {len(logs_df)} logs")
            
            # Probar detección de tendencias
            logger.info("🔍 Probando detección de tendencias...")
            trends = gui._detect_trends(logs_df)
            
            if trends:
                logger.info("📈 TENDENCIAS DETECTADAS:")
                for i, trend in enumerate(trends, 1):
                    logger.info(f"  {i}. {trend}")
            else:
                logger.info("ℹ️ No se detectaron tendencias significativas")
            
            # Mostrar algunos logs de ejemplo
            logger.info("📝 LOGS DE EJEMPLO:")
            for _, row in logs_df.head(5).iterrows():
                timestamp = pd.to_datetime(row['created_at']).strftime("%H:%M:%S")
                data_ts = pd.to_datetime(row['ts']).strftime("%H:%M:%S") if pd.notna(row['ts']) else "N/A"
                agent = row['task']
                symbol = row['symbol']
                tf = row['timeframe']
                label = row['pred_label']
                conf = float(row['pred_conf']) if pd.notna(row['pred_conf']) else 0.0
                
                if data_ts != "N/A":
                    logger.info(f"  [{timestamp}] [{agent.upper()}] {symbol} {tf}: {label} (conf: {conf:.3f}) [data: {data_ts}]")
                else:
                    logger.info(f"  [{timestamp}] [{agent.upper()}] {symbol} {tf}: {label} (conf: {conf:.3f})")
        else:
            logger.warning("⚠️ No hay datos de logs disponibles")
        
        # Cerrar el GUI
        gui.destroy()
        
        logger.info("🎉 Test de detección de tendencias completado")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en test de tendencias: {e}")
        return False

if __name__ == "__main__":
    success = test_trend_detection()
    sys.exit(0 if success else 1)
