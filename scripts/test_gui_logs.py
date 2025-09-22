#!/usr/bin/env python3
"""
Test GUI Logs
=============

Script para probar la funcionalidad de logs del GUI sin abrir la ventana.

Uso:
    python scripts/test_gui_logs.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("TestGUILogs")

def test_gui_logs():
    """Prueba la funcionalidad de logs del GUI"""
    
    try:
        # Importar la clase GUI
        from scripts.data.gui_training_monitor import EnhancedTrainingMonitorGUI
        
        logger.info("🔍 Creando instancia del GUI...")
        
        # Crear instancia del GUI (sin mostrar ventana)
        gui = EnhancedTrainingMonitorGUI(refresh_sec=5)
        
        logger.info("✅ GUI creado exitosamente")
        
        # Verificar que el engine está inicializado
        if hasattr(gui, 'engine'):
            logger.info("✅ Engine de base de datos inicializado")
        else:
            logger.error("❌ Engine de base de datos NO inicializado")
            return False
        
        # Probar la función de fetch de logs
        logger.info("🔍 Probando fetch_phase1_logs...")
        
        try:
            logs_data = gui.fetch_phase1_logs()
            logger.info(f"✅ fetch_phase1_logs exitoso: {len(logs_data.get('logs', []))} logs")
            
            # Probar la función de actualización de logs
            logger.info("🔍 Probando update_phase1_logs_display...")
            gui.update_phase1_logs_display(logs_data)
            logger.info("✅ update_phase1_logs_display exitoso")
            
            # Probar la función de estadísticas
            logger.info("🔍 Probando update_phase1_stats...")
            gui.update_phase1_stats(logs_data)
            logger.info("✅ update_phase1_stats exitoso")
            
        except Exception as e:
            logger.error(f"❌ Error en funciones de logs: {e}")
            return False
        
        # Cerrar el GUI
        gui.destroy()
        
        logger.info("🎉 Test de GUI logs completado exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en test de GUI: {e}")
        return False

if __name__ == "__main__":
    success = test_gui_logs()
    sys.exit(0 if success else 1)
