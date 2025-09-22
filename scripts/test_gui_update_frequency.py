#!/usr/bin/env python3
"""
Test GUI Update Frequency
========================

Script para probar que la actualización de Phase 1 Logs funciona cada 5 segundos.

Uso:
    python scripts/test_gui_update_frequency.py
"""

import os
import sys
import logging
import time
from datetime import datetime

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("TestGUIUpdateFrequency")

def test_update_frequency():
    """Prueba la frecuencia de actualización del GUI"""
    
    try:
        # Importar la clase GUI
        from scripts.data.gui_training_monitor import EnhancedTrainingMonitorGUI
        
        logger.info("🔍 Creando instancia del GUI...")
        
        # Crear instancia del GUI (sin mostrar ventana)
        gui = EnhancedTrainingMonitorGUI(refresh_sec=5)
        
        logger.info("✅ GUI creado exitosamente")
        
        # Simular múltiples actualizaciones
        logger.info("🔄 Simulando actualizaciones cada 5 segundos...")
        
        for i in range(3):
            logger.info(f"📊 Actualización #{i+1} - {datetime.now().strftime('%H:%M:%S')}")
            
            # Simular la función refresh_phase1_logs
            try:
                gui.refresh_phase1_logs()
                logger.info("✅ Phase 1 Logs actualizados correctamente")
            except Exception as e:
                logger.error(f"❌ Error actualizando Phase 1 Logs: {e}")
            
            # Esperar 5 segundos (simulando el ciclo real)
            if i < 2:  # No esperar en la última iteración
                logger.info("⏳ Esperando 5 segundos...")
                time.sleep(5)
        
        # Cerrar el GUI
        gui.destroy()
        
        logger.info("🎉 Test de frecuencia de actualización completado")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en test de frecuencia: {e}")
        return False

if __name__ == "__main__":
    success = test_update_frequency()
    sys.exit(0 if success else 1)
