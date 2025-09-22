#!/usr/bin/env python3
"""
Ejecutar GUI Monitor
===================

Script para ejecutar el GUI de monitoreo de forma robusta.

Uso:
    python scripts/run_gui_monitor.py
"""

import os
import sys
import logging
import signal
import time

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("RunGUIMonitor")

def main():
    """Función principal"""
    
    try:
        logger.info("🚀 Iniciando GUI Training Monitor...")
        
        # Importar y ejecutar el GUI
        from scripts.data.gui_training_monitor import main as gui_main
        
        # Configurar manejo de señales para cierre limpio
        def signal_handler(signum, frame):
            logger.info("🛑 Cerrando GUI...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Ejecutar el GUI
        gui_main()
        
    except KeyboardInterrupt:
        logger.info("🛑 GUI cerrado por usuario")
    except Exception as e:
        logger.error(f"❌ Error ejecutando GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
