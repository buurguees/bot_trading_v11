"""
Feature Updater shim
--------------------
Envuelve feature_engineer para correr en modo tiempo real continuo.
"""

import time
import logging
from core.config.config_loader import load_feature_updater_config

logger = logging.getLogger("FeatureUpdater")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

def main(poll_sec: int = 5):
    """
    Función principal del feature updater.
    Intenta usar feature_engineer en modo realtime, con fallback a modo batch.
    """
    # Cargar configuración
    config = load_feature_updater_config()
    shim_cfg = config.get("feature_updater", {}).get("shim", {})
    engineer_cfg = config.get("feature_updater", {}).get("engineer", {})
    
    poll_interval = shim_cfg.get("poll_interval_sec", poll_sec)
    fallback_enabled = shim_cfg.get("fallback_to_engineer", True)
    
    logger.info(f"Iniciando feature updater (poll_interval={poll_interval}s)")
    
    try:
        # Intentar usar feature_engineer en modo realtime
        from core.ml.feature_engineer import run_realtime_loop, run_realtime_once
        if callable(run_realtime_loop):
            logger.info("Usando feature_engineer.run_realtime_loop()")
            run_realtime_loop(interval_sec=poll_interval)
        elif callable(run_realtime_once):
            logger.info("Usando feature_engineer.run_realtime_once() en loop")
            while True:
                try:
                    run_realtime_once()
                    logger.debug("Feature calculation cycle completed")
                except Exception as e:
                    logger.exception(f"Error en feature calculation: {e}")
                time.sleep(poll_interval)
        else:
            raise ImportError("No se encontraron funciones de realtime en feature_engineer")
            
    except Exception as e:
        if fallback_enabled:
            logger.warning(f"Error con feature_engineer realtime: {e}")
            logger.info("Fallback: usando feature_engineer.run() en loop")
            
            from core.ml.feature_engineer import run as run_batch
            
            while True:
                try:
                    run_batch()
                    logger.debug("Feature calculation cycle completed (batch mode)")
                except Exception as e:
                    logger.exception(f"Error en feature calculation (batch): {e}")
                time.sleep(poll_interval)
        else:
            logger.error(f"Error crítico en feature updater: {e}")
            raise

if __name__ == "__main__":
    main(5)