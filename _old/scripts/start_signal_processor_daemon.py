#!/usr/bin/env python3
"""
Daemon para ejecutar el procesador de señales continuamente
"""

import sys
import os
import time
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from core.ml.signals.signal_processor import create_signal_processor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/signal_processor_daemon.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class SignalProcessorDaemon:
    """Daemon para ejecutar el procesador de señales continuamente"""
    
    def __init__(self, config_path: str = "config/signals/signal_processing.yaml", interval: int = 30):
        self.processor = create_signal_processor(config_path)
        self.interval = interval
        self.running = False
        self.stats = {
            "total_runs": 0,
            "total_predictions": 0,
            "total_signals": 0,
            "total_errors": 0,
            "start_time": None
        }
        
        # Configurar manejadores de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Manejar señales de terminación"""
        logger.info(f"Señal {signum} recibida, deteniendo daemon...")
        self.running = False
    
    def start(self):
        """Iniciar el daemon"""
        logger.info("🚀 Iniciando Signal Processor Daemon")
        logger.info(f"Intervalo: {self.interval} segundos")
        logger.info("Presiona Ctrl+C para detener")
        
        self.running = True
        self.stats["start_time"] = datetime.now(timezone.utc)
        
        try:
            while self.running:
                try:
                    # Procesar señales
                    run_start = time.time()
                    stats = self.processor.process_realtime_predictions()
                    run_time = time.time() - run_start
                    
                    # Actualizar estadísticas
                    self.stats["total_runs"] += 1
                    self.stats["total_predictions"] += stats.get("predictions", 0)
                    self.stats["total_signals"] += stats.get("signals", 0)
                    self.stats["total_errors"] += stats.get("errors", 0)
                    
                    # Log del resultado
                    logger.info(
                        f"Run #{self.stats['total_runs']}: "
                        f"predicciones={stats.get('predictions', 0)}, "
                        f"señales={stats.get('signals', 0)}, "
                        f"guardadas={stats.get('saved_signals', 0)}, "
                        f"errores={stats.get('errors', 0)}, "
                        f"tiempo={run_time:.2f}s"
                    )
                    
                    # Log de estadísticas cada 10 runs
                    if self.stats["total_runs"] % 10 == 0:
                        self._log_summary_stats()
                    
                except Exception as e:
                    logger.error(f"Error en procesamiento: {e}")
                    self.stats["total_errors"] += 1
                
                # Esperar antes del siguiente ciclo
                if self.running:
                    time.sleep(self.interval)
                    
        except KeyboardInterrupt:
            logger.info("Interrupción recibida")
        finally:
            self.stop()
    
    def stop(self):
        """Detener el daemon"""
        self.running = False
        self._log_final_stats()
        logger.info("🛑 Signal Processor Daemon detenido")
    
    def _log_summary_stats(self):
        """Log de estadísticas resumidas"""
        if self.stats["start_time"]:
            uptime = datetime.now(timezone.utc) - self.stats["start_time"]
            avg_predictions = self.stats["total_predictions"] / self.stats["total_runs"] if self.stats["total_runs"] > 0 else 0
            avg_signals = self.stats["total_signals"] / self.stats["total_runs"] if self.stats["total_runs"] > 0 else 0
            error_rate = (self.stats["total_errors"] / self.stats["total_runs"]) * 100 if self.stats["total_runs"] > 0 else 0
            
            logger.info(
                f"📊 Estadísticas (últimos 10 runs): "
                f"uptime={uptime}, "
                f"runs={self.stats['total_runs']}, "
                f"avg_predicciones={avg_predictions:.1f}, "
                f"avg_señales={avg_signals:.1f}, "
                f"error_rate={error_rate:.1f}%"
            )
    
    def _log_final_stats(self):
        """Log de estadísticas finales"""
        if self.stats["start_time"]:
            uptime = datetime.now(timezone.utc) - self.stats["start_time"]
            avg_predictions = self.stats["total_predictions"] / self.stats["total_runs"] if self.stats["total_runs"] > 0 else 0
            avg_signals = self.stats["total_signals"] / self.stats["total_runs"] if self.stats["total_runs"] > 0 else 0
            error_rate = (self.stats["total_errors"] / self.stats["total_runs"]) * 100 if self.stats["total_runs"] > 0 else 0
            
            logger.info("📊 ESTADÍSTICAS FINALES:")
            logger.info(f"  Tiempo activo: {uptime}")
            logger.info(f"  Total de runs: {self.stats['total_runs']}")
            logger.info(f"  Total de predicciones: {self.stats['total_predictions']}")
            logger.info(f"  Total de señales: {self.stats['total_signals']}")
            logger.info(f"  Total de errores: {self.stats['total_errors']}")
            logger.info(f"  Promedio de predicciones/run: {avg_predictions:.1f}")
            logger.info(f"  Promedio de señales/run: {avg_signals:.1f}")
            logger.info(f"  Tasa de error: {error_rate:.1f}%")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Processor Daemon")
    parser.add_argument("--config", default="config/signals/signal_processing.yaml",
                       help="Ruta del archivo de configuración")
    parser.add_argument("--interval", type=int, default=30,
                       help="Intervalo en segundos entre procesamientos")
    
    args = parser.parse_args()
    
    # Crear y ejecutar daemon
    daemon = SignalProcessorDaemon(args.config, args.interval)
    daemon.start()

if __name__ == "__main__":
    main()
