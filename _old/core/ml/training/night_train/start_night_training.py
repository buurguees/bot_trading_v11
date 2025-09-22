#!/usr/bin/env python3
"""
Script de inicio para entrenamiento nocturno optimizado
"""

import os
import sys
import argparse
import subprocess
import time
import signal
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/night_training_launcher.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class NightTrainingLauncher:
    """Lanzador para entrenamiento nocturno con gestión de procesos"""
    
    def __init__(self, config_path: str, dashboard: bool = True, port: int = 5000):
        self.config_path = config_path
        self.dashboard = dashboard
        self.port = port
        self.process = None
        self.start_time = None
        
    def start_training(self):
        """Iniciar el proceso de entrenamiento"""
        logger.info("🚀 Iniciando entrenamiento nocturno optimizado...")
        
        # Verificar configuración
        if not os.path.exists(self.config_path):
            logger.error(f"Archivo de configuración no encontrado: {self.config_path}")
            return False
        
        # Crear directorios necesarios
        os.makedirs("logs", exist_ok=True)
        os.makedirs("artifacts/direction", exist_ok=True)
        
        # Comando de entrenamiento
        cmd = [
            sys.executable, "-m", "core.ml.training.night_train.batch_train",
            "-c", self.config_path
        ]
        
        if self.dashboard:
            cmd.extend(["--dashboard", "--port", str(self.port)])
        
        # Iniciar proceso
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.start_time = time.time()
            logger.info(f"Proceso iniciado con PID: {self.process.pid}")
            
            if self.dashboard:
                logger.info(f"Dashboard disponible en: http://localhost:{self.port}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando proceso: {e}")
            return False
    
    def monitor_process(self):
        """Monitorear el proceso de entrenamiento"""
        if not self.process:
            logger.error("No hay proceso para monitorear")
            return
        
        logger.info("📊 Monitoreando proceso de entrenamiento...")
        
        try:
            while self.process.poll() is None:
                # Leer salida en tiempo real
                if self.process.stdout.readable():
                    line = self.process.stdout.readline()
                    if line:
                        print(line.strip())
                
                # Verificar cada segundo
                time.sleep(1)
            
            # Proceso terminado
            return_code = self.process.returncode
            if return_code == 0:
                logger.info("✅ Entrenamiento completado exitosamente")
            else:
                logger.error(f"❌ Entrenamiento falló con código: {return_code}")
                
                # Mostrar errores
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    logger.error(f"Errores:\n{stderr_output}")
            
            return return_code == 0
            
        except KeyboardInterrupt:
            logger.info("Interrupción recibida, terminando proceso...")
            self.stop_training()
            return False
        except Exception as e:
            logger.error(f"Error monitoreando proceso: {e}")
            return False
    
    def stop_training(self):
        """Detener el proceso de entrenamiento"""
        if self.process:
            logger.info("Deteniendo proceso de entrenamiento...")
            
            # Enviar SIGTERM primero
            self.process.terminate()
            
            # Esperar terminación graceful
            try:
                self.process.wait(timeout=30)
                logger.info("Proceso terminado gracefulmente")
            except subprocess.TimeoutExpired:
                logger.warning("Proceso no terminó en 30s, forzando...")
                self.process.kill()
                self.process.wait()
                logger.info("Proceso terminado forzadamente")
    
    def get_status(self):
        """Obtener estado del proceso"""
        if not self.process:
            return "No iniciado"
        
        if self.process.poll() is None:
            runtime = time.time() - self.start_time if self.start_time else 0
            return f"Ejecutándose (PID: {self.process.pid}, Tiempo: {runtime:.0f}s)"
        else:
            return f"Terminado (Código: {self.process.returncode})"

def setup_signal_handlers(launcher):
    """Configurar manejadores de señales"""
    def signal_handler(signum, frame):
        logger.info(f"Señal {signum} recibida, terminando...")
        launcher.stop_training()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def check_system_requirements():
    """Verificar requisitos del sistema"""
    logger.info("🔍 Verificando requisitos del sistema...")
    
    # Verificar memoria
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        logger.info(f"Memoria total: {total_gb:.1f} GB")
        logger.info(f"Memoria disponible: {available_gb:.1f} GB")
        
        if total_gb < 8:
            logger.warning("⚠️  Menos de 8GB de RAM puede causar problemas")
        else:
            logger.info("✅ Memoria suficiente")
            
    except ImportError:
        logger.error("❌ psutil no instalado")
        return False
    
    # Verificar CPU
    cpu_count = os.cpu_count()
    logger.info(f"CPU cores: {cpu_count}")
    
    if cpu_count < 2:
        logger.warning("⚠️  Menos de 2 cores puede ser lento")
    else:
        logger.info("✅ CPU suficiente")
    
    # Verificar espacio en disco
    disk_usage = psutil.disk_usage('/')
    free_gb = disk_usage.free / (1024**3)
    logger.info(f"Espacio libre en disco: {free_gb:.1f} GB")
    
    if free_gb < 10:
        logger.warning("⚠️  Menos de 10GB libres puede causar problemas")
    else:
        logger.info("✅ Espacio en disco suficiente")
    
    return True

def estimate_runtime(config_path: str):
    """Estimar tiempo de ejecución basado en configuración"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        symbols = len(config.get('symbols', []))
        timeframes = len(config.get('timeframes', []))
        horizons = len(config.get('horizons', []))
        
        total_combinations = symbols * timeframes * horizons
        
        # Estimación: 2-5 minutos por combinación
        min_time = total_combinations * 2
        max_time = total_combinations * 5
        
        logger.info(f"📊 Estimación de tiempo:")
        logger.info(f"  Combinaciones: {total_combinations}")
        logger.info(f"  Tiempo mínimo: {min_time} minutos ({min_time/60:.1f} horas)")
        logger.info(f"  Tiempo máximo: {max_time} minutos ({max_time/60:.1f} horas)")
        
        return total_combinations
        
    except Exception as e:
        logger.warning(f"No se pudo estimar tiempo: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Lanzador de entrenamiento nocturno")
    parser.add_argument("-c", "--config", default="config/ml/night_training.yaml", 
                       help="Archivo de configuración")
    parser.add_argument("--no-dashboard", action="store_true", 
                       help="Deshabilitar dashboard web")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Puerto del dashboard")
    parser.add_argument("--check-requirements", action="store_true",
                       help="Solo verificar requisitos del sistema")
    parser.add_argument("--estimate", action="store_true",
                       help="Solo estimar tiempo de ejecución")
    
    args = parser.parse_args()
    
    logger.info("🌙 LANZADOR DE ENTRENAMIENTO NOCTURNO")
    logger.info("=" * 50)
    
    # Verificar requisitos
    if args.check_requirements:
        if check_system_requirements():
            logger.info("✅ Todos los requisitos cumplidos")
            return 0
        else:
            logger.error("❌ Requisitos no cumplidos")
            return 1
    
    # Estimar tiempo
    if args.estimate:
        estimate_runtime(args.config)
        return 0
    
    # Verificar requisitos básicos
    if not check_system_requirements():
        logger.error("❌ Requisitos del sistema no cumplidos")
        return 1
    
    # Crear lanzador
    launcher = NightTrainingLauncher(
        config_path=args.config,
        dashboard=not args.no_dashboard,
        port=args.port
    )
    
    # Configurar manejadores de señales
    setup_signal_handlers(launcher)
    
    # Estimar tiempo de ejecución
    estimate_runtime(args.config)
    
    # Iniciar entrenamiento
    if not launcher.start_training():
        logger.error("❌ No se pudo iniciar el entrenamiento")
        return 1
    
    # Monitorear proceso
    success = launcher.monitor_process()
    
    if success:
        logger.info("🎉 Entrenamiento nocturno completado exitosamente")
        return 0
    else:
        logger.error("💥 Entrenamiento nocturno falló")
        return 1

if __name__ == "__main__":
    sys.exit(main())
