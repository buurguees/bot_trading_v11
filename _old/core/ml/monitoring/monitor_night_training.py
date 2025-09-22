#!/usr/bin/env python3
"""
Monitor especializado para entrenamiento nocturno
"""

import os
import sys
import time
import json
import argparse
import psutil
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NightTrainingMonitor:
    """Monitor especializado para entrenamiento nocturno"""
    
    def __init__(self, dashboard_url: str = "http://localhost:5000"):
        self.dashboard_url = dashboard_url
        self.log_file = "logs/batch_train_night.log"
        self.checkpoint_file = "logs/night_training_checkpoint.json"
        
    def get_system_metrics(self):
        """Obtener métricas del sistema"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error obteniendo métricas del sistema: {e}")
            return None
    
    def get_dashboard_status(self):
        """Obtener estado del dashboard"""
        try:
            response = requests.get(f"{self.dashboard_url}/api/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            logger.debug(f"Dashboard no disponible: {e}")
            return None
    
    def get_log_status(self):
        """Analizar estado desde logs"""
        if not os.path.exists(self.log_file):
            return None
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Buscar información relevante en las últimas líneas
            recent_lines = lines[-100:]  # Últimas 100 líneas
            
            status = {
                'completed': 0,
                'failed': 0,
                'running': 0,
                'last_activity': None
            }
            
            for line in recent_lines:
                if 'Completado:' in line or '✅' in line:
                    status['completed'] += 1
                elif 'Fallido:' in line or '❌' in line:
                    status['failed'] += 1
                elif 'Iniciado trabajo:' in line:
                    status['running'] += 1
                
                # Extraer timestamp de la línea
                if 'INFO' in line:
                    try:
                        timestamp_str = line.split(' - ')[0]
                        status['last_activity'] = timestamp_str
                    except:
                        pass
            
            return status
            
        except Exception as e:
            logger.error(f"Error analizando logs: {e}")
            return None
    
    def get_checkpoint_status(self):
        """Obtener estado del checkpoint"""
        if not os.path.exists(self.checkpoint_file):
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error leyendo checkpoint: {e}")
            return None
    
    def find_training_processes(self):
        """Encontrar procesos de entrenamiento activos"""
        training_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'batch_train' in cmdline or 'night_train' in cmdline:
                    training_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline,
                        'memory_mb': proc.memory_info().rss / (1024*1024),
                        'cpu_percent': proc.cpu_percent(),
                        'create_time': proc.create_time()
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return training_processes
    
    def display_status(self, detailed: bool = False):
        """Mostrar estado del entrenamiento nocturno"""
        print("🌙 MONITOR DE ENTRENAMIENTO NOCTURNO")
        print("=" * 50)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Métricas del sistema
        system_metrics = self.get_system_metrics()
        if system_metrics:
            print("📊 MÉTRICAS DEL SISTEMA:")
            print(f"  CPU: {system_metrics['cpu_percent']:.1f}%")
            print(f"  RAM: {system_metrics['memory_used_gb']:.1f}GB / {system_metrics['memory_total_gb']:.1f}GB ({system_metrics['memory_percent']:.1f}%)")
            print(f"  Disco: {system_metrics['disk_free_gb']:.1f}GB libres ({system_metrics['disk_percent']:.1f}% usado)")
            print()
        
        # Estado del dashboard
        dashboard_status = self.get_dashboard_status()
        if dashboard_status:
            print("🌐 DASHBOARD DISPONIBLE:")
            print(f"  URL: {self.dashboard_url}")
            print(f"  Estado: Activo")
            print()
        else:
            print("🌐 DASHBOARD: No disponible")
            print()
        
        # Procesos de entrenamiento
        training_processes = self.find_training_processes()
        if training_processes:
            print("🚀 PROCESOS DE ENTRENAMIENTO:")
            for proc in training_processes:
                runtime = time.time() - proc['create_time']
                print(f"  PID {proc['pid']}: {proc['memory_mb']:.0f}MB RAM, {proc['cpu_percent']:.1f}% CPU, {runtime:.0f}s")
            print()
        else:
            print("🚀 PROCESOS: No encontrados")
            print()
        
        # Estado desde logs
        log_status = self.get_log_status()
        if log_status:
            print("📋 ESTADO DESDE LOGS:")
            print(f"  Completados: {log_status['completed']}")
            print(f"  Fallidos: {log_status['failed']}")
            print(f"  En progreso: {log_status['running']}")
            if log_status['last_activity']:
                print(f"  Última actividad: {log_status['last_activity']}")
            print()
        
        # Estado del checkpoint
        checkpoint_status = self.get_checkpoint_status()
        if checkpoint_status:
            print("💾 CHECKPOINT:")
            print(f"  Archivo: {self.checkpoint_file}")
            print(f"  Tamaño: {os.path.getsize(self.checkpoint_file)} bytes")
            print()
        
        if detailed:
            self.display_detailed_status()
    
    def display_detailed_status(self):
        """Mostrar estado detallado"""
        print("📈 ESTADO DETALLADO:")
        print("-" * 30)
        
        # Análisis de logs recientes
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                
                recent_lines = lines[-20:]  # Últimas 20 líneas
                print("📝 ACTIVIDAD RECIENTE:")
                for line in recent_lines[-10:]:  # Mostrar solo las últimas 10
                    print(f"  {line.strip()}")
                print()
                
            except Exception as e:
                print(f"  Error leyendo logs: {e}")
        
        # Análisis de archivos de artefactos
        artifacts_dir = Path("artifacts/direction")
        if artifacts_dir.exists():
            artifacts = list(artifacts_dir.glob("*.pkl"))
            print(f"🎯 ARTEFACTOS GENERADOS: {len(artifacts)}")
            for artifact in artifacts[-5:]:  # Últimos 5
                size_mb = artifact.stat().st_size / (1024*1024)
                mtime = datetime.fromtimestamp(artifact.stat().st_mtime)
                print(f"  {artifact.name}: {size_mb:.1f}MB ({mtime.strftime('%H:%M:%S')})")
            print()
    
    def monitor_continuous(self, interval: int = 30):
        """Monitoreo continuo"""
        print(f"🔄 Iniciando monitoreo continuo (cada {interval}s)")
        print("Presiona Ctrl+C para salir")
        print()
        
        try:
            while True:
                # Limpiar pantalla
                os.system('cls' if os.name == 'nt' else 'clear')
                
                self.display_status()
                
                # Verificar si el entrenamiento sigue activo
                training_processes = self.find_training_processes()
                if not training_processes:
                    print("⚠️  No se detectan procesos de entrenamiento activos")
                    print("   El entrenamiento puede haber terminado o fallado")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoreo detenido por el usuario")
    
    def check_health(self):
        """Verificar salud del sistema"""
        print("🏥 VERIFICACIÓN DE SALUD DEL SISTEMA")
        print("=" * 40)
        
        issues = []
        
        # Verificar métricas del sistema
        system_metrics = self.get_system_metrics()
        if system_metrics:
            if system_metrics['cpu_percent'] > 95:
                issues.append(f"CPU muy alto: {system_metrics['cpu_percent']:.1f}%")
            
            if system_metrics['memory_percent'] > 90:
                issues.append(f"Memoria muy alta: {system_metrics['memory_percent']:.1f}%")
            
            if system_metrics['disk_percent'] > 85:
                issues.append(f"Disco muy lleno: {system_metrics['disk_percent']:.1f}%")
        
        # Verificar procesos
        training_processes = self.find_training_processes()
        if not training_processes:
            issues.append("No hay procesos de entrenamiento activos")
        
        # Verificar logs
        if not os.path.exists(self.log_file):
            issues.append("Archivo de logs no encontrado")
        else:
            # Verificar si los logs están actualizándose
            log_mtime = os.path.getmtime(self.log_file)
            if time.time() - log_mtime > 300:  # 5 minutos sin actividad
                issues.append("Logs sin actividad reciente (>5 min)")
        
        if issues:
            print("❌ PROBLEMAS DETECTADOS:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✅ Sistema saludable")
        
        return len(issues) == 0

def main():
    parser = argparse.ArgumentParser(description="Monitor de entrenamiento nocturno")
    parser.add_argument("--dashboard-url", default="http://localhost:5000",
                       help="URL del dashboard")
    parser.add_argument("--detailed", action="store_true",
                       help="Mostrar información detallada")
    parser.add_argument("--continuous", action="store_true",
                       help="Monitoreo continuo")
    parser.add_argument("--interval", type=int, default=30,
                       help="Intervalo de monitoreo continuo (segundos)")
    parser.add_argument("--health", action="store_true",
                       help="Verificar salud del sistema")
    
    args = parser.parse_args()
    
    monitor = NightTrainingMonitor(args.dashboard_url)
    
    if args.health:
        monitor.check_health()
    elif args.continuous:
        monitor.monitor_continuous(args.interval)
    else:
        monitor.display_status(args.detailed)

if __name__ == "__main__":
    main()
