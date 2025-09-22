#!/usr/bin/env python3
"""
Configurador para entrenamiento nocturno optimizado
"""

import os
import sys
import yaml
import argparse
import psutil
import logging
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NightTrainingConfigurator:
    """Configurador para entrenamiento nocturno"""
    
    def __init__(self, config_path: str = "config/ml/night_training.yaml"):
        self.config_path = config_path
        self.config = {}
        
    def load_config(self):
        """Cargar configuración existente"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            logger.info(f"Creando nueva configuración en {self.config_path}")
            self.config = {}
    
    def save_config(self):
        """Guardar configuración"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuración guardada en {self.config_path}")
    
    def detect_optimal_settings(self):
        """Detectar configuración óptima basada en el sistema"""
        logger.info("🔍 Detectando configuración óptima...")
        
        # Información del sistema
        cpu_count = os.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        disk_free_gb = psutil.disk_usage('/').free / (1024**3)
        
        logger.info(f"CPU cores: {cpu_count}")
        logger.info(f"Memoria total: {memory_gb:.1f} GB")
        logger.info(f"Espacio libre: {disk_free_gb:.1f} GB")
        
        # Calcular configuración óptima
        optimal_workers = max(1, min(cpu_count - 1, 8))
        max_memory_per_process = max(0.5, min(2.0, memory_gb / (optimal_workers * 2)))
        
        # Ajustar según recursos disponibles
        if memory_gb < 8:
            optimal_workers = min(optimal_workers, 2)
            max_memory_per_process = 1.0
        elif memory_gb < 16:
            optimal_workers = min(optimal_workers, 4)
            max_memory_per_process = 1.5
        else:
            max_memory_per_process = 2.0
        
        return {
            'num_workers': optimal_workers,
            'max_memory_per_process': max_memory_per_process,
            'memory_threshold': 0.85,
            'chunk_size': 50000 if memory_gb > 16 else 25000
        }
    
    def create_default_config(self):
        """Crear configuración por defecto"""
        optimal_settings = self.detect_optimal_settings()
        
        self.config = {
            'symbols': [
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 
                'SOLUSDT', 'DOGEUSDT', 'XRPUSDT'
            ],
            'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
            'horizons': [1, 3, 5],
            'days_back': 365,
            'min_rows': 1000,
            'dropna_cols_min_fraction': 0.8,
            'n_splits': 5,
            'embargo_minutes': 30,
            'model': {
                'kind': 'LogisticRegression',
                'max_iter': 2000,
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'class_weight': ['balanced', None],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'num_workers': optimal_settings['num_workers'],
            'max_memory_per_process': optimal_settings['max_memory_per_process'],
            'memory_threshold': optimal_settings['memory_threshold'],
            'retry_attempts': 3,
            'retry_delay': 5.0,
            'skip_threshold': 3,
            'promote_if': {
                'min_auc': 0.52,
                'max_brier': 0.26,
                'min_acc': 0.50
            },
            'artifacts_dir': 'artifacts/direction',
            'version_tag': f'night_v1.0.0_{datetime.now().strftime("%Y%m%d")}',
            'dashboard': {
                'enabled': True,
                'port': 5000,
                'host': '0.0.0.0',
                'refresh_interval': 5
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/batch_train_night.log',
                'max_size': '10MB',
                'backup_count': 5
            },
            'checkpoint': {
                'enabled': True,
                'interval': 30,
                'file': 'logs/night_training_checkpoint.json'
            },
            'cleanup': {
                'enabled': True,
                'interval': 60,
                'max_cache_size': 1000,
                'force_gc_interval': 300
            },
            'monitoring': {
                'memory_check_interval': 10,
                'process_restart_threshold': 0.9,
                'health_check_interval': 30
            },
            'symbol_priorities': {
                'BTCUSDT': 100,
                'ETHUSDT': 90,
                'ADAUSDT': 80,
                'SOLUSDT': 70,
                'DOGEUSDT': 60,
                'XRPUSDT': 50
            },
            'timeframe_resources': {
                '1m': {'max_memory': 1.5, 'priority_boost': 10},
                '5m': {'max_memory': 1.0, 'priority_boost': 5},
                '15m': {'max_memory': 0.8, 'priority_boost': 0},
                '1h': {'max_memory': 0.6, 'priority_boost': 0},
                '4h': {'max_memory': 0.4, 'priority_boost': 0},
                '1d': {'max_memory': 0.3, 'priority_boost': 0}
            },
            'alerts': {
                'enabled': True,
                'email': None,
                'webhook': None,
                'thresholds': {
                    'memory_usage': 0.9,
                    'cpu_usage': 0.95,
                    'disk_usage': 0.85,
                    'failed_jobs': 5
                }
            },
            'auto_scaling': {
                'enabled': True,
                'min_workers': 1,
                'max_workers': 8,
                'scale_up_threshold': 0.7,
                'scale_down_threshold': 0.3,
                'scale_cooldown': 60
            }
        }
    
    def optimize_for_system(self):
        """Optimizar configuración para el sistema actual"""
        logger.info("⚙️  Optimizando configuración para el sistema...")
        
        optimal_settings = self.detect_optimal_settings()
        
        # Actualizar configuración con valores óptimos
        self.config['num_workers'] = optimal_settings['num_workers']
        self.config['max_memory_per_process'] = optimal_settings['max_memory_per_process']
        self.config['memory_threshold'] = optimal_settings['memory_threshold']
        
        # Ajustar chunk size según memoria
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            self.config['cleanup']['max_cache_size'] = 500
        elif memory_gb < 16:
            self.config['cleanup']['max_cache_size'] = 1000
        else:
            self.config['cleanup']['max_cache_size'] = 2000
        
        # Ajustar número de splits según recursos
        if memory_gb < 8:
            self.config['n_splits'] = 3
        elif memory_gb < 16:
            self.config['n_splits'] = 5
        else:
            self.config['n_splits'] = 7
        
        logger.info(f"Workers configurados: {self.config['num_workers']}")
        logger.info(f"Memoria por proceso: {self.config['max_memory_per_process']} GB")
        logger.info(f"Splits de validación: {self.config['n_splits']}")
    
    def add_symbol(self, symbol: str, priority: int = 50):
        """Agregar símbolo a la configuración"""
        if 'symbols' not in self.config:
            self.config['symbols'] = []
        
        if symbol not in self.config['symbols']:
            self.config['symbols'].append(symbol)
            self.config['symbol_priorities'][symbol] = priority
            logger.info(f"Símbolo {symbol} agregado con prioridad {priority}")
        else:
            logger.info(f"Símbolo {symbol} ya existe")
    
    def remove_symbol(self, symbol: str):
        """Remover símbolo de la configuración"""
        if 'symbols' in self.config and symbol in self.config['symbols']:
            self.config['symbols'].remove(symbol)
            if 'symbol_priorities' in self.config:
                self.config['symbol_priorities'].pop(symbol, None)
            logger.info(f"Símbolo {symbol} removido")
        else:
            logger.info(f"Símbolo {symbol} no encontrado")
    
    def set_timeframes(self, timeframes: list):
        """Configurar timeframes"""
        self.config['timeframes'] = timeframes
        logger.info(f"Timeframes configurados: {timeframes}")
    
    def set_horizons(self, horizons: list):
        """Configurar horizontes"""
        self.config['horizons'] = horizons
        logger.info(f"Horizontes configurados: {horizons}")
    
    def enable_dashboard(self, port: int = 5000):
        """Habilitar dashboard"""
        if 'dashboard' not in self.config:
            self.config['dashboard'] = {}
        
        self.config['dashboard']['enabled'] = True
        self.config['dashboard']['port'] = port
        logger.info(f"Dashboard habilitado en puerto {port}")
    
    def disable_dashboard(self):
        """Deshabilitar dashboard"""
        if 'dashboard' not in self.config:
            self.config['dashboard'] = {}
        
        self.config['dashboard']['enabled'] = False
        logger.info("Dashboard deshabilitado")
    
    def set_promotion_thresholds(self, min_auc: float, max_brier: float, min_acc: float):
        """Configurar umbrales de promoción"""
        if 'promote_if' not in self.config:
            self.config['promote_if'] = {}
        
        self.config['promote_if']['min_auc'] = min_auc
        self.config['promote_if']['max_brier'] = max_brier
        self.config['promote_if']['min_acc'] = min_acc
        
        logger.info(f"Umbrales de promoción: AUC>={min_auc}, Brier<={max_brier}, Acc>={min_acc}")
    
    def validate_config(self):
        """Validar configuración"""
        logger.info("🔍 Validando configuración...")
        
        issues = []
        
        # Verificar campos requeridos
        required_fields = ['symbols', 'timeframes', 'horizons', 'num_workers']
        for field in required_fields:
            if field not in self.config:
                issues.append(f"Campo requerido faltante: {field}")
        
        # Verificar símbolos
        if 'symbols' in self.config:
            if not isinstance(self.config['symbols'], list) or len(self.config['symbols']) == 0:
                issues.append("Lista de símbolos vacía o inválida")
        
        # Verificar timeframes
        if 'timeframes' in self.config:
            valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            for tf in self.config['timeframes']:
                if tf not in valid_timeframes:
                    issues.append(f"Timeframe inválido: {tf}")
        
        # Verificar workers
        if 'num_workers' in self.config:
            if self.config['num_workers'] < 1 or self.config['num_workers'] > 16:
                issues.append("Número de workers debe estar entre 1 y 16")
        
        # Verificar memoria
        if 'max_memory_per_process' in self.config:
            if self.config['max_memory_per_process'] < 0.1 or self.config['max_memory_per_process'] > 10:
                issues.append("Memoria por proceso debe estar entre 0.1 y 10 GB")
        
        if issues:
            logger.error("❌ Problemas encontrados:")
            for issue in issues:
                logger.error(f"  • {issue}")
            return False
        else:
            logger.info("✅ Configuración válida")
            return True
    
    def estimate_runtime(self):
        """Estimar tiempo de ejecución"""
        if not self.validate_config():
            return None
        
        symbols = len(self.config.get('symbols', []))
        timeframes = len(self.config.get('timeframes', []))
        horizons = len(self.config.get('horizons', []))
        workers = self.config.get('num_workers', 1)
        
        total_combinations = symbols * timeframes * horizons
        
        # Estimación basada en recursos
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = os.cpu_count()
        
        # Tiempo base por combinación (minutos)
        base_time = 2.0
        if memory_gb < 8:
            base_time *= 1.5
        if cpu_count < 4:
            base_time *= 1.3
        
        # Ajustar por paralelización
        parallel_efficiency = min(0.8, workers / cpu_count)
        parallel_time = base_time / (workers * parallel_efficiency)
        
        total_minutes = total_combinations * parallel_time
        total_hours = total_minutes / 60
        
        logger.info(f"📊 ESTIMACIÓN DE TIEMPO:")
        logger.info(f"  Combinaciones: {total_combinations}")
        logger.info(f"  Workers: {workers}")
        logger.info(f"  Tiempo estimado: {total_hours:.1f} horas ({total_minutes:.0f} minutos)")
        
        return {
            'combinations': total_combinations,
            'workers': workers,
            'estimated_hours': total_hours,
            'estimated_minutes': total_minutes
        }

def main():
    parser = argparse.ArgumentParser(description="Configurador de entrenamiento nocturno")
    parser.add_argument("-c", "--config", default="config/ml/night_training.yaml",
                       help="Archivo de configuración")
    parser.add_argument("--create", action="store_true",
                       help="Crear configuración por defecto")
    parser.add_argument("--optimize", action="store_true",
                       help="Optimizar para el sistema actual")
    parser.add_argument("--add-symbol", type=str,
                       help="Agregar símbolo")
    parser.add_argument("--remove-symbol", type=str,
                       help="Remover símbolo")
    parser.add_argument("--set-timeframes", nargs='+',
                       help="Configurar timeframes")
    parser.add_argument("--set-horizons", type=int, nargs='+',
                       help="Configurar horizontes")
    parser.add_argument("--enable-dashboard", type=int, metavar='PORT',
                       help="Habilitar dashboard en puerto")
    parser.add_argument("--disable-dashboard", action="store_true",
                       help="Deshabilitar dashboard")
    parser.add_argument("--set-thresholds", nargs=3, type=float, metavar=('AUC', 'BRIER', 'ACC'),
                       help="Configurar umbrales de promoción")
    parser.add_argument("--validate", action="store_true",
                       help="Validar configuración")
    parser.add_argument("--estimate", action="store_true",
                       help="Estimar tiempo de ejecución")
    
    args = parser.parse_args()
    
    configurator = NightTrainingConfigurator(args.config)
    configurator.load_config()
    
    if args.create:
        configurator.create_default_config()
        configurator.save_config()
    
    if args.optimize:
        configurator.optimize_for_system()
        configurator.save_config()
    
    if args.add_symbol:
        priority = 50
        if 'symbol_priorities' in configurator.config:
            priority = max(configurator.config['symbol_priorities'].values()) + 10
        configurator.add_symbol(args.add_symbol, priority)
        configurator.save_config()
    
    if args.remove_symbol:
        configurator.remove_symbol(args.remove_symbol)
        configurator.save_config()
    
    if args.set_timeframes:
        configurator.set_timeframes(args.set_timeframes)
        configurator.save_config()
    
    if args.set_horizons:
        configurator.set_horizons(args.set_horizons)
        configurator.save_config()
    
    if args.enable_dashboard is not None:
        configurator.enable_dashboard(args.enable_dashboard)
        configurator.save_config()
    
    if args.disable_dashboard:
        configurator.disable_dashboard()
        configurator.save_config()
    
    if args.set_thresholds:
        configurator.set_promotion_thresholds(*args.set_thresholds)
        configurator.save_config()
    
    if args.validate:
        if configurator.validate_config():
            print("✅ Configuración válida")
            sys.exit(0)
        else:
            print("❌ Configuración inválida")
            sys.exit(1)
    
    if args.estimate:
        configurator.estimate_runtime()

if __name__ == "__main__":
    main()
