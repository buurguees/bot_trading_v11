#!/usr/bin/env python3
"""
Script para validar la configuración de entrenamiento
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingConfigValidator:
    """Validador de configuración de entrenamiento"""
    
    def __init__(self, config_path: str = "config/ml/training.yaml"):
        self.config_path = config_path
        self.config = None
        self.errors = []
        self.warnings = []
    
    def load_config(self) -> bool:
        """Cargar configuración desde archivo"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✅ Configuración cargada desde {self.config_path}")
            return True
        except FileNotFoundError:
            self.errors.append(f"❌ Archivo no encontrado: {self.config_path}")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"❌ Error de sintaxis YAML: {e}")
            return False
        except Exception as e:
            self.errors.append(f"❌ Error cargando configuración: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validar estructura básica de la configuración"""
        required_sections = [
            'resources', 'training', 'models', 'monitoring', 
            'balance', 'backtesting', 'features', 'alerts'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in self.config:
                missing_sections.append(section)
        
        if missing_sections:
            self.errors.append(f"❌ Secciones faltantes: {missing_sections}")
            return False
        
        logger.info("✅ Estructura de configuración válida")
        return True
    
    def validate_resources(self) -> bool:
        """Validar configuración de recursos"""
        resources = self.config.get('resources', {})
        valid = True
        
        # Validar memoria
        max_memory = resources.get('max_memory_gb', 0)
        if not isinstance(max_memory, (int, float)) or max_memory <= 0:
            self.errors.append("❌ max_memory_gb debe ser un número positivo")
            valid = False
        elif max_memory > 64:
            self.warnings.append("⚠️  max_memory_gb muy alto (>64GB)")
        
        # Validar workers
        max_workers = resources.get('max_workers', 0)
        if not isinstance(max_workers, int) or max_workers <= 0:
            self.errors.append("❌ max_workers debe ser un entero positivo")
            valid = False
        elif max_workers > 16:
            self.warnings.append("⚠️  max_workers muy alto (>16)")
        
        # Validar chunk size
        chunk_size = resources.get('chunk_size', 0)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            self.errors.append("❌ chunk_size debe ser un entero positivo")
            valid = False
        elif chunk_size > 1000000:
            self.warnings.append("⚠️  chunk_size muy alto (>1M)")
        
        # Validar cache size
        cache_size = resources.get('cache_size_gb', 0)
        if not isinstance(cache_size, (int, float)) or cache_size < 0:
            self.errors.append("❌ cache_size_gb debe ser un número no negativo")
            valid = False
        elif cache_size > max_memory * 0.5:
            self.warnings.append("⚠️  cache_size_gb muy alto comparado con max_memory_gb")
        
        if valid:
            logger.info("✅ Configuración de recursos válida")
        
        return valid
    
    def validate_training(self) -> bool:
        """Validar configuración de entrenamiento"""
        training = self.config.get('training', {})
        valid = True
        
        # Validar validación
        validation = training.get('validation', {})
        n_splits = validation.get('n_splits', 0)
        if not isinstance(n_splits, int) or n_splits < 2:
            self.errors.append("❌ n_splits debe ser un entero >= 2")
            valid = False
        
        embargo = validation.get('embargo_minutes', 0)
        if not isinstance(embargo, (int, float)) or embargo < 0:
            self.errors.append("❌ embargo_minutes debe ser un número no negativo")
            valid = False
        
        test_size = validation.get('test_size', 0)
        if not isinstance(test_size, (int, float)) or not (0 < test_size < 1):
            self.errors.append("❌ test_size debe estar entre 0 y 1")
            valid = False
        
        # Validar early stopping
        early_stopping = training.get('optimization', {}).get('early_stopping', {})
        patience = early_stopping.get('patience', 0)
        if not isinstance(patience, int) or patience < 0:
            self.errors.append("❌ patience debe ser un entero no negativo")
            valid = False
        
        # Validar max_iter
        max_iter = training.get('optimization', {}).get('max_iter', 0)
        if not isinstance(max_iter, int) or max_iter <= 0:
            self.errors.append("❌ max_iter debe ser un entero positivo")
            valid = False
        
        if valid:
            logger.info("✅ Configuración de entrenamiento válida")
        
        return valid
    
    def validate_models(self) -> bool:
        """Validar configuración de modelos"""
        models = self.config.get('models', {})
        valid = True
        
        # Validar modelo de dirección
        direction = models.get('direction', {})
        if not direction:
            self.warnings.append("⚠️  No hay configuración para modelo de dirección")
            return True
        
        # Validar umbrales de promoción
        thresholds = direction.get('promotion_thresholds', {})
        min_auc = thresholds.get('min_auc', 0)
        if not isinstance(min_auc, (int, float)) or not (0 <= min_auc <= 1):
            self.errors.append("❌ min_auc debe estar entre 0 y 1")
            valid = False
        
        max_brier = thresholds.get('max_brier', 0)
        if not isinstance(max_brier, (int, float)) or not (0 <= max_brier <= 1):
            self.errors.append("❌ max_brier debe estar entre 0 y 1")
            valid = False
        
        min_samples = thresholds.get('min_samples', 0)
        if not isinstance(min_samples, int) or min_samples <= 0:
            self.errors.append("❌ min_samples debe ser un entero positivo")
            valid = False
        
        # Validar parámetros por defecto
        default_params = direction.get('default_params', {})
        if not isinstance(default_params, dict):
            self.errors.append("❌ default_params debe ser un diccionario")
            valid = False
        
        if valid:
            logger.info("✅ Configuración de modelos válida")
        
        return valid
    
    def validate_balance(self) -> bool:
        """Validar configuración de balance"""
        balance = self.config.get('balance', {})
        valid = True
        
        # Validar balance inicial
        initial = balance.get('initial', 0)
        if not isinstance(initial, (int, float)) or initial <= 0:
            self.errors.append("❌ balance.initial debe ser un número positivo")
            valid = False
        
        # Validar balance objetivo
        target = balance.get('target', 0)
        if not isinstance(target, (int, float)) or target <= 0:
            self.errors.append("❌ balance.target debe ser un número positivo")
            valid = False
        
        if target <= initial:
            self.warnings.append("⚠️  balance.target debería ser mayor que balance.initial")
        
        # Validar riesgo por trade
        risk_per_trade = balance.get('risk_per_trade', 0)
        if not isinstance(risk_per_trade, (int, float)) or not (0 < risk_per_trade <= 1):
            self.errors.append("❌ risk_per_trade debe estar entre 0 y 1")
            valid = False
        
        # Validar configuración por símbolo
        symbols = balance.get('symbols', {})
        for symbol, config in symbols.items():
            if not isinstance(config, dict):
                self.errors.append(f"❌ Configuración de {symbol} debe ser un diccionario")
                valid = False
                continue
            
            # Validar balance inicial del símbolo
            sym_initial = config.get('initial', 0)
            if not isinstance(sym_initial, (int, float)) or sym_initial <= 0:
                self.errors.append(f"❌ {symbol}.initial debe ser un número positivo")
                valid = False
            
            # Validar balance objetivo del símbolo
            sym_target = config.get('target', 0)
            if not isinstance(sym_target, (int, float)) or sym_target <= 0:
                self.errors.append(f"❌ {symbol}.target debe ser un número positivo")
                valid = False
        
        if valid:
            logger.info("✅ Configuración de balance válida")
        
        return valid
    
    def validate_monitoring(self) -> bool:
        """Validar configuración de monitoreo"""
        monitoring = self.config.get('monitoring', {})
        valid = True
        
        # Validar nivel de log
        log_level = monitoring.get('log_level', '')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            self.errors.append(f"❌ log_level debe ser uno de: {valid_levels}")
            valid = False
        
        # Validar frecuencia de progreso
        progress_freq = monitoring.get('progress_frequency', 0)
        if not isinstance(progress_freq, (int, float)) or not (0 < progress_freq <= 1):
            self.errors.append("❌ progress_frequency debe estar entre 0 y 1")
            valid = False
        
        # Validar umbrales
        memory_threshold = monitoring.get('memory_threshold_gb', 0)
        if not isinstance(memory_threshold, (int, float)) or memory_threshold <= 0:
            self.errors.append("❌ memory_threshold_gb debe ser un número positivo")
            valid = False
        
        cpu_threshold = monitoring.get('cpu_threshold_percent', 0)
        if not isinstance(cpu_threshold, (int, float)) or not (0 < cpu_threshold <= 100):
            self.errors.append("❌ cpu_threshold_percent debe estar entre 0 y 100")
            valid = False
        
        if valid:
            logger.info("✅ Configuración de monitoreo válida")
        
        return valid
    
    def validate_directories(self) -> bool:
        """Validar que los directorios necesarios existen o pueden crearse"""
        resources = self.config.get('resources', {})
        valid = True
        
        directories = [
            resources.get('temp_dir', 'temp'),
            resources.get('artifacts_dir', 'artifacts'),
            resources.get('logs_dir', 'logs'),
            resources.get('checkpoints_dir', 'checkpoints')
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"✅ Directorio {directory} disponible")
            except Exception as e:
                self.errors.append(f"❌ No se puede crear directorio {directory}: {e}")
                valid = False
        
        return valid
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Validar toda la configuración"""
        logger.info("🔍 INICIANDO VALIDACIÓN DE CONFIGURACIÓN")
        logger.info("=" * 50)
        
        # Cargar configuración
        if not self.load_config():
            return False, self.errors, self.warnings
        
        # Ejecutar validaciones
        validations = [
            ("Estructura", self.validate_structure),
            ("Recursos", self.validate_resources),
            ("Entrenamiento", self.validate_training),
            ("Modelos", self.validate_models),
            ("Balance", self.validate_balance),
            ("Monitoreo", self.validate_monitoring),
            ("Directorios", self.validate_directories)
        ]
        
        all_valid = True
        for name, validation_func in validations:
            logger.info(f"\n🧪 Validando {name}...")
            try:
                if not validation_func():
                    all_valid = False
            except Exception as e:
                self.errors.append(f"❌ Error validando {name}: {e}")
                all_valid = False
        
        return all_valid, self.errors, self.warnings
    
    def print_summary(self):
        """Imprimir resumen de validación"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 RESUMEN DE VALIDACIÓN")
        logger.info(f"{'='*60}")
        
        if self.errors:
            logger.error(f"❌ ERRORES ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"   {error}")
        
        if self.warnings:
            logger.warning(f"⚠️  ADVERTENCIAS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"   {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("✅ ¡CONFIGURACIÓN VÁLIDA!")
        elif not self.errors:
            logger.info("✅ Configuración válida con advertencias")
        else:
            logger.error("❌ Configuración inválida")
        
        return len(self.errors) == 0

def main():
    """Función principal"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/ml/training.yaml"
    
    validator = TrainingConfigValidator(config_path)
    is_valid, errors, warnings = validator.validate_all()
    
    success = validator.print_summary()
    
    if success:
        logger.info("\n🎉 ¡VALIDACIÓN EXITOSA!")
        return 0
    else:
        logger.error("\n💥 VALIDACIÓN FALLÓ")
        return 1

if __name__ == "__main__":
    sys.exit(main())
