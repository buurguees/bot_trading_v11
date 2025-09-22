#!/usr/bin/env python3
"""
Script interactivo para configurar entrenamiento
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingConfigurator:
    """Configurador interactivo de entrenamiento"""
    
    def __init__(self, config_path: str = "config/ml/training.yaml"):
        self.config_path = config_path
        self.config = {}
        self.backup_path = f"{config_path}.backup"
    
    def load_existing_config(self) -> bool:
        """Cargar configuración existente"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"✅ Configuración existente cargada desde {self.config_path}")
                return True
            else:
                logger.info("ℹ️  No hay configuración existente, creando nueva")
                return False
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            return False
    
    def create_backup(self) -> bool:
        """Crear respaldo de configuración existente"""
        try:
            if os.path.exists(self.config_path):
                import shutil
                shutil.copy2(self.config_path, self.backup_path)
                logger.info(f"✅ Respaldo creado en {self.backup_path}")
                return True
            return True
        except Exception as e:
            logger.error(f"❌ Error creando respaldo: {e}")
            return False
    
    def get_user_input(self, prompt: str, default: Any = None, input_type: type = str) -> Any:
        """Obtener entrada del usuario con validación"""
        while True:
            try:
                if default is not None:
                    user_input = input(f"{prompt} [{default}]: ").strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}: ").strip()
                
                if input_type == bool:
                    return user_input.lower() in ['true', '1', 'yes', 'y', 'sí', 'si']
                elif input_type == int:
                    return int(user_input)
                elif input_type == float:
                    return float(user_input)
                else:
                    return user_input
            except ValueError:
                print("❌ Entrada inválida, intenta de nuevo")
            except KeyboardInterrupt:
                print("\n👋 Configuración cancelada")
                sys.exit(0)
    
    def configure_resources(self):
        """Configurar recursos del sistema"""
        print("\n🔧 CONFIGURACIÓN DE RECURSOS")
        print("=" * 40)
        
        # Memoria máxima
        max_memory = self.get_user_input(
            "Memoria máxima (GB)", 
            self.config.get('resources', {}).get('max_memory_gb', 8), 
            float
        )
        
        # Workers máximos
        max_workers = self.get_user_input(
            "Workers máximos", 
            self.config.get('resources', {}).get('max_workers', 4), 
            int
        )
        
        # Tamaño de chunk
        chunk_size = self.get_user_input(
            "Tamaño de chunk", 
            self.config.get('resources', {}).get('chunk_size', 50000), 
            int
        )
        
        # Tamaño de cache
        cache_size = self.get_user_input(
            "Tamaño de cache (GB)", 
            self.config.get('resources', {}).get('cache_size_gb', 2), 
            float
        )
        
        # GPU habilitada
        gpu_enabled = self.get_user_input(
            "¿Habilitar GPU?", 
            self.config.get('resources', {}).get('gpu_enabled', False), 
            bool
        )
        
        self.config['resources'] = {
            'max_memory_gb': max_memory,
            'max_workers': max_workers,
            'chunk_size': chunk_size,
            'cache_size_gb': cache_size,
            'gpu_enabled': gpu_enabled,
            'db_pool_size': 20,
            'db_max_overflow': 30,
            'db_pool_recycle': 3600,
            'temp_dir': 'temp',
            'artifacts_dir': 'artifacts',
            'logs_dir': 'logs',
            'checkpoints_dir': 'checkpoints'
        }
    
    def configure_training(self):
        """Configurar parámetros de entrenamiento"""
        print("\n🎯 CONFIGURACIÓN DE ENTRENAMIENTO")
        print("=" * 40)
        
        # Número de splits
        n_splits = self.get_user_input(
            "Número de folds para validación", 
            self.config.get('training', {}).get('validation', {}).get('n_splits', 5), 
            int
        )
        
        # Embargo en minutos
        embargo = self.get_user_input(
            "Embargo entre train/test (minutos)", 
            self.config.get('training', {}).get('validation', {}).get('embargo_minutes', 30), 
            int
        )
        
        # Tamaño de test
        test_size = self.get_user_input(
            "Proporción de datos para test", 
            self.config.get('training', {}).get('validation', {}).get('test_size', 0.2), 
            float
        )
        
        # Early stopping patience
        patience = self.get_user_input(
            "Paciencia para early stopping", 
            self.config.get('training', {}).get('optimization', {}).get('early_stopping', {}).get('patience', 3), 
            int
        )
        
        # Iteraciones máximas
        max_iter = self.get_user_input(
            "Iteraciones máximas", 
            self.config.get('training', {}).get('optimization', {}).get('max_iter', 5000), 
            int
        )
        
        self.config['training'] = {
            'random_seed': 42,
            'n_jobs': -1,
            'validation': {
                'method': 'walk_forward',
                'n_splits': n_splits,
                'embargo_minutes': embargo,
                'test_size': test_size,
                'gap_minutes': 15,
                'min_train_samples': 1000,
                'min_test_samples': 200
            },
            'optimization': {
                'early_stopping': {
                    'enabled': True,
                    'patience': patience,
                    'min_delta': 0.001,
                    'restore_best_weights': True
                },
                'max_iter': max_iter,
                'auto_tune': True,
                'convergence_tol': 1e-4,
                'hyperparameter_search': {
                    'enabled': True,
                    'method': 'random',
                    'n_iter': 50,
                    'cv_folds': 3,
                    'scoring': 'roc_auc'
                }
            },
            'checkpoints': {
                'enabled': True,
                'frequency': 'per_fold',
                'retention_days': 7,
                'save_best_only': True,
                'monitor': 'val_auc',
                'mode': 'max'
            },
            'data': {
                'min_samples': 1000,
                'max_samples': 1000000,
                'balance_classes': True,
                'feature_selection': True,
                'feature_importance_threshold': 0.01
            },
            'memory': {
                'cleanup_frequency': 10,
                'gc_force': True,
                'memory_monitoring': True,
                'memory_warning_threshold': 0.8
            }
        }
    
    def configure_models(self):
        """Configurar modelos"""
        print("\n🤖 CONFIGURACIÓN DE MODELOS")
        print("=" * 40)
        
        # Umbral de AUC mínimo
        min_auc = self.get_user_input(
            "AUC mínimo para promoción", 
            self.config.get('models', {}).get('direction', {}).get('promotion_thresholds', {}).get('min_auc', 0.52), 
            float
        )
        
        # Umbral de Brier máximo
        max_brier = self.get_user_input(
            "Brier score máximo para promoción", 
            self.config.get('models', {}).get('direction', {}).get('promotion_thresholds', {}).get('max_brier', 0.25), 
            float
        )
        
        # Mínimo de muestras
        min_samples = self.get_user_input(
            "Mínimo de muestras para promoción", 
            self.config.get('models', {}).get('direction', {}).get('promotion_thresholds', {}).get('min_samples', 1000), 
            int
        )
        
        self.config['models'] = {
            'direction': {
                'default_params': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': 3000,
                    'class_weight': 'balanced',
                    'random_state': 42
                },
                'promotion_thresholds': {
                    'min_auc': min_auc,
                    'max_brier': max_brier,
                    'min_accuracy': 0.50,
                    'min_samples': min_samples,
                    'min_f1': 0.45,
                    'max_logloss': 0.7
                },
                'validation': {
                    'min_auc': min_auc - 0.01,
                    'max_brier': max_brier + 0.01,
                    'min_accuracy': 0.49,
                    'stability_threshold': 0.02
                }
            },
            'volatility': {
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'promotion_thresholds': {
                    'min_r2': 0.3,
                    'max_mae': 0.1,
                    'min_samples': 500,
                    'min_correlation': 0.4
                }
            },
            'regime': {
                'default_params': {
                    'n_clusters': 3,
                    'init': 'k-means++',
                    'n_init': 10,
                    'max_iter': 300,
                    'random_state': 42
                },
                'promotion_thresholds': {
                    'min_silhouette': 0.3,
                    'min_samples': 200,
                    'max_inertia': 1000
                }
            }
        }
    
    def configure_balance(self):
        """Configurar balance y riesgo"""
        print("\n💰 CONFIGURACIÓN DE BALANCE Y RIESGO")
        print("=" * 40)
        
        # Balance inicial
        initial_balance = self.get_user_input(
            "Balance inicial", 
            self.config.get('balance', {}).get('initial', 1000.0), 
            float
        )
        
        # Balance objetivo
        target_balance = self.get_user_input(
            "Balance objetivo", 
            self.config.get('balance', {}).get('target', 10000.0), 
            float
        )
        
        # Riesgo por trade
        risk_per_trade = self.get_user_input(
            "Riesgo por trade (0.01 = 1%)", 
            self.config.get('balance', {}).get('risk_per_trade', 0.01), 
            float
        )
        
        self.config['balance'] = {
            'initial': initial_balance,
            'target': target_balance,
            'risk_per_trade': risk_per_trade,
            'symbols': {
                'BTCUSDT': {
                    'initial': initial_balance,
                    'target': target_balance,
                    'risk_per_trade': risk_per_trade,
                    'max_leverage': 5,
                    'min_trade_size': 0.0001
                },
                'ETHUSDT': {
                    'initial': initial_balance,
                    'target': target_balance,
                    'risk_per_trade': risk_per_trade,
                    'max_leverage': 5,
                    'min_trade_size': 0.001
                },
                'ADAUSDT': {
                    'initial': initial_balance * 5,
                    'target': target_balance * 0.55,
                    'risk_per_trade': risk_per_trade * 0.5,
                    'max_leverage': 5,
                    'min_trade_size': 1.0
                },
                'SOLUSDT': {
                    'initial': initial_balance * 6,
                    'target': target_balance * 0.66,
                    'risk_per_trade': risk_per_trade,
                    'max_leverage': 5,
                    'min_trade_size': 0.1
                },
                'XRPUSDT': {
                    'initial': initial_balance * 5,
                    'target': target_balance * 0.55,
                    'risk_per_trade': risk_per_trade * 0.5,
                    'max_leverage': 5,
                    'min_trade_size': 1.0
                },
                'DOGEUSDT': {
                    'initial': initial_balance * 3,
                    'target': target_balance * 0.33,
                    'risk_per_trade': risk_per_trade * 0.5,
                    'max_leverage': 5,
                    'min_trade_size': 10.0
                }
            }
        }
    
    def configure_monitoring(self):
        """Configurar monitoreo"""
        print("\n📊 CONFIGURACIÓN DE MONITOREO")
        print("=" * 40)
        
        # Nivel de log
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        current_level = self.config.get('monitoring', {}).get('log_level', 'INFO')
        print(f"Nivel de log: {', '.join(log_levels)}")
        log_level = self.get_user_input(
            "Selecciona nivel de log", 
            current_level
        )
        
        # Frecuencia de progreso
        progress_freq = self.get_user_input(
            "Frecuencia de progreso (0.1 = cada 10%)", 
            self.config.get('monitoring', {}).get('progress_frequency', 0.1), 
            float
        )
        
        # Umbral de memoria
        memory_threshold = self.get_user_input(
            "Umbral de memoria para alerta (GB)", 
            self.config.get('monitoring', {}).get('memory_threshold_gb', 6), 
            float
        )
        
        self.config['monitoring'] = {
            'log_level': log_level,
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': 'logs/training.log',
            'log_rotation': True,
            'log_max_size': '10MB',
            'log_backup_count': 5,
            'progress_frequency': progress_freq,
            'progress_bar': True,
            'progress_eta': True,
            'metrics_frequency': 'per_fold',
            'metrics_save': True,
            'metrics_file': 'logs/metrics.json',
            'memory_threshold_gb': memory_threshold,
            'cpu_threshold_percent': 90,
            'latency_threshold_seconds': 300,
            'notifications': {
                'enabled': False,
                'email': False,
                'slack': False,
                'webhook': False
            }
        }
    
    def configure_additional_sections(self):
        """Configurar secciones adicionales"""
        print("\n⚙️  CONFIGURACIÓN ADICIONAL")
        print("=" * 40)
        
        # Backtesting
        self.config['backtesting'] = {
            'enabled': True,
            'frequency': 'daily',
            'simulation': {
                'fees_bps': 2,
                'slip_bps': 2,
                'max_hold_bars': 600,
                'initial_balance': 1000.0
            },
            'metrics': {
                'calculate_sharpe': True,
                'calculate_max_dd': True,
                'calculate_win_rate': True,
                'calculate_profit_factor': True
            },
            'reports': {
                'generate_html': True,
                'generate_pdf': False,
                'include_charts': True,
                'save_trades': True
            }
        }
        
        # Features
        self.config['features'] = {
            'enabled': True,
            'real_time': True,
            'batch_size': 1000,
            'technical': {
                'rsi': {'periods': [14, 21, 28], 'enabled': True},
                'ema': {'periods': [20, 50, 200], 'enabled': True},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9, 'enabled': True},
                'bollinger_bands': {'period': 20, 'std': 2, 'enabled': True},
                'atr': {'period': 14, 'enabled': True},
                'obv': {'enabled': True},
                'supertrend': {'period': 10, 'multiplier': 3, 'enabled': True}
            },
            'market': {
                'volume_profile': True,
                'price_action': True,
                'support_resistance': True,
                'trend_detection': True
            },
            'cache': {
                'enabled': True,
                'ttl_minutes': 60,
                'max_size_mb': 100
            }
        }
        
        # Alertas
        self.config['alerts'] = {
            'enabled': True,
            'channels': ['log', 'file'],
            'types': {
                'memory_high': True,
                'cpu_high': True,
                'training_slow': True,
                'model_poor': True,
                'data_quality': True,
                'system_error': True
            },
            'thresholds': {
                'memory_percent': 80,
                'cpu_percent': 90,
                'training_time_minutes': 30,
                'auc_threshold': 0.45,
                'data_missing_percent': 10
            },
            'notifications': {
                'email': {'enabled': False},
                'slack': {'enabled': False},
                'webhook': {'enabled': False}
            }
        }
        
        # Desarrollo
        self.config['development'] = {
            'debug': False,
            'verbose': False,
            'profile': False,
            'test_mode': False,
            'test_samples': 1000,
            'test_timeframes': ['1m', '5m'],
            'log_predictions': False,
            'log_features': False,
            'log_metrics': True,
            'validate_data': True,
            'validate_models': True,
            'validate_predictions': True
        }
        
        # Seguridad
        self.config['security'] = {
            'encrypt_models': False,
            'encrypt_data': False,
            'require_auth': False,
            'api_key': '',
            'allowed_ips': [],
            'audit_log': True,
            'audit_file': 'logs/audit.log',
            'audit_retention_days': 30
        }
        
        # Mantenimiento
        self.config['maintenance'] = {
            'cleanup': {
                'enabled': True,
                'frequency': 'daily',
                'retention_days': 7,
                'temp_files': True,
                'old_logs': True,
                'old_checkpoints': True,
                'old_artifacts': True
            },
            'backup': {
                'enabled': False,
                'frequency': 'daily',
                'retention_days': 30,
                'compress': True,
                'encrypt': False
            },
            'update': {
                'auto_check': False,
                'check_frequency': 'weekly',
                'auto_update': False,
                'backup_before_update': True
            }
        }
    
    def save_config(self) -> bool:
        """Guardar configuración"""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Guardar configuración
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2, allow_unicode=True)
            
            logger.info(f"✅ Configuración guardada en {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando configuración: {e}")
            return False
    
    def run_interactive_config(self):
        """Ejecutar configuración interactiva"""
        print("🚀 CONFIGURADOR INTERACTIVO DE ENTRENAMIENTO")
        print("=" * 60)
        
        # Cargar configuración existente
        self.load_existing_config()
        
        # Crear respaldo
        if not self.create_backup():
            return False
        
        # Configurar secciones
        try:
            self.configure_resources()
            self.configure_training()
            self.configure_models()
            self.configure_balance()
            self.configure_monitoring()
            self.configure_additional_sections()
            
            # Guardar configuración
            if self.save_config():
                print("\n🎉 ¡CONFIGURACIÓN COMPLETADA!")
                print(f"📁 Archivo guardado en: {self.config_path}")
                print(f"💾 Respaldo creado en: {self.backup_path}")
                return True
            else:
                print("\n❌ Error guardando configuración")
                return False
                
        except KeyboardInterrupt:
            print("\n👋 Configuración cancelada por el usuario")
            return False
        except Exception as e:
            print(f"\n❌ Error durante la configuración: {e}")
            return False

def main():
    """Función principal"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/ml/training.yaml"
    
    configurator = TrainingConfigurator(config_path)
    success = configurator.run_interactive_config()
    
    if success:
        print("\n✅ ¡Configuración exitosa!")
        print("\n📋 Próximos pasos:")
        print("   1. Revisar la configuración generada")
        print("   2. Ejecutar: python validate_training_config.py")
        print("   3. Iniciar entrenamiento con la nueva configuración")
        return 0
    else:
        print("\n❌ Configuración falló")
        return 1

if __name__ == "__main__":
    sys.exit(main())
