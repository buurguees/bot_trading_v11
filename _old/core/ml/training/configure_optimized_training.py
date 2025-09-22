#!/usr/bin/env python3
"""
Script de configuración para entrenamiento optimizado
"""

import os
import yaml
import argparse
from pathlib import Path

def create_optimized_config():
    """Crear configuración optimizada para entrenamiento"""
    
    config = {
        "training": {
            "optimized": {
                "enabled": True,
                "chunk_size": 50000,
                "memory_threshold": 0.85,
                "n_splits": 5,
                "embargo_minutes": 30,
                "early_stopping_patience": 3,
                "adaptive_max_iter": True,
                "use_gpu": False,  # Cambiar a True si tienes GPU
                "checkpoint_enabled": True,
                "logging_level": "INFO"
            },
            "memory_management": {
                "force_gc_frequency": 5,  # Cada 5 chunks
                "memory_monitoring": True,
                "chunk_processing": True
            },
            "validation": {
                "walk_forward": True,
                "time_series_split": True,
                "embargo_temporal": True,
                "min_folds": 3,
                "max_folds": 10
            }
        }
    }
    
    return config

def update_training_yaml(config_path: str = "config/ml/training.yaml"):
    """Actualizar training.yaml con configuración optimizada"""
    
    # Cargar configuración existente
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            existing_config = yaml.safe_load(f) or {}
    else:
        existing_config = {}
    
    # Crear configuración optimizada
    optimized_config = create_optimized_config()
    
    # Fusionar configuraciones
    if "training" not in existing_config:
        existing_config["training"] = {}
    
    existing_config["training"].update(optimized_config["training"])
    
    # Guardar configuración actualizada
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Configuración optimizada guardada en {config_path}")

def create_logging_config():
    """Crear configuración de logging para entrenamiento optimizado"""
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/train_direction_optimized.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }
    
    # Guardar configuración de logging
    os.makedirs("logs", exist_ok=True)
    with open("logs/logging_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(logging_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ Configuración de logging creada en logs/logging_config.yaml")

def check_system_requirements():
    """Verificar requisitos del sistema"""
    
    print("🔍 Verificando requisitos del sistema...")
    
    # Verificar memoria disponible
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"  💾 Memoria total: {total_gb:.1f} GB")
        print(f"  💾 Memoria disponible: {available_gb:.1f} GB")
        
        if total_gb < 8:
            print("  ⚠️  Advertencia: Menos de 8GB de RAM puede causar problemas")
        else:
            print("  ✅ Memoria suficiente")
            
    except ImportError:
        print("  ❌ psutil no instalado - ejecuta: pip install psutil")
        return False
    
    # Verificar GPU
    gpu_available = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  🚀 GPU detectada: {gpu_count} dispositivo(s)")
            gpu_available = True
        else:
            print("  💻 Solo CPU disponible")
    except ImportError:
        print("  💻 PyTorch no instalado - solo CPU")
    
    # Verificar directorios
    required_dirs = ["logs", "artifacts/direction", "config/ml"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"  📁 Directorio creado: {dir_path}")
        else:
            print(f"  ✅ Directorio existe: {dir_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Configurar entrenamiento optimizado")
    parser.add_argument("--config", default="config/ml/training.yaml", help="Ruta del archivo de configuración")
    parser.add_argument("--check-requirements", action="store_true", help="Verificar requisitos del sistema")
    parser.add_argument("--create-logging", action="store_true", help="Crear configuración de logging")
    
    args = parser.parse_args()
    
    print("🚀 CONFIGURANDO ENTRENAMIENTO OPTIMIZADO")
    print("=" * 50)
    
    # Verificar requisitos
    if args.check_requirements:
        if not check_system_requirements():
            print("❌ Requisitos del sistema no cumplidos")
            return
    
    # Actualizar configuración
    update_training_yaml(args.config)
    
    # Crear configuración de logging
    if args.create_logging:
        create_logging_config()
    
    print("\n✅ Configuración completada")
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. Instalar dependencias: pip install -r requirements_optimized.txt")
    print("2. Ejecutar entrenamiento: python -m core.ml.training.train_direction --help")
    print("3. Monitorear logs: tail -f logs/train_direction_optimized.log")

if __name__ == "__main__":
    main()
