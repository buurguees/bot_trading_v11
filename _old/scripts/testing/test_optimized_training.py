#!/usr/bin/env python3
"""
Script de prueba para entrenamiento optimizado
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def test_system_requirements():
    """Probar requisitos del sistema"""
    print("🔍 PROBANDO REQUISITOS DEL SISTEMA")
    print("=" * 40)
    
    # Verificar psutil
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ psutil: {psutil.__version__}")
        print(f"   Memoria total: {memory.total / (1024**3):.1f} GB")
    except ImportError:
        print("❌ psutil no instalado")
        return False
    
    # Verificar scikit-learn
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn no instalado")
        return False
    
    # Verificar numpy
    try:
        import numpy as np
        print(f"✅ numpy: {np.__version__}")
    except ImportError:
        print("❌ numpy no instalado")
        return False
    
    # Verificar pandas
    try:
        import pandas as pd
        print(f"✅ pandas: {pd.__version__}")
    except ImportError:
        print("❌ pandas no instalado")
        return False
    
    return True

def test_optimized_training(symbol: str = "BTCUSDT", timeframe: str = "1m", 
                          horizon: int = 1, max_bars: int = 1000):
    """Probar entrenamiento optimizado con dataset pequeño"""
    
    print(f"\n🚀 PROBANDO ENTRENAMIENTO OPTIMIZADO")
    print("=" * 40)
    print(f"Símbolo: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Horizonte: {horizon}")
    print(f"Max bars: {max_bars}")
    print()
    
    # Comando de prueba
    cmd = [
        "python", "-m", "core.ml.training.train_direction",
        "--symbol", symbol,
        "--tf", timeframe,
        "--horizon", str(horizon),
        "--max-bars", str(max_bars),
        "--n-splits", "3",  # Menos folds para prueba rápida
        "--embargo-minutes", "15",  # Menos embargo para prueba rápida
        "--chunk-size", "1000"  # Chunks más pequeños para prueba
    ]
    
    print(f"Comando: {' '.join(cmd)}")
    print()
    
    try:
        # Ejecutar entrenamiento
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"⏱️  Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        print(f"📊 Código de salida: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Entrenamiento completado exitosamente")
            print("\n📝 SALIDA:")
            print(result.stdout)
        else:
            print("❌ Error en entrenamiento")
            print("\n📝 ERROR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout - entrenamiento tardó más de 5 minutos")
        return False
    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento: {e}")
        return False
    
    return True

def test_monitoring(symbol: str = "BTCUSDT", timeframe: str = "1m", horizon: int = 1):
    """Probar sistema de monitoreo"""
    
    print(f"\n📊 PROBANDO SISTEMA DE MONITOREO")
    print("=" * 40)
    
    # Verificar si existe checkpoint
    checkpoint_path = f"logs/checkpoint_{symbol}_{timeframe}_H{horizon}.pkl"
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint encontrado: {checkpoint_path}")
        
        # Probar comando de monitoreo
        cmd = [
            "python", "core/ml/monitoring/monitor_training_progress.py",
            "--symbol", symbol,
            "--tf", timeframe,
            "--horizon", str(horizon),
            "--summary"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Monitor funcionando correctamente")
                print("\n📝 RESUMEN:")
                print(result.stdout)
            else:
                print("❌ Error en monitor")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Error ejecutando monitor: {e}")
    else:
        print(f"❌ No se encontró checkpoint: {checkpoint_path}")
        print("   Ejecuta primero el entrenamiento de prueba")

def main():
    parser = argparse.ArgumentParser(description="Probar entrenamiento optimizado")
    parser.add_argument("--symbol", default="BTCUSDT", help="Símbolo para prueba")
    parser.add_argument("--tf", default="1m", help="Timeframe para prueba")
    parser.add_argument("--horizon", type=int, default=1, help="Horizonte para prueba")
    parser.add_argument("--max-bars", type=int, default=1000, help="Máximo de barras para prueba")
    parser.add_argument("--skip-training", action="store_true", help="Saltar prueba de entrenamiento")
    parser.add_argument("--skip-monitoring", action="store_true", help="Saltar prueba de monitoreo")
    
    args = parser.parse_args()
    
    print("🧪 PRUEBAS DE ENTRENAMIENTO OPTIMIZADO")
    print("=" * 50)
    
    # Crear directorios necesarios
    os.makedirs("logs", exist_ok=True)
    os.makedirs("artifacts/direction", exist_ok=True)
    
    # Probar requisitos
    if not test_system_requirements():
        print("\n❌ Requisitos del sistema no cumplidos")
        sys.exit(1)
    
    # Probar entrenamiento
    if not args.skip_training:
        if not test_optimized_training(args.symbol, args.tf, args.horizon, args.max_bars):
            print("\n❌ Prueba de entrenamiento falló")
            sys.exit(1)
    
    # Probar monitoreo
    if not args.skip_monitoring:
        test_monitoring(args.symbol, args.tf, args.horizon)
    
    print("\n✅ TODAS LAS PRUEBAS COMPLETADAS")
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. Ejecutar entrenamiento completo: python -m core.ml.training.train_direction --help")
    print("2. Monitorear progreso: python core/ml/monitoring/monitor_training_progress.py --symbol BTCUSDT --tf 1m")
    print("3. Verificar logs: tail -f logs/train_direction_optimized.log")

if __name__ == "__main__":
    main()
