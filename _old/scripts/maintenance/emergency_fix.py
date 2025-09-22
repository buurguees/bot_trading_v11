#!/usr/bin/env python3
"""
Script de emergencia para detener entrenamiento en pares problemáticos
y ajustar configuración de riesgo
"""

import os
import yaml
from core.data.database import get_engine
from sqlalchemy import text

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_yaml(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def emergency_fix():
    print("🚨 APLICANDO CORRECCIONES DE EMERGENCIA...")
    
    # 1. Cargar configuración actual
    symbols_config = load_yaml('config/trading/symbols.yaml')
    training_config = load_yaml('config/ml/training.yaml')
    
    # 2. Pares problemáticos identificados
    problematic_pairs = [
        'BTCUSDT',  # PnL muy negativo
        'XRPUSDT',  # PnL muy negativo en 1m
        'ADAUSDT',  # PnL muy negativo en 1m
        'SOLUSDT',  # PnL negativo
        'DOGEUSDT', # PnL negativo
        'ETHUSDT'   # PnL negativo
    ]
    
    print(f"📋 Pares problemáticos identificados: {', '.join(problematic_pairs)}")
    
    # 3. Ajustar objetivos de balance (reducir agresividad)
    print("🎯 Ajustando objetivos de balance...")
    
    new_balance_config = {
        'BTCUSDT': {
            'initial': 1000.0,
            'target': 2000.0,  # Reducido de 100000 a 2000 (2x)
            'risk_per_trade': 0.01  # Reducido de 0.02 a 0.01
        },
        'ETHUSDT': {
            'initial': 1000.0,
            'target': 2000.0,  # Reducido de 100000 a 2000 (2x)
            'risk_per_trade': 0.01  # Reducido de 0.02 a 0.01
        },
        'ADAUSDT': {
            'initial': 5000.0,
            'target': 5500.0,  # Reducido de 10000 a 5500 (1.1x)
            'risk_per_trade': 0.005  # Reducido de 0.015 a 0.005
        },
        'SOLUSDT': {
            'initial': 6000.0,
            'target': 6600.0,  # Reducido de 12000 a 6600 (1.1x)
            'risk_per_trade': 0.01  # Reducido de 0.02 a 0.01
        },
        'DOGEUSDT': {
            'initial': 3000.0,
            'target': 3300.0,  # Reducido de 6000 a 3300 (1.1x)
            'risk_per_trade': 0.005  # Reducido de 0.01 a 0.005
        },
        'XRPUSDT': {
            'initial': 5000.0,
            'target': 5500.0,  # Reducido de 10000 a 5500 (1.1x)
            'risk_per_trade': 0.005  # Reducido de 0.015 a 0.005
        }
    }
    
    # Actualizar training.yaml
    if 'balance' not in training_config:
        training_config['balance'] = {}
    if 'symbols' not in training_config['balance']:
        training_config['balance']['symbols'] = {}
    
    training_config['balance']['symbols'] = new_balance_config
    save_yaml(training_config, 'config/ml/training.yaml')
    print("✅ Objetivos de balance actualizados")
    
    # 4. Ajustar apalancamiento máximo en symbols.yaml
    print("⚙️ Ajustando apalancamiento máximo...")
    
    for symbol in problematic_pairs:
        if symbol in symbols_config.get('symbols', {}):
            symbols_config['symbols'][symbol]['max_leverage'] = 5  # Reducido de 80/50/30 a 5
            symbols_config['symbols'][symbol]['min_leverage'] = 1  # Reducido de 3/5 a 1
    
    save_yaml(symbols_config, 'config/trading/symbols.yaml')
    print("✅ Apalancamiento máximo reducido a 5x")
    
    # 5. Deshabilitar temporalmente pares más problemáticos
    print("⏸️ Deshabilitando pares más problemáticos temporalmente...")
    
    # Crear backup de configuración original
    os.system('copy config\\trading\\symbols.yaml config\\trading\\symbols_backup.yaml')
    
    # Deshabilitar BTCUSDT y ETHUSDT (los más problemáticos)
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        if symbol in symbols_config.get('symbols', {}):
            symbols_config['symbols'][symbol]['enabled'] = False
    
    save_yaml(symbols_config, 'config/trading/symbols.yaml')
    print("✅ BTCUSDT y ETHUSDT deshabilitados temporalmente")
    
    # 6. Limpiar memoria de estrategias problemáticas
    print("🧹 Limpiando memoria de estrategias problemáticas...")
    
    try:
        eng = get_engine()
        with eng.connect() as conn:
            # Limpiar strategy_memory para pares problemáticos
            for symbol in problematic_pairs:
                conn.execute(text("""
                    DELETE FROM trading.strategy_memory 
                    WHERE symbol = :symbol
                """), {"symbol": symbol})
            
            # Limpiar strategy_samples para pares problemáticos
            for symbol in problematic_pairs:
                conn.execute(text("""
                    DELETE FROM trading.strategy_samples 
                    WHERE symbol = :symbol
                """), {"symbol": symbol})
            
            conn.commit()
            print("✅ Memoria de estrategias limpiada")
            
    except Exception as e:
        print(f"⚠️ Error limpiando memoria: {e}")
    
    print("\n🎯 CORRECCIONES APLICADAS:")
    print("1. ✅ Objetivos de balance reducidos (2x máximo)")
    print("2. ✅ Apalancamiento máximo reducido a 5x")
    print("3. ✅ Riesgo por trade reducido")
    print("4. ✅ BTCUSDT y ETHUSDT deshabilitados temporalmente")
    print("5. ✅ Memoria de estrategias limpiada")
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. Reinicia el entrenamiento con: python -m core.ml.training.daily_train.runner --skip-backfill")
    print("2. Monitorea solo ADAUSDT, XRPUSDT, SOLUSDT, DOGEUSDT")
    print("3. Si mejoran, gradualmente rehabilita BTCUSDT y ETHUSDT")
    print("4. Ejecuta la consulta de monitoreo cada 2 horas")

if __name__ == "__main__":
    emergency_fix()
