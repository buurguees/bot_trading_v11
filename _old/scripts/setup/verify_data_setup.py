#!/usr/bin/env python3
"""
verify_data_setup.py - Verificación rápida del estado de los datos

Verifica:
1. Conexión a la base de datos
2. Cobertura de datos históricos
3. Cobertura de features
4. Preparación para entrenamiento
"""

import sys
from datetime import datetime, timezone
from core.data.database import get_engine
from sqlalchemy import text

def verify_database_connection():
    """Verificar conexión a la base de datos"""
    print("🔌 Verificando conexión a la base de datos...")
    try:
        engine = get_engine()
        with engine.begin() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        print("✅ Conexión a la base de datos exitosa")
        return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

def verify_historical_data():
    """Verificar datos históricos"""
    print("\n📊 Verificando datos históricos...")
    try:
        engine = get_engine()
        with engine.begin() as conn:
            # Verificar que existen datos
            result = conn.execute(text("""
                SELECT COUNT(*) as total_bars
                FROM trading.historicaldata
                WHERE timestamp >= NOW() - INTERVAL '365 days'
            """)).scalar()
            
            if result == 0:
                print("❌ No hay datos históricos en los últimos 365 días")
                return False
            
            print(f"✅ {result:,} barras de datos históricos encontradas")
            
            # Verificar cobertura por símbolo
            result = conn.execute(text("""
                SELECT symbol, COUNT(*) as bars
                FROM trading.historicaldata
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY symbol
                ORDER BY symbol
            """))
            
            print("📈 Cobertura por símbolo:")
            for row in result:
                print(f"  {row[0]}: {row[1]:,} barras")
            
            return True
    except Exception as e:
        print(f"❌ Error verificando datos históricos: {e}")
        return False

def verify_features():
    """Verificar features calculados"""
    print("\n📈 Verificando features...")
    try:
        engine = get_engine()
        with engine.begin() as conn:
            # Verificar que existen features
            result = conn.execute(text("""
                SELECT COUNT(*) as total_features
                FROM trading.features
                WHERE timestamp >= NOW() - INTERVAL '365 days'
            """)).scalar()
            
            if result == 0:
                print("❌ No hay features calculados en los últimos 365 días")
                return False
            
            print(f"✅ {result:,} features calculados encontrados")
            
            # Verificar cobertura por símbolo y timeframe
            result = conn.execute(text("""
                SELECT symbol, timeframe, COUNT(*) as features
                FROM trading.features
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """))
            
            print("📊 Cobertura por símbolo y timeframe:")
            current_symbol = None
            for row in result:
                if current_symbol != row[0]:
                    current_symbol = row[0]
                    print(f"  {row[0]}:")
                print(f"    {row[1]}: {row[2]:,} features")
            
            return True
    except Exception as e:
        print(f"❌ Error verificando features: {e}")
        return False

def verify_training_readiness():
    """Verificar preparación para entrenamiento"""
    print("\n🎯 Verificando preparación para entrenamiento...")
    try:
        engine = get_engine()
        with engine.begin() as conn:
            # Verificar que tenemos datos suficientes para cada símbolo/TF
            # Umbrales diferentes por timeframe (1d tiene menos barras por año)
            result = conn.execute(text("""
                SELECT symbol, timeframe, COUNT(*) as features,
                       CASE 
                           WHEN timeframe = '1d' THEN 200
                           WHEN timeframe = '4h' THEN 500
                           WHEN timeframe = '1h' THEN 1000
                           WHEN timeframe = '15m' THEN 2000
                           WHEN timeframe = '5m' THEN 3000
                           WHEN timeframe = '1m' THEN 3000
                           ELSE 1000
                       END as min_required
                FROM trading.features
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY symbol, timeframe
                HAVING COUNT(*) < CASE 
                           WHEN timeframe = '1d' THEN 200
                           WHEN timeframe = '4h' THEN 500
                           WHEN timeframe = '1h' THEN 1000
                           WHEN timeframe = '15m' THEN 2000
                           WHEN timeframe = '5m' THEN 3000
                           WHEN timeframe = '1m' THEN 3000
                           ELSE 1000
                       END
                ORDER BY symbol, timeframe
            """))
            
            insufficient = list(result)
            if insufficient:
                print("⚠️  Símbolos/TFs con datos insuficientes:")
                for row in insufficient:
                    print(f"  {row[0]} {row[1]}: {row[2]} features (mínimo requerido: {row[3]})")
                return False
            
            # Verificar cobertura temporal (umbrales más realistas por timeframe)
            result = conn.execute(text("""
                SELECT symbol, timeframe, 
                       MIN(timestamp) as desde,
                       MAX(timestamp) as hasta,
                       EXTRACT(DAYS FROM (MAX(timestamp) - MIN(timestamp))) as dias,
                       CASE 
                           WHEN timeframe = '1d' THEN 200
                           WHEN timeframe = '4h' THEN 100
                           WHEN timeframe = '1h' THEN 50
                           WHEN timeframe = '15m' THEN 30
                           WHEN timeframe = '5m' THEN 15
                           WHEN timeframe = '1m' THEN 3
                           ELSE 30
                       END as min_dias
                FROM trading.features
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY symbol, timeframe
                HAVING EXTRACT(DAYS FROM (MAX(timestamp) - MIN(timestamp))) < CASE 
                           WHEN timeframe = '1d' THEN 200
                           WHEN timeframe = '4h' THEN 100
                           WHEN timeframe = '1h' THEN 50
                           WHEN timeframe = '15m' THEN 30
                           WHEN timeframe = '5m' THEN 15
                           WHEN timeframe = '1m' THEN 3
                           ELSE 30
                       END
                ORDER BY symbol, timeframe
            """))
            
            short_coverage = list(result)
            if short_coverage:
                print("⚠️  Símbolos/TFs con cobertura temporal insuficiente:")
                for row in short_coverage:
                    print(f"  {row[0]} {row[1]}: {row[4]} días (mínimo requerido: {row[5]}) | {row[2]} - {row[3]}")
                return False
            
            print("✅ Todos los símbolos/TFs tienen datos suficientes para entrenamiento")
            return True
    except Exception as e:
        print(f"❌ Error verificando preparación: {e}")
        return False

def main():
    """Función principal"""
    print("🔍 VERIFICACIÓN DEL ESTADO DE LOS DATOS")
    print("=" * 50)
    
    # Verificar conexión
    if not verify_database_connection():
        return False
    
    # Verificar datos históricos
    if not verify_historical_data():
        print("\n💡 Ejecuta 'python setup_full_data.py' para descargar datos históricos")
        return False
    
    # Verificar features
    if not verify_features():
        print("\n💡 Ejecuta 'python setup_full_data.py' para calcular features")
        return False
    
    # Verificar preparación para entrenamiento
    if not verify_training_readiness():
        print("\n💡 Ejecuta 'python setup_full_data.py' para completar la configuración")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 SISTEMA COMPLETAMENTE CONFIGURADO")
    print("✅ Listo para entrenamiento nocturno")
    print("🚀 Ejecuta 'python -m core.ml.training.daily_train.runner' para comenzar")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
