#!/usr/bin/env python3
"""
Script para verificar duplicados en las tablas del sistema de trading
"""

import pandas as pd
from core.data.database import get_engine
from sqlalchemy import text

def check_duplicates():
    print("🔍 VERIFICANDO DUPLICADOS EN EL SISTEMA")
    print("=" * 50)
    
    eng = get_engine()
    
    # 1. Verificar duplicados en agentversions
    print("\n📊 1. DUPLICADOS EN AGENTVERSIONS:")
    print("-" * 30)
    
    query_agentversions = """
    SELECT 
        (params->>'symbol') as symbol,
        (params->>'timeframe') as timeframe,
        COUNT(*) as total_registros,
        COUNT(DISTINCT id) as registros_unicos,
        COUNT(*) - COUNT(DISTINCT id) as duplicados_id,
        MIN(created_at) as primer_registro,
        MAX(created_at) as ultimo_registro
    FROM trading.agentversions
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY (params->>'symbol'), (params->>'timeframe')
    ORDER BY total_registros DESC;
    """
    
    try:
        with eng.connect() as conn:
            result = conn.execute(text(query_agentversions))
            df_agentversions = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_agentversions.empty:
                print("Registros por símbolo/timeframe en las últimas 24h:")
                for _, row in df_agentversions.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    total = row['total_registros']
                    unicos = row['registros_unicos']
                    duplicados = row['duplicados_id']
                    
                    if duplicados > 0:
                        print(f"  🔴 {symbol} {timeframe}: {total} registros, {duplicados} duplicados ID")
                    else:
                        print(f"  ✅ {symbol} {timeframe}: {total} registros, 0 duplicados ID")
            else:
                print("  ❌ No hay registros en las últimas 24 horas")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # 2. Verificar duplicados en backtests
    print("\n📊 2. DUPLICADOS EN BACKTESTS:")
    print("-" * 30)
    
    query_backtests = """
    SELECT 
        symbol,
        timeframe,
        COUNT(*) as total_registros,
        COUNT(DISTINCT id) as registros_unicos,
        COUNT(*) - COUNT(DISTINCT id) as duplicados_id,
        MIN(run_ts) as primer_backtest,
        MAX(run_ts) as ultimo_backtest
    FROM trading.backtests
    WHERE run_ts >= NOW() - INTERVAL '24 hours'
    GROUP BY symbol, timeframe
    ORDER BY total_registros DESC;
    """
    
    try:
        with eng.connect() as conn:
            result = conn.execute(text(query_backtests))
            df_backtests = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_backtests.empty:
                print("Backtests por símbolo/timeframe en las últimas 24h:")
                for _, row in df_backtests.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    total = row['total_registros']
                    unicos = row['registros_unicos']
                    duplicados = row['duplicados_id']
                    
                    if duplicados > 0:
                        print(f"  🔴 {symbol} {timeframe}: {total} backtests, {duplicados} duplicados ID")
                    else:
                        print(f"  ✅ {symbol} {timeframe}: {total} backtests, 0 duplicados ID")
            else:
                print("  ❌ No hay backtests en las últimas 24 horas")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # 3. Verificar duplicados en tradeplans
    print("\n📊 3. DUPLICADOS EN TRADEPLANS:")
    print("-" * 30)
    
    query_tradeplans = """
    SELECT 
        symbol,
        timeframe,
        COUNT(*) as total_registros,
        COUNT(DISTINCT id) as registros_unicos,
        COUNT(*) - COUNT(DISTINCT id) as duplicados_id,
        MIN(created_at) as primer_plan,
        MAX(created_at) as ultimo_plan
    FROM trading.tradeplans
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY symbol, timeframe
    ORDER BY total_registros DESC;
    """
    
    try:
        with eng.connect() as conn:
            result = conn.execute(text(query_tradeplans))
            df_tradeplans = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_tradeplans.empty:
                print("Planes por símbolo/timeframe en las últimas 24h:")
                for _, row in df_tradeplans.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    total = row['total_registros']
                    unicos = row['registros_unicos']
                    duplicados = row['duplicados_id']
                    
                    if duplicados > 0:
                        print(f"  🔴 {symbol} {timeframe}: {total} planes, {duplicados} duplicados ID")
                    else:
                        print(f"  ✅ {symbol} {timeframe}: {total} planes, 0 duplicados ID")
            else:
                print("  ❌ No hay planes en las últimas 24 horas")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # 4. Verificar duplicados específicos por timestamp
    print("\n📊 4. DUPLICADOS POR TIMESTAMP (AGENTVERSIONS):")
    print("-" * 30)
    
    query_timestamp_duplicates = """
    SELECT 
        (params->>'symbol') as symbol,
        (params->>'timeframe') as timeframe,
        created_at,
        COUNT(*) as duplicados_timestamp
    FROM trading.agentversions
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY (params->>'symbol'), (params->>'timeframe'), created_at
    HAVING COUNT(*) > 1
    ORDER BY duplicados_timestamp DESC, created_at DESC;
    """
    
    try:
        with eng.connect() as conn:
            result = conn.execute(text(query_timestamp_duplicates))
            df_timestamp_duplicates = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_timestamp_duplicates.empty:
                print("⚠️ DUPLICADOS POR TIMESTAMP ENCONTRADOS:")
                for _, row in df_timestamp_duplicates.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    timestamp = row['created_at']
                    duplicados = row['duplicados_timestamp']
                    print(f"  🔴 {symbol} {timeframe} en {timestamp}: {duplicados} registros idénticos")
            else:
                print("  ✅ No hay duplicados por timestamp")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # 5. Verificar duplicados en agentpreds
    print("\n📊 5. DUPLICADOS EN AGENTPREDS:")
    print("-" * 30)
    
    query_agentpreds = """
    SELECT 
        symbol,
        timeframe,
        timestamp,
        COUNT(*) as duplicados_timestamp
    FROM trading.agentpreds
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY symbol, timeframe, timestamp
    HAVING COUNT(*) > 1
    ORDER BY duplicados_timestamp DESC, timestamp DESC
    LIMIT 10;
    """
    
    try:
        with eng.connect() as conn:
            result = conn.execute(text(query_agentpreds))
            df_agentpreds = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_agentpreds.empty:
                print("⚠️ DUPLICADOS EN PREDICCIONES:")
                for _, row in df_agentpreds.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    timestamp = row['timestamp']
                    duplicados = row['duplicados_timestamp']
                    print(f"  🔴 {symbol} {timeframe} en {timestamp}: {duplicados} predicciones idénticas")
            else:
                print("  ✅ No hay duplicados en predicciones")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ VERIFICACIÓN COMPLETADA")

if __name__ == "__main__":
    check_duplicates()
