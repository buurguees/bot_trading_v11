#!/usr/bin/env python3
"""
Script para debuggear problemas del pipeline
Identifica exactamente qué está fallando en Data Quality y Feature Pipeline
"""

import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")
APP_TZ = ZoneInfo("Europe/Madrid")

def check_database_connection():
    """Verifica conexión a la base de datos"""
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
        with engine.begin() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        print("✅ Conexión a BD: OK")
        return engine
    except Exception as e:
        print(f"❌ Error conexión BD: {e}")
        return None

def check_tables_exist(engine):
    """Verifica qué tablas existen"""
    print("\n🔍 VERIFICANDO TABLAS:")
    
    tables_to_check = [
        ("market", "historical_data"),
        ("market", "features"),
        ("ml", "agent_preds"),
        ("ml", "strategies"),
        ("ml", "backtest_runs"),
        ("trading", "trade_plans")
    ]
    
    with engine.begin() as conn:
        for schema, table in tables_to_check:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table}")).scalar()
                print(f"✅ {schema}.{table}: {count} registros")
            except Exception as e:
                print(f"❌ {schema}.{table}: ERROR - {e}")

def check_data_quality(engine):
    """Analiza problemas específicos de calidad de datos"""
    print("\n📊 ANÁLISIS DETALLADO DE DATA QUALITY:")
    
    try:
        with engine.begin() as conn:
            # 1. Inicio y fin por símbolo/timeframe
            print("\n1. INICIO/FIN POR SÍMBOLO:")
            result = conn.execute(text("""
                SELECT 
                    symbol,
                    timeframe,
                    MIN(ts) as first_ts,
                    MAX(ts) as last_ts,
                    COUNT(*) as total_records,
                    EXTRACT(EPOCH FROM (NOW() - MAX(ts))) / 60 as minutes_old
                FROM market.historical_data
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """)).mappings().all()
            
            for row in result:
                age_indicator = "🔴" if row['minutes_old'] > 60 else "🟡" if row['minutes_old'] > 10 else "🟢"
                print(f"   {age_indicator} {row['symbol']:10} {row['timeframe']:4} | "
                      f"Registros: {row['total_records']:>6} | "
                      f"Inicio: {row['first_ts']} | "
                      f"Fin: {row['last_ts']} ({row['minutes_old']:.1f}min)")

            # 2. Verificar gaps en datos
            print("\n2. GAPS EN DATOS (últimas 24h):")
            gap_check = conn.execute(text("""
                WITH time_gaps AS (
                    SELECT 
                        symbol, timeframe, ts,
                        LAG(ts) OVER (PARTITION BY symbol, timeframe ORDER BY ts) as prev_ts,
                        CASE 
                            WHEN timeframe = '1m' THEN INTERVAL '2 minutes'
                            WHEN timeframe = '5m' THEN INTERVAL '10 minutes'
                            WHEN timeframe = '15m' THEN INTERVAL '30 minutes'
                            WHEN timeframe = '1h' THEN INTERVAL '2 hours'
                            WHEN timeframe = '4h' THEN INTERVAL '8 hours'
                            WHEN timeframe = '1d' THEN INTERVAL '2 days'
                        END as max_gap
                    FROM market.historical_data
                    WHERE ts >= NOW() - INTERVAL '24 hours'
                )
                SELECT 
                    symbol, timeframe,
                    COUNT(*) as gap_count
                FROM time_gaps
                WHERE prev_ts IS NOT NULL 
                  AND ts - prev_ts > max_gap
                GROUP BY symbol, timeframe
                HAVING COUNT(*) > 0
                ORDER BY gap_count DESC
            """)).mappings().all()
            
            if gap_check:
                for row in gap_check:
                    print(f"   ⚠️  {row['symbol']} {row['timeframe']}: {row['gap_count']} gaps detectados")
            else:
                print("   ✅ No se detectaron gaps significativos")

            # 3. Rango global de histórico y de features
            print("\n3. RANGO GLOBAL (TODOS LOS SÍMBOLOS/TFs):")
            g_hd = conn.execute(text("""
                SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n
                FROM market.historical_data
            """)).mappings().first()
            g_ft = conn.execute(text("""
                SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n
                FROM market.features
            """)).mappings().first()
            print(f"   historical_data → Inicio: {g_hd['first_ts']} | Fin: {g_hd['last_ts']} | Registros: {g_hd['n'] if g_hd else 0}")
            print(f"   features        → Inicio: {g_ft['first_ts']} | Fin: {g_ft['last_ts']} | Registros: {g_ft['n'] if g_ft else 0}")

    except Exception as e:
        print(f"❌ Error en análisis de datos: {e}")

def check_feature_pipeline(engine):
    """Analiza problemas del feature pipeline"""
    print("\n🔧 ANÁLISIS DEL FEATURE PIPELINE:")
    
    try:
        with engine.begin() as conn:
            # 1. Features vs datos históricos
            print("\n1. COBERTURA DE FEATURES:")
            result = conn.execute(text("""
                SELECT 
                    h.symbol,
                    h.timeframe,
                    COUNT(h.ts) as historical_count,
                    COUNT(f.ts) as features_count,
                    ROUND(CAST((COUNT(f.ts)::float / COUNT(h.ts)) * 100 AS NUMERIC), 1) as coverage_pct,
                    MAX(f.ts) as last_feature,
                    EXTRACT(EPOCH FROM (NOW() - MAX(f.ts))) / 60 as feature_lag_minutes
                FROM market.historical_data h
                LEFT JOIN market.features f ON h.symbol = f.symbol 
                    AND h.timeframe = f.timeframe 
                    AND h.ts = f.ts
                WHERE h.ts >= NOW() - INTERVAL '7 days'
                GROUP BY h.symbol, h.timeframe
                ORDER BY coverage_pct ASC
            """)).mappings().all()
            
            for row in result:
                coverage = row['coverage_pct'] or 0
                status = "🟢" if coverage >= 95 else "🟡" if coverage >= 80 else "🔴"
                lag = row['feature_lag_minutes'] if row['feature_lag_minutes'] else 999
                
                print(f"   {status} {row['symbol']:10} {row['timeframe']:4} | "
                      f"Cobertura: {coverage:5.1f}% | "
                      f"Features: {row['features_count']:>5}/{row['historical_count']:<5} | "
                      f"Lag: {lag:.1f}min")

            # 2. Verificar columnas de features
            print("\n2. ESTRUCTURA DE FEATURES:")
            columns_check = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'market' AND table_name = 'features'
                ORDER BY ordinal_position
            """)).mappings().all()
            
            expected_columns = [
                'symbol', 'timeframe', 'ts', 'rsi_14', 'macd', 'macd_signal', 
                'macd_hist', 'ema_20', 'ema_50', 'ema_200', 'atr_14', 'obv', 
                'supertrend', 'st_direction', 'smc_flags'
            ]
            
            existing_columns = [row['column_name'] for row in columns_check]
            
            for col in expected_columns:
                if col in existing_columns:
                    print(f"   ✅ {col}")
                else:
                    print(f"   ❌ {col} - FALTANTE")

    except Exception as e:
        print(f"❌ Error en análisis de features: {e}")

def check_predictions(engine):
    """Verifica las predicciones"""
    print("\n🤖 ANÁLISIS DE PREDICCIONES:")
    
    try:
        with engine.begin() as conn:
            # Últimas predicciones por task
            result = conn.execute(text("""
                SELECT 
                    task,
                    COUNT(*) as predictions_1h,
                    AVG(pred_conf) as avg_confidence,
                    MAX(created_at) as last_prediction,
                    EXTRACT(EPOCH FROM (NOW() - MAX(created_at))) / 60 as last_pred_minutes_ago
                FROM ml.agent_preds
                WHERE created_at >= NOW() - INTERVAL '1 hour'
                GROUP BY task
                ORDER BY task
            """)).mappings().all()
            
            if result:
                for row in result:
                    status = "🟢" if row['last_pred_minutes_ago'] < 30 else "🟡" if row['last_pred_minutes_ago'] < 120 else "🔴"
                    print(f"   {status} {row['task']:12} | "
                          f"Predicciones: {row['predictions_1h']:>3} | "
                          f"Confianza: {row['avg_confidence']:.3f} | "
                          f"Última: {row['last_pred_minutes_ago']:.1f}min")
            else:
                print("   ❌ No hay predicciones en la última hora")

    except Exception as e:
        print(f"❌ Error en análisis de predicciones: {e}")

def suggest_fixes():
    """Sugiere soluciones basadas en los problemas encontrados"""
    print("\n💡 POSIBLES SOLUCIONES:")
    print("="*50)
    print("1. Si hay datos desactualizados (>60min):")
    print("   → Ejecutar: python -m core.data.realtime_updater")
    print("")
    print("2. Si faltan features:")
    print("   → Ejecutar: python -m core.ml.feature_engineer")
    print("")
    print("3. Si hay gaps en datos:")
    print("   → Ejecutar: python -m core.data.historical_downloader --mode repair")
    print("")
    print("4. Si no hay predicciones:")
    print("   → Verificar: python -m core.agents.agent_direction")
    print("   → Verificar: python -m core.agents.agent_regime")
    print("   → Verificar: python -m core.agents.agent_smc")
    print("")
    print("5. Para restart completo del pipeline:")
    print("   → Ejecutar: python scripts/run_phase1.py")

def main():
    print("🔍 DIAGNÓSTICO DEL PIPELINE BOT TRADING v11")
    print("="*60)
    print(f"Timestamp: {datetime.now(APP_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # 1. Verificar conexión
    engine = check_database_connection()
    if not engine:
        return
    
    # 2. Verificar tablas
    check_tables_exist(engine)
    
    # 3. Análisis detallado
    check_data_quality(engine)
    check_feature_pipeline(engine)
    check_predictions(engine)
    
    # 4. Sugerencias
    suggest_fixes()
    
    print("\n" + "="*60)
    print("✅ Diagnóstico completado")

if __name__ == "__main__":
    main()