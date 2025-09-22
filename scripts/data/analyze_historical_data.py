#!/usr/bin/env python3
"""
Script para analizar datos históricos por símbolo
Uso: python scripts/data/analyze_historical_data.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data.database import MarketDB
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

def analyze_historical_data():
    print('=== ANÁLISIS DE DATOS HISTÓRICOS POR SÍMBOLO ===')
    print('')
    
    with engine.begin() as conn:
        # 1. Resumen general por símbolo
        print('📊 RESUMEN GENERAL POR SÍMBOLO:')
        result = conn.execute(text('''
            SELECT 
                symbol,
                COUNT(*) as total_records,
                MIN(ts) as first_date,
                MAX(ts) as last_date,
                COUNT(DISTINCT timeframe) as timeframes_count,
                EXTRACT(DAYS FROM MAX(ts) - MIN(ts)) as total_days
            FROM market.historical_data 
            GROUP BY symbol
            ORDER BY total_records DESC
        '''))
        
        historical_data = result.fetchall()
        for row in historical_data:
            print(f'   {row[0]:<10} | {row[1]:>8} registros | {row[2]} → {row[3]} | {row[4]:>2} TFs | {row[5]:>3} días')
        
        print('')
        
        # 2. Detalle por símbolo y timeframe
        print('📈 DETALLE POR SÍMBOLO Y TIMEFRAME:')
        result = conn.execute(text('''
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as records,
                MIN(ts) as first_date,
                MAX(ts) as last_date,
                EXTRACT(DAYS FROM MAX(ts) - MIN(ts)) as days_span
            FROM market.historical_data 
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        '''))
        
        tf_data = result.fetchall()
        current_symbol = None
        for row in tf_data:
            if current_symbol != row[0]:
                current_symbol = row[0]
                print(f'\\n   {row[0]}:')
            print(f'     {row[1]:<4} | {row[2]:>6} registros | {row[3]} → {row[4]} | {row[5]:>3} días')
        
        print('')
        
        # 3. Análisis de completitud (últimos 7 días)
        print('🔍 COMPLETITUD ÚLTIMOS 7 DÍAS:')
        result = conn.execute(text('''
            WITH expected_counts AS (
                SELECT 
                    '1m' as timeframe, 7 * 24 * 60 as expected
                UNION ALL SELECT '5m', 7 * 24 * 12
                UNION ALL SELECT '15m', 7 * 24 * 4
                UNION ALL SELECT '1h', 7 * 24
                UNION ALL SELECT '4h', 7 * 6
                UNION ALL SELECT '1d', 7
            ),
            actual_counts AS (
                SELECT 
                    symbol,
                    timeframe,
                    COUNT(*) as actual
                FROM market.historical_data 
                WHERE ts >= NOW() - INTERVAL '7 days'
                GROUP BY symbol, timeframe
            )
            SELECT 
                a.symbol,
                a.timeframe,
                a.actual,
                e.expected,
                ROUND(a.actual::numeric / e.expected * 100, 1) as completeness_pct
            FROM actual_counts a
            JOIN expected_counts e ON a.timeframe = e.timeframe
            ORDER BY a.symbol, a.timeframe
        '''))
        
        completeness_data = result.fetchall()
        current_symbol = None
        for row in completeness_data:
            if current_symbol != row[0]:
                current_symbol = row[0]
                print(f'\\n   {row[0]}:')
            
            status = '✅' if row[4] >= 95 else '⚠️' if row[4] >= 80 else '❌'
            print(f'     {status} {row[1]:<4} | {row[2]:>4}/{row[3]:<4} | {row[4]:>5}%')
        
        print('')
        
        # 4. Verificar gaps recientes
        print('⚠️ GAPS DETECTADOS (últimos 3 días):')
        result = conn.execute(text('''
            WITH time_series AS (
                SELECT 
                    symbol,
                    timeframe,
                    ts,
                    LAG(ts) OVER (PARTITION BY symbol, timeframe ORDER BY ts) as prev_ts
                FROM market.historical_data 
                WHERE ts >= NOW() - INTERVAL '3 days'
            )
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as gap_count
            FROM time_series
            WHERE prev_ts IS NOT NULL 
              AND ts - prev_ts > CASE 
                  WHEN timeframe = '1m' THEN INTERVAL '2 minutes'
                  WHEN timeframe = '5m' THEN INTERVAL '10 minutes'
                  WHEN timeframe = '15m' THEN INTERVAL '30 minutes'
                  WHEN timeframe = '1h' THEN INTERVAL '2 hours'
                  WHEN timeframe = '4h' THEN INTERVAL '8 hours'
                  WHEN timeframe = '1d' THEN INTERVAL '2 days'
              END
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        '''))
        
        gaps_data = result.fetchall()
        if gaps_data:
            for row in gaps_data:
                print(f'   ⚠️ {row[0]:<10} {row[1]:<4} | {row[2]:>2} gaps')
        else:
            print('   ✅ No se detectaron gaps significativos')
        
        print('')
        
        # 5. Resumen de calidad
        print('📊 RESUMEN DE CALIDAD:')
        result = conn.execute(text('''
            SELECT 
                symbol,
                COUNT(DISTINCT timeframe) as timeframes_available,
                MIN(ts) as data_start,
                MAX(ts) as data_end,
                EXTRACT(DAYS FROM MAX(ts) - MIN(ts)) as total_days,
                COUNT(*) as total_records
            FROM market.historical_data 
            GROUP BY symbol
            ORDER BY total_days DESC
        '''))
        
        summary_data = result.fetchall()
        print('   Símbolo    | TFs | Inicio                | Fin                  | Días | Registros')
        print('   -----------|-----|-----------------------|----------------------|------|----------')
        for row in summary_data:
            print(f'   {row[0]:<10} | {row[1]:>3} | {row[2]} | {row[3]} | {row[4]:>4} | {row[5]:>9}')
        
        print('')
        
        # 6. Análisis de features
        print('🔧 ANÁLISIS DE FEATURES:')
        result = conn.execute(text('''
            SELECT 
                symbol,
                COUNT(*) as feature_records,
                MIN(ts) as first_feature,
                MAX(ts) as last_feature
            FROM market.features 
            GROUP BY symbol
            ORDER BY feature_records DESC
        '''))
        
        features_data = result.fetchall()
        for row in features_data:
            print(f'   {row[0]:<10} | {row[1]:>8} features | {row[2]} → {row[3]}')
        
        print('')
        
        # 7. Comparación historical vs features
        print('📊 COMPARACIÓN HISTORICAL vs FEATURES:')
        result = conn.execute(text('''
            WITH historical_summary AS (
                SELECT symbol, COUNT(*) as hist_count, MAX(ts) as hist_max
                FROM market.historical_data 
                GROUP BY symbol
            ),
            features_summary AS (
                SELECT symbol, COUNT(*) as feat_count, MAX(ts) as feat_max
                FROM market.features 
                GROUP BY symbol
            )
            SELECT 
                h.symbol,
                h.hist_count,
                f.feat_count,
                h.hist_max,
                f.feat_max,
                CASE 
                    WHEN f.feat_count IS NULL THEN '❌ Sin features'
                    WHEN f.feat_count < h.hist_count * 0.8 THEN '⚠️ Features incompletos'
                    ELSE '✅ Features OK'
                END as status
            FROM historical_summary h
            LEFT JOIN features_summary f ON h.symbol = f.symbol
            ORDER BY h.symbol
        '''))
        
        comparison_data = result.fetchall()
        for row in comparison_data:
            print(f'   {row[0]:<10} | {row[1]:>8} hist | {row[2]:>8} feat | {row[5]}')

def show_help():
    print('=== AYUDA: ANÁLISIS DE DATOS HISTÓRICOS ===')
    print('')
    print('USO:')
    print('   python scripts/data/analyze_historical_data.py')
    print('')
    print('FUNCIONES:')
    print('   - Resumen general por símbolo')
    print('   - Detalle por símbolo y timeframe')
    print('   - Análisis de completitud (últimos 7 días)')
    print('   - Detección de gaps')
    print('   - Comparación historical vs features')
    print('')
    print('SÍMBOLOS DISPONIBLES:')
    print('   - ADAUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, SOLUSDT, XRPUSDT')
    print('')
    print('TIMEFRAMES:')
    print('   - 1m, 5m, 15m, 1h, 4h, 1d')
    print('')
    print('ESTADOS:')
    print('   ✅ = Completitud >= 95%')
    print('   ⚠️ = Completitud 80-94%')
    print('   ❌ = Completitud < 80%')

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
    else:
        try:
            analyze_historical_data()
        except Exception as e:
            print(f'❌ Error: {e}')
            print('')
            print('💡 Sugerencias:')
            print('   - Verificar conexión a la base de datos')
            print('   - Verificar que DB_URL esté configurado en .env')
            print('   - Ejecutar: python scripts/data/analyze_historical_data.py --help')
