#!/usr/bin/env python3
"""
Script de monitoreo de emergencia para pares problemáticos
"""

import pandas as pd
from core.data.database import get_engine
from sqlalchemy import text

def monitor_emergency():
    print("🚨 MONITOREO DE EMERGENCIA - PARES PROBLEMÁTICOS")
    print("=" * 60)
    
    eng = get_engine()
    
    # Consulta de monitoreo enfocada en pares problemáticos
    query = """
    WITH balance_config AS (
        SELECT 'ADAUSDT' as symbol, 5000.0 as balance_inicial, 5500.0 as balance_objetivo
        UNION ALL SELECT 'XRPUSDT', 5000.0, 5500.0
        UNION ALL SELECT 'SOLUSDT', 6000.0, 6600.0
        UNION ALL SELECT 'DOGEUSDT', 3000.0, 3300.0
        UNION ALL SELECT 'BTCUSDT', 1000.0, 2000.0
        UNION ALL SELECT 'ETHUSDT', 1000.0, 2000.0
    ),
    performance_summary AS (
        SELECT 
            b.symbol,
            b.timeframe,
            COUNT(*) as backtests_count,
            ROUND(AVG(b.net_pnl)::numeric, 2) as avg_pnl_total,
            ROUND((AVG(b.net_pnl) / NULLIF(AVG(b.n_trades), 0))::numeric, 2) as avg_pnl_per_trade,
            ROUND((AVG(b.win_rate) * 100)::numeric, 1) as winrate_pct,
            ROUND(AVG(b.n_trades), 0) as avg_trades,
            ROUND(AVG(p.leverage)::numeric, 1) as avg_leverage,
            COUNT(CASE WHEN p.side = 1 THEN 1 END) as longs,
            COUNT(CASE WHEN p.side = -1 THEN 1 END) as shorts
        FROM trading.backtests b
        LEFT JOIN trading.tradeplans p ON b.symbol = p.symbol AND b.timeframe = p.timeframe
        WHERE b.run_ts >= NOW() - INTERVAL '24 hours'
        GROUP BY b.symbol, b.timeframe
    )
    SELECT 
        ps.symbol,
        ps.timeframe,
        bc.balance_inicial,
        bc.balance_objetivo,
        ROUND((bc.balance_objetivo / bc.balance_inicial)::numeric, 1) as objetivo_x,
        ps.avg_pnl_total,
        ps.avg_pnl_per_trade,
        ps.winrate_pct,
        ps.avg_trades,
        ps.avg_leverage,
        ps.longs,
        ps.shorts,
        CASE 
            WHEN ps.longs + ps.shorts = 0 THEN 0
            ELSE ROUND((ps.longs::numeric / (ps.longs + ps.shorts) * 100)::numeric, 1)
        END as longs_pct,
        CASE 
            WHEN ps.longs + ps.shorts = 0 THEN 0
            ELSE ROUND((ps.shorts::numeric / (ps.longs + ps.shorts) * 100)::numeric, 1)
        END as shorts_pct,
        CASE 
            WHEN ps.avg_pnl_total > 0 THEN '✅ RENTABLE'
            WHEN ps.avg_pnl_total > -100 THEN '⚠️ PÉRDIDAS BAJAS'
            WHEN ps.avg_pnl_total > -1000 THEN '🔴 PÉRDIDAS ALTAS'
            ELSE '💀 PÉRDIDAS CRÍTICAS'
        END as estado_pnl
    FROM performance_summary ps
    LEFT JOIN balance_config bc ON ps.symbol = bc.symbol
    ORDER BY ps.avg_pnl_total DESC;
    """
    
    try:
        with eng.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if df.empty:
                print("❌ No hay datos de backtesting en las últimas 24 horas")
                return
            
            print(f"📊 RESULTADOS DE LAS ÚLTIMAS 24 HORAS:")
            print(f"Total de registros: {len(df)}")
            print()
            
            # Análisis por estado
            rentables = df[df['estado_pnl'] == '✅ RENTABLE']
            perdidas_bajas = df[df['estado_pnl'] == '⚠️ PÉRDIDAS BAJAS']
            perdidas_altas = df[df['estado_pnl'] == '🔴 PÉRDIDAS ALTAS']
            perdidas_criticas = df[df['estado_pnl'] == '💀 PÉRDIDAS CRÍTICAS']
            
            print("🎯 RESUMEN POR ESTADO:")
            print(f"✅ RENTABLES: {len(rentables)} pares")
            print(f"⚠️ PÉRDIDAS BAJAS: {len(perdidas_bajas)} pares")
            print(f"🔴 PÉRDIDAS ALTAS: {len(perdidas_altas)} pares")
            print(f"💀 PÉRDIDAS CRÍTICAS: {len(perdidas_criticas)} pares")
            print()
            
            # Mostrar detalles
            for _, row in df.iterrows():
                symbol = row['symbol']
                timeframe = row['timeframe']
                pnl = row['avg_pnl_total']
                winrate = row['winrate_pct']
                objetivo = row['objetivo_x']
                estado = row['estado_pnl']
                
                print(f"{symbol} {timeframe}: PnL={pnl:>8.2f}, WinRate={winrate:>5.1f}%, Objetivo={objetivo:>4.1f}x {estado}")
            
            print()
            
            # Recomendaciones
            if len(perdidas_criticas) > 0:
                print("🚨 ACCIÓN INMEDIATA REQUERIDA:")
                for _, row in perdidas_criticas.iterrows():
                    print(f"   - {row['symbol']} {row['timeframe']}: PnL {row['avg_pnl_total']:.2f}")
                print("   Considera deshabilitar estos pares temporalmente")
                print()
            
            if len(rentables) > 0:
                print("✅ PARES RENTABLES (mantener):")
                for _, row in rentables.iterrows():
                    print(f"   - {row['symbol']} {row['timeframe']}: PnL {row['avg_pnl_total']:.2f}")
                print()
            
            # PnL total
            pnl_total = df['avg_pnl_total'].sum()
            print(f"💰 PnL TOTAL: {pnl_total:>8.2f}")
            
            if pnl_total < -1000:
                print("🚨 PÉRDIDAS TOTALES CRÍTICAS - REVISAR CONFIGURACIÓN")
            elif pnl_total < 0:
                print("⚠️ PÉRDIDAS TOTALES - MONITOREAR DE CERCA")
            else:
                print("✅ PnL TOTAL POSITIVO - SISTEMA FUNCIONANDO")
                
    except Exception as e:
        print(f"❌ Error en monitoreo: {e}")

if __name__ == "__main__":
    monitor_emergency()
