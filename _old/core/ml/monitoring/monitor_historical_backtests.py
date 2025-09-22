#!/usr/bin/env python3
"""
Script para monitorear backtests con datos históricos completos
"""

import pandas as pd
from core.data.database import get_engine
from sqlalchemy import text

def monitor_historical_backtests():
    print("📊 MONITOREO DE BACKTESTS HISTÓRICOS")
    print("=" * 50)
    
    eng = get_engine()
    
    try:
        with eng.connect() as conn:
            # 1. Backtests recientes con datos históricos
            print("1. 📈 BACKTESTS CON DATOS HISTÓRICOS (últimas 24h):")
            print("-" * 50)
            
            query = """
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as total_backtests,
                ROUND(AVG(net_pnl), 2) as avg_pnl_total,
                ROUND(AVG(n_trades), 0) as avg_trades,
                ROUND(AVG(win_rate), 4) as win_rate,
                ROUND(AVG(gross_pnl), 2) as avg_gross_pnl,
                ROUND(AVG(fees), 2) as avg_fees,
                ROUND(AVG(max_dd), 4) as avg_max_drawdown,
                MIN(run_ts) as primer_backtest,
                MAX(run_ts) as ultimo_backtest,
                ROUND(EXTRACT(EPOCH FROM (MAX(run_ts) - MIN(run_ts)))/3600, 1) as horas_cobertura
            FROM trading.backtests
            WHERE run_ts >= NOW() - INTERVAL '24 hours'
            GROUP BY symbol, timeframe
            ORDER BY avg_pnl_total DESC;
            """
            
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df.empty:
                print("   Backtests con datos históricos completos:")
                for _, row in df.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    pnl = row['avg_pnl_total']
                    trades = row['avg_trades']
                    winrate = row['win_rate']
                    gross_pnl = row['avg_gross_pnl']
                    fees = row['avg_fees']
                    max_dd = row['avg_max_drawdown']
                    cobertura = row['horas_cobertura']
                    total = row['total_backtests']
                    
                    status = "✅" if pnl > 0 else "❌"
                    print(f"   {status} {symbol} {timeframe}:")
                    print(f"      PnL: {pnl:>8.2f} | Trades: {trades:>4.0f} | WR: {winrate:.1%}")
                    print(f"      Gross: {gross_pnl:>6.2f} | Fees: {fees:>6.2f} | MaxDD: {max_dd:.2%}")
                    print(f"      Cobertura: {cobertura:>4.1f}h | Backtests: {total}")
                    print()
            else:
                print("   ❌ No hay backtests con datos históricos en las últimas 24 horas")
            
            # 2. Análisis de cobertura temporal
            print("2. ⏰ ANÁLISIS DE COBERTURA TEMPORAL:")
            print("-" * 50)
            
            query_coverage = """
            SELECT 
                symbol,
                timeframe,
                MIN(run_ts) as fecha_inicio,
                MAX(run_ts) as fecha_fin,
                ROUND(EXTRACT(EPOCH FROM (MAX(run_ts) - MIN(run_ts)))/86400, 1) as dias_cobertura,
                COUNT(*) as total_backtests
            FROM trading.backtests
            WHERE run_ts >= NOW() - INTERVAL '7 days'
            GROUP BY symbol, timeframe
            ORDER BY dias_cobertura DESC;
            """
            
            result = conn.execute(text(query_coverage))
            df_coverage = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_coverage.empty:
                print("   Cobertura temporal de backtests:")
                for _, row in df_coverage.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    inicio = row['fecha_inicio']
                    fin = row['fecha_fin']
                    dias = row['dias_cobertura']
                    total = row['total_backtests']
                    
                    print(f"   {symbol} {timeframe}: {inicio.strftime('%Y-%m-%d')} → {fin.strftime('%Y-%m-%d')} ({dias} días, {total} backtests)")
            else:
                print("   ❌ No hay datos de cobertura")
            
            # 3. Resumen de rendimiento
            print("\n3. 📊 RESUMEN DE RENDIMIENTO:")
            print("-" * 50)
            
            if not df.empty:
                positive_pairs = len(df[df['avg_pnl_total'] > 0])
                total_pairs = len(df)
                avg_pnl = df['avg_pnl_total'].mean()
                total_trades = df['avg_trades'].sum()
                avg_winrate = df['win_rate'].mean()
                
                print(f"   Pares rentables: {positive_pairs}/{total_pairs} ({positive_pairs/total_pairs*100:.1f}%)")
                print(f"   PnL promedio: {avg_pnl:>8.2f}")
                print(f"   Total trades: {total_trades:>8.0f}")
                print(f"   Win rate promedio: {avg_winrate:.1%}")
                
                # Mejores y peores performers
                best = df.loc[df['avg_pnl_total'].idxmax()]
                worst = df.loc[df['avg_pnl_total'].idxmin()]
                
                print(f"\n   🏆 Mejor: {best['symbol']} {best['timeframe']} (PnL: {best['avg_pnl_total']:.2f})")
                print(f"   🔻 Peor: {worst['symbol']} {worst['timeframe']} (PnL: {worst['avg_pnl_total']:.2f})")
            
            # 4. Próximos backtests esperados
            print("\n4. ⏰ PRÓXIMOS BACKTESTS HISTÓRICOS:")
            print("-" * 50)
            
            query_promoted = """
            SELECT 
                (params->>'symbol') as symbol,
                (params->>'timeframe') as timeframe,
                created_at,
                COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8) as auc
            FROM trading.agentversions
            WHERE promoted = true
            AND created_at >= NOW() - INTERVAL '2 hours'
            ORDER BY created_at DESC;
            """
            
            result = conn.execute(text(query_promoted))
            df_promoted = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_promoted.empty:
                print("   Modelos promovidos recientes (generarán backtests históricos):")
                for _, row in df_promoted.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    auc = row['auc']
                    created = row['created_at']
                    print(f"   ✅ {symbol} {timeframe}: AUC={auc:.4f} (promovido: {created.strftime('%H:%M')})")
            else:
                print("   ❌ No hay modelos promovidos recientes")
            
            print("\n💡 NOTA: Los backtests ahora usan TODO el historial disponible (365+ días)")
            print("   Esto proporciona una evaluación más robusta del rendimiento.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    monitor_historical_backtests()
