#!/usr/bin/env python3
"""
Script para verificar cambios en PnL recientes
"""

import pandas as pd
from core.data.database import get_engine
from sqlalchemy import text

def check_pnl_changes():
    print("📊 VERIFICANDO CAMBIOS EN PnL RECIENTES")
    print("=" * 50)
    
    eng = get_engine()
    
    try:
        with eng.connect() as conn:
            # 1. PnL de backtests recientes (últimas 24 horas)
            print("1. 📈 PnL DE BACKTESTS RECIENTES (24h):")
            print("-" * 40)
            
            query_recent = """
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as total_backtests,
                ROUND(AVG(net_pnl), 2) as avg_pnl_total,
                ROUND(AVG(n_trades), 0) as avg_trades,
                ROUND(AVG(win_rate), 4) as win_rate,
                MIN(run_ts) as primer_backtest,
                MAX(run_ts) as ultimo_backtest
            FROM trading.backtests
            WHERE run_ts >= NOW() - INTERVAL '24 hours'
            GROUP BY symbol, timeframe
            ORDER BY avg_pnl_total DESC;
            """
            
            result = conn.execute(text(query_recent))
            df_recent = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_recent.empty:
                print("   Backtests de las últimas 24 horas:")
                for _, row in df_recent.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    pnl = row['avg_pnl_total']
                    trades = row['avg_trades']
                    winrate = row['win_rate']
                    total = row['total_backtests']
                    
                    status = "✅" if pnl > 0 else "❌"
                    print(f"   {status} {symbol} {timeframe}: PnL={pnl:>8.2f}, Trades={trades:>4.0f}, WR={winrate:.1%} ({total} backtests)")
            else:
                print("   ❌ No hay backtests en las últimas 24 horas")
            
            print()
            
            # 2. PnL de backtests históricos (últimos 7 días)
            print("2. 📊 PnL DE BACKTESTS HISTÓRICOS (7d):")
            print("-" * 40)
            
            query_historical = """
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as total_backtests,
                ROUND(AVG(net_pnl), 2) as avg_pnl_total,
                ROUND(AVG(n_trades), 0) as avg_trades,
                ROUND(AVG(win_rate), 4) as win_rate,
                MIN(run_ts) as primer_backtest,
                MAX(run_ts) as ultimo_backtest
            FROM trading.backtests
            WHERE run_ts >= NOW() - INTERVAL '7 days'
            GROUP BY symbol, timeframe
            ORDER BY avg_pnl_total DESC;
            """
            
            result = conn.execute(text(query_historical))
            df_historical = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_historical.empty:
                print("   Backtests de los últimos 7 días:")
                for _, row in df_historical.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    pnl = row['avg_pnl_total']
                    trades = row['avg_trades']
                    winrate = row['win_rate']
                    total = row['total_backtests']
                    
                    status = "✅" if pnl > 0 else "❌"
                    print(f"   {status} {symbol} {timeframe}: PnL={pnl:>8.2f}, Trades={trades:>4.0f}, WR={winrate:.1%} ({total} backtests)")
            else:
                print("   ❌ No hay backtests en los últimos 7 días")
            
            print()
            
            # 3. Comparar cambios
            print("3. 🔄 COMPARACIÓN DE CAMBIOS:")
            print("-" * 40)
            
            if not df_recent.empty and not df_historical.empty:
                # Crear un merge para comparar
                df_compare = pd.merge(
                    df_recent[['symbol', 'timeframe', 'avg_pnl_total']].rename(columns={'avg_pnl_total': 'pnl_24h'}),
                    df_historical[['symbol', 'timeframe', 'avg_pnl_total']].rename(columns={'avg_pnl_total': 'pnl_7d'}),
                    on=['symbol', 'timeframe'],
                    how='outer'
                )
                
                print("   Cambios en PnL (24h vs 7d):")
                for _, row in df_compare.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    pnl_24h = row['pnl_24h'] if pd.notna(row['pnl_24h']) else 0
                    pnl_7d = row['pnl_7d'] if pd.notna(row['pnl_7d']) else 0
                    change = pnl_24h - pnl_7d
                    
                    if change > 0:
                        trend = "📈 MEJORANDO"
                    elif change < 0:
                        trend = "📉 EMPEORANDO"
                    else:
                        trend = "➡️ ESTABLE"
                    
                    print(f"   {symbol} {timeframe}: {pnl_7d:>8.2f} → {pnl_24h:>8.2f} ({change:>+8.2f}) {trend}")
            else:
                print("   ❌ No hay datos suficientes para comparar")
            
            print()
            
            # 4. Próximos backtests esperados
            print("4. ⏰ PRÓXIMOS BACKTESTS ESPERADOS:")
            print("-" * 40)
            
            # Verificar modelos promovidos recientes que podrían generar backtests
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
                print("   Modelos promovidos recientes (podrían generar backtests):")
                for _, row in df_promoted.iterrows():
                    symbol = row['symbol']
                    timeframe = row['timeframe']
                    auc = row['auc']
                    created = row['created_at']
                    print(f"   ✅ {symbol} {timeframe}: AUC={auc:.4f} (promovido: {created.strftime('%H:%M')})")
            else:
                print("   ❌ No hay modelos promovidos recientes")
            
            print()
            
            # 5. Resumen
            print("5. 📋 RESUMEN:")
            print("-" * 40)
            
            if not df_recent.empty:
                positive_pnl = len(df_recent[df_recent['avg_pnl_total'] > 0])
                total_pairs = len(df_recent)
                print(f"   📊 Pares rentables (24h): {positive_pnl}/{total_pairs}")
                print(f"   📈 Mejores performers: {df_recent.head(3)['symbol'].tolist()}")
                print(f"   📉 Peores performers: {df_recent.tail(3)['symbol'].tolist()}")
            else:
                print("   ❌ No hay datos recientes para analizar")
            
            print("\n💡 NOTA: Los PnL cambian cuando se ejecutan nuevos backtests.")
            print("   Los backtests se ejecutan después de cada promoción de modelo.")
            print("   Monitorea cada 30 minutos para ver cambios.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_pnl_changes()
