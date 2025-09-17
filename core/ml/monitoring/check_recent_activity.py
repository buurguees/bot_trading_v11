#!/usr/bin/env python3
"""
Script para verificar actividad reciente del sistema
"""

import pandas as pd
from core.data.database import get_engine
from sqlalchemy import text

def check_recent_activity():
    print("🔄 VERIFICANDO ACTIVIDAD RECIENTE DEL SISTEMA")
    print("=" * 60)
    
    eng = get_engine()
    
    try:
        with eng.connect() as conn:
            # 1. Verificar entrenamientos recientes
            print("1. 🎯 ENTRENAMIENTOS (últimas 2 horas):")
            print("-" * 40)
            
            query_trainings = """
            SELECT 
                (params->>'symbol') as symbol,
                (params->>'timeframe') as timeframe,
                created_at,
                promoted,
                COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8) as auc
            FROM trading.agentversions
            WHERE created_at >= NOW() - INTERVAL '2 hours'
            ORDER BY created_at DESC;
            """
            
            result = conn.execute(text(query_trainings))
            df_trainings = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_trainings.empty:
                print(f"   Total entrenamientos: {len(df_trainings)}")
                promoted = df_trainings[df_trainings['promoted'] == True]
                print(f"   Promovidos: {len(promoted)}")
                
                for _, row in df_trainings.head(10).iterrows():
                    status = "✅ PROMOVIDO" if row['promoted'] else "⏳ ENTRENANDO"
                    print(f"   {row['symbol']} {row['timeframe']}: AUC={row['auc']:.4f} {status}")
            else:
                print("   ❌ No hay entrenamientos en las últimas 2 horas")
            
            print()
            
            # 2. Verificar predicciones recientes
            print("2. 🔮 PREDICCIONES (últimas 2 horas):")
            print("-" * 40)
            
            query_predictions = """
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as total_predicciones,
                MIN(created_at) as primera,
                MAX(created_at) as ultima
            FROM trading.agentpreds
            WHERE created_at >= NOW() - INTERVAL '2 hours'
            GROUP BY symbol, timeframe
            ORDER BY total_predicciones DESC;
            """
            
            result = conn.execute(text(query_predictions))
            df_predictions = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_predictions.empty:
                print(f"   Total predicciones: {df_predictions['total_predicciones'].sum()}")
                for _, row in df_predictions.iterrows():
                    print(f"   {row['symbol']} {row['timeframe']}: {row['total_predicciones']} predicciones")
            else:
                print("   ❌ No hay predicciones en las últimas 2 horas")
            
            print()
            
            # 3. Verificar planes recientes
            print("3. 📋 PLANES DE TRADING (últimas 2 horas):")
            print("-" * 40)
            
            query_plans = """
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as total_planes,
                COUNT(CASE WHEN side = 1 THEN 1 END) as longs,
                COUNT(CASE WHEN side = -1 THEN 1 END) as shorts,
                MIN(created_at) as primero,
                MAX(created_at) as ultimo
            FROM trading.tradeplans
            WHERE created_at >= NOW() - INTERVAL '2 hours'
            GROUP BY symbol, timeframe
            ORDER BY total_planes DESC;
            """
            
            result = conn.execute(text(query_plans))
            df_plans = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_plans.empty:
                print(f"   Total planes: {df_plans['total_planes'].sum()}")
                for _, row in df_plans.iterrows():
                    print(f"   {row['symbol']} {row['timeframe']}: {row['total_planes']} planes ({row['longs']}L/{row['shorts']}S)")
            else:
                print("   ❌ No hay planes en las últimas 2 horas")
            
            print()
            
            # 4. Verificar backtests recientes
            print("4. 📊 BACKTESTS (últimas 2 horas):")
            print("-" * 40)
            
            query_backtests = """
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as total_backtests,
                ROUND(AVG(net_pnl), 2) as avg_pnl,
                ROUND(AVG(n_trades), 0) as avg_trades,
                MAX(run_ts) as ultimo_backtest
            FROM trading.backtests
            WHERE run_ts >= NOW() - INTERVAL '2 hours'
            GROUP BY symbol, timeframe
            ORDER BY avg_pnl DESC;
            """
            
            result = conn.execute(text(query_backtests))
            df_backtests = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df_backtests.empty:
                print(f"   Total backtests: {len(df_backtests)}")
                for _, row in df_backtests.iterrows():
                    print(f"   {row['symbol']} {row['timeframe']}: PnL={row['avg_pnl']:>8.2f}, Trades={row['avg_trades']:>4.0f}")
            else:
                print("   ❌ No hay backtests en las últimas 2 horas")
            
            print()
            
            # 5. Verificar el estado del runner
            print("5. 🏃 ESTADO DEL RUNNER:")
            print("-" * 40)
            
            # Verificar si hay locks activos
            query_locks = """
            SELECT 
                lockname,
                granted,
                pid,
                mode,
                granted_at
            FROM pg_locks 
            WHERE lockname LIKE 'daily_train:%'
            ORDER BY granted_at DESC;
            """
            
            result = conn.execute(text(query_locks))
            locks = result.fetchall()
            
            if locks:
                print(f"   🔒 Locks activos: {len(locks)}")
                for lock in locks:
                    status = "✅ ACTIVO" if lock[1] else "⏳ ESPERANDO"
                    print(f"   {lock[0]}: {status} (PID: {lock[2]})")
            else:
                print("   ❌ No hay locks activos - Runner no está ejecutándose")
            
            print()
            
            # 6. Resumen de actividad
            print("6. 📈 RESUMEN DE ACTIVIDAD:")
            print("-" * 40)
            
            total_activity = len(df_trainings) + len(df_predictions) + len(df_plans) + len(df_backtests)
            
            if total_activity > 0:
                print(f"   ✅ Sistema ACTIVO - {total_activity} actividades en las últimas 2 horas")
                print("   📋 Próximos pasos:")
                print("      - Los PnL se actualizarán cuando se ejecuten nuevos backtests")
                print("      - Los backtests se ejecutan después de cada promoción")
                print("      - Monitorea cada 30 minutos para ver cambios")
            else:
                print("   ❌ Sistema INACTIVO - No hay actividad reciente")
                print("   📋 Acciones recomendadas:")
                print("      - Verificar que el runner esté ejecutándose")
                print("      - Revisar logs del sistema")
                print("      - Reiniciar el entrenamiento si es necesario")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_recent_activity()
