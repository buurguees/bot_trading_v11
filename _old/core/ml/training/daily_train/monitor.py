# core/ml/training/daily_train/monitor.py

import time
import datetime as dt
from sqlalchemy import text
from core.data.database import get_engine

def monitor_system():
    """Monitorea el estado del sistema de entrenamiento en tiempo real"""
    engine = get_engine()
    
    while True:
        try:
            with engine.begin() as conn:
                # Estado de versiones recientes
                result = conn.execute(text("""
                    SELECT 
                        (params->>'symbol') as symbol,
                        (params->>'timeframe') as timeframe,
                        COUNT(*) as total_versions,
                        COUNT(CASE WHEN promoted = true THEN 1 END) as promoted_versions,
                        MAX(created_at) as last_training,
                        AVG((metrics->>'auc')::float8) as avg_auc,
                        AVG((metrics->>'brier')::float8) as avg_brier,
                        AVG((metrics->>'acc')::float8) as avg_acc
                    FROM trading.agentversions 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY (params->>'symbol'), (params->>'timeframe')
                    ORDER BY last_training DESC
                """))
                
                print(f"\n=== ESTADO DEL SISTEMA - {dt.datetime.now()} ===")
                print("Símbolo | TF  | Versiones | Promovidas | Último Entrenamiento | AUC Promedio | Brier Promedio | Acc Promedio")
                print("-" * 120)
                
                for row in result:
                    last_training = row[4].strftime("%Y-%m-%d %H:%M") if row[4] else "N/A"
                    print(f"{row[0]:8} | {row[1]:2} | {row[2]:9} | {row[3]:11} | {last_training:19} | {row[5]:11.4f} | {row[6]:13.4f} | {row[7]:11.4f}")
                
                # Estado de predicciones
                result = conn.execute(text("""
                    SELECT 
                        symbol,
                        timeframe,
                        COUNT(*) as total_preds,
                        MIN(timestamp) as desde,
                        MAX(timestamp) as hasta
                    FROM trading.agentpreds 
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                    GROUP BY symbol, timeframe
                    ORDER BY total_preds DESC
                """))
                
                print(f"\n=== PREDICCIONES (última hora) ===")
                print("Símbolo | TF  | Predicciones | Desde | Hasta")
                print("-" * 60)
                
                for row in result:
                    desde = row[3].strftime("%H:%M:%S") if row[3] else "N/A"
                    hasta = row[4].strftime("%H:%M:%S") if row[4] else "N/A"
                    print(f"{row[0]:8} | {row[1]:2} | {row[2]:12} | {desde:19} | {hasta:19}")
                
                # Estado de planes de trading
                result = conn.execute(text("""
                    SELECT 
                        symbol,
                        timeframe,
                        COUNT(*) as total_plans,
                        COUNT(CASE WHEN status = 'planned' THEN 1 END) as planned,
                        COUNT(CASE WHEN status = 'openable' THEN 1 END) as openable,
                        AVG(leverage) as avg_leverage
                    FROM trading.tradeplans 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY symbol, timeframe
                    ORDER BY total_plans DESC
                """))
                
                print(f"\n=== PLANES DE TRADING (última hora) ===")
                print("Símbolo | TF  | Total | Planned | Openable | Avg Leverage")
                print("-" * 70)
                
                for row in result:
                    print(f"{row[0]:8} | {row[1]:2} | {row[2]:5} | {row[3]:7} | {row[4]:8} | {row[5]:12.2f}")
                
                # Estado de backtests
                result = conn.execute(text("""
                    SELECT 
                        symbol,
                        timeframe,
                        COUNT(*) as total_backtests,
                        AVG(n_trades) as avg_trades,
                        AVG(pnl) as avg_pnl,
                        AVG(win_rate) as avg_win_rate
                    FROM trading.backtests 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY symbol, timeframe
                    ORDER BY total_backtests DESC
                """))
                
                print(f"\n=== BACKTESTS (últimas 24h) ===")
                print("Símbolo | TF  | Backtests | Avg Trades | Avg PnL | Avg Win Rate")
                print("-" * 70)
                
                for row in result:
                    print(f"{row[0]:8} | {row[1]:2} | {row[2]:9} | {row[3]:10} | {row[4]:7.2f} | {row[5]:12.4f}")
                
                print(f"\n{'='*120}")
                
        except Exception as e:
            print(f"Error en monitoreo: {e}")
            
        time.sleep(60)  # Actualizar cada minuto

if __name__ == "__main__":
    monitor_system()
