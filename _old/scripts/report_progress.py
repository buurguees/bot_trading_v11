#!/usr/bin/env python3
import os
import json
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv("config/.env")
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://trading_user:160501@192.168.10.109:5432/trading_db')
engine = create_engine(DB_URL)

def main():
    sections = {}
    with engine.connect() as conn:
        # 1) Velas (historicaldata)
        try:
            q = text(
                """
                SELECT symbol, timeframe, COUNT(*) AS rows
                FROM trading.historicaldata
                WHERE timestamp >= now() - interval '24 hours'
                GROUP BY 1,2
                ORDER BY 1,2
                """
            )
            rows = conn.execute(q).fetchall()
            sections["historicaldata_24h"] = [dict(r._mapping) for r in rows]
        except Exception as e:
            sections["historicaldata_24h_error"] = str(e)

        # 2) Features (si existe)
        try:
            exists = conn.execute(text(
                """
                SELECT EXISTS(
                  SELECT 1 FROM information_schema.tables
                  WHERE table_schema='trading' AND table_name='features'
                )
                """
            )).scalar()
            if exists:
                q = text(
                    """
                    SELECT symbol, timeframe, COUNT(*) AS rows
                    FROM trading.features
                    WHERE timestamp >= now() - interval '24 hours'
                    GROUP BY 1,2
                    ORDER BY 1,2
                    """
                )
                rows = conn.execute(q).fetchall()
                sections["features_24h"] = [dict(r._mapping) for r in rows]
            else:
                sections["features_24h"] = []
        except Exception as e:
            sections["features_24h_error"] = str(e)

        # 3) Strategy samples últimas 24h
        try:
            q = text(
                """
                SELECT m.symbol, m.timeframe,
                       COUNT(*) AS trades,
                       AVG(CASE WHEN ss.pnl>0 THEN 1.0 ELSE 0.0 END) AS win_rate,
                       AVG(ss.pnl) AS avg_pnl,
                       AVG(ss.leverage) AS avg_lev
                FROM trading.strategy_samples ss
                JOIN trading.strategy_memory m ON m.id = ss.memory_id
                WHERE ss.entry_ts >= now() - interval '24 hours'
                GROUP BY m.symbol, m.timeframe
                ORDER BY m.symbol, m.timeframe
                """
            )
            rows = conn.execute(q).fetchall()
            sections["strategy_samples_24h"] = [dict(r._mapping) for r in rows]
        except Exception as e:
            sections["strategy_samples_24h_error"] = str(e)

        # 4) Strategy memory actualizado últimas 24h
        try:
            q = text(
                """
                SELECT symbol, timeframe,
                       COUNT(*) as strategies,
                       AVG(win_rate) as avg_wr,
                       AVG(avg_pnl) as avg_pnl,
                       AVG(avg_leverage) as avg_lev
                FROM trading.strategy_memory
                WHERE last_updated >= now() - interval '24 hours'
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
                """
            )
            rows = conn.execute(q).fetchall()
            sections["strategy_memory_24h"] = [dict(r._mapping) for r in rows]
        except Exception as e:
            sections["strategy_memory_24h_error"] = str(e)

        # 5) MLStrategies creadas últimas 24h
        try:
            q = text(
                """
                SELECT symbol, action,
                       COUNT(*) as n,
                       AVG(performance) as avg_perf,
                       AVG(confidence_score) as avg_conf
                FROM trading.mlstrategies
                WHERE timestamp >= now() - interval '24 hours'
                GROUP BY symbol, action
                ORDER BY symbol, action
                """
            )
            rows = conn.execute(q).fetchall()
            sections["mlstrategies_24h"] = [dict(r._mapping) for r in rows]
        except Exception as e:
            sections["mlstrategies_24h_error"] = str(e)

        # 6) Agentversions (creados/promovidos últimas 24h)
        try:
            q = text(
                """
                SELECT 
                  COUNT(*) FILTER (WHERE created_at >= now() - interval '24 hours') as created_24h,
                  COUNT(*) FILTER (WHERE promoted = true AND created_at >= now() - interval '24 hours') as promoted_24h,
                  AVG( (metrics->>'auc')::numeric ) FILTER (WHERE created_at >= now() - interval '24 hours' AND (metrics->>'auc') IS NOT NULL) as avg_auc_24h,
                  AVG( (metrics->>'acc')::numeric ) FILTER (WHERE created_at >= now() - interval '24 hours' AND (metrics->>'acc') IS NOT NULL) as avg_acc_24h
                FROM trading.agentversions
                """
            )
            row = conn.execute(q).fetchone()
            sections["agentversions_24h"] = dict(row._mapping)

            q2 = text(
                """
                SELECT (params->>'symbol') as symbol, (params->>'timeframe') as timeframe,
                       COUNT(*) FILTER (WHERE created_at >= now() - interval '24 hours') as created,
                       AVG( (metrics->>'auc')::numeric ) FILTER (WHERE created_at >= now() - interval '24 hours' AND (metrics->>'auc') IS NOT NULL) as avg_auc,
                       AVG( (metrics->>'acc')::numeric ) FILTER (WHERE created_at >= now() - interval '24 hours' AND (metrics->>'acc') IS NOT NULL) as avg_acc
                FROM trading.agentversions
                GROUP BY (params->>'symbol'), (params->>'timeframe')
                ORDER BY 1,2
                """
            )
            rows = conn.execute(q2).fetchall()
            sections["agentversions_by_pair"] = [dict(r._mapping) for r in rows]
        except Exception as e:
            sections["agentversions_error"] = str(e)

    print(json.dumps(sections, default=str, indent=2))

if __name__ == "__main__":
    main()
