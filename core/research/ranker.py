"""
Ranker & Combiner
-----------------
Lee:
  - ml.backtest_runs (engine='vectorized' o 'event_driven')
Escribe:
  - (opcional) estrategias combinadas en ml.strategies (source='ensemble', status='testing')

Qué hace:
  - Crea un ranking por 'score' (fórmula simple, ajustable).
  - (Opcional) genera ensembles de 2-3 estrategias top del mismo símbolo/timeframe.

Funciones:
  - rank_strategies(metric:str='score', engine:str='vectorized', top:int=20) -> list[dict]
  - build_ensembles(symbol:str, timeframe:str, k:int=3) -> list[strategy_id]
"""

from __future__ import annotations
import os, json, logging
from typing import List, Dict

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("Ranker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _score(m: dict) -> float:
    sharpe = float(m.get("sharpe", 0.0))
    pf     = float(m.get("profit_factor", 0.0))
    dd     = float(m.get("max_dd", 0.0))
    stab   = float(m.get("stability", 0.0))  # si lo calculas
    dd_term = (1.0 - min(dd, 0.99))
    return 0.4*sharpe + 0.3*min(pf, 3.0) + 0.2*dd_term + 0.1*stab


def rank_strategies(metric: str = "score", engine: str = "vectorized", top: int = 20) -> List[Dict]:
    engine_db = create_engine(DB_URL, pool_pre_ping=True, future=True)
    sql = text("""
        SELECT r.run_id, r.strategy_id, r.symbol, r.timeframe, r.engine, r.metrics, s.rules
        FROM ml.backtest_runs r
        JOIN ml.strategies s ON s.strategy_id=r.strategy_id
        WHERE r.engine=:eng
    """)
    out = []
    with engine_db.begin() as conn:
        for r in conn.execute(sql, {"eng": engine}).mappings().all():
            m = r["metrics"]
            if isinstance(m, str):
                try: m = json.loads(m)
                except Exception: m = {}
            sc = _score(m) if metric == "score" else float(m.get(metric, 0.0))
            out.append({"score": sc, **dict(r)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top]


def build_ensembles(symbol: str, timeframe: str, k: int = 3) -> List[str]:
    """
    Crea una estrategia 'ensemble' juntando top-k reglas como OR (simple).
    Inserta en ml.strategies con source='ensemble'.
    """
    engine_db = create_engine(DB_URL, pool_pre_ping=True, future=True)
    sql = text("""
        SELECT s.strategy_id, s.rules, r.metrics
        FROM ml.strategies s
        JOIN LATERAL (
           SELECT metrics FROM ml.backtest_runs r
           WHERE r.strategy_id = s.strategy_id
           ORDER BY started_at DESC
           LIMIT 1
        ) r ON TRUE
        WHERE s.symbol=:sym AND s.timeframe=:tf AND s.status IN ('testing','ready_for_training','promoted','shadow')
    """)
    with engine_db.begin() as conn:
        rows = [dict(r) for r in conn.execute(sql, {"sym": symbol, "tf": timeframe}).mappings().all()]
        if not rows:
            return []
        # ranking local
        ranked = []
        for r in rows:
            m = r["metrics"]; 
            if isinstance(m, str):
                try: m = json.loads(m)
                except Exception: m = {}
            ranked.append((r["strategy_id"], _score(m), r["rules"]))
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_rules = [json.loads(rr[2]) if isinstance(rr[2], str) else rr[2] for rr in ranked[:k]]

        ensemble_rules = {"ensemble_or": top_rules}
        conn.execute(text("""
            INSERT INTO ml.strategies (symbol,timeframe,strategy_key,description,source,rules,risk_profile,metrics_summary,status,tags)
            VALUES (:sym,:tf,:key,:desc,'ensemble',:rules::jsonb,'{}'::jsonb,'{}'::jsonb,'testing',ARRAY['ensemble'])
        """), {
            "sym": symbol, "tf": timeframe,
            "key": f"ensemble-{symbol}-{timeframe}-{k}",
            "desc": f"Ensemble OR top-{k}",
            "rules": json.dumps(ensemble_rules),
        })
    return []


if __name__ == "__main__":
    top = rank_strategies()
    logger.info(f"Top (vectorized): {[(t['strategy_id'], round(t['score'],2)) for t in top]}")
