"""
Scorer & Gate
-------------
Lee:
  - ml.backtest_runs (últimos runs por strategy_id)
  - config/promotion/promotion.yaml (umbrales)
Escribe:
  - ml.strategies.metrics_summary (con score y métricas finales)
  - ml.strategies.status -> 'ready_for_training' | 'rejected'

Qué hace:
  - Calcula un score compuesto (Sharpe, PF, MaxDD inverso, Winrate).
  - Decide si la estrategia está lista para entrenar la PPO o se rechaza.

Funciones:
  - compute_score(metrics:dict) -> float
  - gate_ready_for_training(strategy_id:str) -> bool
  - score_and_gate_all() -> dict {ready, rejected}
"""

from __future__ import annotations
import os, json, logging
from typing import Dict
import numpy as np

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import load_promotion_config

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("Scorer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def compute_score(m: Dict) -> float:
    """Score balanceado y realista para desarrollo."""
    sharpe = float(m.get("sharpe", 0.0))
    pf = float(m.get("profit_factor", 1.0))
    dd = float(m.get("max_dd", 1.0))
    wr = float(m.get("winrate", 0.0))
    trades = int(m.get("trades", 0))

    trade_penalty = 1.0 if trades >= 20 else (trades / 20.0)
    dd_term = max(0.1, 1.0 - min(dd, 0.95))
    pf_capped = min(pf, 5.0) if pf != np.inf else 5.0

    score = (0.30 * sharpe +
             0.25 * (pf_capped - 1.0) +
             0.25 * dd_term +
             0.15 * wr +
             0.05 * trade_penalty)
    return float(max(0.0, score))


def gate_ready_for_training(strategy_metrics: Dict, thresholds: Dict) -> bool:
    """Gate más permisivo para desarrollo (aún controlado por thresholds)."""
    t = thresholds.get("global", {})
    min_trades = int(t.get("min_trades", 20))
    if int(strategy_metrics.get("trades", 0)) < min_trades:
        return False
    if float(strategy_metrics.get("sharpe", 0.0)) < -1.0:
        return False
    if float(strategy_metrics.get("profit_factor", 1.0)) < 0.8:
        return False
    if float(strategy_metrics.get("max_dd", 0.0)) > 0.80:
        return False
    # thresholds siguen aplicando si están por encima de estos mínimos
    if float(strategy_metrics.get("sharpe", 0.0)) < float(t.get("sharpe_min", -1.0)):
        return False
    if float(strategy_metrics.get("profit_factor", 0.0)) < float(t.get("pf_min", 0.8)):
        return False
    if float(strategy_metrics.get("max_dd", 1.0)) > float(t.get("max_dd_max", 0.80)):
        return False
    if float(strategy_metrics.get("stability", 1.0)) < float(t.get("stability_min", 0.0)):
        return False
    return True


def score_and_gate_all():
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    thr = load_promotion_config().get("thresholds", {})

    # Tomar última métrica por strategy_id (prefiere event_driven; si no, vectorized)
    sql = text("""
        WITH last_runs AS (
          SELECT DISTINCT ON (strategy_id)
                 strategy_id, engine, metrics, started_at
          FROM ml.backtest_runs
          ORDER BY strategy_id, started_at DESC
        )
        SELECT s.strategy_id, s.symbol, s.timeframe, s.metrics_summary, lr.engine, lr.metrics
        FROM ml.strategies s
        JOIN last_runs lr ON lr.strategy_id = s.strategy_id
        WHERE s.status IN ('testing','ready_for_training')
    """)
    ready = rejected = 0
    with engine.begin() as conn:
        for r in conn.execute(sql).mappings().all():
            m = r["metrics"]; m = json.loads(m) if isinstance(m, str) else (m or {})
            # estabilidad placeholder (si hay múltiples runs futuros, aquí se computaría varianza por splits)
            m["stability"] = float(m.get("stability", 0.75))
            m["score"] = compute_score(m)
            ok = gate_ready_for_training(m, thr)
            new_status = "ready_for_training" if ok else "rejected"
            if ok: ready += 1
            else: rejected += 1

            # merge en metrics_summary de la estrategia
            ms = r["metrics_summary"]; 
            if isinstance(ms, str):
                try: ms = json.loads(ms)
                except Exception: ms = {}
            ms.update({"last_engine": r["engine"], **m})

            conn.execute(text("""
                UPDATE ml.strategies
                SET metrics_summary = CAST(:ms AS jsonb), status = :st, updated_at = NOW()
                WHERE strategy_id = :sid
            """), {"ms": json.dumps(ms), "st": new_status, "sid": r["strategy_id"]})

    logger.info(f"Estrategias listas: {ready} | rechazadas: {rejected}")
    return {"ready": ready, "rejected": rejected}


if __name__ == "__main__":
    score_and_gate_all()
