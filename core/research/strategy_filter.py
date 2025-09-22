"""
Strategy Filter
---------------
Lee:
  - ml.strategies con status='candidate'
  - (opcional) trading.trade_plans para validar dispersión temporal
Escribe:
  - ml.strategies: status -> 'testing' (si pasa) o 'deprecated' (si no),
    actualizando metrics_summary con quick stats.

Qué hace:
  - Aplica filtros mínimos: soporte (nº apariciones) y dispersión en el tiempo.
  - Sirve para cribar antes del backtest.

Funciones:
  - filter_candidates(min_occurrences:int=None, min_span_days:int=None) -> dict {kept, dropped}

Nota:
  - Si los parámetros son None, lee desde variables de entorno:
    - PH2_FILTER_MIN_OCC (default: 5)
    - PH2_FILTER_MIN_SPAN_DAYS (default: 2)
"""

from __future__ import annotations
import os, json, logging
from datetime import timedelta

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("StrategyFilter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _json(v):
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return {}
    return v or {}


def filter_candidates(min_occurrences: int = None, min_span_days: int = None):
    """
    Filtra estrategias candidatas basándose en soporte mínimo y span temporal.
    Si los parámetros son None, lee la configuración desde variables de entorno.
    """
    # Leer configuración desde variables de entorno si no se especifican
    if min_occurrences is None:
        min_occurrences = int(os.getenv("PH2_FILTER_MIN_OCC", "5"))
    if min_span_days is None:
        min_span_days = int(os.getenv("PH2_FILTER_MIN_SPAN_DAYS", "2"))
    
    logger.info(f"Filtrando candidatas: min_occurrences={min_occurrences}, min_span_days={min_span_days}")
    
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

    # 1) Cargar candidatas
    with engine.begin() as conn:
        rows = [dict(r) for r in conn.execute(text("""
            SELECT strategy_id, symbol, timeframe, strategy_key, metrics_summary
            FROM ml.strategies WHERE status='candidate'
        """)).mappings().all()]

    if not rows:
        logger.info("No hay candidatas.")
        return {"kept": 0, "dropped": 0}

    kept = dropped = 0
    with engine.begin() as conn:
        for r in rows:
            m = _json(r["metrics_summary"])
            support = int(m.get("support_n", 0))
            first_seen = pd.Timestamp(m.get("first_seen") or "1970-01-01", tz="UTC")
            last_seen  = pd.Timestamp(m.get("last_seen")  or "1970-01-01", tz="UTC")
            span_days = (last_seen - first_seen).days

            ok = (support >= min_occurrences) and (span_days >= min_span_days)
            new_status = "testing" if ok else "deprecated"
            if ok: kept += 1
            else:  dropped += 1

            m["filter_min_occurrences"] = min_occurrences
            m["filter_min_span_days"] = min_span_days
            m["filter_pass"] = ok

            conn.execute(text("""
                UPDATE ml.strategies
                SET status=:st, metrics_summary=:m, updated_at=NOW()
                WHERE strategy_id=:sid
            """), {"st": new_status, "m": json.dumps(m), "sid": r["strategy_id"]})

    logger.info(f"Candidatas -> testing: {kept} | deprecated: {dropped}")
    return {"kept": kept, "dropped": dropped}


if __name__ == "__main__":
    filter_candidates()
