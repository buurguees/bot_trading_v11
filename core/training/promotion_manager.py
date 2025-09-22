"""
Promotion Manager (champion–challenger)
---------------------------------------
Lee:
  - ml.agents (candidate vs promoted) por símbolo ('execution')
  - config/promotion/promotion.yaml (tie-breakers)
Escribe:
  - ml.agents: status -> 'promoted' (único por (symbol, task)) y 'shadow' para el anterior
  - agents/{SYMBOL}_PPO.zip: sobrescribe con el artefacto del campeón

Qué hace:
  - Compara métricas (score simple) entre candidatos y campeón actual.
  - Si el candidato supera umbrales (ya verificados por scorer) y gana por tie-breaker,
    promueve y mantiene unicidad con el índice parcial.

Funciones:
  - evaluate_and_promote(symbol:str) -> promoted_agent_id | None
  - promote_all() -> int
"""

from __future__ import annotations
import os, json, shutil, logging
from typing import Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import load_promotion_config

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("PromotionManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _score(m: dict) -> float:
    sharpe = float(m.get("sharpe", 0.0))
    pf     = float(m.get("profit_factor", 0.0))
    dd     = float(m.get("max_dd", 0.0))
    wr     = float(m.get("winrate", 0.0))
    dd_term = (1.0 - min(dd, 0.99))
    return 0.4*sharpe + 0.3*min(pf, 3.0) + 0.2*dd_term + 0.1*wr


def _better(a: dict, b: dict, tie_breaker: list) -> bool:
    sa, sb = _score(a), _score(b)
    if sa != sb:
        return sa > sb
    # desempate
    for k in tie_breaker:
        va, vb = float(a.get(k, 0.0)), float(b.get(k, 0.0))
        if k in ("max_dd",):  # menor es mejor
            if va != vb: return va < vb
        else:                 # mayor es mejor
            if va != vb: return va > vb
    return False


def evaluate_and_promote(symbol: str) -> Optional[str]:
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    tie = load_promotion_config().get("tie_breaker", ["stability","max_dd","sharpe"])

    with engine.begin() as conn:
        prom = conn.execute(text("""
            SELECT agent_id, metrics, artifact_uri FROM ml.agents
            WHERE symbol=:s AND task='execution' AND status='promoted'
            LIMIT 1
        """), {"s": symbol}).mappings().first()
        cand = conn.execute(text("""
            SELECT agent_id, metrics, artifact_uri FROM ml.agents
            WHERE symbol=:s AND task='execution' AND status='candidate'
        """), {"s": symbol}).mappings().all()

        if not cand:
            return None

        # elegir mejor candidato
        best = None
        for c in cand:
            m = c["metrics"]; m = json.loads(m) if isinstance(m, str) else (m or {})
            if best is None or _better(m, json.loads(best["metrics"]) if isinstance(best["metrics"], str) else best["metrics"], tie):
                best = {"agent_id": c["agent_id"], "metrics": m, "artifact_uri": c["artifact_uri"]}

        if best is None:
            return None

        # comparar con promovido actual (si hay)
        if prom:
            m_prom = prom["metrics"]; m_prom = json.loads(m_prom) if isinstance(m_prom, str) else (m_prom or {})
            if not _better(best["metrics"], m_prom, tie):
                logger.info(f"[{symbol}] el candidato no supera al campeón actual.")
                return None

        # PROMOVER
        # 1) set champion único
        conn.execute(text("""UPDATE ml.agents SET status='shadow' WHERE symbol=:s AND task='execution' AND status='promoted'"""),
                     {"s": symbol})
        conn.execute(text("""UPDATE ml.agents SET status='promoted', promoted_at=NOW() WHERE agent_id=:aid"""),
                     {"aid": best["agent_id"]})

        # 2) actualizar artifact principal agents/{SYMBOL}_PPO.zip
        target = os.path.join("agents", f"{symbol}_PPO.zip")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.abspath(best["artifact_uri"]) != os.path.abspath(target):
            try:
                shutil.copyfile(best["artifact_uri"], target)
            except Exception as e:
                logger.warning(f"No se pudo copiar artifact: {e}")

        logger.info(f"[{symbol}] PROMOTED: {best['agent_id']}")
        return best["agent_id"]


def promote_all() -> int:
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    with engine.begin() as conn:
        syms = [r[0] for r in conn.execute(text("""SELECT DISTINCT symbol FROM ml.agents WHERE task='execution'""")).fetchall()]
    n = 0
    for s in syms:
        if evaluate_and_promote(s):
            n += 1
    logger.info(f"Agentes promovidos: {n}")
    return n


if __name__ == "__main__":
    promote_all()
