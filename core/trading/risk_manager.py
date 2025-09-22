"""
Risk Manager (Capa 7)
=====================
Lee:
  - trading.trade_plans (status='planned' y sin expirar)
  - trading.positions (exposición y estado por símbolo)
  - risk/rules desde config/risk.yaml (opcional; si no existe usa defaults)
Escribe:
  - risk.risk_events (cada violación/reporte)
  - trading.trade_plans (status -> 'queued' si pasa filtros; 'invalid' si no)

Qué controla (baseline):
  - Exposición por símbolo (máx USDT)
  - Nº máximo de posiciones abiertas
  - Apalancamiento máximo
  - Tamaño mínimo/máximo de orden (qty)
  - Circuit breaker por drawdown día (placeholder)
  - Enfriamiento (cooldown) por símbolo tras cierre (placeholder sencillo)

Funciones:
  - load_risk_config()  -> dict de límites
  - check_plan(plan, exposure, limits) -> lista de eventos (vacía si OK)
  - write_event(engine, event)  -> inserta en risk.risk_events
  - mark_plan(engine, plan_id, status)
  - run_once()           -> procesa todos los 'planned'
"""

from __future__ import annotations
import os, json, logging
from datetime import datetime, timedelta
from typing import Dict, List

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import load_risk_config

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("RiskManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

def load_risk_limits() -> Dict:
    """
    Carga límites de riesgo desde config/trading/risk.yaml.
    """
    return load_risk_config()

def _get_open_positions(engine) -> Dict[str, Dict]:
    sql = text("""SELECT account_id, symbol, side, qty, avg_entry, status FROM trading.positions WHERE status='open'""")
    ex = {}
    with engine.begin() as conn:
        for r in conn.execute(sql).mappings():
            ex[r["symbol"]] = dict(r)
    return ex

def _get_planned_plans(engine) -> List[Dict]:
    sql = text("""
      SELECT *
      FROM trading.trade_plans
      WHERE status='planned' AND (valid_until IS NULL OR valid_until > NOW())
      ORDER BY ts ASC
    """)
    with engine.begin() as conn:
        return [dict(r) for r in conn.execute(sql).mappings().all()]

def write_event(engine, ev: Dict):
    sql = text("""
      INSERT INTO risk.risk_events (ts, symbol, timeframe, plan_id, type, severity, message, details)
      VALUES (:ts, :symbol, :timeframe, :plan_id, :type, :severity, :message, :details)
    """)
    with engine.begin() as conn:
        # Convertir details a JSON string si es necesario
        ev_copy = ev.copy()
        if 'details' in ev_copy and ev_copy['details'] is not None:
            import json
            ev_copy['details'] = json.dumps(ev_copy['details'])
        conn.execute(sql, ev_copy)

def mark_plan(engine, plan_id, status: str):
    sql = text("""UPDATE trading.trade_plans SET status=:st WHERE plan_id=:pid""")
    with engine.begin() as conn:
        conn.execute(sql, {"st": status, "pid": plan_id})

def check_plan(plan: Dict, exposure: Dict[str, Dict], limits: Dict) -> List[Dict]:
    evs = []
    symbol = plan["symbol"]
    side   = plan["side"]
    qty    = float(plan["qty"])
    lev    = float(plan.get("leverage") or 1.0)
    price  = float(plan.get("entry_price") or 0.0)
    ts     = plan["ts"]

    # leverage
    if lev > limits["max_leverage"]:
        evs.append({"ts": ts, "symbol": symbol, "timeframe": plan["timeframe"], "plan_id": plan["plan_id"],
                    "type": "leverage_max", "severity": "error",
                    "message": "Leverage above max", "details": {"value": lev, "limit": limits["max_leverage"]}})

    # qty
    if qty < limits["min_qty"] or qty > limits["max_qty"]:
        evs.append({"ts": ts, "symbol": symbol, "timeframe": plan["timeframe"], "plan_id": plan["plan_id"],
                    "type": "qty_bounds", "severity": "error",
                    "message": "Qty out of bounds", "details": {"qty": qty, "min": limits["min_qty"], "max": limits["max_qty"]}})

    # exposición por símbolo
    cur = 0.0
    if symbol in exposure and exposure[symbol]["status"] == "open":
        cur = float(exposure[symbol]["qty"]) * float(exposure[symbol]["avg_entry"] or 0.0)
    new_exp = cur + qty * price
    if new_exp > limits["max_exposure_per_symbol_usdt"]:
        evs.append({"ts": ts, "symbol": symbol, "timeframe": plan["timeframe"], "plan_id": plan["plan_id"],
                    "type": "exposure_symbol", "severity": "error",
                    "message": "Exposure per symbol exceeded", "details": {"new": new_exp, "limit": limits["max_exposure_per_symbol_usdt"]}})

    # (placeholders) cooldown, dd_day, max_open_positions -> implementar con métricas reales cuando tengamos PnL
    return evs

def run_once() -> int:
    limits = load_risk_limits()
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

    exposure = _get_open_positions(engine)
    plans = _get_planned_plans(engine)
    promoted = 0

    for p in plans:
        evs = check_plan(p, exposure, limits)
        if evs:
            for e in evs: write_event(engine, e)
            mark_plan(engine, p["plan_id"], "invalid")
            continue
        mark_plan(engine, p["plan_id"], "queued")
        promoted += 1

    logger.info(f"Plans queued: {promoted} / {len(plans)}")
    return promoted

if __name__ == "__main__":
    run_once()
