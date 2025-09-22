"""
OMS Router (Capa 7)
===================
Lee:
  - trading.trade_plans (status='queued')
  - market.features (para close si necesitas lógica de simulación)
Escribe:
  - trading.orders (crea orden a partir de plan)
  - trading.fills  (simula fill inmediato a entry_price)
  - trading.positions (vía positions.apply_fill)
  - trading.trade_plans (status -> 'sent' / 'filled')

Funciones:
  - fetch_queued_plans()
  - create_order_from_plan()
  - simulate_fill(order)
  - run_once(mode='sim'): procesa en modo simulación (paper)

Nota:
  - Esto es un OMS mínimo (paper). En 'live' sustituirás simulate_fill por integración Bitget.
"""

from __future__ import annotations
import os, json, logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timezone

from core.trading.oms.positions import apply_fill

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("OMSRouter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

def fetch_queued_plans(conn):
    return [dict(r) for r in conn.execute(text("""
        SELECT * FROM trading.trade_plans
        WHERE status='queued' AND (valid_until IS NULL OR valid_until > NOW())
        ORDER BY ts ASC
    """)).mappings().all()]

def create_order_from_plan(conn, plan: dict) -> dict:
    sql = text("""
        INSERT INTO trading.orders
          (plan_id, symbol, timeframe, side, order_type, price, qty, route, status)
        VALUES (:plan_id, :symbol, :timeframe, :side, :order_type, :price, :qty, 'sim', 'sent')
        RETURNING *
    """)
    row = conn.execute(sql, {
        "plan_id": plan["plan_id"],
        "symbol": plan["symbol"],
        "timeframe": plan["timeframe"],
        "side": plan["side"],
        "order_type": plan["entry_type"],
        "price": plan["entry_price"],
        "qty": plan["qty"],
    }).mappings().first()
    conn.execute(text("UPDATE trading.trade_plans SET status='sent' WHERE plan_id=:pid"), {"pid": plan["plan_id"]})
    return dict(row)

def simulate_fill(conn, order: dict):
    # fill inmediato a price del plan (paper)
    fill = {
        "order_id": order["order_id"],
        "symbol": order["symbol"],
        "side": order["side"],
        "price": order["price"],
        "qty": order["qty"],
        "fee_usdt": 0.0,
        "liquidity": "maker",
        "ts": datetime.now(tz=timezone.utc)
    }
    conn.execute(text("""
        INSERT INTO trading.fills (order_id, symbol, side, price, qty, fee_usdt, liquidity, ts)
        VALUES (:order_id,:symbol,:side,:price,:qty,:fee_usdt,:liquidity,:ts)
    """), fill)
    conn.execute(text("UPDATE trading.orders SET status='filled', updated_at=NOW() WHERE order_id=:oid"),
                 {"oid": order["order_id"]})
    apply_fill(conn, fill)

def run_once(mode: str = "sim") -> int:
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    processed = 0
    with engine.begin() as conn:
        for plan in fetch_queued_plans(conn):
            order = create_order_from_plan(conn, plan)
            if mode == "sim":
                simulate_fill(conn, order)
                conn.execute(text("UPDATE trading.trade_plans SET status='filled' WHERE plan_id=:pid"),
                             {"pid": plan["plan_id"]})
            processed += 1
    logger.info(f"OMS processed orders: {processed}")
    return processed

if __name__ == "__main__":
    run_once()
