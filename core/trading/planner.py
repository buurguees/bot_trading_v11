import os, json
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from .position_sizer import load_risk_params, plan_from_price

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

def _get_entry_price(c, symbol: str, tf: str, ts) -> Optional[float]:
    q = text("""
        SELECT close FROM trading.HistoricalData
        WHERE symbol=:s AND timeframe=:tf AND timestamp<=:ts
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    row = c.execute(q, {"s":symbol,"tf":tf,"ts":ts}).fetchone()
    return float(row[0]) if row else None

def _get_atr(c, symbol: str, tf: str, ts) -> Optional[float]:
    # Primero intenta leer desde Features (columna atr14)
    qf = text("""
        SELECT atr14 FROM trading.Features
        WHERE symbol=:s AND timeframe=:tf AND timestamp=:ts
        LIMIT 1
    """)
    row = c.execute(qf, {"s":symbol,"tf":tf,"ts":ts}).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    # Si no hay, toma la última disponible antes de ts
    qf2 = text("""
        SELECT atr14 FROM trading.Features
        WHERE symbol=:s AND timeframe=:tf AND timestamp<=:ts
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    row = c.execute(qf2, {"s":symbol,"tf":tf,"ts":ts}).fetchone()
    return float(row[0]) if row and row[0] is not None else None

def plan_and_store(symbol: str, tf: str, ts, side: int, strength: float,
                   direction_ver_id: int, engine=ENGINE) -> int:
    if side == 0:
        return -1
    with engine.begin() as c:
        entry_px = _get_entry_price(c, symbol, tf, ts)
        atr      = _get_atr(c, symbol, tf, ts)
        if entry_px is None or atr is None:
            raise RuntimeError("No se pudo obtener entry_px o ATR para planificar el trade.")

        rp = load_risk_params(tf)  # de config/risk.yaml
        plan = plan_from_price(entry_px, atr, side, tf, rp)

        q = text("""
            INSERT INTO trading.TradePlans
            (created_at, bar_ts, symbol, timeframe, side, entry_px, sl_px, tp_px, risk_pct, qty, leverage, margin_mode, reason, status)
            VALUES (now(), :bt, :s, :tf, :sd, :e, :sl, :tp, :r, :q, :lv, :mm, :rs, 'planned')
            RETURNING id
        """)
        reason = {
            "direction_ver_id": direction_ver_id,
            "strength": strength,
            "atr": atr,
            **plan["params_used"]
        }
        plan_id = c.execute(q, {
            "bt": ts, "s": symbol, "tf": tf, "sd": side,
            "e": plan["entry_px"], "sl": plan["sl_px"], "tp": plan["tp_px"],
            "r": plan["risk_pct"], "q": plan["qty"], "lv": plan["leverage"],
            "mm": plan["margin_mode"], "rs": reason
        }).scalar()
        return int(plan_id)
