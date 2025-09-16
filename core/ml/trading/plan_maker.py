import os
from typing import Dict, Any, Union
from dataclasses import dataclass
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

# -- permite acceder a r["a"]["b"] o r.a.b indistintamente
def _dig(obj, *path, default=None):
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur

def make_trade_plan(symbol: str, tf: str, ts, side: int, strength: float,
                   direction_ver_id: int, risk: Union[Dict, Any], engine=ENGINE) -> int:
    """
    Crea un plan de trade usando la función helper _dig para leer la configuración.
    """
    if side == 0:
        return -1
    
    with engine.begin() as c:
        # Obtener precio de entrada y ATR
        entry_px = _get_entry_price(c, symbol, tf, ts)
        atr = _get_atr(c, symbol, tf, ts)
        if entry_px is None or atr is None:
            raise RuntimeError("No se pudo obtener entry_px o ATR para planificar el trade.")
        
        # Leer configuración usando _dig
        risk_pct  = _dig(risk, "portfolio", "risk_model", "risk_pct", default=0.003)
        k_sl      = _dig(risk, "sl_tp", "k_sl", default=1.5)
        k_tp      = _dig(risk, "sl_tp", "k_tp", default=2.0)
        atr_p     = _dig(risk, "sl_tp", "atr_period", default=14)
        
        lev_min   = _dig(risk, "futures", "leverage_min", default=5)
        lev_max   = _dig(risk, "futures", "leverage_max", default=50)
        liq_buf   = _dig(risk, "futures", "liq_buffer_atr", default=3.0)
        use_liq   = _dig(risk, "futures", "use_leverage_liq_guard", default=True)
        
        fees_taker = _dig(risk, "futures", "fees", "taker", default=0.0006)
        slip_bps   = _dig(risk, "futures", "slippage_bps", default=2)
        
        # Calcular SL y TP
        if side == 1:  # LONG
            sl_px = entry_px - k_sl * atr
            tp_px = entry_px + k_tp * atr
        else:  # SHORT
            sl_px = entry_px + k_sl * atr
            tp_px = entry_px - k_tp * atr
        
        # Calcular leverage dinámico
        leverage = _choose_leverage(entry_px, sl_px, side, atr, liq_buf, lev_min, lev_max)
        
        # Calcular cantidad
        equity = _dig(risk, "portfolio", "equity", default=10000.0)
        qty = (equity * risk_pct) / max(1e-9, abs(entry_px - sl_px))
        
        # Guardar razonamiento
        reason = {
            "direction_ver_id": direction_ver_id,
            "strength": float(strength),
            "atr": float(atr),
            "tf": tf,
            "k_sl": float(k_sl),
            "k_tp": float(k_tp),
            "lev_min": float(lev_min),
            "lev_max": float(lev_max),
            "liq_buf_atr": float(liq_buf),
            "equity": float(equity),
            "fees_taker": float(fees_taker),
            "slip_bps": float(slip_bps),
        }
        
        # Insertar en la base de datos
        q = text("""
            INSERT INTO trading.TradePlans
            (created_at, bar_ts, symbol, timeframe, side, entry_px, sl_px, tp_px, risk_pct, qty, leverage, margin_mode, reason, status)
            VALUES (now(), :bt, :s, :tf, :sd, :e, :sl, :tp, :r, :q, :lv, :mm, :rs, 'planned')
            RETURNING id
        """).bindparams(bindparam("rs", type_=JSONB()))
        
        plan_id = c.execute(q, {
            "bt": ts, "s": symbol, "tf": tf, "sd": side,
            "e": float(entry_px), "sl": float(sl_px), "tp": float(tp_px),
            "r": float(risk_pct), "q": float(qty), "lv": float(leverage),
            "mm": "isolated", "rs": reason
        }).scalar()
        return int(plan_id)

def _choose_leverage(entry_px: float, sl_px: float, side: int,
                    atr: float, liq_buf_atr: float,
                    lev_min: float, lev_max: float) -> float:
    """
    Aproximación para perps USDT lineales en modo aislado.
    Distancia aproximada a liquidación ~ entry / L (ignorando margen de mantenimiento).
    Exigimos que la distancia a liq sea MAYOR que (distancia al SL + buffer ATRs).
    """
    stop_dist = abs(entry_px - sl_px)
    min_liq_dist = stop_dist + liq_buf_atr * atr
    if min_liq_dist <= 0:
        return float(lev_min)

    lev_cap = entry_px / min_liq_dist           # L máximo permitido por buffer
    L = max(lev_min, min(lev_max, float(lev_cap)))
    return round(L, 2)

def _get_entry_price(c, symbol: str, tf: str, ts) -> float:
    """Obtiene el precio de entrada (close) para el timestamp dado."""
    q = text("""
        SELECT close FROM trading.HistoricalData
        WHERE symbol=:s AND timeframe=:tf AND timestamp=:ts
        LIMIT 1
    """)
    row = c.execute(q, {"s":symbol,"tf":tf,"ts":ts}).fetchone()
    return float(row[0]) if row and row[0] is not None else None

def _get_atr(c, symbol: str, tf: str, ts) -> float:
    """Obtiene el ATR para el timestamp dado."""
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
