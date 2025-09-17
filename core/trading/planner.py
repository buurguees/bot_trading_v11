import os, json
from typing import Optional, Dict, Any
from dataclasses import is_dataclass, asdict
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv
from .position_sizer import load_risk_params, plan_from_price

load_dotenv("config/.env")

def _to_plain_dict(obj):
    """Convierte dataclasses / SimpleNamespace / objetos a dict recursivamente."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: _to_plain_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj  # tipos primitivos

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

def choose_leverage(entry_px: float, sl_px: float, side: int,
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
ENGINE = create_engine(os.getenv("DB_URL"))

# Cache simple para saber si existe la columna bar_ts en trading.TradePlans
_HAS_BAR_TS: Optional[bool] = None

def _check_has_bar_ts(conn) -> bool:
    global _HAS_BAR_TS
    if _HAS_BAR_TS is not None:
        return _HAS_BAR_TS
    q = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='trading' AND table_name='tradeplans' AND column_name='bar_ts'
        LIMIT 1
        """
    )
    _HAS_BAR_TS = conn.execute(q).fetchone() is not None
    return _HAS_BAR_TS

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
        
        # Acceso directo a atributos del dataclass RiskParams
        mmode   = rp.margin_mode
        equity  = float(rp.equity)
        risk_pct= float(rp.risk_pct)
        LEV_MIN = 5.0                    # mínimo operativo (o léelo del YAML si lo tienes)
        LEV_MAX = float(rp.lev_max)
        LIQ_BUF = float(rp.liq_buf_atr)
        K_SL    = float(rp.k_sl)
        K_TP    = float(rp.k_tp)
        
        if side == 1:  # LONG
            sl_px = entry_px - K_SL * atr
            tp_px = entry_px + K_TP * atr
        else:  # SHORT
            sl_px = entry_px + K_SL * atr
            tp_px = entry_px - K_TP * atr
        
        # Calcular leverage dinámico
        leverage = choose_leverage(entry_px, sl_px, side, atr, LIQ_BUF, LEV_MIN, LEV_MAX)
        
        # tamaño de posición (no depende del leverage; el riesgo lo marca el SL)
        # risk_pct * equity / distancia al SL
        qty = (equity * risk_pct) / max(1e-9, abs(entry_px - sl_px))
        
        # guarda el razonamiento útil:
        reason = {
            "direction_ver_id": direction_ver_id,
            "strength": float(strength),
            "atr": float(atr),
            "tf": tf,
            "k_sl": float(K_SL),
            "k_tp": float(K_TP),
            "lev_min": float(LEV_MIN),
            "lev_max": float(LEV_MAX),
            "liq_buf_atr": float(LIQ_BUF),
            "equity": float(equity),
        }

        # Insert con o sin bar_ts según exista la columna en la tabla
        if _check_has_bar_ts(c):
            q = text(
                """
                INSERT INTO trading.TradePlans
                (created_at, bar_ts, symbol, timeframe, side, entry_px, sl_px, tp_px, risk_pct, qty, leverage, margin_mode, reason, status)
                VALUES (now(), :bt, :s, :tf, :sd, :e, :sl, :tp, :r, :q, :lv, :mm, :rs, 'planned')
                RETURNING id
                """
            ).bindparams(bindparam("rs", type_=JSONB()))
            params = {
                "bt": ts,
                "s": symbol,
                "tf": tf,
                "sd": side,
                "e": float(entry_px),
                "sl": float(sl_px),
                "tp": float(tp_px),
                "r": float(risk_pct),
                "q": float(qty),
                "lv": float(leverage),
                "mm": mmode,
                "rs": reason,
            }
        else:
            q = text(
                """
                INSERT INTO trading.TradePlans
                (created_at, symbol, timeframe, side, entry_px, sl_px, tp_px, risk_pct, qty, leverage, margin_mode, reason, status)
                VALUES (now(), :s, :tf, :sd, :e, :sl, :tp, :r, :q, :lv, :mm, :rs, 'planned')
                RETURNING id
                """
            ).bindparams(bindparam("rs", type_=JSONB()))
            params = {
                "s": symbol,
                "tf": tf,
                "sd": side,
                "e": float(entry_px),
                "sl": float(sl_px),
                "tp": float(tp_px),
                "r": float(risk_pct),
                "q": float(qty),
                "lv": float(leverage),
                "mm": mmode,
                "rs": reason,
            }

        plan_id = c.execute(q, params).scalar()
        return int(plan_id)
