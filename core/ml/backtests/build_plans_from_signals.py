import argparse, json, math, datetime as dt
from typing import Dict, Any, Optional

import pandas as pd
from sqlalchemy import text
from math import tanh

from core.data.database import get_engine

# --- Parser de fechas robusto -------------------------------------------------
def parse_utc(s: str) -> pd.Timestamp:
    if s is None:
        return None
    ts = pd.Timestamp(s)
    # si viene naive -> localiza en UTC; si viene con tz -> convierte a UTC
    return ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')

# --- Utilidades de riesgo ----------------------------------------------------
def load_risk_for(symbol: str, tf: str) -> Dict[str, Any]:
    """Lee risk.yaml desde BD/archivo si ya tienes un loader; aquí valores por defecto robustos."""
    # Ajusta a tu loader si ya tienes uno; estos defaults son sensatos
    return {
        "risk_pct": 0.003,      # 0.3% por operación
        "k_sl": 1.5,            # SL = k_sl * ATR
        "k_tp": 2.0,            # TP = k_tp * ATR (simétrico)
        "lev_max": 10.0,        # límite de apalancamiento por defecto
        "liq_buf_atr": 3.0,     # colchón para liquidación (si lo usas)
        "min_qty": 0.001,       # mínimo absoluto de qty
        "max_qty": None,        # sin tope si None
        "margin_mode": "isolated",
        "equity": 10_000.0      # capital notional para dimensionar
    }

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# --- Lecturas en BD ----------------------------------------------------------
def fetch_signals(conn, symbol, tf, ver_id, ts_from, ts_to):
    q = text("""
        SELECT timestamp, side, strength
        FROM trading.AgentSignals
        WHERE symbol=:s AND timeframe=:tf
          AND (meta->>'direction_ver_id')::int = :vid
          AND timestamp >= :a AND timestamp < :b
        ORDER BY timestamp
    """)
    return pd.read_sql(q, conn, params={"s":symbol, "tf":tf, "vid":ver_id, "a":ts_from, "b":ts_to})

def fetch_px_atr(conn, symbol, tf, ts):
    q = text("""
        SELECT h.close, f.atr14
        FROM trading.HistoricalData h
        LEFT JOIN trading.Features f
               ON f.symbol=h.symbol AND f.timeframe=h.timeframe AND f.timestamp=h.timestamp
        WHERE h.symbol=:s AND h.timeframe=:tf AND h.timestamp=:ts
        LIMIT 1
    """)
    row = conn.execute(q, {"s":symbol, "tf":tf, "ts":ts}).mappings().first()
    if not row:
        return None, None
    return float(row["close"]), (float(row["atr14"]) if row["atr14"] is not None else None)

def ensure_idx_tradeplans(conn):
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_tradeplans_sym_tf_bar
        ON trading.tradeplans (symbol, timeframe, bar_ts DESC)
    """))

# --- Inserción de TradePlan ---------------------------------------------------
def insert_plan(conn, *, bar_ts, symbol, tf, side, entry, sl, tp,
                risk_pct, qty, leverage, margin_mode, reason: Dict[str, Any]):
    sql = (
        """
        INSERT INTO trading.tradeplans
        (created_at, bar_ts, symbol, timeframe, side, entry_px, sl_px, tp_px,
         risk_pct, qty, leverage, margin_mode, reason, status)
        VALUES (now(), %(bt)s, %(s)s, %(tf)s, %(sd)s, %(e)s, %(sl)s, %(tp)s, %(r)s, %(q)s, %(lv)s, %(mm)s, %(rs)s::jsonb, 'planned')
        ON CONFLICT (symbol, timeframe, bar_ts, side) DO NOTHING
        RETURNING id
        """
    )
    return conn.exec_driver_sql(sql, {
        "bt": bar_ts, "s": symbol, "tf": tf, "sd": int(side),
        "e": float(entry), "sl": float(sl), "tp": float(tp),
        "r": float(risk_pct), "q": float(qty), "lv": float(leverage),
        "mm": margin_mode, "rs": json.dumps(reason)
    }).scalar()

# --- Dimensionado ------------------------------------------------------------
def size_and_targets(side: int, close_px: float, atr: float, risk: Dict[str, Any]):
    k_sl = float(risk.get("k_sl", 1.5))
    k_tp = float(risk.get("k_tp", 2.0))
    risk_pct = float(risk.get("risk_pct", 0.003))
    lev_max = float(risk.get("lev_max", 10.0))
    equity = float(risk.get("equity", 10_000.0))
    margin_mode = risk.get("margin_mode", "isolated")

    if atr is None or atr <= 0:
        return None  # No podemos dimensionar

    # SL/TP simétricos por ATR
    if side > 0:   # long
        sl = close_px - k_sl * atr
        tp = close_px + k_tp * atr
        stop_gap = close_px - sl
    else:          # short
        sl = close_px + k_sl * atr
        tp = close_px - k_tp * atr
        stop_gap = sl - close_px

    stop_gap = max(stop_gap, 1e-9)

    # Apalancamiento: por ahora usa el máximo del perfil (puedes hacer función por “strength”)
    leverage = clamp(lev_max, 1.0, 100.0)

    # Qty por riesgo: equity * risk_pct = pérdida si toca SL
    # pérdida ≈ (stop_gap / close_px) * leverage * notional  => notional = equity * risk_pct * close_px / (stop_gap * leverage)
    notional = equity * risk_pct * close_px / (stop_gap * leverage)
    qty = notional / close_px

    min_qty = float(risk.get("min_qty", 0.0) or 0.0)
    max_qty = risk.get("max_qty", None)
    if max_qty is not None:
        max_qty = float(max_qty)
        qty = min(qty, max_qty)
    qty = max(qty, min_qty)

    return sl, tp, risk_pct, qty, leverage, margin_mode

# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--ver-id", type=int, required=True, help="AgentVersions.id a usar")
    ap.add_argument("--from", dest="dt_from", required=True)
    ap.add_argument("--to",   dest="dt_to",   required=True)
    args = ap.parse_args()

    ts_from = parse_utc(args.dt_from)
    ts_to   = parse_utc(args.dt_to)

    eng = get_engine()
    with eng.begin() as conn:
        ensure_idx_tradeplans(conn)

        sig = fetch_signals(conn, args.symbol, args.tf, args.ver_id, ts_from, ts_to)
        if sig.empty:
            print("No hay AgentSignals en ese rango para ese ver_id.")
            return

        risk = load_risk_for(args.symbol, args.tf)
        created = 0

        for row in sig.itertuples(index=False):
            ts, side, strength = row.timestamp.to_pydatetime(), int(row.side), float(row.strength)
            close_px, atr = fetch_px_atr(conn, args.symbol, args.tf, ts)
            if close_px is None:
                continue
            sized = size_and_targets(side, close_px, atr, risk)
            if not sized:
                continue
            sl, tp, risk_pct, qty, lev, mm = sized

            # --- Modular leverage con strategy_memory ---
            # construir firma sencilla con los mismos buckets (si tienes los campos en features/meta)
            reason_probe = {"rsi": None, "ema_state": None, "macd": None, "supertrend_dir": None, "trend_1h": None}
            signature = None
            try:
                from .strategy_memory import build_signature  # type: ignore
                signature, _ = build_signature(reason_probe)
            except Exception:
                signature = None
            if signature:
                mem_q = text("""
                  SELECT win_rate, avg_pnl, sharpe, n_trades
                  FROM trading.strategy_memory
                  WHERE symbol=:s AND timeframe=:tf AND signature=:sig
                """)
                memory = conn.execute(mem_q, {"s":args.symbol, "tf":args.tf, "sig":signature}).mappings().first()
                alpha = 0.25
                mem_score = 0.5*((memory["win_rate"]) if memory and memory.get("win_rate") is not None else 0.5) \
                           + 0.5*(tanh((memory.get("sharpe") or 0.0)) if memory else 0.0)
                lev_min = 1.0
                lev_max = float(risk.get("lev_max", 10.0))
                lev = clamp(lev * (1 + alpha*(mem_score-0.5)), lev_min, lev_max)

            reason = {
                "direction_ver_id": args.ver_id,
                "strength": strength,
                "atr": atr,
                "tf": args.tf,
                "k_sl": risk.get("k_sl"),
                "k_tp": risk.get("k_tp"),
                "lev_max": risk.get("lev_max"),
                "equity": risk.get("equity"),
            }
            plan_id = insert_plan(conn,
                                  bar_ts=ts, symbol=args.symbol, tf=args.tf, side=side,
                                  entry=close_px, sl=sl, tp=tp,
                                  risk_pct=risk_pct, qty=qty, leverage=lev,
                                  margin_mode=mm, reason=reason)
            if plan_id:
                created += 1

        print(f"TradePlans creados: {created}")

if __name__ == "__main__":
    main()
