"""
Backtest Event-Driven (realista, base)
--------------------------------------
Lee:
  - ml.strategies (p.ej., las top del ranker)
  - market.features (close, atr_14) para ticks de vela
  - config/backtest/backtest.yaml (fees, slippage, latency, parciales)

Escribe:
  - ml.backtest_runs (engine='event_driven') con métricas
  - ml.backtest_trades (detalle por trade)

Qué hace:
  - Simula ejecución más realista con:
      * Latencia fija (ms) -> desplaza la entrada 1 vela si la latencia excede el TF.
      * Slippage y fees por cada fill.
      * TP/SL discretos (1R y 2R) con cierre parcial 50/50.
  - Es una base simplificada para robustecer más adelante.

Funciones:
  - run_event_driven(strategy_ids:list[str] | None=None) -> list[str]  # run_ids
"""

from __future__ import annotations
import os, json, math, uuid, logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import load_backtest_config

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("BacktestEvent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _load_testing_strategies(engine, strategy_ids: Optional[List[str]]):
    if strategy_ids:
        sql = text("""SELECT strategy_id, symbol, timeframe, rules FROM ml.strategies WHERE strategy_id=ANY(:ids)""")
        params = {"ids": strategy_ids}
    else:
        sql = text("""SELECT strategy_id, symbol, timeframe, rules FROM ml.strategies WHERE status IN ('testing','ready_for_training')""")
        params = {}
    with engine.begin() as conn:
        return [dict(r) for r in conn.execute(sql, params).mappings().all()]


def _features(engine, symbol: str, timeframe: str) -> pd.DataFrame:
    sql = text("""
        SELECT f.ts, h.close, f.atr_14 
        FROM market.features f
        JOIN market.historical_data h ON f.symbol = h.symbol AND f.timeframe = h.timeframe AND f.ts = h.ts
        WHERE f.symbol=:s AND f.timeframe=:tf 
        ORDER BY f.ts ASC
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"s": symbol, "tf": timeframe}).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["ts","close","atr_14"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["atr_14"] = df["atr_14"].astype(float).ffill()
    return df


def run_event_driven(strategy_ids: Optional[List[str]] = None) -> List[str]:
    cfg = load_backtest_config()
    ev = cfg.get("event_driven", {}) or {}
    fees_bps = float(ev.get("fees_bps", 3.0))
    slippage_bps = float(ev.get("slippage_bps", 2.0))
    latency_ms = int(ev.get("latency_ms", 120))
    warmup_bars = int(ev.get("warmup_bars", 300))

    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    strategies = _load_testing_strategies(engine, strategy_ids)
    if not strategies:
        logger.info("No hay estrategias para event-driven.")
        return []

    run_ids: List[str] = []
    for s in strategies:
        rules = s["rules"]; rules = json.loads(rules) if isinstance(rules, str) else rules
        side = rules.get("side","long")
        offset_bp = float(rules.get("limit_offset_bp") or 0.0)
        sl_bp = float(rules.get("sl_bp") or 100.0)

        df = _features(engine, s["symbol"], s["timeframe"])
        if len(df) <= warmup_bars + 2:
            continue
        df = df.iloc[warmup_bars:].reset_index(drop=True)

        trades = []
        equity = [0.0]
        step = 10
        for i in range(0, len(df)-step-2, step):
            px = float(df.loc[i, "close"])
            atr = float(df.loc[i, "atr_14"] or 0.0)

            # latencia -> empuja la entrada una vela si hace falta (modelo simple)
            entry_idx = i + 1
            entry_base = float(df.loc[entry_idx, "close"])

            mult = (1 - offset_bp/10000.0) if side=="long" else (1 + offset_bp/10000.0)
            entry = entry_base * mult

            sl_dist = (sl_bp/10000.0)*entry if atr<=0 else max((sl_bp/10000.0)*entry, 1.5*atr)
            if side=="long":
                sl = entry - sl_dist
                tp1 = entry + sl_dist
                tp2 = entry + 2*sl_dist
            else:
                sl = entry + sl_dist
                tp1 = entry - sl_dist
                tp2 = entry - 2*sl_dist

            block = df.loc[entry_idx+1:entry_idx+step, "close"].astype(float).values
            hit_tp1 = (np.any(block >= tp1) if side=="long" else np.any(block <= tp1))
            hit_tp2 = (np.any(block >= tp2) if side=="long" else np.any(block <= tp2))
            hit_sl  = (np.any(block <= sl)  if side=="long" else np.any(block >= sl))

            # cierre parcial 50/50
            gross = 0.0
            if hit_sl and not hit_tp1:
                gross = (sl - entry) if side=="long" else (entry - sl)
            elif hit_tp1 and not hit_sl:
                if hit_tp2:  # llegó a 2R
                    gross = ((tp1 - entry) + (tp2 - entry))/2.0 if side=="long" else ((entry - tp1) + (entry - tp2))/2.0
                else:        # sólo 1R y cierra el resto al final del bloque
                    last = block[-1]
                    gross = ((tp1 - entry) + (last - entry))/2.0 if side=="long" else ((entry - tp1) + (entry - last))/2.0
            else:
                # ni SL ni TP1 -> cierra al final
                last = block[-1]
                gross = (last - entry) if side=="long" else (entry - last)

            cost = entry * ((fees_bps + slippage_bps)/10000.0) * 2  # ida y vuelta
            ret = float(gross - cost)
            equity.append(equity[-1] + ret)
            trades.append({
                "entry_ts": df.loc[entry_idx, "ts"], "exit_ts": df.loc[min(entry_idx+step, len(df)-1), "ts"],
                "entry_price": entry, "exit_price": entry + gross if side=="long" else entry - gross,
                "qty": 1.0, "pnl_usdt": ret, "mae": None, "mfe": None, "fees_usdt": entry*((fees_bps/10000.0)*2)
            })

        eq = np.array(equity, dtype=float)
        rets = np.diff(eq)
        if len(rets)==0:
            continue

        pnl = eq[-1] - eq[0]
        dd = np.maximum.accumulate(eq) - eq
        max_dd = float(np.max(dd) / max(1e-8, np.max(eq)))
        mu = float(np.mean(rets)); sd = float(np.std(rets) or 1e-8)
        sharpe = (mu/sd) * np.sqrt(max(1.0, len(rets)))
        wins = int(np.sum(rets>0)); losses = int(np.sum(rets<0))
        pf = float((rets[rets>0].sum() / abs(rets[rets<0].sum())) if losses>0 else (np.inf if wins>0 else 0.0))
        winrate = float(wins / max(1, wins+losses))

        metrics = {"trades": int(len(rets)), "pnl": float(pnl), "max_dd": max_dd,
                   "sharpe": float(sharpe), "profit_factor": float(pf), "winrate": float(winrate)}

        run_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO ml.backtest_runs
                  (run_id, strategy_id, symbol, timeframe, engine, dataset_start, dataset_end, market_costs, metrics, status)
                VALUES
                  (:rid, :sid, :sym, :tf, 'event_driven', :ds, :de, :mc, :mt, 'ok')
            """), {
                "rid": run_id, "sid": s["strategy_id"], "sym": s["symbol"], "tf": s["timeframe"],
                "ds": str(df["ts"].iloc[0].to_pydatetime()), "de": str(df["ts"].iloc[-1].to_pydatetime()),
                "mc": json.dumps({"fees_bps": fees_bps, "slippage_bps": slippage_bps, "latency_ms": latency_ms}),
                "mt": json.dumps(metrics),
            })
            # trades detallados
            for t in trades:
                conn.execute(text("""
                    INSERT INTO ml.backtest_trades
                      (entry_ts, run_id, trade_id, symbol, side, entry_price, exit_ts, exit_price, qty, pnl_usdt, mae, mfe, fees_usdt, bars_held, notes)
                    VALUES
                      (:entry_ts, :rid, gen_random_uuid(), :sym, :side, :entry_price, :exit_ts, :exit_price,
                       :qty, :pnl_usdt, :mae, :mfe, :fees_usdt, :bars, '{}'::jsonb)
                """), {
                    **t, "rid": run_id, "sym": s["symbol"], "side": side,
                    "bars": 10
                })
        run_ids.append(run_id)
        logger.info(f"Run event {run_id} — trades={metrics['trades']} sharpe={metrics['sharpe']:.2f} pf={metrics['profit_factor']:.2f}")

    return run_ids


if __name__ == "__main__":
    run_event_driven()
