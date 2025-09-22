"""
Backtest Vectorized (rápido)
----------------------------
Lee:
  - ml.strategies (status='testing') para una (symbol,timeframe)
  - market.features (close, atr_14) para simular reglas simples (entry limit con offset, SL ATR, TP 1R)
  - config/backtest/backtest.yaml (parámetros de costes y ventanas)

Escribe:
  - ml.backtest_runs (engine='vectorized') con métricas globales (PnL, Sharpe aprox, PF, maxDD, winrate)
  - (opcional) un resumen de trades; el detallado se deja al event-driven

Qué hace:
  - Para cada strategy_id, recorre el histórico y genera operaciones simples:
      * Entrada: precio de cierre de la vela +/− offset bps (aprox)
      * SL: múltiplo de ATR
      * TP: 1R (distancia igual a la del stop)
  - Calcula PnL neto con fees/slippage agregados y agrega métricas.

Funciones:
  - run_vectorized(strategy_ids:list[str] | None=None) -> list[str]  # run_ids
Limitaciones:
  - Simplificado (no cola de órdenes ni parciales).
  - Útil para cribar rápido antes del event-driven.
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

logger = logging.getLogger("BacktestVectorized")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _load_strategies(engine, strategy_ids: Optional[List[str]] = None):
    if strategy_ids:
        sql = text("""
            SELECT strategy_id, symbol, timeframe, rules, risk_profile
            FROM ml.strategies
            WHERE strategy_id = ANY(:ids) AND status IN ('testing','ready_for_training','promoted','shadow')
        """)
        params = {"ids": strategy_ids}
    else:
        sql = text("""
            SELECT strategy_id, symbol, timeframe, rules, risk_profile
            FROM ml.strategies
            WHERE status='testing'
        """)
        params = {}
    with engine.begin() as conn:
        return [dict(r) for r in conn.execute(sql, params).mappings().all()]


def _fetch_features(engine, symbol: str, timeframe: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    sql = text("""
        SELECT f.ts, h.close, f.atr_14
        FROM market.features f
        JOIN market.historical_data h ON f.symbol = h.symbol AND f.timeframe = h.timeframe AND f.ts = h.ts
        WHERE f.symbol=:s AND f.timeframe=:tf
          AND (:start IS NULL OR f.ts >= :start)
          AND (:end IS NULL OR f.ts <= :end)
        ORDER BY f.ts ASC
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"s": symbol, "tf": timeframe, "start": start, "end": end}).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["ts","close","atr_14"]).astype({"ts":"datetime64[ns]"})
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["atr_14"] = df["atr_14"].astype(float).ffill()
    return df


def _metrics_from_pnl(equity_curve: np.ndarray, rets: np.ndarray) -> Dict:
    if len(equity_curve) < 2:
        return {"trades": 0, "sharpe": 0.0, "profit_factor": 0.0, "max_dd": 0.0, "winrate": 0.0, "pnl": 0.0}
    pnl = equity_curve[-1] - equity_curve[0]
    dd = np.maximum.accumulate(equity_curve) - equity_curve
    max_dd = float(np.max(dd) / max(1e-8, np.max(equity_curve)))
    # sharpe diario aprox con rets por trade
    mu = np.mean(rets) if len(rets) else 0.0
    sd = np.std(rets) if len(rets) else 1e-8
    sharpe = float((mu / sd) * np.sqrt(max(1.0, len(rets))))
    wins = np.sum(rets > 0)
    losses = np.sum(rets < 0)
    pf = float((rets[rets>0].sum() / abs(rets[rets<0].sum())) if losses>0 else (np.inf if wins>0 else 0.0))
    winrate = float(wins / max(1, wins+losses))
    return {"trades": int(len(rets)), "sharpe": sharpe, "profit_factor": pf, "max_dd": max_dd, "winrate": winrate, "pnl": float(pnl)}


def run_vectorized(strategy_ids: Optional[List[str]] = None) -> List[str]:
    cfg = load_backtest_config()
    fees_bps = float(cfg.get("vectorized", {}).get("fees_bps", 2.5))
    slippage_bps = float(cfg.get("vectorized", {}).get("slippage_bps", 1.0))
    warmup_bars = int(cfg.get("vectorized", {}).get("warmup_bars", 300))

    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    strategies = _load_strategies(engine, strategy_ids)
    if not strategies:
        logger.info("No hay estrategias 'testing'.")
        return []

    run_ids: List[str] = []

    for s in strategies:
        rules = s["rules"]; 
        if isinstance(rules, str):
            rules = json.loads(rules)
        side = rules.get("side", "long")
        offset_bp = float(rules.get("limit_offset_bp") or 0.0)
        sl_bp = float(rules.get("sl_bp") or 100.0)

        # datos
        df = _fetch_features(engine, s["symbol"], s["timeframe"],
                             start=None, end=None)
        if len(df) <= warmup_bars:
            logger.info(f"[{s['symbol']} {s['timeframe']}] pocos datos.")
            continue
        df = df.iloc[warmup_bars:].reset_index(drop=True)

        # Simulación simple por barras: 1 trade por X barras (heurística ligera)
        equity = [0.0]; rets = []
        step = 10  # cada 10 velas intentamos entrada
        for i in range(0, len(df)-step-1, step):
            px = float(df.loc[i, "close"])
            atr = float(df.loc[i, "atr_14"] or 0.0)

            # Entrada limitada ~ offset bps desde close actual
            mult = (1 - offset_bp/10000.0) if side=="long" else (1 + offset_bp/10000.0)
            entry = px * mult

            # SL por bps (aprox) si no hay ATR -> usa bps; si hay ATR usa 1.5 ATR como fallback
            sl_dist = (sl_bp/10000.0)*entry if atr<=0 else max((sl_bp/10000.0)*entry, 1.5*atr)
            if side=="long":
                sl = entry - sl_dist
                tp = entry + sl_dist
            else:
                sl = entry + sl_dist
                tp = entry - sl_dist

            # Siguiente bloque de barras para evaluar
            block = df.loc[i+1:i+step, "close"].astype(float).values
            if side=="long":
                hit_tp = np.any(block >= tp)
                hit_sl = np.any(block <= sl)
                # prioridad: si ambos, asumimos primero que toque (aprox con distancia)
                if hit_tp and not hit_sl:
                    gross = tp - entry
                elif hit_sl and not hit_tp:
                    gross = sl - entry
                else:
                    # ninguno → cierra en la última
                    gross = block[-1] - entry
            else:
                hit_tp = np.any(block <= tp)
                hit_sl = np.any(block >= sl)
                if hit_tp and not hit_sl:
                    gross = entry - tp
                elif hit_sl and not hit_tp:
                    gross = entry - sl
                else:
                    gross = entry - block[-1]

            # costes
            roundtrip_bps = fees_bps*2 + slippage_bps*2
            cost = entry * (roundtrip_bps/10000.0)
            ret = float(gross - cost)
            equity.append(equity[-1] + ret)
            rets.append(ret)

        eq = np.array(equity, dtype=float)
        rets = np.array(rets, dtype=float)
        metrics = _metrics_from_pnl(eq, rets)

        # Guardar run
        run_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO ml.backtest_runs
                  (run_id, strategy_id, symbol, timeframe, engine, dataset_start, dataset_end,
                   cv_schema, market_costs, config, metrics, status)
                VALUES
                  (:rid, :sid, :sym, :tf, 'vectorized', :ds, :de,
                   :cv, :mc, :cfg, :mt, 'ok')
            """), {
                "rid": run_id, "sid": s["strategy_id"], "sym": s["symbol"], "tf": s["timeframe"],
                "ds": str(df["ts"].iloc[0].to_pydatetime()), "de": str(df["ts"].iloc[-1].to_pydatetime()),
                "cv": json.dumps({"type":"single_window"}),
                "mc": json.dumps({"fees_bps": fees_bps, "slippage_bps": slippage_bps}),
                "cfg": json.dumps({"rules": rules}),
                "mt": json.dumps(metrics),
            })
        run_ids.append(run_id)
        logger.info(f"Run vectorized {run_id} — trades={metrics['trades']} sharpe={metrics['sharpe']:.2f} pf={metrics['profit_factor']:.2f}")

    return run_ids


if __name__ == "__main__":
    run_vectorized()
