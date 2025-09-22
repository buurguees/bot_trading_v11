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

# Configuración desde ENV
BACKTEST_RANDOM_WINDOWS = os.getenv("BACKTEST_RANDOM_WINDOWS", "false").lower() in ("true", "1", "yes", "y", "on")
BACKTEST_RANDOM_WINDOWS_COUNT = int(os.getenv("BACKTEST_RANDOM_WINDOWS_COUNT", "3"))

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
    """Calcula métricas sobre una curva de equity multiplicativa (base=1.0)."""
    if len(equity_curve) < 2:
        return {"trades": 0, "sharpe": 0.0, "profit_factor": 0.0, "max_dd": 0.0, "winrate": 0.0, "pnl": 0.0}
    pnl = equity_curve[-1] - 1.0
    peaks = np.maximum.accumulate(equity_curve)
    dd = peaks - equity_curve
    peak = np.maximum(peaks, 1e-8)
    max_dd = float(np.max(dd / peak))
    # Sharpe por trade (no anualizado)
    if len(rets) > 1:
        mu = float(np.mean(rets))
        sd = float(np.std(rets, ddof=1) or 1e-8)
        sharpe = float(mu / sd) if sd > 0 else 0.0
    else:
        sharpe = 0.0
    wins = int(np.sum(rets > 0))
    losses = int(np.sum(rets < 0))
    if losses > 0:
        avg_win = float(np.mean(rets[rets > 0])) if wins > 0 else 0.0
        avg_loss = float(abs(np.mean(rets[rets < 0])))
        pf = float((avg_win * wins) / (avg_loss * losses))
    else:
        pf = float(np.inf if wins > 0 else 0.0)
    winrate = float(wins / max(1, wins + losses))
    return {"trades": int(len(rets)), "sharpe": sharpe, "profit_factor": pf, "max_dd": max_dd, "winrate": winrate, "pnl": float(pnl)}


def run_vectorized(strategy_ids: Optional[List[str]] = None) -> List[str]:
    cfg = load_backtest_config()
    vcfg = cfg.get("vectorized", {})
    fees_bps = float(vcfg.get("fees_bps", 2.5))
    slippage_bps = float(vcfg.get("slippage_bps", 1.0))
    warmup_bars = int(vcfg.get("warmup_bars", 300))
    # Random windows config (opcional)
    rw_cfg = (vcfg.get("random_windows") or {})
    rw_enabled = bool(rw_cfg.get("enabled", False))
    rw_n = int(rw_cfg.get("n", 0))
    rw_min_days = int(rw_cfg.get("min_days", 3))
    rw_max_days = int(rw_cfg.get("max_days", 7))

    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    strategies = _load_strategies(engine, strategy_ids)
    if not strategies:
        logger.info("No hay estrategias 'testing'.")
        return []

    run_ids: List[str] = []

    # Mapa TF -> barras por día para aproximar ventanas en días
    TF_BARS_PER_DAY = {
        "1m": 1440,
        "5m": 288,
        "15m": 96,
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }

    def simulate_on_df(df_sim: pd.DataFrame, rules: Dict) -> Dict:
        side = rules.get("side", "long")
        offset_bp = float(rules.get("limit_offset_bp") or 0.0)
        sl_bp = float(rules.get("sl_bp") or 100.0)
        equity: List[float] = [1.0]
        rets: List[float] = []
        step = 10
        for i in range(0, len(df_sim)-step-1, step):
            px = float(df_sim.loc[i, "close"])
            atr = float(df_sim.loc[i, "atr_14"] or 0.0)
            mult = (1 - offset_bp/10000.0) if side=="long" else (1 + offset_bp/10000.0)
            entry = px * mult
            sl_dist = (sl_bp/10000.0)*entry if atr<=0 else max((sl_bp/10000.0)*entry, 1.5*atr)
            if side=="long":
                sl = entry - sl_dist; tp = entry + sl_dist
            else:
                sl = entry + sl_dist; tp = entry - sl_dist
            block = df_sim.loc[i+1:i+step, "close"].astype(float).values
            if side=="long":
                hit_tp = np.any(block >= tp); hit_sl = np.any(block <= sl)
                if hit_tp and not hit_sl: gross = tp - entry
                elif hit_sl and not hit_tp: gross = sl - entry
                else: gross = block[-1] - entry
            else:
                hit_tp = np.any(block <= tp); hit_sl = np.any(block >= sl)
                if hit_tp and not hit_sl: gross = entry - tp
                elif hit_sl and not hit_tp: gross = entry - sl
                else: gross = entry - block[-1]
            roundtrip_bps = fees_bps*2 + slippage_bps*2
            cost_prop = (roundtrip_bps/10000.0)
            ret_prop = float(gross/entry - cost_prop)
            equity.append(equity[-1] * (1.0 + ret_prop))
            rets.append(ret_prop)
        return _metrics_from_pnl(np.array(equity, dtype=float), np.array(rets, dtype=float))

    for s in strategies:
        rules = s["rules"]; 
        if isinstance(rules, str):
            rules = json.loads(rules)

        # datos
        df = _fetch_features(engine, s["symbol"], s["timeframe"],
                             start=None, end=None)
        if len(df) <= warmup_bars:
            logger.info(f"[{s['symbol']} {s['timeframe']}] pocos datos.")
            continue
        df = df.iloc[warmup_bars:].reset_index(drop=True)

        # Si random windows está activo, generar N runs por ventanas aleatorias
        if rw_enabled and rw_n > 0:
            bars_per_day = TF_BARS_PER_DAY.get(s["timeframe"], 24)
            for j in range(rw_n):
                win_days = np.random.randint(rw_min_days, rw_max_days+1)
                win_bars = max(bars_per_day * win_days, 50)
                if len(df) <= win_bars:
                    continue
                start_idx = np.random.randint(0, len(df) - win_bars)
                df_win = df.iloc[start_idx:start_idx+win_bars].reset_index(drop=True)
                if len(df_win) < 60:
                    continue
                metrics = simulate_on_df(df_win, rules)
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
                        "ds": str(df_win["ts"].iloc[0].to_pydatetime()),
                        "de": str(df_win["ts"].iloc[-1].to_pydatetime()),
                        "cv": json.dumps({"type":"random_window","days":win_days}),
                        "mc": json.dumps({"fees_bps": fees_bps, "slippage_bps": slippage_bps}),
                        "cfg": json.dumps({"rules": rules}),
                        "mt": json.dumps(metrics),
                    })
                run_ids.append(run_id)
                logger.info(f"Run vectorized(random) {run_id} — trades={metrics['trades']} sharpe={metrics['sharpe']:.2f} pf={metrics['profit_factor']:.2f}")
            # además, continuar con el full-run estándar para tener referencia global

        # Full-run estándar sobre todo el df restante
        metrics = simulate_on_df(df, rules)

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
