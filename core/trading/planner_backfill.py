"""
Planner Backfill
-----------------
Genera planes históricos en trading.trade_plans a partir de market.features
para habilitar minería de estrategias sobre TODO el histórico.

Control por ENV (config/.env):
  PLANNER_BACKFILL_ENABLE=true|false
  PLANNER_BACKFILL_WINDOW_DAYS=730
  PLANNER_BACKFILL_CHUNK_DAYS=30
  PLANNER_BACKFILL_TIMEFRAME=1m

Reglas:
  - Usa el mismo gating básico que el planner online (direction/regime/smc)
  - Calcula entry/SL/TP igual que planner.py
  - Upsert por (symbol,timeframe,ts,plan_id) para no duplicar
"""

from __future__ import annotations

import os, json, uuid, logging
from datetime import timedelta
from typing import Dict, List

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import (
    load_training_config, load_symbols_config, extract_symbols_and_tfs, get_planner_config, load_risk_config_yaml
)

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("PlannerBackfill")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _bps_to_multiplier(bps: int, side: str, invert: bool = False) -> float:
    m = bps/10000.0
    return (1.0 - m if side=="long" else 1.0 + m) if not invert else (1.0 + m if side=="long" else 1.0 - m)


def _fetch_preds_at(engine, symbol: str, qtf: str, ts: pd.Timestamp) -> Dict:
    # Buscar predicciones exactas para este timestamp, o la más cercana si no existe exacta
    sql = text(
        """
        SELECT task, pred_label, pred_conf, probs, ts as pred_ts
        FROM ml.agent_preds
        WHERE symbol=:s AND timeframe=:tf AND ts <= :t
        ORDER BY ts DESC
        LIMIT 3
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"s": symbol, "tf": qtf, "t": ts}).mappings().all()
    
    if not rows:
        return {}
    
    # Agrupar por task y tomar la más reciente para cada uno
    out = {}
    seen_tasks = set()
    
    for r in rows:
        task = r["task"]
        if task in seen_tasks:
            continue
        seen_tasks.add(task)
        
        probs = r["probs"]
        if isinstance(probs, str):
            try:
                probs = json.loads(probs)
            except Exception:
                probs = {}
        
        out[task] = {
            "label": r["pred_label"], 
            "conf": float(r.get("pred_conf") or 0), 
            "probs": probs,
            "pred_ts": r["pred_ts"]
        }
    
    return out


def _load_features_chunk(engine, symbol: str, tf: str, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    sql = text(
        """
        SELECT f.ts, f.atr_14, h.close
        FROM market.features f
        JOIN market.historical_data h ON f.symbol=h.symbol AND f.timeframe=h.timeframe AND f.ts=h.ts
        WHERE f.symbol=:s AND f.timeframe=:tf AND f.ts>=:a AND f.ts<:b
        ORDER BY f.ts ASC
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"s": symbol, "tf": tf, "a": a, "b": b}).mappings().all()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _upsert_plan(engine, plan: Dict):
    sql = text(
        """
        INSERT INTO trading.trade_plans
        (symbol,timeframe,ts,plan_id,side,entry_type,entry_price,limit_offset_bp,stop_loss,tp_targets,
         qty,leverage,risk_pct,max_loss_usdt,take_profit_usdt,valid_until,route,account_id,
         source,confidence,rationale,tags,status)
        VALUES
        (:symbol,:timeframe,:ts,:plan_id,:side,:entry_type,:entry_price,:limit_offset_bp,:stop_loss,:tp_targets,
         :qty,:leverage,:risk_pct,:max_loss_usdt,:take_profit_usdt,:valid_until,:route,:account_id,
         :source,:confidence,:rationale,:tags,:status)
        ON CONFLICT (symbol,timeframe,ts,plan_id) DO UPDATE SET
          side=EXCLUDED.side, entry_type=EXCLUDED.entry_type, entry_price=EXCLUDED.entry_price,
          limit_offset_bp=EXCLUDED.limit_offset_bp, stop_loss=EXCLUDED.stop_loss, tp_targets=EXCLUDED.tp_targets,
          qty=EXCLUDED.qty, leverage=EXCLUDED.leverage, risk_pct=EXCLUDED.risk_pct,
          max_loss_usdt=EXCLUDED.max_loss_usdt, take_profit_usdt=EXCLUDED.take_profit_usdt,
          valid_until=EXCLUDED.valid_until, route=EXCLUDED.route, account_id=EXCLUDED.account_id,
          source=EXCLUDED.source, confidence=EXCLUDED.confidence, rationale=EXCLUDED.rationale,
          tags=EXCLUDED.tags, status=EXCLUDED.status;
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, plan)


def _build_plan(symbol: str, qtf: str, preds: Dict, row: Dict, cfg: Dict) -> Dict | None:
    # Cargar thresholds del risk.yaml
    risk_cfg = load_risk_config_yaml()
    planner_cfg = risk_cfg.get("planner", {})
    thresholds = planner_cfg.get("thresholds", {})
    gating = planner_cfg.get("gating", {})

    thr_dir = thresholds.get("direction_confidence_min", 0.30)
    regime_required = thresholds.get("regime_required", "trend")
    smc_alignment = thresholds.get("smc_alignment", True)

    d = preds.get("direction", {}).get("label")
    r = preds.get("regime", {}).get("label")
    s = preds.get("smc", {}).get("label")

    if not (d and r and s):
        return None
    if r != regime_required:
        return None
    if smc_alignment:
        direction_long_smc = gating.get("direction_long_smc", "bull")
        direction_short_smc = gating.get("direction_short_smc", "bear")
        if d == "long" and s != direction_long_smc:
            return None
        if d == "short" and s != direction_short_smc:
            return None
    if float(preds.get("direction", {}).get("conf", 0.0)) < thr_dir:
        return None

    close = float(row["close"]) if row.get("close") is not None else 0.0
    atr   = float(row.get("atr_14") or 0.0)
    ts    = pd.Timestamp(row["ts"]).to_pydatetime()

    # Parámetros desde training.yaml
    planner_cfg_tr = get_planner_config(cfg)
    ppo_env = (cfg.get("ppo_execution") or {}).get("env", {})
    max_offset_bp = int(ppo_env.get("action_space", {}).get("max_offset_bp", 25))
    offset_bp = min(10, max_offset_bp)

    side = "long" if d=="long" else "short"
    entry_price = close * _bps_to_multiplier(offset_bp, side, invert=False)

    atr_sl_mult = planner_cfg_tr["atr_sl_mult"]
    atr_tp_mult_1 = planner_cfg_tr["atr_tp_mult_1"]
    atr_tp_mult_2 = planner_cfg_tr["atr_tp_mult_2"]

    if side=="long":
        sl = entry_price - atr_sl_mult*atr
        tp1 = entry_price + atr_tp_mult_1*atr
        tp2 = entry_price + atr_tp_mult_2*atr
    else:
        sl = entry_price + atr_sl_mult*atr
        tp1 = entry_price - atr_tp_mult_1*atr
        tp2 = entry_price - atr_tp_mult_2*atr

    leverage = planner_cfg_tr["leverage"]
    balance = planner_cfg_tr["account_balance_usdt"]
    risk_pct = planner_cfg_tr["risk_pct"]

    # qty basada en riesgo nominal (igual que planner)
    dist = abs(entry_price - sl) or 1.0
    risk_usdt = balance * risk_pct / 100.0
    qty = max(0.0, (risk_usdt * leverage) / dist)

    plan = {
      "symbol": symbol, "timeframe": qtf, "ts": ts,
      "plan_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{symbol}|{qtf}|{ts.isoformat()}|planner")),
      "side": side, "entry_type": "limit",
      "entry_price": round(entry_price,6), "limit_offset_bp": offset_bp,
      "stop_loss": round(sl,6),
      "tp_targets": json.dumps([{"p": round(tp1,6), "qty_pct": 50}, {"p": round(tp2,6), "qty_pct": 50}]),
      "qty": qty, "leverage": leverage, "risk_pct": risk_pct,
      "max_loss_usdt": abs(entry_price-sl)*qty,
      "take_profit_usdt": abs(tp1-entry_price)*qty,
      "valid_until": ts + timedelta(minutes=planner_cfg_tr["ttl_minutes"]),
      "route": "bitget_perp", "account_id": "default",
      "source": json.dumps({"from": "planner_backfill"}),
      "confidence": float(preds.get("direction", {}).get("conf", 0.0)),
      "rationale": json.dumps({"gating": "dir>=thr & regime=trend & smc aligned"}),
      "tags": ["planner","backfill"], "status": "planned"
    }
    return plan


def backfill(window_days: int | None = None) -> int:
    cfg = load_training_config()
    cfg_symbols = load_symbols_config()
    symbols, _ = extract_symbols_and_tfs(cfg_symbols, cfg)

    qtf = os.getenv("PLANNER_BACKFILL_TIMEFRAME", (cfg.get("heads", {}).get("execution", {}) or {}).get("query_tf", "1m"))
    win_days = int(os.getenv("PLANNER_BACKFILL_WINDOW_DAYS", str(window_days or 730)))
    chunk_days = int(os.getenv("PLANNER_BACKFILL_CHUNK_DAYS", "30"))

    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

    # Rango temporal
    with engine.begin() as conn:
        row = conn.execute(text("SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts FROM market.features WHERE timeframe=:tf"), {"tf": qtf}).mappings().first()
    if not row or not row["min_ts"]:
        logger.info("Sin features para backfill")
        return 0

    end_ts = pd.Timestamp.now(tz="UTC")
    min_ts = pd.to_datetime(row["min_ts"], utc=True)
    start_ts = max(min_ts, end_ts - timedelta(days=win_days))

    written = 0
    cur = pd.to_datetime(start_ts, utc=True)
    total_chunks = ((end_ts - cur).days // chunk_days) + 1
    chunk_num = 0
    
    while cur < end_ts:
        nxt = min(cur + timedelta(days=chunk_days), end_ts)
        chunk_num += 1
        chunk_plans = 0
        
        logger.info(f"📅 Procesando chunk {chunk_num}/{total_chunks}: {cur.strftime('%Y-%m-%d %H:%M')} → {nxt.strftime('%Y-%m-%d %H:%M')}")
        
        for sym in symbols:
            df = _load_features_chunk(engine, sym, qtf, cur, nxt)
            if df.empty:
                continue
                
            sym_plans = 0
            for _, r in df.iterrows():
                preds = _fetch_preds_at(engine, sym, qtf, r["ts"])
                if not preds:  # No hay predicciones para este timestamp
                    continue
                    
                plan = _build_plan(sym, qtf, preds, r, cfg)
                if plan:
                    _upsert_plan(engine, plan)
                    written += 1
                    chunk_plans += 1
                    sym_plans += 1
            
            if sym_plans > 0:
                logger.info(f"  📊 {sym}: {sym_plans} planes generados")
        
        logger.info(f"✅ Chunk {chunk_num} completado: {chunk_plans} planes | Total acumulado: {written}")
        cur = nxt

    logger.info(f"Backfill completado: {written} planes")
    return written


if __name__ == "__main__":
    backfill()
