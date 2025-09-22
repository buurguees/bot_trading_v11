"""
Agentes Backfill
================
Genera predicciones históricas para direction, regime y smc en todo el histórico
disponible para habilitar minería de estrategias basada en patrones del pasado.

Control por ENV:
  AGENTS_BACKFILL_ENABLE=true|false
  AGENTS_BACKFILL_WINDOW_DAYS=730
  AGENTS_BACKFILL_CHUNK_DAYS=30
  AGENTS_BACKFILL_TIMEFRAME=1m

Procesa en chunks para evitar cargar todo en memoria.
"""

from __future__ import annotations

import os, json, logging
from datetime import timedelta
from typing import Dict, List

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import (
    load_training_config, load_symbols_config,
    build_encoder_config_from_training, extract_symbols_and_tfs, get_window_from_training
)
from core.ml.models.direction import DirectionModel
from core.ml.models.regime import RegimeModel
from core.ml.models.smc import SMCModel

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("AgentsBackfill")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _upsert_agent_pred(engine, *, symbol: str, timeframe: str, ts, task: str,
                       label: str, probs: dict, model_name: str, model_version: str):
    pred_conf = float(probs.get(label, 0.0))
    
    sql = text("""
        INSERT INTO ml.agent_preds
            (symbol, timeframe, ts, task, pred_label, pred_conf, probs)
        VALUES
            (:symbol, :tf, :ts, :task, :label, :pred_conf, :probs)
        ON CONFLICT (symbol, timeframe, ts, task) DO UPDATE SET
            pred_label   = EXCLUDED.pred_label,
            pred_conf    = EXCLUDED.pred_conf,
            probs        = EXCLUDED.probs,
            created_at   = NOW();
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "symbol": symbol, "tf": timeframe, "ts": ts, "task": task,
            "label": label, "pred_conf": pred_conf, "probs": json.dumps(probs)
        })


def _load_features_chunk(engine, symbol: str, tf: str, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    sql = text("""
        SELECT f.ts, f.atr_14, h.close, f.rsi_14, f.macd, f.macd_signal, f.ema_20, f.ema_50, f.ema_200,
               f.obv, f.supertrend_dir, f.smc_flags
        FROM market.features f
        JOIN market.historical_data h ON f.symbol=h.symbol AND f.timeframe=h.timeframe AND f.ts=h.ts
        WHERE f.symbol=:s AND f.timeframe=:tf AND f.ts>=:a AND f.ts<:b
        ORDER BY f.ts ASC
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"s": symbol, "tf": tf, "a": a, "b": b}).mappings().all()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _predict_at_ts(model, symbol: str, tfs: List[str], window: int, query_tf: str, 
                   ts: pd.Timestamp, engine) -> Dict | None:
    """Predice en un timestamp específico usando datos históricos hasta ese punto"""
    try:
        # Cargar ventana de datos hasta el timestamp
        sql = text("""
            SELECT f.ts, f.atr_14, h.close, f.rsi_14, f.macd, f.macd_signal, 
                   f.ema_20, f.ema_50, f.ema_200, f.obv, f.supertrend_dir, f.smc_flags
            FROM market.features f
            JOIN market.historical_data h ON f.symbol=h.symbol AND f.timeframe=h.timeframe AND f.ts=h.ts
            WHERE f.symbol=:s AND f.timeframe=:tf AND f.ts<=:ts
            ORDER BY f.ts DESC
            LIMIT :window
        """)
        
        with engine.begin() as conn:
            rows = conn.execute(sql, {"s": symbol, "tf": query_tf, "ts": ts, "window": window}).mappings().all()
        
        if len(rows) < window:
            return None
            
        # Convertir a DataFrame y ordenar cronológicamente
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        
        # Simular predict_from_db con datos históricos
        # (esto es una simplificación - en la práctica necesitarías adaptar el modelo)
        return model._predict_from_features(df)
        
    except Exception as e:
        logger.warning(f"Error prediciendo {symbol} en {ts}: {e}")
        return None


def backfill_agent(agent_type: str, window_days: int | None = None) -> int:
    """Backfill para un tipo de agente específico"""
    cfg_train = load_training_config()
    cfg_symbols = load_symbols_config()
    
    enc_cfg = build_encoder_config_from_training(cfg_train)
    heads_cfg = cfg_train.get("heads", {})
    qtf = heads_cfg.get(agent_type, {}).get("query_tf", enc_cfg.query_tf_default)
    
    # Override con ENV si está disponible
    qtf = os.getenv("AGENTS_BACKFILL_TIMEFRAME", qtf)
    
    symbols, tfs = extract_symbols_and_tfs(cfg_symbols, cfg_train)
    window = get_window_from_training(cfg_train)
    
    win_days = int(os.getenv("AGENTS_BACKFILL_WINDOW_DAYS", str(window_days or 730)))
    chunk_days = int(os.getenv("AGENTS_BACKFILL_CHUNK_DAYS", "30"))
    
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    
    # Cargar modelo apropiado
    if agent_type == "direction":
        model = DirectionModel(enc_cfg)
        model_name = "direction_backfill_v1"
    elif agent_type == "regime":
        model = RegimeModel(enc_cfg)
        model_name = "regime_backfill_v1"
    elif agent_type == "smc":
        model = SMCModel(enc_cfg)
        model_name = "smc_backfill_v1"
    else:
        logger.error(f"Tipo de agente no soportado: {agent_type}")
        return 0
    
    # Rango temporal
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts 
            FROM market.features 
            WHERE timeframe=:tf
        """), {"tf": qtf}).mappings().first()
    
    if not row or not row["min_ts"]:
        logger.info(f"Sin features para backfill de {agent_type}")
        return 0
    
    end_ts = pd.Timestamp.utcnow()
    start_ts = max(pd.Timestamp(row["min_ts"]).to_pydatetime(), 
                   (end_ts - timedelta(days=win_days)).to_pydatetime())
    
    written = 0
    cur = pd.Timestamp(start_ts, tz="UTC")
    
    while cur < end_ts:
        nxt = min(cur + timedelta(days=chunk_days), end_ts)
        
        for sym in symbols:
            df = _load_features_chunk(engine, sym, qtf, cur, nxt)
            if df.empty:
                continue
                
            for _, row in df.iterrows():
                ts = row["ts"]
                
                # Verificar si ya existe predicción para este timestamp
                with engine.begin() as conn:
                    exists = conn.execute(text("""
                        SELECT 1 FROM ml.agent_preds 
                        WHERE symbol=:s AND timeframe=:tf AND ts=:ts AND task=:task
                    """), {"s": sym, "tf": qtf, "ts": ts, "task": agent_type}).scalar()
                
                if exists:
                    continue  # Ya existe, saltar
                
                # Predecir en este timestamp
                pred = _predict_at_ts(model, sym, tfs, window, qtf, ts, engine)
                if not pred:
                    continue
                
                _upsert_agent_pred(
                    engine,
                    symbol=sym, timeframe=qtf, ts=ts, task=agent_type,
                    label=pred["label"], probs=pred["probs"],
                    model_name=model_name, model_version="backfill"
                )
                written += 1
        
        logger.info(f"Backfill {agent_type} {cur} → {nxt}: {written} predicciones")
        cur = nxt
    
    logger.info(f"Backfill {agent_type} completado: {written} predicciones")
    return written


def backfill_all_agents(window_days: int | None = None) -> int:
    """Backfill para todos los agentes"""
    total = 0
    for agent_type in ["direction", "regime", "smc"]:
        logger.info(f"🔄 Iniciando backfill de {agent_type}...")
        count = backfill_agent(agent_type, window_days)
        total += count
        logger.info(f"✅ {agent_type}: {count} predicciones")
    return total


if __name__ == "__main__":
    backfill_all_agents()
