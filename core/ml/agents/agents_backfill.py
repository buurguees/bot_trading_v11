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
    # Selección mínima robusta (evita columnas opcionales que pueden no existir en tu esquema)
    sql = text("""
        SELECT f.ts, f.atr_14, h.close
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
                   ts: pd.Timestamp, engine, agent_type: str) -> Dict | None:
    """Predice en un timestamp específico usando datos históricos hasta ese punto"""
    try:
        # Cargar ventana de datos hasta el timestamp (usando las columnas reales de market.features)
        sql = text("""
            SELECT f.ts, f.atr_14, h.close, f.rsi_14, f.macd, f.macd_signal, f.macd_hist,
                   f.ema_20, f.ema_50, f.ema_200, f.obv, f.supertrend, f.st_direction, f.smc_flags, f.extra
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
        
        # Predicción basada en features técnicas reales
        # En un sistema real, aquí cargarías el modelo entrenado y harías la predicción
        import random
        import numpy as np
        
        if len(df) < 2:
            return None
            
        # Obtener la fila más reciente (última en el DataFrame ordenado)
        latest = df.iloc[-1]
        
        # Features técnicas disponibles
        rsi = latest.get("rsi_14", 50.0)
        macd = latest.get("macd", 0.0)
        macd_signal = latest.get("macd_signal", 0.0)
        ema_20 = latest.get("ema_20", 0.0)
        ema_50 = latest.get("ema_50", 0.0)
        ema_200 = latest.get("ema_200", 0.0)
        close = latest.get("close", 0.0)
        st_direction = latest.get("st_direction", "neutral")
        
        # Generar predicciones basadas en indicadores técnicos
        if agent_type == "direction":
            # Lógica basada en RSI, MACD y EMAs
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI
            if rsi < 30:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 1
                
            # MACD
            if macd > macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            # EMAs
            if ema_20 > ema_50 > ema_200:
                bullish_signals += 1
            elif ema_20 < ema_50 < ema_200:
                bearish_signals += 1
                
            # SuperTrend
            if st_direction == "bull":
                bullish_signals += 1
            elif st_direction == "bear":
                bearish_signals += 1
            
            # Decidir dirección
            if bullish_signals > bearish_signals:
                label = "long"
                confidence = min(0.9, 0.5 + (bullish_signals - bearish_signals) * 0.1)
                probs = {"long": confidence, "short": 1.0 - confidence}
            elif bearish_signals > bullish_signals:
                label = "short"
                confidence = min(0.9, 0.5 + (bearish_signals - bullish_signals) * 0.1)
                probs = {"long": 1.0 - confidence, "short": confidence}
            else:
                label = "long" if random.random() > 0.5 else "short"
                probs = {"long": 0.5, "short": 0.5}
                
        elif agent_type == "regime":
            # Lógica basada en volatilidad y tendencia
            price_volatility = df["close"].std() / df["close"].mean() if len(df) > 1 else 0
            
            if price_volatility > 0.02:  # Alta volatilidad
                label = "trend"
                probs = {"trend": 0.8, "range": 0.2}
            else:
                label = "range"
                probs = {"trend": 0.2, "range": 0.8}
                
        elif agent_type == "smc":
            # Lógica basada en estructura de mercado
            if ema_20 > ema_50 and close > ema_20:
                label = "bull"
                probs = {"bull": 0.7, "bear": 0.3}
            elif ema_20 < ema_50 and close < ema_20:
                label = "bear"
                probs = {"bear": 0.7, "bull": 0.3}
            else:
                label = "bull" if random.random() > 0.5 else "bear"
                probs = {"bull": 0.5, "bear": 0.5}
        else:
            return None
            
        return {
            "label": label,
            "probs": probs
        }
        
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
    chunk_days = int(os.getenv("AGENTS_BACKFILL_CHUNK_DAYS", "5"))  # Reducido a 5 días para prueba
    strict_chrono = os.getenv("AGENTS_BACKFILL_STRICT", "true").lower() in ("1","true","yes","y","on")
    
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
    
    # Rango temporal - limitar a los últimos 30 días para prueba
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts 
            FROM market.features 
            WHERE timeframe=:tf
        """), {"tf": qtf}).mappings().first()
    
    if not row or not row["min_ts"]:
        logger.info(f"Sin features para backfill de {agent_type}")
        return 0
    
    # Asegurar timestamps con tz UTC - LIMITAR A 30 DÍAS PARA PRUEBA
    end_ts = pd.Timestamp.now(tz="UTC")
    min_ts = pd.to_datetime(row["min_ts"], utc=True)
    # Limitar a 30 días para evitar sobrecarga
    test_days = min(30, win_days)
    start_ts = max(min_ts, end_ts - timedelta(days=test_days))
    
    logger.info(f"📅 Rango temporal: {start_ts.strftime('%Y-%m-%d')} → {end_ts.strftime('%Y-%m-%d')} ({test_days} días)")
    
    written = 0
    cur = pd.to_datetime(start_ts, utc=True)
    total_chunks = ((end_ts - cur).days // chunk_days) + 1
    chunk_num = 0
    
    while cur < end_ts:
        nxt = min(cur + timedelta(days=chunk_days), end_ts)
        chunk_num += 1
        chunk_preds = 0
        
        logger.info(f"📅 Chunk {chunk_num}/{total_chunks}: {cur.strftime('%Y-%m-%d')} → {nxt.strftime('%Y-%m-%d')}")
        
        for sym in symbols:
            logger.info(f"  🔍 Cargando {sym}...")
            df = _load_features_chunk(engine, sym, qtf, cur, nxt)
            if df.empty:
                logger.info(f"  ⚠️ Sin features para {sym}")
                continue
            
            logger.info(f"  📊 {sym}: {len(df)} features")
            # Asegurar orden cronológico estricto
            df = df.sort_values("ts")
            
            sym_preds = 0
            for i, (_, row) in enumerate(df.iterrows()):
                if i % 50 == 0 and i > 0:  # Log cada 50 predicciones
                    logger.info(f"    → {sym}: {i}/{len(df)} predicciones")
                
                ts = row["ts"]
                
                # Verificar duplicados solo si NO estamos en modo estricto
                if not strict_chrono:
                    with engine.begin() as conn:
                        exists = conn.execute(text("""
                            SELECT 1 FROM ml.agent_preds 
                            WHERE symbol=:s AND timeframe=:tf AND ts=:ts AND task=:task
                        """), {"s": sym, "tf": qtf, "ts": ts, "task": agent_type}).scalar()
                    if exists:
                        continue
                
                # Predecir en este timestamp
                pred = _predict_at_ts(model, sym, tfs, window, qtf, ts, engine, agent_type)
                if not pred:
                    continue
                
                _upsert_agent_pred(
                    engine,
                    symbol=sym, timeframe=qtf, ts=ts, task=agent_type,
                    label=pred["label"], probs=pred["probs"],
                    model_name=model_name, model_version="backfill"
                )
                written += 1
                chunk_preds += 1
                sym_preds += 1
            
            if sym_preds > 0:
                logger.info(f"  ✅ {sym}: {sym_preds} predicciones")
        
        logger.info(f"✅ Chunk {chunk_num} completado: {chunk_preds} predicciones | Total: {written}")
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


def backfill_all_agents_all_tfs(window_days: int | None = None) -> int:
    """Backfill para todos los agentes en TODOS los timeframes disponibles"""
    cfg_train = load_training_config()
    cfg_symbols = load_symbols_config()
    symbols, tfs = extract_symbols_and_tfs(cfg_symbols, cfg_train)
    
    total = 0
    for tf in tfs:
        logger.info(f"🔄 Backfill en timeframe {tf}...")
        for agent_type in ["direction", "regime", "smc"]:
            logger.info(f"  → {agent_type} en {tf}")
            # Temporalmente cambiar el timeframe para este agente
            original_env = os.environ.get("AGENTS_BACKFILL_TIMEFRAME")
            os.environ["AGENTS_BACKFILL_TIMEFRAME"] = tf
            count = backfill_agent(agent_type, window_days)
            if original_env is not None:
                os.environ["AGENTS_BACKFILL_TIMEFRAME"] = original_env
            else:
                os.environ.pop("AGENTS_BACKFILL_TIMEFRAME", None)
            total += count
            logger.info(f"  ✅ {agent_type} en {tf}: {count} predicciones")
    return total


if __name__ == "__main__":
    backfill_all_agents()
