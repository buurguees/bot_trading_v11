"""
Common helpers para agentes:
- Conexión DB
- Lectura de símbolos/TF de configs
- Cálculo del último TS cerrado por TF
- UPSERT de predicción en ml.agent_preds
"""

from __future__ import annotations
import os, json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import load_symbols_config, load_training_config, extract_symbols_and_tfs

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")
ENGINE = create_engine(DB_URL, pool_pre_ping=True, future=True)

TF_MS = {"1m":60_000,"5m":300_000,"15m":900_000,"1h":3_600_000,"4h":14_400_000,"1d":86_400_000}

def last_closed_ts(tf: str) -> pd.Timestamp:
    tf_ms = TF_MS.get(tf, 60_000)
    now_ms = int(pd.Timestamp.utcnow().timestamp()*1000)
    last_ms = (now_ms // tf_ms) * tf_ms - tf_ms
    return pd.to_datetime(last_ms, unit="ms", utc=True)

def load_symbols_and_query_tf(task: str, default_tf: str="1m"):
    sym_cfg = load_symbols_config()
    train_cfg = load_training_config()
    symbols, _ = extract_symbols_and_tfs(sym_cfg, train_cfg)
    heads = (train_cfg or {}).get("heads", {})
    qtf = (heads.get(task, {}) or {}).get("query_tf",
          (train_cfg.get("encoder", {}) or {}).get("query_tf_default", default_tf))
    return symbols, qtf

def fetch_feature_row(symbol: str, timeframe: str):
    lc = last_closed_ts(timeframe)
    sql = text("""
        SELECT ts, close, rsi_14, macd_hist, ema_21, ema_50, ema_200, atr_14,
               obv, supertrend_dir, smc_bull, smc_bear
        FROM market.features
        WHERE symbol=:s AND timeframe=:tf AND ts<=:lc
        ORDER BY ts DESC LIMIT 1
    """)
    with ENGINE.begin() as conn:
        row = conn.execute(sql, {"s": symbol, "tf": timeframe, "lc": lc}).mappings().first()
    return dict(row) if row else None

def upsert_pred(ts, symbol, timeframe, task, label, conf: float, probs: dict):
    sql = text("""
      INSERT INTO ml.agent_preds (ts,symbol,timeframe,task,pred_label,pred_conf,probs)
      VALUES (:ts,:s,:tf,:task,:lab,:conf,:probs::jsonb)
      ON CONFLICT (ts,symbol,timeframe,task) DO UPDATE SET
        pred_label=EXCLUDED.pred_label, pred_conf=EXCLUDED.pred_conf, probs=EXCLUDED.probs, created_at=NOW()
    """)
    with ENGINE.begin() as conn:
        conn.execute(sql, {"ts": ts, "s": symbol, "tf": timeframe, "task": task,
                           "lab": label, "conf": float(conf),
                           "probs": json.dumps(probs or {})})
