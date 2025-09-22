"""
core/training/ppo_trainer.py
----------------------------
Entrenamiento PPO real con Stable-Baselines3 si está disponible.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.environments.trading_env import TradingEnv

logger = logging.getLogger("PPOTrainer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")


def _load_df(engine, symbol: str, timeframe: str) -> pd.DataFrame:
    sql = text(
        """
        SELECT f.ts, h.close, f.atr_14
        FROM market.features f
        JOIN market.historical_data h ON f.symbol=h.symbol AND f.timeframe=h.timeframe AND f.ts=h.ts
        WHERE f.symbol=:s AND f.timeframe=:tf
        ORDER BY f.ts ASC
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"s": symbol, "tf": timeframe}).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["ts","close","atr_14"]).astype({"ts":"datetime64[ns]"})
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def train_ppo_for_strategy(strategy_id: str, total_timesteps: int = 100_000) -> Dict:
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        logger.warning(f"Stable-Baselines3 no disponible: {e}. Saltando entrenamiento.")
        return {"trained": False, "reason": "sb3_missing"}

    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    with engine.begin() as conn:
        r = conn.execute(text(
            """
            SELECT symbol, timeframe FROM ml.strategies WHERE strategy_id=:sid
            """
        ), {"sid": strategy_id}).mappings().first()
    if not r:
        raise ValueError("strategy_id no encontrado")

    symbol = r["symbol"]; timeframe = r["timeframe"]
    df = _load_df(engine, symbol, timeframe)
    if len(df) < 500:
        return {"trained": False, "reason": "insufficient_data"}

    env = TradingEnv(df=df, lookback=60)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)

    artifact_uri = os.path.join("agents", f"{symbol}_PPO.zip")
    os.makedirs("agents", exist_ok=True)
    model.save(artifact_uri)

    return {"trained": True, "artifact_uri": artifact_uri}


