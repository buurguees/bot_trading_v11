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
from core.config.config_loader import load_training_config

logger = logging.getLogger("PPOTrainer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")


def _load_df(engine, symbol: str, timeframe: str, features: list[str]) -> pd.DataFrame:
    sql = text(
        """
        SELECT f.ts, h.close, {feat_cols}
        FROM market.features f
        JOIN market.historical_data h ON f.symbol=h.symbol AND f.timeframe=h.timeframe AND f.ts=h.ts
        WHERE f.symbol=:s AND f.timeframe=:tf
        ORDER BY f.ts ASC
        """
    )
    feat_cols = ", ".join([f"f.{c}" for c in features if c not in ("close",)])
    q = text(sql.text.format(feat_cols=feat_cols if feat_cols else ""))
    with engine.begin() as conn:
        rows = conn.execute(q, {"s": symbol, "tf": timeframe}).mappings().all()
    if not rows:
        cols = ["ts","close"] + [c for c in features if c != "close"]
        return pd.DataFrame(columns=cols).astype({"ts":"datetime64[ns]"})
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def train_ppo_for_strategy(strategy_id: str, total_timesteps: int = 100_000) -> Dict:
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        logger.warning(f"Stable-Baselines3 no disponible: {e}. Saltando entrenamiento.")
        return {"trained": False, "reason": "sb3_missing"}

    cfg = load_training_config()
    env_cfg = cfg.get("ppo_execution", {}).get("env", {})
    obs_cfg = env_cfg.get("observation_space", {})
    features = obs_cfg.get("features", ["close", "atr_14"])  # fallback
    lookback = int(obs_cfg.get("lookback_steps", 60))
    normalization = str(obs_cfg.get("normalization", "zscore"))
    reward_cfg = env_cfg.get("reward", {})
    risk_penalty = float(reward_cfg.get("risk_penalty", 0.0))
    costs = env_cfg.get("costs", {})
    fee = float(costs.get("taker_fee_bp", 1)) / 10_000.0
    slippage = float(costs.get("slippage_bp", 1)) / 10_000.0

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
    df = _load_df(engine, symbol, timeframe, features=features)
    if len(df) < 500:
        return {"trained": False, "reason": "insufficient_data"}

    env = TradingEnv(df=df, lookback=lookback, fee=fee, slippage=slippage,
                     features=features, normalization=normalization, risk_penalty=risk_penalty)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)

    artifact_uri = os.path.join("agents", f"{symbol}_PPO.zip")
    os.makedirs("agents", exist_ok=True)
    model.save(artifact_uri)

    # evaluación básica: retorno medio por episodio (simulación simple)
    # Nota: aquí podría usarse un conjunto de evaluación; por simplicidad, omitimos.
    metrics = {"episodes": 1, "note": "basic_train_only"}
    return {"trained": True, "artifact_uri": artifact_uri, "metrics": metrics}


