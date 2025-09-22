#!/usr/bin/env python3
"""
Runner Fase 3 (Entrenamiento PPO Continuo)
=========================================
Entrena y mejora modelos PPO de forma continua para cada símbolo.

Proceso por símbolo:
1. Buscar estrategias 'ready_for_training' ordenadas cronológicamente
2. Entrenar PPO incremental usando datos históricos crecientes
3. Evaluar contra champion actual (promoted)
4. Promover si mejora métricas clave
5. Actualizar estado de estrategias a 'trained'

ENV variables:
  PH3_LOOP_SLEEP_SEC=30         # Pausa entre ciclos
  PH3_TRAINING_EVERY_SEC=1800   # Cadencia de entrenamiento (30min default)
  PH3_MIN_TIMESTEPS=50000       # Timesteps mínimos de entrenamiento
  PH3_MAX_TIMESTEPS=200000      # Timesteps máximos por sesión
  PH3_EVAL_EPISODES=10          # Episodios para evaluación
  PH3_PROMOTION_THRESHOLD=0.05  # Mejora mínima para promoción (5%)

Orden cronológico de entrenamiento:
- Para cada símbolo, entrena con estrategias más antiguas primero
- Acumula aprendizaje sobre diferentes condiciones de mercado
- Permite especialización por símbolo manteniendo generalización temporal
"""

from __future__ import annotations
import os, sys, time, signal, logging, json, uuid
from datetime import datetime, timezone
from typing import Optional, Dict, List
from collections import defaultdict
import numpy as np

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Setup (añadir raíz al path antes de importar core.*)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from core.training.ppo_trainer import train_ppo_for_strategy
from core.environments.trading_env import TradingEnv
from core.config.config_loader import load_training_config

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("RunnerPhase3")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

# Configuration from ENV
LOOP_SLEEP = int(os.getenv("PH3_LOOP_SLEEP_SEC", "30"))
TRAINING_EVERY = int(os.getenv("PH3_TRAINING_EVERY_SEC", "1800"))  # 30 minutes
MIN_TIMESTEPS = int(os.getenv("PH3_MIN_TIMESTEPS", "50000"))
MAX_TIMESTEPS = int(os.getenv("PH3_MAX_TIMESTEPS", "200000"))
EVAL_EPISODES = int(os.getenv("PH3_EVAL_EPISODES", "10"))
PROMOTION_THRESHOLD = float(os.getenv("PH3_PROMOTION_THRESHOLD", "0.05"))

def _advisory_lock(engine, key: int) -> bool:
    with engine.begin() as conn:
        got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar()
    return bool(got)

def _advisory_unlock(engine, key: int):
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})

def get_ready_strategies_by_symbol(engine) -> Dict[str, List[Dict]]:
    """
    Obtiene estrategias ready_for_training agrupadas por símbolo,
    ordenadas cronológicamente para entrenamiento incremental.
    """
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT strategy_id, symbol, timeframe, created_at, metrics_summary
            FROM ml.strategies 
            WHERE status = 'ready_for_training'
            ORDER BY symbol, created_at ASC
        """)).mappings().all()
    
    strategies_by_symbol = defaultdict(list)
    for row in rows:
        strategies_by_symbol[row["symbol"]].append({
            "strategy_id": str(row["strategy_id"]),  # Convertir UUID a string
            "symbol": row["symbol"], 
            "timeframe": row["timeframe"],
            "created_at": row["created_at"],
            "metrics_summary": row["metrics_summary"]
        })
    
    return dict(strategies_by_symbol)

def load_historical_data(engine, symbol: str, timeframe: str, features: List[str]) -> pd.DataFrame:
    """
    Carga datos históricos completos para entrenamiento.
    """
    feat_cols = ", ".join([f"f.{c}" for c in features if c not in ("close",)])
    feat_cols = feat_cols if feat_cols else ""
    
    sql = f"""
        SELECT f.ts, h.close{', ' + feat_cols if feat_cols else ''}
        FROM market.features f
        JOIN market.historical_data h ON f.symbol=h.symbol AND f.timeframe=h.timeframe AND f.ts=h.ts
        WHERE f.symbol=:s AND f.timeframe=:tf
        ORDER BY f.ts ASC
    """
    
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"s": symbol, "tf": timeframe}).mappings().all()
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df

def evaluate_model(model, env, episodes: int = 10) -> Dict:
    """
    Evalúa un modelo PPO en el environment.
    Retorna métricas de rendimiento promedio.
    """
    episode_returns = []
    episode_lengths = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_return += reward
            episode_length += 1
            
            if episode_length >= 1000:  # Max episode length safety
                break
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    
    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "total_episodes": episodes,
        "sharpe": float(np.mean(episode_returns) / (np.std(episode_returns) + 1e-8))
    }

def get_current_champion(engine, symbol: str) -> Optional[Dict]:
    """
    Obtiene el agente campeón actual para un símbolo.
    """
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT agent_id, artifact_uri, metrics, promoted_at
            FROM ml.agents
            WHERE symbol = :s AND task = 'execution' AND status = 'promoted'
            ORDER BY promoted_at DESC
            LIMIT 1
        """), {"s": symbol}).mappings().first()
    
    if row:
        return {
            "agent_id": row["agent_id"],
            "artifact_uri": row["artifact_uri"],
            "metrics": json.loads(row["metrics"]) if isinstance(row["metrics"], str) else row["metrics"],
            "promoted_at": row["promoted_at"]
        }
    return None

def should_promote(challenger_metrics: Dict, champion_metrics: Optional[Dict]) -> bool:
    """
    Decide si el challenger debe ser promovido basado en métricas.
    """
    if not champion_metrics:
        return True  # Si no hay campeón, promover automáticamente
    
    challenger_return = challenger_metrics.get("mean_return", 0.0)
    champion_return = champion_metrics.get("mean_return", 0.0)
    
    # Requiere mejora mínima del 5% (configurable)
    improvement = (challenger_return - champion_return) / max(abs(champion_return), 1e-8)
    
    return improvement > PROMOTION_THRESHOLD

def promote_agent(engine, symbol: str, agent_id: str, metrics: Dict) -> bool:
    """
    Promueve un agente a campeón y archiva el anterior.
    """
    try:
        with engine.begin() as conn:
            # Archivar campeón anterior
            conn.execute(text("""
                UPDATE ml.agents 
                SET status = 'shadow', promoted_at = NULL
                WHERE symbol = :s AND task = 'execution' AND status = 'promoted'
            """), {"s": symbol})
            
            # Promover nuevo campeón
            conn.execute(text("""
                UPDATE ml.agents
                SET status = 'promoted', promoted_at = NOW(), metrics = :metrics
                WHERE agent_id = :aid
            """), {"aid": agent_id, "metrics": json.dumps(metrics)})
            
            logger.info(f"[{symbol}] Promovido agente {agent_id[:8]} con return {metrics.get('mean_return', 0):.4f}")
            return True
            
    except Exception as e:
        logger.error(f"Error promoviendo agente {agent_id}: {e}")
        return False

def train_symbol_strategies(engine, symbol: str, strategies: List[Dict]) -> int:
    """
    Entrena estrategias de un símbolo en orden cronológico.
    """
    if not strategies:
        return 0
    
    logger.info(f"[{symbol}] Entrenando {len(strategies)} estrategias...")
    
    # Configuración de entrenamiento
    cfg = load_training_config()
    env_cfg = cfg.get("ppo_execution", {}).get("env", {})
    features = env_cfg.get("observation_space", {}).get("features", ["close", "atr_14"])
    
    # Cargar datos históricos completos
    first_timeframe = strategies[0]["timeframe"]  # Usar TF de primera estrategia
    df = load_historical_data(engine, symbol, first_timeframe, features)
    
    if len(df) < 1000:
        logger.warning(f"[{symbol}] Datos insuficientes: {len(df)} barras")
        return 0
    
    trained_count = 0
    
    for strategy in strategies:
        try:
            strategy_id = strategy["strategy_id"]
            logger.info(f"[{symbol}] Entrenando estrategia {strategy_id[:8]}")
            
            # Calcular timesteps basado en data disponible
            timesteps = min(MAX_TIMESTEPS, max(MIN_TIMESTEPS, len(df) * 10))
            
            # Entrenar modelo
            result = train_ppo_for_strategy(strategy_id, total_timesteps=timesteps)
            
            if result.get("trained", False):
                # Evaluar modelo entrenado
                try:
                    from stable_baselines3 import PPO
                    model = PPO.load(result["artifact_uri"])
                    
                    # Crear environment para evaluación
                    eval_env = TradingEnv(
                        df=df.tail(2000),  # Evaluar en últimas 2000 barras
                        lookback=60,
                        fee=0.001,
                        slippage=0.0005,
                        features=features
                    )
                    
                    eval_metrics = evaluate_model(model, eval_env, episodes=EVAL_EPISODES)
                    
                    # Registrar agente candidato
                    agent_id = str(uuid.uuid4())
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO ml.agents
                            (agent_id, symbol, task, version, artifact_uri, train_run_ref, metrics, status)
                            VALUES (:aid, :sym, 'execution', 'v1', :uri, :ref, :metrics, 'candidate')
                        """), {
                            "aid": agent_id,
                            "sym": symbol,
                            "uri": result["artifact_uri"],
                            "ref": strategy_id,
                            "metrics": json.dumps(eval_metrics)
                        })
                    
                    # Comparar con campeón actual
                    champion = get_current_champion(engine, symbol)
                    if should_promote(eval_metrics, champion["metrics"] if champion else None):
                        promote_agent(engine, symbol, agent_id, eval_metrics)
                    
                    # Marcar estrategia como entrenada
                    with engine.begin() as conn:
                        conn.execute(text("""
                            UPDATE ml.strategies 
                            SET status = 'trained', updated_at = NOW()
                            WHERE strategy_id = :sid
                        """), {"sid": strategy_id})
                    
                    trained_count += 1
                    logger.info(f"[{symbol}] Estrategia {strategy_id[:8]} entrenada exitosamente")
                    
                except Exception as e:
                    logger.error(f"Error evaluando modelo para {strategy_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error entrenando estrategia {strategy_id}: {e}")
            continue
    
    return trained_count

def main():
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    running = True
    last_training = 0
    
    def _graceful(signum, frame):
        nonlocal running
        logger.info("Shutting down Phase 3...")
        running = False
    
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)
    
    logger.info("🤖 Phase 3: Entrenamiento PPO Continuo iniciado")
    
    while running:
        cycle_start = time.monotonic()
        
        # Lock para evitar múltiples instancias
        if not _advisory_lock(engine, key=0xF3A5E003):
            time.sleep(LOOP_SLEEP)
            continue
        
        try:
            now = time.monotonic()
            
            if now - last_training >= TRAINING_EVERY:
                logger.info("🔄 Iniciando ciclo de entrenamiento...")
                
                # Obtener estrategias por símbolo
                strategies_by_symbol = get_ready_strategies_by_symbol(engine)
                
                if strategies_by_symbol:
                    total_trained = 0
                    
                    # Entrenar cada símbolo en orden alfabético (para consistencia)
                    for symbol in sorted(strategies_by_symbol.keys()):
                        strategies = strategies_by_symbol[symbol]
                        trained = train_symbol_strategies(engine, symbol, strategies)
                        total_trained += trained
                        
                        if not running:  # Check si se interrumpió
                            break
                    
                    logger.info(f"✅ Ciclo completado: {total_trained} estrategias entrenadas")
                else:
                    logger.info("⏳ No hay estrategias ready_for_training")
                
                last_training = now
        
        except Exception as e:
            logger.error(f"Error en ciclo de entrenamiento: {e}")
        
        finally:
            _advisory_unlock(engine, key=0xF3A5E003)
        
        # Sleep hasta próximo ciclo
        elapsed = time.monotonic() - cycle_start
        time.sleep(max(0.0, LOOP_SLEEP - elapsed))

if __name__ == "__main__":
    main()