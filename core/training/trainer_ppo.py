"""
Trainer PPO (multitf, placeholder de integración)
-------------------------------------------------
Lee:
  - ml.strategies (status='ready_for_training') por símbolo
  - market.features (para construir observaciones)
  - config/backtest/backtest.yaml (costes) y config/train/training.yaml (arquitectura, si entrenas de verdad)

Escribe:
  - agents/{SYMBOL}_PPO.zip (staging)  [placeholder si no hay entrenamiento real]
  - ml.agents (candidate) con métricas de validación/backtest

Qué hace:
  - Prepara datos y (si tienes SB3) entrena PPO. Aquí dejamos una rutina de "registro"
    que simula el artefacto y registra al candidato con las métricas de la estrategia.

Funciones:
  - register_from_strategy(strategy_id:str) -> agent_id
  - train_and_register_all() -> int
"""

from __future__ import annotations
import os, json, uuid, logging

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import load_backtest_config, load_training_config
from core.training.ppo_trainer import train_ppo_for_strategy

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("TrainerPPO")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

AGENTS_DIR = "agents"


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def register_from_strategy(strategy_id: str) -> str:
    """
    Toma metrics_summary de la estrategia y registra un agente 'candidate'.
    Crea/actualiza agents/{SYMBOL}_PPO.zip como 'staging' (placeholder).
    """
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    _ensure_dir(AGENTS_DIR)

    with engine.begin() as conn:
        r = conn.execute(text("""
            SELECT symbol, timeframe, metrics_summary
            FROM ml.strategies
            WHERE strategy_id=:sid
        """), {"sid": strategy_id}).mappings().first()
        if not r:
            raise ValueError("strategy_id no encontrado")

        sym = r["symbol"]; ms = r["metrics_summary"]
        ms = json.loads(ms) if isinstance(ms, str) else (ms or {})

        # decidir si entrenar realmente PPO
        cfg = load_training_config()
        enable_rl = bool(cfg.get("ppo_execution", {}).get("enable", False))
        artifact_uri = os.path.join(AGENTS_DIR, f"{sym}_PPO.zip")
        os.makedirs(AGENTS_DIR, exist_ok=True)
        if enable_rl:
            try:
                res = train_ppo_for_strategy(strategy_id=strategy_id, total_timesteps=int(cfg.get("ppo_execution", {}).get("training", {}).get("total_timesteps", 100000)))
                if res.get("trained") and res.get("artifact_uri"):
                    artifact_uri = res["artifact_uri"]
                else:
                    # fallback placeholder
                    with open(artifact_uri, "wb") as f:
                        f.write(b"placeholder-ppo-artifact")
            except Exception as e:
                logger.warning(f"Fallo entrenando PPO real: {e}. Usando placeholder.")
                with open(artifact_uri, "wb") as f:
                    f.write(b"placeholder-ppo-artifact")
        else:
            # placeholder si el flag está deshabilitado
            with open(artifact_uri, "wb") as f:
                f.write(b"placeholder-ppo-artifact")

        agent_id = str(uuid.uuid4())
        # si el entrenador devolvió métricas, úsalas; si no, usa las de estrategia
        mt_payload = ms
        try:
            if enable_rl and 'res' in locals() and isinstance(res, dict) and res.get('metrics'):
                mt_payload = res['metrics']
        except Exception:
            pass

        conn.execute(text("""
            INSERT INTO ml.agents
              (agent_id, symbol, task, version, components, artifact_uri, train_run_ref, metrics, status)
            VALUES
              (:aid, :sym, 'execution', 'v0', '{}'::jsonb, :uri, :ref, :mt, 'candidate')
        """), {"aid": agent_id, "sym": sym, "uri": artifact_uri, "ref": strategy_id, "mt": mt_payload})

    logger.info(f"Agente candidate registrado para {sym}: {agent_id}")
    return agent_id


def train_and_register_all() -> int:
    """
    Registra agentes 'candidate' a partir de todas las estrategias 'ready_for_training'.
    """
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    with engine.begin() as conn:
        rows = [dict(r) for r in conn.execute(text("""
            SELECT strategy_id FROM ml.strategies WHERE status='ready_for_training'
        """)).mappings().all()]

    n = 0
    for r in rows:
        register_from_strategy(r["strategy_id"])
        n += 1

    logger.info(f"Agentes registrados: {n}")
    return n


if __name__ == "__main__":
    train_and_register_all()
