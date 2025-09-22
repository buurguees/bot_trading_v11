"""
Runner Fase 2 (loop infinito) - ACTUALIZADO SIN ENTRENAMIENTO PPO
================================================================
Pipeline continuo de minería → backtests → scoring → (preparación para Phase 3).

Orden (con cadencias independientes):
  - strategy_miner             (PH2_MINER_EVERY_SEC, default 300)
  - strategy_filter            (PH2_FILTER_EVERY_SEC, default 300)  
  - backtest.vectorized_runner (PH2_VECBT_EVERY_SEC, default 900)
  - research.ranker (opcional ensembles) (PH2_RANK_EVERY_SEC, default 900)
  - backtest.event_runner      (PH2_EVTBT_EVERY_SEC, default 1800)
  - scoring.scorer             (PH2_SCORE_EVERY_SEC, default 1800)
  
REMOVIDO:
  - training.trainer_ppo       -> Movido a Phase 3
  - training.promotion_manager -> Movido a Phase 3

La Phase 2 ahora se enfoca en:
1. Minería continua de estrategias
2. Filtrado y backtesting
3. Scoring y preparación para entrenamiento
4. Las estrategias llegan hasta 'ready_for_training'
5. Phase 3 se encarga del entrenamiento y promoción

ENV toggles:
  PH2_ENABLE_ENSEMBLES=true|false

Usa pg_advisory_lock por ciclo para evitar concurrencia múltiple de este runner.
"""

from __future__ import annotations
import os, sys, time, signal, logging, importlib
from typing import Optional

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("RunnerPhase2")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

def _to_bool(v: Optional[str], default: bool) -> bool:
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def _advisory_lock(engine, key: int) -> bool:
    with engine.begin() as conn:
        got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar()
    return bool(got)

def _advisory_unlock(engine, key: int):
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})

def _run_module(module_path: str, fn_name: str | None = None) -> bool:
    try:
        mod = importlib.import_module(module_path)
        if fn_name and hasattr(mod, fn_name):
            getattr(mod, fn_name)()
            logger.info(f"{module_path}.{fn_name}() OK")
        else:
            logger.info(f"{module_path} OK")
        return True
    except Exception as e:
        logger.error(f"Error running {module_path}: {e}")
        return False

def main():
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    running = True
    
    # Configuración de cadencias (en segundos)
    LOOP_SLEEP = int(os.getenv("PH2_LOOP_SLEEP_SEC", "5"))
    MINER_EVERY = int(os.getenv("PH2_MINER_EVERY_SEC", "5"))         # 5 seg (más frecuente para minería)
    FILTER_EVERY = int(os.getenv("PH2_FILTER_EVERY_SEC", "30"))      # 30 seg
    VECBT_EVERY = int(os.getenv("PH2_VECBT_EVERY_SEC", "60"))        # 1 min
    RANK_EVERY = int(os.getenv("PH2_RANK_EVERY_SEC", "900"))         # 15 min
    EVTBT_EVERY = int(os.getenv("PH2_EVTBT_EVERY_SEC", "1800"))      # 30 min
    SCORE_EVERY = int(os.getenv("PH2_SCORE_EVERY_SEC", "120"))       # 2 min
    
    # Configuración de features opcionales
    ENABLE_ENSEMBLE = _to_bool(os.getenv("PH2_ENABLE_ENSEMBLES"), False)
    ENABLE_BACKFILL = _to_bool(os.getenv("PLANNER_BACKFILL_ENABLE"), False)
    ENABLE_AGENTS_BACKFILL = _to_bool(os.getenv("AGENTS_BACKFILL_ENABLE"), False)
    
    # Tracking de última ejecución por módulo
    last = {
        "miner": 0,
        "filter": 0, 
        "vecbt": 0,
        "rank": 0,
        "evtbt": 0,
        "score": 0,
        "backfill": 0,
        "agents_backfill": 0
    }
    
    def _graceful(signum, frame):
        nonlocal running
        logger.info("Phase 2 shutting down...")
        running = False
    
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)
    
    logger.info("🔄 Phase 2: Strategy Mining & Backtesting iniciado")
    logger.info("⚠️  Entrenamiento PPO movido a Phase 3")
    
    while running:
        cycle_start = time.monotonic()
        
        # Lock por ciclo para evitar concurrencia múltiple
        if not _advisory_lock(engine, key=0xF2A5E002):
            time.sleep(LOOP_SLEEP)
            continue
        
        try:
            now = time.monotonic()
            
            # 0. Agents Backfill (una sola vez si está habilitado)
            if ENABLE_AGENTS_BACKFILL and last["agents_backfill"] == 0:
                logger.info("🔄 Ejecutando backfill de predicciones históricas...")
                _run_module("core.ml.agents.agents_backfill", "backfill_all_agents")
                last["agents_backfill"] = now
                logger.info("✅ Agents backfill completado")
            
            # 0.1. Planner Backfill (una sola vez si está habilitado)
            if ENABLE_BACKFILL and last["backfill"] == 0:
                logger.info("🔄 Ejecutando backfill de planes históricos...")
                _run_module("core.trading.planner_backfill", "backfill")
                last["backfill"] = now
                logger.info("✅ Planner backfill completado")
            
            # 1. Strategy Mining (cada 5 segundos)
            if now - last["miner"] >= MINER_EVERY:
                _run_module("core.research.strategy_miner", "mine_candidates")
                last["miner"] = now
            
            # 2. Strategy Filtering (cada 5 minutos)
            if now - last["filter"] >= FILTER_EVERY:
                _run_module("core.research.strategy_filter", "filter_candidates")
                last["filter"] = now
            
            # 3. Vectorized Backtesting (cada 15 minutos)
            if now - last["vecbt"] >= VECBT_EVERY:
                _run_module("core.backtest.vectorized_runner", "run_vectorized")
                last["vecbt"] = now
            
            # 4. Research Ranking (opcional, cada 15 minutos)
            if ENABLE_ENSEMBLE and (now - last["rank"] >= RANK_EVERY):
                _run_module("core.research.ranker", "rank_strategies")
                # Si implementas build_ensembles, puedes llamarlo aquí también
                last["rank"] = now
            
            # 5. Event-driven Backtesting (cada 30 minutos)
            if now - last["evtbt"] >= EVTBT_EVERY:
                _run_module("core.backtest.event_runner", "run_event_driven")
                last["evtbt"] = now
            
            # 6. Scoring & Gating (cada 30 minutos)
            if now - last["score"] >= SCORE_EVERY:
                _run_module("core.scoring.scorer", "score_and_gate_all")
                last["score"] = now
                
                # Log estado después del scoring
                with engine.begin() as conn:
                    counts = conn.execute(text("""
                        SELECT status, COUNT(*) as count 
                        FROM ml.strategies 
                        GROUP BY status
                    """)).mappings().all()
                    
                    status_summary = {r["status"]: r["count"] for r in counts}
                    ready_count = status_summary.get("ready_for_training", 0)
                    
                    if ready_count > 0:
                        logger.info(f"📋 Estrategias ready_for_training: {ready_count} (listas para Phase 3)")
                    else:
                        logger.info("⏳ Sin estrategias ready_for_training todavía")
        
        except Exception as e:
            logger.error(f"Error en ciclo Phase 2: {e}")
        
        finally:
            _advisory_unlock(engine, key=0xF2A5E002)
        
        # Sleep hasta el próximo ciclo
        elapsed = time.monotonic() - cycle_start
        time.sleep(max(0.0, LOOP_SLEEP - elapsed))

if __name__ == "__main__":
    main()