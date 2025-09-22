"""
Runner Fase 2 (loop infinito)
=============================
Pipeline continuo de minería → backtests → scoring → training → promoción.

Orden (con cadencias independientes):
  - strategy_miner             (PH2_MINER_EVERY_SEC, default 300)
  - strategy_filter            (PH2_FILTER_EVERY_SEC, default 300)
  - backtest.vectorized_runner (PH2_VECBT_EVERY_SEC, default 900)
  - research.ranker (opcional ensembles) (PH2_RANK_EVERY_SEC, default 900)
  - backtest.event_runner      (PH2_EVTBT_EVERY_SEC, default 1800)
  - scoring.scorer             (PH2_SCORE_EVERY_SEC, default 1800)
  - training.trainer_ppo       (PH2_TRAIN_EVERY_SEC, default 3600)
  - training.promotion_manager (PH2_PROMO_EVERY_SEC, default 3600)

ENV toggles:
  PH2_ENABLE_ENSEMBLES=true|false

Usa pg_advisory_lock por ciclo para evitar concurrencia múltiple de este runner.
"""

from __future__ import annotations
import os, sys, time, signal, logging, importlib, subprocess
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
            logger.info(f"{module_path}.{fn_name} OK")
            return True
        # fallback: __main__
        cp = subprocess.run([sys.executable, "-m", module_path], check=False)
        logger.info(f"{module_path} exitcode={cp.returncode}")
        return cp.returncode == 0
    except Exception as e:
        logger.exception(f"Error running {module_path}: {e}")
        return False

def main():
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

    LOOP_SLEEP      = int(os.getenv("PH2_LOOP_SLEEP_SEC", "5"))
    MINER_EVERY     = int(os.getenv("PH2_MINER_EVERY_SEC", "300"))
    FILTER_EVERY    = int(os.getenv("PH2_FILTER_EVERY_SEC", "300"))
    VECBT_EVERY     = int(os.getenv("PH2_VECBT_EVERY_SEC", "900"))
    RANK_EVERY      = int(os.getenv("PH2_RANK_EVERY_SEC", "900"))
    EVTBT_EVERY     = int(os.getenv("PH2_EVTBT_EVERY_SEC", "1800"))
    SCORE_EVERY     = int(os.getenv("PH2_SCORE_EVERY_SEC", "1800"))
    TRAIN_EVERY     = int(os.getenv("PH2_TRAIN_EVERY_SEC", "3600"))
    PROMO_EVERY     = int(os.getenv("PH2_PROMO_EVERY_SEC", "3600"))
    ENABLE_ENSEMBLE = _to_bool(os.getenv("PH2_ENABLE_ENSEMBLES", "false"), False)

    t0 = time.monotonic()
    last = {
        "miner": t0 - MINER_EVERY,    # fuerza primera pasada inmediata
        "filter": t0 - FILTER_EVERY,
        "vecbt": t0 - VECBT_EVERY,
        "rank": t0 - RANK_EVERY,
        "evtbt": t0 - EVTBT_EVERY,
        "score": t0 - SCORE_EVERY,
        "train": t0 - TRAIN_EVERY,
        "promo": t0 - PROMO_EVERY,
    }

    running = True
    def _graceful(sig, frm):
        nonlocal running
        logger.info(f"Signal {sig} received. Shutting down...")
        running = False
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    while running:
        cycle_start = time.monotonic()

        # lock por ciclo
        if not _advisory_lock(engine, key=0xF2A5E002):
            time.sleep(LOOP_SLEEP)
            continue

        try:
            now = time.monotonic()

            if now - last["miner"] >= MINER_EVERY:
                _run_module("core.research.strategy_miner", "mine_candidates")
                last["miner"] = now

            if now - last["filter"] >= FILTER_EVERY:
                _run_module("core.research.strategy_filter", "filter_candidates")
                last["filter"] = now

            if now - last["vecbt"] >= VECBT_EVERY:
                _run_module("core.backtest.vectorized_runner", "run_vectorized")
                last["vecbt"] = now

            if ENABLE_ENSEMBLE and (now - last["rank"] >= RANK_EVERY):
                _run_module("core.research.ranker", "rank_strategies")
                # si implementas build_ensembles, puedes llamarlo aquí también
                last["rank"] = now

            if now - last["evtbt"] >= EVTBT_EVERY:
                _run_module("core.backtest.event_runner", "run_event_driven")
                last["evtbt"] = now

            if now - last["score"] >= SCORE_EVERY:
                _run_module("core.scoring.scorer", "score_and_gate_all")
                last["score"] = now

            if now - last["train"] >= TRAIN_EVERY:
                _run_module("core.training.trainer_ppo", "train_and_register_all")
                last["train"] = now

            if now - last["promo"] >= PROMO_EVERY:
                _run_module("core.training.promotion_manager", "promote_all")
                last["promo"] = now

        finally:
            _advisory_unlock(engine, key=0xF2A5E002)

        time.sleep(max(0.0, LOOP_SLEEP - (time.monotonic() - cycle_start)))


if __name__ == "__main__":
    main()
