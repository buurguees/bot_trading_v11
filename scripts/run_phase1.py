"""
Runner Fase 1 (loop infinito)
=============================
Arranca y supervisa:
  - core.data.realtime_updater         (daemon de datos OHLCV en tiempo real)
  - core.ml.feature_updater            (daemon de features en tiempo real)

Ejecuta periódicamente (en orden):
  - core.ml.agents.agent_direction     (predicciones -> ml.agent_preds)
  - core.ml.agents.agent_regime
  - core.ml.agents.agent_smc
  - core.trading.planner (opcional)    (planes -> trading.trade_plans)
  - core.ml.agents.agent_execution (opcional, PPO/heurístico)
  - core.trading.risk_manager          (filtra -> queued/invalid)
  - core.trading.oms.router            (paper/live -> orders/fills/positions)

ENV (opcionales):
  PH1_LOOP_SLEEP_SEC=5
  PH1_PREDICT_CADENCE_SEC=30
  PH1_PLANNER_CADENCE_SEC=30
  PH1_RISK_CADENCE_SEC=5
  PH1_OMS_CADENCE_SEC=5
  PH1_ENABLE_PLANNER=true|false
  PH1_ENABLE_EXECUTION_AGENT=true|false
  PH1_START_REALTIME_UPDATER=true|false
  PH1_START_FEATURE_UPDATER=true|false
  OMS_MODE=sim|live

Requisitos:
  - DB_URL en config/.env
  - Los módulos importados deben exponer run_once() o tener entrypoint -m
"""

from __future__ import annotations
import os, sys, time, signal, subprocess, logging, importlib
from datetime import datetime
from typing import Optional, List

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ---- paths & env ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

# ---- logging ----
logger = logging.getLogger("RunnerPhase1")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

# ---- helpers ----
def _to_bool(v: Optional[str], default: bool) -> bool:
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def _advisory_lock(engine, key: int) -> bool:
    # key must fit in BIGINT; use a fixed value for phase1
    with engine.begin() as conn:
        got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar()
    return bool(got)

def _advisory_unlock(engine, key: int):
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})

def _spawn_module(mod: str, args: Optional[List[str]] = None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", mod]
    if args: cmd += args
    logger.info(f"Spawning: {' '.join(cmd)}")
    # Start detached so we can restart if it dies
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _ensure_proc(name: str, proc: Optional[subprocess.Popen], mod: str, args: Optional[List[str]] = None) -> subprocess.Popen:
    if proc is None or proc.poll() is not None:
        if proc and proc.poll() is not None:
            logger.warning(f"{name} exited with code {proc.returncode}, restarting...")
        return _spawn_module(mod, args)
    return proc

def _try_run_once(module_path: str) -> bool:
    """
    Intenta importar y ejecutar run_once(); si falla, intenta -m module (one-shot si lo soporta).
    """
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, "run_once", None)
        if callable(fn):
            n = fn()  # puede devolver número de filas/planes/etc.
            logger.info(f"{module_path}.run_once() -> {n}")
            return True
    except Exception as e:
        logger.exception(f"Error importing {module_path}: {e}")
    # fallback: lanzar como módulo (si su __main__ hace una pasada corta)
    try:
        cp = subprocess.run([sys.executable, "-m", module_path], check=False)
        logger.info(f"{module_path} exitcode={cp.returncode}")
        return cp.returncode == 0
    except Exception as e:
        logger.exception(f"Cannot exec -m {module_path}: {e}")
        return False

# ---- main loop ----
def main():
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

    # ENV cadencias y toggles
    LOOP_SLEEP   = int(os.getenv("PH1_LOOP_SLEEP_SEC", "5"))
    PRED_EVERY   = int(os.getenv("PH1_PREDICT_CADENCE_SEC", "30"))
    PLAN_EVERY   = int(os.getenv("PH1_PLANNER_CADENCE_SEC", "30"))
    RISK_EVERY   = int(os.getenv("PH1_RISK_CADENCE_SEC", "5"))
    OMS_EVERY    = int(os.getenv("PH1_OMS_CADENCE_SEC", "5"))

    EN_PLANNER   = _to_bool(os.getenv("PH1_ENABLE_PLANNER", "true"), True)
    EN_EXEC_AGENT= _to_bool(os.getenv("PH1_ENABLE_EXECUTION_AGENT", "true"), True)
    ST_RT_UPD    = _to_bool(os.getenv("PH1_START_REALTIME_UPDATER", "true"), True)
    ST_FE_UPD    = _to_bool(os.getenv("PH1_START_FEATURE_UPDATER", "true"), True)
    OMS_MODE     = os.getenv("OMS_MODE", "sim").lower()

    # child daemons
    p_realtime = None
    p_features = None

    # timers
    t0 = time.monotonic()
    last_pred = last_plan = last_risk = last_oms = t0

    running = True
    def _graceful(sig, frm):
        nonlocal running
        logger.info(f"Signal {sig} received. Shutting down...")
        running = False
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    while running:
        cycle_start = time.monotonic()

        # aseguramos lock por ciclo
        if not _advisory_lock(engine, key=0xF1A5E001):
            # Otro runner activo: dormimos y reintentamos
            time.sleep(LOOP_SLEEP)
            continue

        try:
            # 1) asegurar daemons
            if ST_RT_UPD:
                p_realtime = _ensure_proc("realtime_updater", p_realtime, "core.data.realtime_updater")
            if ST_FE_UPD:
                # si tienes core.ml.feature_updater como módulo daemonizado; si no, usa feature_engineer con --realtime
                try:
                    p_features = _ensure_proc("feature_updater", p_features, "core.ml.feature_updater")
                except Exception:
                    p_features = _ensure_proc("feature_engineer", p_features, "core.ml.feature_engineer", ["--realtime"])

            now = time.monotonic()

            # 2) Predicciones heads
            if now - last_pred >= PRED_EVERY:
                _try_run_once("core.ml.agents.agent_direction")
                _try_run_once("core.ml.agents.agent_regime")
                _try_run_once("core.ml.agents.agent_smc")
                last_pred = now

            # 3) Planner / Execution agent -> trade_plans
            if now - last_plan >= PLAN_EVERY:
                if EN_PLANNER:
                    _try_run_once("core.trading.planner")
                if EN_EXEC_AGENT:
                    _try_run_once("core.ml.agents.agent_execution")
                last_plan = now

            # 4) Risk manager
            if now - last_risk >= RISK_EVERY:
                _try_run_once("core.trading.risk_manager")
                last_risk = now

            # 5) OMS (paper/live)
            if now - last_oms >= OMS_EVERY:
                # pasamos modo con ENV OMS_MODE; si el router usa __main__ con arg, mejor llamar run_once()
                ok = _try_run_once("core.trading.oms.router")
                if not ok:
                    # fallback a -m con arg (no estándar, por si lo añadiste)
                    subprocess.run([sys.executable, "-m", "core.trading.oms.router", f"--mode={OMS_MODE}"], check=False)
                last_oms = now

        finally:
            _advisory_unlock(engine, key=0xF1A5E001)

        # dormir hasta siguiente tick
        time.sleep(max(0.0, LOOP_SLEEP - (time.monotonic() - cycle_start)))

    # intentar parar hijos
    for p in (p_realtime, p_features):
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
