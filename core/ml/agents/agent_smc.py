"""
Agente SMC
==========
Lee:
  - config/train/training.yaml (encoder + heads.smc.query_tf si existe)
  - config/market/symbols.yaml
  - market.features
Escribe:
  - ml.agent_preds (UPSERT) con task='smc' (pred_label bull|bear|neutral)

Funciones:
- run_once(model_name, model_version): mismo patrón que direction/regime.
"""

from __future__ import annotations
import os
import json
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import (
    load_training_config, load_symbols_config,
    build_encoder_config_from_training, extract_symbols_and_tfs, get_window_from_training
)
from core.ml.models.smc import SMCModel

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("AgentSMC")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _get_last_closed_ts(engine, symbol: str, timeframe: str):
    sql = text("""SELECT MAX(ts) AS ts FROM market.features WHERE symbol=:s AND timeframe=:tf""")
    with engine.begin() as conn:
        row = conn.execute(sql, {"s": symbol, "tf": timeframe}).mappings().first()
        return row["ts"] if row and row["ts"] else None


def _upsert_agent_pred(engine, *, symbol: str, timeframe: str, ts, task: str,
                       label: str, probs: dict, model_name: str, model_version: str):
    # Calcular pred_conf en Python para evitar problemas de sintaxis SQL
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


def run_once(model_name: str = "smc_v1", model_version: str = "dev") -> int:
    cfg_train = load_training_config()
    cfg_symbols = load_symbols_config()

    enc_cfg = build_encoder_config_from_training(cfg_train)
    heads_cfg = cfg_train.get("heads", {})
    qtf = heads_cfg.get("smc", {}).get("query_tf", enc_cfg.query_tf_default)

    symbols, tfs = extract_symbols_and_tfs(cfg_symbols, cfg_train)
    window = get_window_from_training(cfg_train)

    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    model = SMCModel(enc_cfg)

    written = 0
    for sym in symbols:
        ts = _get_last_closed_ts(engine, sym, qtf)
        if ts is None:
            logger.info(f"[{sym}] sin features para {qtf}.")
            continue

        pred = model.predict_from_db(symbol=sym, tfs=tfs, window=window, query_tf=qtf)
        if not pred:
            logger.info(f"[{sym}] predict_from_db vacío.")
            continue

        _upsert_agent_pred(
            engine,
            symbol=sym, timeframe=qtf, ts=ts, task="smc",
            label=pred["label"], probs=pred["probs"],
            model_name=model_name, model_version=model_version
        )
        written += 1
        logger.info(f"[{sym}] smc={pred['label']} conf={pred['probs'][pred['label']]:.3f} ts={ts}")

    return written


if __name__ == "__main__":
    run_once()
