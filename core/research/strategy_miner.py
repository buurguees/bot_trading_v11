"""
Strategy Miner
--------------
Lee:
  - trading.trade_plans (ventana configurable o TODO el histórico)
  - (opcional) ml.agent_preds para recuperar 'heads' si no están en source
Escribe:
  - ml.strategies (status='candidate'), actualizando o creando por (symbol,timeframe,strategy_key)

Qué hace:
  - Normaliza cada trade_plan a una "fingerprint" de estrategia (rules + risk_profile).
  - Calcula un strategy_key (hash estable) y agrega soporte (nº ocurrencias).
  - Upserta en ml.strategies con metrics_summary preliminar: {support_n, first_seen, last_seen}.

Funciones:
  - mine_candidates(window_days:int=None) -> int  # nº estrategias nuevas/actualizadas
  - _fingerprint(plan_row: Mapping) -> (strategy_key:str, rules:dict, risk_profile:dict)

Nota:
  - Si window_days=None, lee la configuración desde training.yaml (strategy_mining.window_days)
  - Si training.yaml indica full_history (strategy_mining.full_history=true) o window_days <= 0,
    entonces usa TODO el histórico disponible de trading.trade_plans (desde MIN(ts)).
  - Si no hay configuración, usa default_years_history de config/market/symbols.yaml.
"""

from __future__ import annotations
import os, json, logging, hashlib
from datetime import timedelta
from typing import Dict, Mapping, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import load_training_config
import yaml

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("StrategyMiner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


def _json_hash(obj: dict) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _safe_json(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return {}


def _fingerprint(plan: Mapping) -> Tuple[str, Dict, Dict]:
    """
    Construye 'rules' y 'risk_profile' "genéricos" para agrupar planes equivalentes.
    No usa precios exactos: usa offsets/relaciones para robustez.

    Devuelve:
      - strategy_key: hash estable de (rules + risk_profile)
      - rules: side, entry_type, limit_offset_bp, sl_relation, tp_scheme, gating...
      - risk_profile: leverage, risk_pct, tags
    """
    # Gating heads (si está en source)
    source = _safe_json(plan.get("source"))
    heads = source.get("heads") or source.get("preds") or {}

    rules = {
        "side": plan["side"],                           # 'long'|'short'
        "entry_type": plan["entry_type"],               # 'limit'|'market'|...
        "limit_offset_bp": plan.get("limit_offset_bp") or 0,
        # Relación SL/entrada en bps (aprox robusto)
        "sl_bp": None,
        "tp_scheme": "2_targets_equal_50",              # baseline (puedes enriquecerlo)
        "gating": {                                     # etiquetas de heads (si existen)
            "direction": (heads.get("direction") or {}).get("label") if isinstance(heads.get("direction"), dict) else heads.get("direction"),
            "regime":    (heads.get("regime")    or {}).get("label") if isinstance(heads.get("regime"), dict) else heads.get("regime"),
            "smc":       (heads.get("smc")       or {}).get("label") if isinstance(heads.get("smc"), dict) else heads.get("smc"),
        },
    }

    entry = float(plan.get("entry_price") or 0.0)
    sl    = float(plan.get("stop_loss") or 0.0)
    if entry > 0 and sl > 0:
        bp = abs(entry - sl) / entry * 10000.0
        rules["sl_bp"] = round(bp, 1)

    risk_profile = {
        "leverage": float(plan.get("leverage") or 1.0),
        "risk_pct": float(plan.get("risk_pct") or 0.0),
        "tags": plan.get("tags") or [],
        "route": plan.get("route") or "bitget_perp",
    }

    key_obj = {"rules": rules, "risk_profile": risk_profile}
    strategy_key = _json_hash(key_obj)
    return strategy_key, rules, risk_profile


def _default_years_history() -> int:
    try:
        with open(os.path.join("config","market","symbols.yaml"), "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        return int(y.get("default_years_history", 2))
    except Exception:
        return 2


def mine_candidates(window_days: int = None) -> int:
    """
    Busca trade_plans recientes, agrega por fingerprint y upserta ml.strategies.
    Si window_days es None, lee la configuración desde training.yaml.
    """
    # Si no se especifica window_days, leer desde training.yaml
    use_full_history = False
    if window_days is None:
        try:
            cfg = load_training_config()
            sm = cfg.get("strategy_mining", {}) if cfg else {}
            use_full_history = bool(sm.get("full_history", False))
            window_days = sm.get("window_days")
        except Exception as e:
            logger.warning(f"Error cargando training.yaml: {e}")
            window_days = None

    # Determinar 'since'
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    since = None
    if use_full_history or (window_days is not None and window_days <= 0):
        # MIN(ts) en trade_plans
        with engine.begin() as conn:
            row = conn.execute(text("SELECT MIN(ts) AS first_ts FROM trading.trade_plans" )).mappings().first()
        if not row or not row["first_ts"]:
            logger.info("No hay trade_plans en la base de datos.")
            return 0
        since = pd.to_datetime(row["first_ts"])  # tz-naive o tz-aware, SQLA lo maneja
        logger.info("Minería con TODO el histórico: desde %s", since)
    else:
        # Si no hay window_days en training.yaml, usar años por defecto del YAML de símbolos
        if window_days is None:
            years = _default_years_history()
            window_days = int(years * 365)
        since = pd.Timestamp.utcnow() - timedelta(days=int(window_days))
        logger.info("Minería con ventana de %d días (desde %s)", window_days, since)
    
    sql = text("""
        SELECT symbol, timeframe, ts, plan_id, side, entry_type, entry_price, limit_offset_bp,
               stop_loss, tp_targets, leverage, risk_pct, route, account_id, source, tags
        FROM trading.trade_plans
        WHERE ts >= :since
          AND status IN ('planned','sent','filled')   -- puedes ajustar
    """)
    with engine.begin() as conn:
        rows = [dict(r) for r in conn.execute(sql, {"since": since}).mappings().all()]

    if not rows:
        logger.info("No hay trade_plans en ventana.")
        return 0

    # Agregación por (symbol, timeframe, strategy_key)
    bucket: Dict[tuple, Dict] = {}
    for r in rows:
        skey, rules, rprof = _fingerprint(r)
        key = (r["symbol"], r["timeframe"], skey)
        b = bucket.get(key)
        if not b:
            bucket[key] = {
                "symbol": r["symbol"], "timeframe": r["timeframe"],
                "strategy_key": skey, "rules": rules, "risk_profile": rprof,
                "support_n": 1, "first_seen": r["ts"], "last_seen": r["ts"],
                "description": f"auto: {rules['side']} limit {rules['limit_offset_bp']}bp, SL≈{rules['sl_bp']}bp",
                "source": "planner",
                "tags": list(set((r.get("tags") or []) + ["auto-mined"]))
            }
        else:
            b["support_n"] += 1
            if r["ts"] < b["first_seen"]: b["first_seen"] = r["ts"]
            if r["ts"] > b["last_seen"]:  b["last_seen"] = r["ts"]

    # Upsert en ml.strategies (sin UNIQUE por strategy_key, hacemos SELECT/UPDATE o INSERT)
    upserted = 0
    with engine.begin() as conn:
        for (sym, tf, skey), v in bucket.items():
            # ¿Existe ya?
            row = conn.execute(text("""
                SELECT created_at, strategy_id, metrics_summary, status
                FROM ml.strategies
                WHERE symbol=:s AND timeframe=:tf AND strategy_key=:k
                ORDER BY created_at DESC
                LIMIT 1
            """), {"s": sym, "tf": tf, "k": skey}).mappings().first()

            metrics = (row and row["metrics_summary"]) or {}
            if isinstance(metrics, str):
                try: metrics = json.loads(metrics)
                except Exception: metrics = {}

            # merge de soporte
            metrics["support_n"] = int(metrics.get("support_n", 0)) + v["support_n"]
            metrics["first_seen"] = str(min(pd.Timestamp(metrics.get("first_seen") or v["first_seen"]), pd.Timestamp(v["first_seen"])))
            metrics["last_seen"]  = str(max(pd.Timestamp(metrics.get("last_seen")  or v["last_seen"]),  pd.Timestamp(v["last_seen"])))

            if row:
                conn.execute(text("""
                    UPDATE ml.strategies
                    SET description=:d, source=:src, rules=:rules, risk_profile=:rprof,
                        metrics_summary=:m, tags=:tags, status=CASE WHEN status='candidate' THEN 'candidate' ELSE status END,
                        updated_at=NOW()
                    WHERE strategy_id=:sid
                """), {
                    "d": v["description"], "src": v["source"],
                    "rules": json.dumps(v["rules"]), "rprof": json.dumps(v["risk_profile"]),
                    "m": json.dumps(metrics), "tags": v["tags"], "sid": row["strategy_id"]
                })
            else:
                conn.execute(text("""
                    INSERT INTO ml.strategies
                      (symbol, timeframe, strategy_key, description, source, rules, risk_profile, metrics_summary, status, tags)
                    VALUES
                      (:s, :tf, :k, :d, :src, :rules, :rprof, :m, 'candidate', :tags)
                """), {
                    "s": sym, "tf": tf, "k": skey, "d": v["description"], "src": v["source"],
                    "rules": json.dumps(v["rules"]), "rprof": json.dumps(v["risk_profile"]),
                    "m": json.dumps(metrics), "tags": v["tags"]
                })
            upserted += 1

    logger.info(f"Estrategias minadas/upsertadas: {upserted}")
    return upserted


if __name__ == "__main__":
    mine_candidates(30)
