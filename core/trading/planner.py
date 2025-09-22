"""
Planner (Capa 6)
================
Lee:
  - ml.agent_preds (última predicción por símbolo/TF base)
  - market.features (para close/ATR de la vela cerrada en TF de ejecución)
  - config/train/training.yaml (query_tf por head opcional; parámetros genéricos)
Escribe:
  - trading.trade_plans (INSERT/UPSERT) con status='planned'

Qué hace:
  - Gating jerárquico simple y explicable: si (direction.conf >= thr_dir) & (regime == 'trend') &
    (smc == 'bull' / 'bear') -> propone plan long/short.
  - Calcula entry_limit mediante offset en bps; SL/TP por múltiplos de ATR; sizing por riesgo.
  - Genera fields de explicación (source/rationale/confidence).

Funciones:
  - fetch_latest_preds(engine, symbol, qtf) -> dict por task
  - fetch_last_feature_row(engine, symbol, qtf) -> Series(close, atr, ts)
  - build_plan(symbol, preds, feat) -> dict plan listo para UPSERT
  - upsert_plan(engine, plan)
"""

from __future__ import annotations

import os, json, uuid, logging
from datetime import timedelta
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import load_training_config, load_symbols_config, extract_symbols_and_tfs, get_planner_config, load_risk_config_yaml
from core.ml.encoders.multitf_encoder import TF_MS
from core.ml.policy.ppo_execution import PPOExecutionPolicy   # para utilidades (_bps_to_multiplier)

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("Planner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)

def _last_closed_ts(tf: str) -> pd.Timestamp:
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    tf_ms = TF_MS.get(tf, 60_000)
    last_ms = (now_ms // tf_ms) * tf_ms - tf_ms
    return pd.to_datetime(last_ms, unit="ms", utc=True)

def fetch_latest_preds(engine, symbol: str, qtf: str) -> Dict:
    sql = text("""
        SELECT task, pred_label, pred_conf, probs
        FROM ml.agent_preds
        WHERE symbol=:s AND timeframe=:tf AND ts = (
            SELECT MAX(ts) FROM ml.agent_preds WHERE symbol=:s AND timeframe=:tf
        )
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"s": symbol, "tf": qtf}).mappings().all()
    out = {}
    for r in rows:
        probs = r["probs"]
        if isinstance(probs, str):
            try: probs = json.loads(probs)
            except Exception: probs = {}
        out[r["task"]] = {"label": r["pred_label"], "conf": float(r.get("pred_conf") or 0), "probs": probs}
    return out

def fetch_last_feature_row(engine, symbol: str, qtf: str) -> Optional[Dict]:
    lc = _last_closed_ts(qtf)
    sql = text("""
        SELECT f.ts, f.atr_14, h.close
        FROM market.features f
        JOIN market.historical_data h ON f.symbol = h.symbol 
            AND f.timeframe = h.timeframe 
            AND f.ts = h.ts
        WHERE f.symbol=:s AND f.timeframe=:tf AND f.ts<=:lc
        ORDER BY f.ts DESC LIMIT 1;
    """)
    with engine.begin() as conn:
        row = conn.execute(sql, {"s": symbol, "tf": qtf, "lc": lc}).mappings().first()
    return dict(row) if row else None

def _bps_to_multiplier(bps: int, side: str, invert: bool = False) -> float:
    m = bps/10000.0
    return (1.0 - m if side=="long" else 1.0 + m) if not invert else (1.0 + m if side=="long" else 1.0 - m)

def _risk_qty(close: float, sl: float, risk_pct: float, balance_usdt: float, leverage: float) -> float:
    dist = abs(close - sl)
    if dist <= 0: return 0.0
    risk_usdt = balance_usdt * risk_pct / 100.0
    return max(0.0, (risk_usdt * leverage) / dist)

def _conf(preds: Dict, key: str) -> float:
    if key not in preds: return 0.0
    lab = preds[key]["label"]
    return float((preds[key]["probs"] or {}).get(lab, preds[key].get("conf", 0.0)))

def build_plan(symbol: str, qtf: str, preds: Dict, feat: Dict, cfg: Dict) -> Optional[Dict]:
    # Cargar configuración del planner desde risk.yaml
    risk_cfg = load_risk_config_yaml()
    planner_cfg = risk_cfg.get("planner", {})
    thresholds = planner_cfg.get("thresholds", {})
    gating = planner_cfg.get("gating", {})
    
    # Umbrales configurables
    thr_dir = thresholds.get("direction_confidence_min", 0.30)
    regime_required = thresholds.get("regime_required", "trend")
    smc_alignment = thresholds.get("smc_alignment", True)
    
    # gating
    d = preds.get("direction", {}).get("label")
    r = preds.get("regime", {}).get("label")
    s = preds.get("smc", {}).get("label")
    
    # Verificaciones de gating
    if not (d and r and s): return None
    if r != regime_required: return None
    
    if smc_alignment:
        direction_long_smc = gating.get("direction_long_smc", "bull")
        direction_short_smc = gating.get("direction_short_smc", "bear")
        if d == "long" and s != direction_long_smc: return None
        if d == "short" and s != direction_short_smc: return None
    
    if _conf(preds, "direction") < thr_dir: return None

    close = float(feat["close"])
    atr   = float(feat.get("atr_14") or 0.0)
    ts    = pd.Timestamp(feat["ts"]).to_pydatetime()

    # Cargar configuración del planner desde training.yaml
    planner_cfg = get_planner_config(cfg)
    
    # Parámetros básicos
    ppo_env = (cfg.get("ppo_execution") or {}).get("env", {})
    max_offset_bp = int(ppo_env.get("action_space", {}).get("max_offset_bp", 25))
    offset_bp = min(10, max_offset_bp)

    side = "long" if d=="long" else "short"
    entry_price = close * _bps_to_multiplier(offset_bp, side, invert=False)
    
    # Usar configuración del planner
    atr_sl_mult = planner_cfg["atr_sl_mult"]
    atr_tp_mult_1 = planner_cfg["atr_tp_mult_1"]
    atr_tp_mult_2 = planner_cfg["atr_tp_mult_2"]
    
    if side=="long":
        sl = entry_price - atr_sl_mult*atr
        tp1 = entry_price + atr_tp_mult_1*atr
        tp2 = entry_price + atr_tp_mult_2*atr
    else:
        sl = entry_price + atr_sl_mult*atr
        tp1 = entry_price - atr_tp_mult_1*atr
        tp2 = entry_price - atr_tp_mult_2*atr

    leverage = planner_cfg["leverage"]
    balance = planner_cfg["account_balance_usdt"]
    risk_pct = planner_cfg["risk_pct"]
    qty = _risk_qty(entry_price, sl, risk_pct, balance, leverage)

    plan = {
      "symbol": symbol, "timeframe": qtf, "ts": ts,
      "plan_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{symbol}|{qtf}|{ts.isoformat()}|planner")),
      "side": side, "entry_type": "limit",
      "entry_price": round(entry_price,6), "limit_offset_bp": offset_bp,
      "stop_loss": round(sl,6),
      "tp_targets": json.dumps([{"p": round(tp1,6), "qty_pct": 50},
                                {"p": round(tp2,6), "qty_pct": 50}]),
      "qty": qty, "leverage": leverage, "risk_pct": risk_pct,
      "max_loss_usdt": abs(entry_price-sl)*qty,
      "take_profit_usdt": abs(tp1-entry_price)*qty,
      "valid_until": ts + timedelta(minutes=planner_cfg["ttl_minutes"]),
      "route": "bitget_perp", "account_id": "default",
      "source": json.dumps({"from": "planner", "preds": {k: v["label"] for k,v in preds.items()}}),
      "confidence": (_conf(preds,"direction")*0.5 + _conf(preds,"smc")*0.3 + _conf(preds,"regime")*0.2),
      "rationale": json.dumps({"gating": "dir>=thr & regime=trend & smc aligned"}),
      "tags": ["planner","gated"], "status": "planned"
    }
    return plan

def upsert_plan(engine, plan: Dict):
    sql = text("""
      INSERT INTO trading.trade_plans
      (symbol,timeframe,ts,plan_id,side,entry_type,entry_price,limit_offset_bp,stop_loss,tp_targets,
       qty,leverage,risk_pct,max_loss_usdt,take_profit_usdt,valid_until,route,account_id,
       source,confidence,rationale,tags,status)
      VALUES
      (:symbol,:timeframe,:ts,:plan_id,:side,:entry_type,:entry_price,:limit_offset_bp,:stop_loss,:tp_targets,
       :qty,:leverage,:risk_pct,:max_loss_usdt,:take_profit_usdt,:valid_until,:route,:account_id,
       :source,:confidence,:rationale,:tags,:status)
      ON CONFLICT (symbol,timeframe,ts,plan_id) DO UPDATE SET
        side=EXCLUDED.side, entry_type=EXCLUDED.entry_type, entry_price=EXCLUDED.entry_price,
        limit_offset_bp=EXCLUDED.limit_offset_bp, stop_loss=EXCLUDED.stop_loss, tp_targets=EXCLUDED.tp_targets,
        qty=EXCLUDED.qty, leverage=EXCLUDED.leverage, risk_pct=EXCLUDED.risk_pct,
        max_loss_usdt=EXCLUDED.max_loss_usdt, take_profit_usdt=EXCLUDED.take_profit_usdt,
        valid_until=EXCLUDED.valid_until, route=EXCLUDED.route, account_id=EXCLUDED.account_id,
        source=EXCLUDED.source, confidence=EXCLUDED.confidence, rationale=EXCLUDED.rationale,
        tags=EXCLUDED.tags, status=EXCLUDED.status;
    """)
    with engine.begin() as conn:
        conn.execute(sql, plan)

def run_once() -> int:
    cfg = load_training_config()
    cfg_symbols = load_symbols_config()
    symbols, _ = extract_symbols_and_tfs(cfg_symbols, cfg)

    heads = cfg.get("heads", {})
    qtf = heads.get("execution", {}).get("query_tf",
          (cfg.get("encoder", {}) or {}).get("query_tf_default", "1m"))

    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    written = 0
    for sym in symbols:
        preds = fetch_latest_preds(engine, sym, qtf)
        feat  = fetch_last_feature_row(engine, sym, qtf)
        if not preds or not feat: 
            logger.info(f"[{sym}] sin preds/feat para {qtf}")
            continue
        plan = build_plan(sym, qtf, preds, feat, cfg)
        if not plan: 
            continue
        upsert_plan(engine, plan); written += 1
        logger.info(f"[{sym}] plan {plan['side']} @ {plan['entry_price']} ts={plan['ts']}")
    return written

if __name__ == "__main__":
    run_once()
