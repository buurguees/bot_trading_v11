"""
Agente de Ejecución (RL/PPO)
============================
Lee:
  - config/train/training.yaml (ppo_execution + heads.execution.query_tf + sizing por defecto)
  - config/market/symbols.yaml (símbolos a procesar)
  - market.features (para close/ATR de la última vela cerrada)
  - ml.agent_preds (para las señales direction/regime/smc del mismo ts)

Escribe:
  - trading.trade_plans (INSERT/UPSERT idempotente con plan_id determinista)

Flujo:
  1) Para cada símbolo:
     - Construye observación con PPOExecutionPolicy.build_observation()
     - Obtiene acción con PPOExecutionPolicy.act()
  2) Si la acción es 'open_long' o 'open_short' → genera un plan:
     - entry_type: 'limit' (por defecto) con 'limit_offset_bp'
     - entry_price: close * (1 -/+ offset bps)
     - stop_loss: ATR-multiple (configurable)
     - tp_targets: array JSON (1R y 2R por defecto)
     - qty: sizing por % riesgo sobre distancia al SL
     - valid_until: ts + TTL configurable
     - source/rationale/confidence para auditoría
  3) Inserta en trading.trade_plans (UPSERT por plan_id determinista
     para evitar duplicados si re-ejecutas el agente).

Necesita:
  - Tabla trading.trade_plans ya creada (la que definimos antes).
"""

from __future__ import annotations

import os
import json
import uuid
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from core.config.config_loader import (
    load_training_config, load_symbols_config, extract_symbols_and_tfs,
    get_planner_config, get_execution_config
)
from core.ml.policy.ppo_execution import PPOExecutionPolicy

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("AgentExecution")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


# -------------------- utilidades de sizing / precios --------------------
def _bps_to_multiplier(bps: int, side: str, invert: bool = False) -> float:
    """
    Convierte un offset en basis points a multiplicador de precio.
    side='long' => precio * (1 - bps/10000) si invert=False (limit por debajo)
    side='short'=> precio * (1 + bps/10000) si invert=False (limit por encima)
    """
    m = bps / 10000.0
    if side == "long":
        return 1.0 - m if not invert else 1.0 + m
    else:
        return 1.0 + m if not invert else 1.0 - m


def _risk_qty(close: float, sl: float, risk_pct: float, balance_usdt: float, leverage: float) -> float:
    """
    Sizing por riesgo fijo:
      riesgo_USDT = balance * risk_pct
      distancia   = |close - sl|
      qty (en coin) ~ (riesgo_USDT * leverage) / distancia
    """
    risk_usdt = balance_usdt * max(0.0, risk_pct) / 100.0
    dist = abs(close - sl)
    if dist <= 0:
        return 0.0
    qty = (risk_usdt * leverage) / dist
    return max(0.0, qty)


def _deterministic_plan_id(symbol: str, timeframe: str, ts: datetime, side: str, entry_type: str, entry_price: float, stop_loss: float) -> str:
    """
    Genera un UUID5 determinista para idempotencia del plan.
    """
    seed = f"{symbol}|{timeframe}|{ts.isoformat()}|{side}|{entry_type}|{entry_price:.6f}|{stop_loss:.6f}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


# -------------------- agente --------------------
class ExecutionAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.cfg = load_training_config() or {}
        self.engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
        self.policy = PPOExecutionPolicy(model_path=model_path)

        # Cargar configuración centralizada desde training.yaml
        planner_cfg = get_planner_config(self.cfg)
        execution_cfg = get_execution_config(self.cfg)
        
        # Configuración del planner
        self.ttl_minutes = planner_cfg["ttl_minutes"]
        self.risk_pct = planner_cfg["risk_pct"]
        self.leverage = planner_cfg["leverage"]
        self.balance_usdt = planner_cfg["account_balance_usdt"]
        self.atr_sl_mult = planner_cfg["atr_sl_mult"]
        self.atr_tp_mult_1 = planner_cfg["atr_tp_mult_1"]
        self.atr_tp_mult_2 = planner_cfg["atr_tp_mult_2"]
        
        # Configuración específica de execution
        self.query_tf = execution_cfg["query_tf"]
        self.max_offset_bp = execution_cfg["max_offset_bp"]

    # --------------- escritura en BD ---------------
    def _insert_trade_plan(self, plan: Dict) -> None:
        """
        Inserta/actualiza un plan en trading.trade_plans.
        Usa plan_id determinista para ser idempotente.
        """
        sql = text("""
            INSERT INTO trading.trade_plans
            (symbol, timeframe, ts, plan_id, side, entry_type, entry_price, limit_offset_bp,
             stop_loss, tp_targets, qty, leverage, risk_pct, max_loss_usdt, take_profit_usdt,
             valid_until, route, account_id, source, confidence, rationale, tags, status)
            VALUES
            (:symbol, :timeframe, :ts, :plan_id, :side, :entry_type, :entry_price, :limit_offset_bp,
             :stop_loss, :tp_targets, :qty, :leverage, :risk_pct, :max_loss_usdt, :take_profit_usdt,
             :valid_until, :route, :account_id, :source, :confidence, :rationale, :tags, :status)
            ON CONFLICT (symbol, timeframe, ts, plan_id) DO UPDATE SET
              side            = EXCLUDED.side,
              entry_type      = EXCLUDED.entry_type,
              entry_price     = EXCLUDED.entry_price,
              limit_offset_bp = EXCLUDED.limit_offset_bp,
              stop_loss       = EXCLUDED.stop_loss,
              tp_targets      = EXCLUDED.tp_targets,
              qty             = EXCLUDED.qty,
              leverage        = EXCLUDED.leverage,
              risk_pct        = EXCLUDED.risk_pct,
              max_loss_usdt   = EXCLUDED.max_loss_usdt,
              take_profit_usdt= EXCLUDED.take_profit_usdt,
              valid_until     = EXCLUDED.valid_until,
              route           = EXCLUDED.route,
              account_id      = EXCLUDED.account_id,
              source          = EXCLUDED.source,
              confidence      = EXCLUDED.confidence,
              rationale       = EXCLUDED.rationale,
              tags            = EXCLUDED.tags,
              status          = EXCLUDED.status;
        """)
        with self.engine.begin() as conn:
            # Los campos JSON se pasan directamente como objetos Python
            # SQLAlchemy los serializará automáticamente a JSONB
            conn.execute(sql, plan)

    # --------------- composición de planes ---------------
    def _build_plan_from_action(self, symbol: str, ctx: Dict, decision: Dict) -> Optional[Dict]:
        action = decision.get("action", "hold")
        if action not in ("open_long", "open_short"):
            return None  # sólo planeamos aperturas; cierre lo hará el OMS sobre posiciones

        side = "long" if action == "open_long" else "short"
        offset_bp = int(decision.get("limit_offset_bp", 5))
        close = float(ctx["close"])
        atr = float(ctx["atr"])
        tf = ctx["timeframe"]
        ts = ctx["ts"]

        # Precio de entrada limit: por debajo si long, por encima si short
        entry_price = close * _bps_to_multiplier(offset_bp, side, invert=False)

        # Stop y Take Profits por ATR
        if side == "long":
            sl = entry_price - self.atr_sl_mult * atr
            tp1 = entry_price + self.atr_tp_mult_1 * atr
            tp2 = entry_price + self.atr_tp_mult_2 * atr
        else:
            sl = entry_price + self.atr_sl_mult * atr
            tp1 = entry_price - self.atr_tp_mult_1 * atr
            tp2 = entry_price - self.atr_tp_mult_2 * atr

        # Tamaño por riesgo
        qty = _risk_qty(close=entry_price, sl=sl, risk_pct=self.risk_pct,
                        balance_usdt=self.balance_usdt, leverage=self.leverage)

        # Plan meta
        valid_until = ts + timedelta(minutes=self.ttl_minutes)
        source = {
            "policy": decision.get("meta", {}).get("source", "unknown"),
            "heads": {
                k: {"label": v.get("label"), "conf": max(v.get("probs", {}).get(v.get("label",""), 0.0), 0.0)}
                for k, v in (ctx.get("preds") or {}).items()
            }
        }
        rationale = {
            "query_tf": tf,
            "rules": ["atr_stop", "tp_1R_2R", "offset_limit_bps"],
        }
        tp_targets = json.dumps([
            {"p": round(tp1, 6), "qty_pct": 50},
            {"p": round(tp2, 6), "qty_pct": 50},
        ])

        confidence = 0.0
        try:
            d = ctx.get("preds", {}).get("direction", {})
            s = ctx.get("preds", {}).get("smc", {})
            r = ctx.get("preds", {}).get("regime", {})
            confidence = float(d.get("probs", {}).get(d.get("label",""), 0.0)) * 0.5 \
                       + float(s.get("probs", {}).get(s.get("label",""), 0.0)) * 0.3 \
                       + float(r.get("probs", {}).get(r.get("label",""), 0.0)) * 0.2
        except Exception:
            confidence = 0.5

        plan_id = _deterministic_plan_id(symbol, tf, ts, side, "limit", entry_price, sl)

        plan = {
            "symbol": symbol,
            "timeframe": tf,
            "ts": ts,
            "plan_id": plan_id,
            "side": side,
            "entry_type": "limit",
            "entry_price": round(entry_price, 6),
            "limit_offset_bp": offset_bp,
            "stop_loss": round(sl, 6),
            "tp_targets": tp_targets,
            "qty": qty,
            "leverage": self.leverage,
            "risk_pct": self.risk_pct,
            "max_loss_usdt": abs(entry_price - sl) * qty / max(1e-8, 1.0 / self.leverage),
            "take_profit_usdt": abs(tp1 - entry_price) * qty,  # sólo 1R para estimación
            "valid_until": valid_until,
            "route": "bitget_perp",
            "account_id": "default",
            "source": json.dumps(source),
            "confidence": confidence,
            "rationale": json.dumps(rationale),
            "tags": ["ppo", "auto"],
            "status": "planned",
        }
        return plan

    # --------------- ciclo principal ---------------
    def run_once(self, symbols: Optional[list] = None, model_path: Optional[str] = None) -> int:
        """
        Procesa una pasada por todos los símbolos y genera trade_plans.
        """
        cfg_symbols = load_symbols_config()
        _symbols, _ = extract_symbols_and_tfs(cfg_symbols, self.cfg)
        symbols = symbols or _symbols

        written = 0
        for sym in symbols:
            obs, ctx = self.policy.build_observation(sym)
            if obs is None or ctx is None:
                logger.info(f"[{sym}] sin observación lista (features/preds faltantes).")
                continue

            decision = self.policy.act(obs, ctx)
            plan = self._build_plan_from_action(sym, ctx, decision)
            if not plan:
                logger.info(f"[{sym}] acción={decision.get('action')} → no se genera plan.")
                continue

            self._insert_trade_plan(plan)
            written += 1
            logger.info(f"[{sym}] plan {plan['side']} @ {plan['entry_price']} (ts={ctx['ts']})")

        return written


def run_once() -> int:
    """
    Función de conveniencia para ejecutar el agente una vez.
    """
    agent = ExecutionAgent(model_path=None)  # pon ruta a .zip SB3 si tienes PPO entrenada
    return agent.run_once()


if __name__ == "__main__":
    result = run_once()
    print(f"Plans generated: {result}")
