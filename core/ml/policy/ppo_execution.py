
"""
PPOExecutionPolicy
==================
Lee:
  - config/train/training.yaml (sección ppo_execution.env/algo + heads.execution.query_tf opcional)
  - market.features (última vela cerrada del TF elegido: precio, ATR, etc.)
  - ml.agent_preds (predicciones más recientes de direction/regime/smc para ese ts)

Escribe:
  - No escribe en BD. Devuelve una decisión de ejecución y parámetros auxiliares.

Qué hace:
  1) Construye una observación vectorizada con:
     - features clave del TF de ejecución (close, atr_14/close, rsi, macd, st_direction, flags SMC)
     - confidencias de las cabezas: direction, regime, smc
  2) Si hay un modelo PPO entrenado (Stable-Baselines3 .zip), lo carga y usa `predict`.
     Si no existe o no hay SB3 instalado → fallback determinístico:
        * Si direction=long & regime=trend & smc=bull → 'open_long'
        * Si direction=short & regime=trend & smc=bear → 'open_short'
        * En otro caso → 'hold'
  3) Devuelve `{"action": str, "limit_offset_bp": int, "meta": {...}}`.

Funciones:
  - build_observation(engine, symbol, timeframe) -> (obs: np.ndarray, ctx: dict)
  - act(obs, ctx) -> dict con 'action' y parámetros
  - map_action(idx) -> str (para modelos PPO con acción discreta)
  - _load_sb3(path) -> objeto policy (si hay SB3), en caso contrario None

Notas:
  - La PPO se entrena fuera. Aquí sólo inferimos.
  - El offset de entrada por defecto se toma de training.yaml (ppo_execution.env.action_space.max_offset_bp).
"""

from __future__ import annotations

import os
import json
import math
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# SB3 es opcional: si no está, usamos fallback
try:
    from stable_baselines3 import PPO  # type: ignore
    _HAS_SB3 = True
except Exception:  # pragma: no cover
    _HAS_SB3 = False

from core.config.config_loader import load_training_config
from core.ml.encoders.multitf_encoder import TF_MS  # para granularidades

load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

logger = logging.getLogger("PPOExecutionPolicy")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


class PPOExecutionPolicy:
    """
    Wrapper de política de ejecución. Puede usar PPO (SB3) si hay modelo; si no, fallback.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.cfg = load_training_config() or {}
        self.engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

        env_cfg = (self.cfg.get("ppo_execution") or {}).get("env", {})
        act_cfg = env_cfg.get("action_space", {})
        self.discrete_actions = act_cfg.get("discrete_actions", ["open_long", "open_short", "hold", "close"])
        self.max_offset_bp = int(act_cfg.get("max_offset_bp", 25))

        heads_cfg = self.cfg.get("heads", {})
        self.query_tf = heads_cfg.get("execution", {}).get("query_tf",
                           (self.cfg.get("encoder", {}) or {}).get("query_tf_default", "1m"))

        # intenta cargar SB3 PPO si se pide path y está disponible
        self.model = None
        if model_path and _HAS_SB3:
            self.model = self._load_sb3(model_path)
            if self.model:
                logger.info(f"SB3 PPO cargado desde: {model_path}")
        elif model_path and not _HAS_SB3:
            logger.warning("Stable-Baselines3 no está instalado. Usaré fallback determinístico.")

    # ------------------- IO BD -------------------
    def _last_closed_ts(self) -> pd.Timestamp:
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        tf_ms = TF_MS.get(self.query_tf, 60_000)
        last_ms = (now_ms // tf_ms) * tf_ms - tf_ms
        return pd.to_datetime(last_ms, unit="ms", utc=True)

    def _fetch_last_features(self, symbol: str) -> Optional[pd.Series]:
        sql = text("""
            SELECT f.ts, h.close, f.rsi_14, f.macd, f.macd_signal, f.macd_hist, f.atr_14, f.st_direction, f.supertrend, f.smc_flags
            FROM market.features f
            JOIN market.historical_data h ON f.symbol = h.symbol AND f.timeframe = h.timeframe AND f.ts = h.ts
            WHERE f.symbol = :s AND f.timeframe = :tf AND f.ts <= :lc
            ORDER BY f.ts DESC
            LIMIT 1
        """)
        lc = self._last_closed_ts()
        with self.engine.begin() as conn:
            row = conn.execute(sql, {"s": symbol, "tf": self.query_tf, "lc": lc}).mappings().first()
        if not row:
            return None
        ser = pd.Series(dict(row))
        if isinstance(ser.get("smc_flags"), str):
            try:
                ser["smc_flags"] = json.loads(ser["smc_flags"])
            except Exception:
                ser["smc_flags"] = {}
        return ser

    def _fetch_preds_at_ts(self, symbol: str, ts) -> Dict[str, Dict]:
        sql = text("""
            SELECT task, pred_label, probs
            FROM ml.agent_preds
            WHERE symbol=:s AND timeframe=:tf AND ts=:ts AND task IN ('direction','regime','smc')
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"s": symbol, "tf": self.query_tf, "ts": ts}).mappings().all()
        out = {}
        for r in rows:
            probs = r["probs"]
            if isinstance(probs, str):
                try:
                    probs = json.loads(probs)
                except Exception:
                    probs = {}
            out[r["task"]] = {"label": r["pred_label"], "probs": probs}
        return out

    # ------------------- Build observation -------------------
    def build_observation(self, symbol: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Devuelve (obs, ctx)
         - obs: vector np.float32 con features normalizadas para la política PPO (o fallback)
         - ctx: dict con info necesaria para construir el plan (ts, close, atr, preds, etc.)
        """
        feat = self._fetch_last_features(symbol)
        if feat is None:
            return None, None

        preds = self._fetch_preds_at_ts(symbol, feat.ts)
        # features numéricos básicos
        close = float(feat.close)
        atr = float(feat.atr_14) if feat.atr_14 is not None else 0.0
        atr_n = atr / max(1e-8, close)

        rsi = float(feat.rsi_14) if feat.rsi_14 is not None else 50.0
        macd = float(feat.macd) if feat.macd is not None else 0.0
        macd_sig = float(feat.macd_signal) if feat.macd_signal is not None else 0.0
        macd_hist = float(feat.macd_hist) if feat.macd_hist is not None else 0.0
        st_dir = float(feat.st_direction) if feat.st_direction is not None else 0.0

        # flags SMC (baseline)
        flags = feat.get("smc_flags") or {}
        f_bos = 1.0 if flags.get("bos") else 0.0
        f_fvg_up = 1.0 if flags.get("fvg_up") else 0.0
        f_fvg_dn = 1.0 if flags.get("fvg_dn") else 0.0
        f_ob_bull = 1.0 if flags.get("ob_bull") else 0.0
        f_ob_bear = 1.0 if flags.get("ob_bear") else 0.0

        # confidencias de heads
        dir_probs = (preds.get("direction") or {}).get("probs", {})
        reg_probs = (preds.get("regime") or {}).get("probs", {})
        smc_probs = (preds.get("smc") or {}).get("probs", {})

        p_long = float(dir_probs.get("long", 0.0))
        p_short = float(dir_probs.get("short", 0.0))
        p_flat = float(dir_probs.get("flat", 0.0))

        p_trend = float(reg_probs.get("trend", 0.0))
        p_range = float(reg_probs.get("range", 0.0))

        p_bull = float(smc_probs.get("bull", 0.0))
        p_bear = float(smc_probs.get("bear", 0.0))

        obs = np.array([
            rsi / 100.0,
            macd, macd_sig, macd_hist,
            atr_n,
            st_dir,          # {-1,1}
            f_bos, f_fvg_up, f_fvg_dn, f_ob_bull, f_ob_bear,
            p_long, p_short, p_flat,
            p_trend, p_range,
            p_bull, p_bear,
        ], dtype=np.float32)

        ctx = {
            "symbol": symbol,
            "timeframe": self.query_tf,
            "ts": pd.Timestamp(feat.ts).to_pydatetime(),
            "close": close,
            "atr": atr,
            "preds": preds,
        }
        return obs, ctx

    # ------------------- Act -------------------
    def act(self, obs: np.ndarray, ctx: Dict) -> Dict:
        """
        Devuelve dict con:
          - action: 'open_long' | 'open_short' | 'hold' | 'close'
          - limit_offset_bp: int (si aplica)
          - meta: info auxiliar (scores del policy, etc.)
        """
        # Con PPO
        if self.model is not None:
            try:
                act_idx, _ = self.model.predict(obs, deterministic=True)
                action = self.map_action(int(act_idx))
                # para el offset, puedes usar red continua aparte o fija:
                offset_bp = min(max(int(self.max_offset_bp * 0.4), 1), self.max_offset_bp)
                return {"action": action, "limit_offset_bp": offset_bp, "meta": {"source": "ppo"}}
            except Exception as e:  # pragma: no cover
                logger.exception(f"Error en predict PPO: {e}")

        # Fallback determinístico (gating con heads)
        preds = ctx.get("preds", {})
        d = (preds.get("direction") or {}).get("label", "")
        r = (preds.get("regime") or {}).get("label", "")
        s = (preds.get("smc") or {}).get("label", "")

        if d == "long" and r == "trend" and s == "bull":
            return {"action": "open_long", "limit_offset_bp": min(10, self.max_offset_bp), "meta": {"source": "gated"}}
        if d == "short" and r == "trend" and s == "bear":
            return {"action": "open_short", "limit_offset_bp": min(10, self.max_offset_bp), "meta": {"source": "gated"}}
        # si direction=flat o regime=range o smc neutral → no hacer nada
        return {"action": "hold", "limit_offset_bp": 0, "meta": {"source": "gated"}}

    # ------------------- Helpers -------------------
    def map_action(self, idx: int) -> str:
        if 0 <= idx < len(self.discrete_actions):
            return self.discrete_actions[idx]
        return "hold"

    def _load_sb3(self, path: str):
        try:
            return PPO.load(path, device="cpu")
        except Exception as e:  # pragma: no cover
            logger.warning(f"No pude cargar PPO desde {path}: {e}")
            return None
