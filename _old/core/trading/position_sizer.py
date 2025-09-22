import os, math, yaml
from dataclasses import dataclass
from typing import Dict, Any

DEFAULTS = {
    "equity": 10_000.0,
    "margin_mode": "isolated",
    "per_trade_pct": {"1m": 0.003, "5m": 0.003},
    "k_sl_atr":      {"1m": 1.5,   "5m": 1.5},
    "k_tp_atr":      {"1m": 2.0,   "5m": 2.0},
    "leverage_max":  {"1m": 3.0,   "5m": 3.0},
    "liq_buffer_atr":{"1m": 3.0,   "5m": 3.0},
}

@dataclass
class RiskParams:
    equity: float
    risk_pct: float
    k_sl: float
    k_tp: float
    lev_max: float
    liq_buf_atr: float
    margin_mode: str

def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def load_risk_params(tf: str, path: str = "config/risk.yaml") -> RiskParams:
    cfg = DEFAULTS.copy()
    user = _read_yaml(path)
    # mezcla superficial
    for k, v in (user or {}).items():
        if isinstance(v, dict) and k in cfg:
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v

    equity    = float(cfg.get("equity", DEFAULTS["equity"]))
    risk_pct  = float(cfg["per_trade_pct"].get(tf, list(cfg["per_trade_pct"].values())[0]))
    k_sl      = float(cfg["k_sl_atr"].get(tf, list(cfg["k_sl_atr"].values())[0]))
    k_tp      = float(cfg["k_tp_atr"].get(tf, list(cfg["k_tp_atr"].values())[0]))
    lev_max   = float(cfg["leverage_max"].get(tf, list(cfg["leverage_max"].values())[0]))
    liq_buf   = float(cfg["liq_buffer_atr"].get(tf, list(cfg["liq_buffer_atr"].values())[0]))
    margin_md = str(cfg.get("margin_mode", "isolated")).lower()
    return RiskParams(equity, risk_pct, k_sl, k_tp, lev_max, liq_buf, margin_md)

def plan_from_price(entry_px: float, atr: float, side: int, tf: str, rp: RiskParams) -> Dict[str, Any]:
    """
    Devuelve SL/TP, qty y leverage calculados con reglas deterministas.
    """
    if atr <= 0 or entry_px <= 0:
        raise ValueError("ATR y precio de entrada deben ser positivos.")

    stop_dist = rp.k_sl * atr
    tp_dist   = rp.k_tp * atr

    sl_px = entry_px - stop_dist if side == 1 else entry_px + stop_dist
    tp_px = entry_px + tp_dist   if side == 1 else entry_px - tp_dist

    risk_cash = rp.risk_pct * rp.equity
    qty = risk_cash / stop_dist  # riesgo monetario / distancia al stop

    # leverage implícito y recorte por límite
    notional = qty * entry_px
    lev = notional / rp.equity if rp.equity > 0 else 0.0
    if lev > rp.lev_max and lev > 0:
        scale = rp.lev_max / lev
        qty *= scale
        notional = qty * entry_px
        lev = rp.lev_max

    # buffer de “liquidación” aproximado con ATR (conservador)
    min_stop_dist = rp.liq_buf_atr * atr
    if abs(entry_px - sl_px) < min_stop_dist:
        # aleja SL lo suficiente y ajusta qty para mantener el mismo risk_cash
        sl_px = entry_px - min_stop_dist if side == 1 else entry_px + min_stop_dist
        stop_dist = min_stop_dist
        qty = risk_cash / stop_dist
        notional = qty * entry_px
        lev = min(rp.lev_max, (notional / rp.equity) if rp.equity > 0 else 0.0)

    return {
        "entry_px": float(entry_px),
        "sl_px":    float(sl_px),
        "tp_px":    float(tp_px),
        "risk_pct": float(rp.risk_pct),
        "qty":      float(qty),
        "leverage": float(lev),
        "margin_mode": rp.margin_mode,
        "params_used": {
            "tf": tf, "k_sl": rp.k_sl, "k_tp": rp.k_tp,
            "lev_max": rp.lev_max, "liq_buf_atr": rp.liq_buf_atr, "equity": rp.equity
        }
    }
